"""
Ollama Client Adapter
=====================

Priority order:
  1. Groq API (GROQ_API_KEY set) — fastest, 30 req/min free tier
  2. HuggingFace Router API (HF_TOKEN set) — fast but rate limited
  3. Ollama local fallback — slow but always available

This module is used by the GuardrailSystem to:
    1) Generate RAW LLM output
    2) Generate Guarded (RAG + Safety) output
"""

import os
import requests
import time

OLLAMA_HOST = "http://localhost:11434"

# Groq model IDs (fast, free tier, 30 req/min)
GROQ_MODELS = {
    "qwen0.5":   "llama-3.1-8b-instant",
    "qwen2.5":   "llama-3.3-70b-versatile",
    "llama3":    "llama-3.1-8b-instant",
    "mistral":   "llama-3.1-8b-instant",
    "phi3":      "llama-3.1-8b-instant",
    "gemma3:1b": "llama-3.1-8b-instant",
    "gemma:2b":  "llama-3.1-8b-instant",
}

# HF Router fallback model IDs (only confirmed working ones)
HF_MODELS = {
    "qwen0.5":   "Qwen/Qwen2.5-7B-Instruct:together",
    "qwen2.5":   "Qwen/Qwen2.5-7B-Instruct:together",
    "llama3":    "meta-llama/Llama-3.1-8B-Instruct:novita",
}

# Ollama fallback model names
SUPPORTED_MODELS = {
    "qwen0.5":   "qwen:0.5b",
    "qwen2.5":   "qwen:0.5b",
    "llama3":    "llama3.2:3b",
    "mistral":   "qwen:0.5b",
    "phi3":      "qwen:0.5b",
    "gemma3:1b": "qwen:0.5b",
    "gemma:2b":  "qwen:0.5b",
}


def _groq_generate(prompt: str, system: str, groq_model: str, max_tokens: int) -> str:
    """Call Groq API — very fast, free tier, no HTTP/2 issues."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": groq_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }

    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[Groq Retry {attempt+1}/3] Error: {e}")
            if attempt < 2:
                time.sleep(2)

    raise RuntimeError("Groq API failed after retries")


def _hf_generate(prompt: str, system: str, hf_model: str, max_tokens: int) -> str:
    """Call HuggingFace Router API via huggingface_hub InferenceClient."""
    token = os.getenv("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN not set")

    from huggingface_hub import InferenceClient
    client = InferenceClient(api_key=token)

    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=hf_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"[HF Retry {attempt+1}/3] Error: {e}")
            if attempt < 2:
                time.sleep(2)

    raise RuntimeError("HF API failed after retries")


def ollama_generate(
    prompt: str,
    model_name: str = "qwen0.5",
    system: str = "You are a helpful assistant.",
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> str:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Model '{model_name}' not supported. "
            f"Choose from {list(SUPPORTED_MODELS.keys())}"
        )

    # 1. Groq (fastest, preferred)
    if os.getenv("GROQ_API_KEY") and model_name in GROQ_MODELS:
        try:
            print(f"[Groq] Using Groq for {model_name} ({GROQ_MODELS[model_name]})")
            return _groq_generate(prompt, system, GROQ_MODELS[model_name], max_tokens)
        except Exception as e:
            print(f"[Groq] Failed ({e}), trying HF API")

    # 2. HuggingFace Router
    if model_name in HF_MODELS and os.getenv("HF_TOKEN"):
        try:
            print(f"[HF API] Using HuggingFace router for {model_name} ({HF_MODELS[model_name]})")
            return _hf_generate(prompt, system, HF_MODELS[model_name], max_tokens)
        except Exception as e:
            print(f"[HF API] Failed ({e}), falling back to Ollama")

    # 3. Ollama local fallback
    model = SUPPORTED_MODELS[model_name]
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    last_error = None
    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, timeout=600)
            response.raise_for_status()
            return response.json()["message"]["content"].strip()
        except Exception as e:
            last_error = e
            print(f"[Ollama Retry {attempt + 1}/3] Error: {e}")
            time.sleep(2)

    raise RuntimeError(f"Ollama request failed after retries: {last_error}")
