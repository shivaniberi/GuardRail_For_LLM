"""
Ollama Client Adapter
=====================

What this file does:

- Primary: HuggingFace Router API via huggingface_hub.InferenceClient
  Uses :novita provider suffix for fast serverless inference
- Fallback: Locally running Ollama server (http://localhost:11434)
- Supports multiple LLM models (qwen, llama3, mistral, phi3, etc.)
- Implements retry logic and clean error handling
- Returns only the final model text response (non-streaming)

This module is used by the GuardrailSystem to:
    1) Generate RAW LLM output
    2) Generate Guarded (RAG + Safety) output
"""

import os
import requests
import time

OLLAMA_HOST = "http://localhost:11434"

# HF Router model IDs with :novita provider suffix (fast, free tier)
HF_MODELS = {
    "qwen0.5":   "Qwen/Qwen2.5-0.5B-Instruct:novita",
    "qwen2.5":   "Qwen/Qwen2.5-7B-Instruct:novita",
    "llama3":    "meta-llama/Llama-3.1-8B-Instruct:novita",
    "mistral":   "mistralai/Mistral-7B-Instruct-v0.3:novita",
    "phi3":      "meta-llama/Llama-3.1-8B-Instruct:novita",
    "gemma3:1b": "meta-llama/Llama-3.1-8B-Instruct:novita",
    "gemma:2b":  "meta-llama/Llama-3.1-8B-Instruct:novita",
}

# Ollama fallback model names (used when HF_TOKEN not set or HF call fails)
SUPPORTED_MODELS = {
    "qwen0.5":   "qwen:0.5b",
    "qwen2.5":   "qwen:0.5b",
    "llama3":    "llama3.2:3b",
    "mistral":   "qwen:0.5b",
    "phi3":      "qwen:0.5b",
    "gemma3:1b": "qwen:0.5b",
    "gemma:2b":  "qwen:0.5b",
}


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

    # Use HF Router API if HF_TOKEN is available (much faster than Ollama)
    if model_name in HF_MODELS and os.getenv("HF_TOKEN"):
        try:
            print(f"[HF API] Using HuggingFace router for {model_name} ({HF_MODELS[model_name]})")
            return _hf_generate(prompt, system, HF_MODELS[model_name], max_tokens)
        except Exception as e:
            print(f"[HF API] Failed ({e}), falling back to Ollama")

    # Ollama fallback
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
