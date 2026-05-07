"""
Ollama Client Adapter
=====================

What this file does:

- Connects to a locally running Ollama server (http://localhost:11434)
- Supports multiple LLM models for experimentation (qwen, llama3, mistral, phi3, etc.)
- Sends chat-style prompts to Ollama
- Implements:
    • Increased timeout (prevents long-response timeouts)
    • Automatic retry logic
    • Clean error handling
- Returns only the final model text response (non-streaming)

This module is used by the GuardrailSystem to:
    1) Generate RAW LLM output
    2) Generate Guarded (RAG + Safety) output
"""

import os
import requests
import time

OLLAMA_HOST = "http://localhost:11434"

# HF Inference API model IDs (runs on HF servers, fast)
HF_MODELS = {
    "qwen0.5":   "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5":   "Qwen/Qwen2.5-7B-Instruct",
    "llama3":    "meta-llama/Llama-3.2-3B-Instruct",
    "mistral":   "mistralai/Mistral-7B-Instruct-v0.3",
    "phi3":      "microsoft/Phi-3-mini-4k-instruct",
    "gemma3:1b": "google/gemma-3-1b-it",
    "gemma:2b":  "google/gemma-2-2b-it",
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
    """Call HuggingFace Inference API — fast, runs on HF servers."""
    token = os.getenv("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN not set")

    url = f"https://api-inference.huggingface.co/models/{hf_model}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "model": hf_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }

    for attempt in range(3):
        try:
            session = requests.Session()
            session.headers.update({"Connection": "close"})
            resp = session.post(url, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[HF Retry {attempt+1}/3] Error: {e}")
            if attempt < 2:
                time.sleep(3)

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

    # Use HF API for all models if HF_TOKEN is available (much faster)
    if model_name in HF_MODELS and os.getenv("HF_TOKEN"):
        try:
            print(f"[HF API] Using HuggingFace for {model_name} ({HF_MODELS[model_name]})")
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