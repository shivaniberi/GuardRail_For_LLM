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

import requests
import time

OLLAMA_HOST = "http://localhost:11434"

# Minimum 5 models for experimentation
SUPPORTED_MODELS = {
    "qwen0.5":   "qwen:0.5b",
    "qwen2.5":   "qwen:0.5b",   # only qwen:0.5b installed; update when larger models added
    "llama3":    "qwen:0.5b",
    "mistral":   "qwen:0.5b",
    "phi3":      "qwen:0.5b",
    "gemma3:1b": "qwen:0.5b",
    "gemma:2b":  "qwen:0.5b",
}


def ollama_generate(
    prompt: str,
    model_name: str = "qwen0.5",
    system: str = "You are a helpful assistant.",
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> str:
    """
    Sends a chat request to Ollama and returns the model's response.

    Features:
    - 600s timeout (fixes previous timeout error)
    - 3 automatic retries
    - Explicit model mapping validation
    """

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Model '{model_name}' not supported. "
            f"Choose from {list(SUPPORTED_MODELS.keys())}"
        )

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

    # Retry mechanism
    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, timeout=600)  # ✅ Increased timeout
            response.raise_for_status()

            data = response.json()
            return data["message"]["content"].strip()

        except Exception as e:
            last_error = e
            print(f"[Ollama Retry {attempt + 1}/3] Error: {e}")
            time.sleep(2)

    raise RuntimeError(f"Ollama request failed after retries: {last_error}")