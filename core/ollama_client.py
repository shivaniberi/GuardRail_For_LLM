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
import re
import requests
import time


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> chain-of-thought blocks from Qwen3/DeepSeek responses.
    If stripping leaves nothing (model put everything inside think tags), return the
    content of the last think block as a fallback so the response is never empty."""
    original = text
    stripped = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE).strip()
    stripped = re.sub(r'<think>.*', '', stripped, flags=re.DOTALL | re.IGNORECASE).strip()
    if stripped:
        return stripped
    # Fallback: extract text from inside the last <think> block
    match = re.search(r'<think>(.*?)(?:</think>|$)', original, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return original.strip()

OLLAMA_HOST = "http://localhost:11434"

# ── LlamaGuard 3 via Groq ─────────────────────────────────────────────────────

LLAMAGUARD_CATEGORIES = {
    "S1": "violent_crimes",
    "S2": "non_violent_crimes",
    "S3": "sex_related_crimes",
    "S4": "child_sexual_exploitation",
    "S5": "defamation",
    "S6": "specialized_advice",
    "S7": "privacy",
    "S8": "intellectual_property",
    "S9": "indiscriminate_weapons",
    "S10": "hate",
    "S11": "suicide_self_harm",
    "S12": "sexual_content",
    "S13": "elections",
    "S14": "code_interpreter_abuse",
}

# Map LlamaGuard categories to our internal block categories
_LG_TO_INTERNAL = {
    "S1":  "violence_illegal",
    "S2":  "violence_illegal",
    "S3":  "violence_illegal",
    "S4":  "violence_illegal",
    "S7":  "privacy",
    "S9":  "drug_synthesis",
    "S10": "hate",
    "S11": "self_harm",
    "S14": "prompt_injection",
}


def llamaguard_check(prompt: str) -> dict:
    """
    Call LlamaGuard 3 on Groq to classify prompt intent.
    Returns {"safe": bool, "category": str|None, "block_category": str|None, "raw": str}

    Only called when ML classifier is in the uncertain zone (0.2–0.5).
    Falls back gracefully — never raises, always returns a result.
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return {"safe": True, "category": None, "block_category": None, "raw": "no_groq_key"}

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "meta-llama/llama-guard-3-8b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 20,
        "temperature": 0.0,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        # LlamaGuard output: "safe" or "unsafe\nS1" etc.
        if raw.lower().startswith("unsafe"):
            lines = raw.strip().splitlines()
            lg_cat = lines[1].strip() if len(lines) > 1 else ""
            block_category = _LG_TO_INTERNAL.get(lg_cat, "violence_illegal")
            return {"safe": False, "category": lg_cat, "block_category": block_category, "raw": raw}
        return {"safe": True, "category": None, "block_category": None, "raw": raw}
    except Exception as e:
        print(f"[LlamaGuard] Error: {e} — defaulting to safe")
        return {"safe": True, "category": None, "block_category": None, "raw": f"error:{e}"}

# Groq model IDs (fast, free tier, 30 req/min)
# Only models actually available on Groq — mistral/phi3/gemma fall through to HF/Ollama
GROQ_MODELS = {
    "qwen0.5":   "qwen/qwen3-32b",
    "qwen2.5":   "qwen/qwen3-32b",
    "llama3":    "llama-3.1-8b-instant",
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
    # For Qwen3 thinking models, append /no_think to suppress chain-of-thought output
    is_thinking_model = "qwen3" in groq_model.lower()
    effective_system = (system + " /no_think") if is_thinking_model else system
    payload = {
        "model": groq_model,
        "messages": [
            {"role": "system", "content": effective_system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }

    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            return _strip_think_tags(resp.json()["choices"][0]["message"]["content"].strip())
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
            return _strip_think_tags(completion.choices[0].message.content.strip())
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
            return _strip_think_tags(response.json()["message"]["content"].strip())
        except Exception as e:
            last_error = e
            print(f"[Ollama Retry {attempt + 1}/3] Error: {e}")
            time.sleep(2)

    raise RuntimeError(f"Ollama request failed after retries: {last_error}")
