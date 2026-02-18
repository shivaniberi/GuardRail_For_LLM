import requests

OLLAMA_HOST = "http://localhost:11434"

# Define minimum 5 models for experimentation
SUPPORTED_MODELS = {
    "qwen0.5": "qwen:0.5b",
    "qwen2.5": "qwen2.5:3b",
    "llama3": "llama3.1",
    "mistral": "mistral",
    "phi3": "phi3"
}

def ollama_generate(prompt: str, model_name: str = "qwen0.5",
                     system: str = "You are a helpful assistant.") -> str:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Model '{model_name}' not supported. Choose from {list(SUPPORTED_MODELS.keys())}")

    model = SUPPORTED_MODELS[model_name]

    url = f"{OLLAMA_HOST}/api/chat"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()

    return response.json()["message"]["content"]
