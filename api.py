"""
GuardRail FastAPI Server
========================
Endpoints:
  POST /api/guardrail   — single-model guardrail query (used by frontend)
  POST /api/multi-agent — multi-agent debate query (used by frontend)
  POST /query           — alias for /api/guardrail
  GET  /health          — liveness check
  GET  /models          — list supported Ollama model keys
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.guardrail_implementation import GuardrailConfig, GuardrailSystem
from core.multi_agent import ChainOfDebateOrchestrator

_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "core")

SUPPORTED_MODELS = ["qwen0.5", "qwen2.5", "llama3", "mistral", "phi3", "gemma3:1b", "gemma:2b"]

# Map frontend model keys to installed Ollama model names
MODEL_MAP = {
    "qwen0.5":   "qwen:0.5b",
    "qwen2.5":   "qwen:0.5b",   # fallback until qwen2.5 is installed
    "llama3":    "qwen:0.5b",
    "mistral":   "qwen:0.5b",
    "phi3":      "qwen:0.5b",
    "gemma3:1b": "qwen:0.5b",
    "gemma:2b":  "qwen:0.5b",
}

_system: GuardrailSystem | None = None
_debate_busy = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _system
    config = GuardrailConfig(
        enable_input_validation=True,
        enable_output_verification=True,
        enable_rag=True,
        enable_logging=True,
        rag_dataset_path=os.getenv(
            "RAG_DATASET_PATH",
            "s3://guardraildemo/data/dolly_clean_final.parquet",
        ),
        wiki_rag_dataset_path=os.getenv(
            "WIKI_RAG_DATASET_PATH",
            "",  # disabled by default — 206MB file makes startup too slow
        ),
        ml_guardrail_model_path=os.path.join(_CORE, "input_safety_all7.joblib"),
        ml_guardrail_threshold=0.5,
        hallucination_threshold=0.72,
        context_relevance_threshold=0.35,
        ollama_model_name="qwen0.5",
        always_return_raw_llm=False,
        rag_max_chunks=int(os.getenv("RAG_MAX_CHUNKS", "2000")),
    )
    _system = GuardrailSystem(config)
    yield
    _system = None


app = FastAPI(title="GuardRail for LLM", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PromptRequest(BaseModel):
    prompt: str
    model: str = "qwen0.5"
    max_tokens: int = 256
    use_rag: bool = True


class MultiAgentRequest(BaseModel):
    prompt: str
    model: str = "qwen0.5"
    num_agents: int = 2
    rounds: int = 1


def _run_guardrail(req: PromptRequest):
    if _system is None:
        raise HTTPException(status_code=503, detail="Guardrail system not ready")
    if req.model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model '{req.model}'. Choose from: {SUPPORTED_MODELS}")

    _system.config.ollama_model_name = req.model  # key expected by ollama_client.py
    result = _system.generate_with_guardrails(
        prompt=req.prompt,
        max_new_tokens=req.max_tokens,
        use_rag=req.use_rag,
    )
    return {
        "raw_llm_response": result.get("raw_llm_response", ""),
        "final_response":   result.get("final_response", ""),
        "response":         result.get("final_response", ""),
        "guarded_response": result.get("final_response", ""),
        "verdict":          result.get("verdict", "unknown"),
        "block_reason":     result.get("block_reason"),
        "input_guardrail": {
            "rule_based": {
                "valid": not bool(result.get("block_reason")),
                "block_category": result.get("block_reason"),
            },
            "ml_based": {
                "valid": result.get("metadata", {}).get("ml_unsafe_probability", 0) < 0.5,
                "unsafe_probability": result.get("metadata", {}).get("ml_unsafe_probability"),
            },
        },
        "output_guardrail": {
            "hallucination_similarity": result.get("factual_flags", {}).get("hallucination_sim"),
            "valid": not result.get("factual_flags", {}).get("hallucination_detected", False),
        },
        "rag_metadata": {
            "rag_used":   result.get("metadata", {}).get("rag_used"),
            "total_docs": result.get("metadata", {}).get("retrieved_docs_total"),
            "kb_sources": result.get("metadata", {}).get("kb_sources"),
        },
        "safety_flags":  result.get("safety_flags", {}),
        "factual_flags": result.get("factual_flags", {}),
    }


@app.post("/api/guardrail")
async def api_guardrail(req: PromptRequest):
    return _run_guardrail(req)


@app.post("/query")
async def query(req: PromptRequest):
    return _run_guardrail(req)


@app.post("/api/multi-agent")
async def api_multi_agent(req: MultiAgentRequest):
    global _debate_busy
    if _debate_busy:
        raise HTTPException(status_code=429, detail="A debate is already running.")
    if _system is None:
        raise HTTPException(status_code=503, detail="Guardrail system not ready")

    _debate_busy = True
    try:
        from core.multi_agent import run_debate_sync
        import asyncio, functools
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            functools.partial(
                run_debate_sync,
                guardrail_system=_system,
                question=req.prompt,
                num_agents=req.num_agents,
                rounds=req.rounds,
                model_name=req.model,
            ),
        )
        return result
    finally:
        _debate_busy = False


@app.get("/health")
async def health():
    return {"status": "ok", "system_ready": _system is not None}


@app.get("/models")
async def models():
    return {"supported_models": SUPPORTED_MODELS}
