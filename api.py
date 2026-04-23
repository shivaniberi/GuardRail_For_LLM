"""
GuardRail FastAPI Server
========================
Wraps GuardrailSystem as an HTTP REST API for the frontend and CI/CD pipeline.

Endpoints:
  POST /query    — run a prompt through the full guardrail pipeline
  GET  /health   — liveness check
  GET  /models   — list supported Ollama model keys
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.guardrail_implementation import GuardrailConfig, GuardrailSystem

_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "core")

SUPPORTED_MODELS = ["qwen0.5", "qwen2.5", "llama3", "mistral", "phi3"]

_system: GuardrailSystem | None = None


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
            "s3://guardraildemo/RAG/nli_fact_checking.parquet",
        ),
        ml_guardrail_model_path=os.path.join(_CORE, "input_safety_all7.joblib"),
        ml_guardrail_threshold=0.5,
        hallucination_threshold=0.72,
        context_relevance_threshold=0.35,
        ollama_model_name="qwen2.5",
        always_return_raw_llm=False,
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
    model: str = "qwen2.5"
    max_tokens: int = 256
    use_rag: bool = True


@app.post("/query")
async def query(req: PromptRequest):
    if _system is None:
        raise HTTPException(status_code=503, detail="Guardrail system not ready")

    if req.model not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{req.model}'. Choose from: {SUPPORTED_MODELS}",
        )

    # Swap model on the live config so one server can serve multiple model requests
    _system.config.ollama_model_name = req.model

    result = _system.generate_with_guardrails(
        prompt=req.prompt,
        max_new_tokens=req.max_tokens,
        use_rag=req.use_rag,
    )

    return {
        "raw_llm_response": result.get("raw_llm_response", ""),
        "guarded_response": result.get("final_response", ""),
        "verdict": result.get("verdict", "unknown"),
        "block_reason": result.get("block_reason"),
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
            "rag_used": result.get("metadata", {}).get("rag_used"),
            "total_docs": result.get("metadata", {}).get("retrieved_docs_total"),
            "kb_sources": result.get("metadata", {}).get("kb_sources"),
        },
        "safety_flags": result.get("safety_flags", {}),
        "factual_flags": result.get("factual_flags", {}),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "system_ready": _system is not None}


@app.get("/models")
async def models():
    return {"supported_models": SUPPORTED_MODELS}
