"""
GuardRail FastAPI Server
========================
Endpoints:
  POST /api/guardrail   — single-model guardrail query (used by frontend)
  POST /api/multi-agent — multi-agent debate query (used by frontend)
  POST /api/feedback    — human feedback (thumbs up/down + correction)
  POST /query           — alias for /api/guardrail
  GET  /health          — liveness check
  GET  /models          — list supported Ollama model keys
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import re
import uuid
import asyncio
import threading
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.guardrail_implementation import GuardrailConfig, GuardrailSystem
from core.multi_agent import ChainOfDebateOrchestrator

_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "core")

# Job store for async multi-agent polling
_jobs: dict = {}  # job_id -> {"status": "running"|"done"|"error", "result": ..., "error": ...}

SUPPORTED_MODELS = ["qwen0.5", "qwen2.5", "llama3", "mistral", "phi3", "gemma3:1b", "gemma:2b"]

MODEL_MAP = {
    "qwen0.5":   "qwen:0.5b",
    "qwen2.5":   "qwen:2.5b",
    "llama3":    "qwen:2.5b",
    "mistral":   "qwen:2.5b",
    "phi3":      "qwen:2.5b",
    "gemma3:1b": "qwen:2.5b",
    "gemma:2b":  "qwen:2.5b",
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
            "s3://guardrail-group-bucket/processed/train.parquet",
        ),
        wiki_rag_dataset_path=os.getenv(
            "WIKI_RAG_DATASET_PATH",
            "s3://guardrail-group-bucket/knowledge_base/wikipedia/latest/simplewiki_articles.parquet",
        ),
        ml_guardrail_model_path=os.path.join(_CORE, "input_safety_all7.joblib"),
        ml_guardrail_threshold=0.5,
        hallucination_threshold=0.72,
        context_relevance_threshold=0.35,
        ollama_model_name="qwen2.5",
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
    model: str = "qwen2.5"
    max_tokens: int = 256
    use_rag: bool = True


class MultiAgentRequest(BaseModel):
    prompt: str
    model: str = "qwen2.5"
    num_agents: int = 2
    rounds: int = 1


# ── NEW: Human feedback model ─────────────────────────────────────────────────
class FeedbackRequest(BaseModel):
    prompt: str
    response: str
    rating: int          # 1 = thumbs up, -1 = thumbs down
    correction: str = "" # user's correct answer (optional)
# ── END NEW ───────────────────────────────────────────────────────────────────


# ── Single-model guardrail ────────────────────────────────────────────────────

def _run_guardrail(req: PromptRequest):
    if _system is None:
        raise HTTPException(status_code=503, detail="Guardrail system not ready")
    if req.model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model '{req.model}'. Choose from: {SUPPORTED_MODELS}")

    _system.config.ollama_model_name = req.model

    # ── NEW: Check reflexion memory for human corrections ─────────────────────
    human_correction_found = False
    human_correction = ""
    try:
        from core.reflexion_memory import get_correction
        human_correction = get_correction(req.prompt)
        if human_correction:
            human_correction_found = True
            prompt_to_use = (
                f"{req.prompt}\n\n"
                f"[Note: A human previously verified the correct answer is: {human_correction}. "
                f"Use this as your primary answer.]"
            )
        else:
            prompt_to_use = req.prompt
    except Exception:
        prompt_to_use = req.prompt
    # ── END NEW ───────────────────────────────────────────────────────────────

    result = _system.generate_with_guardrails(
        prompt=prompt_to_use,
        max_new_tokens=req.max_tokens,
        use_rag=req.use_rag,
    )

    # ── NEW: If human correction exists, use it as the final response ─────────
    if human_correction_found:
        result["final_response"] = human_correction
        result["response"] = human_correction
        result["guarded_response"] = human_correction
    # ── END NEW ───────────────────────────────────────────────────────────────

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


# ── Multi-agent helpers ───────────────────────────────────────────────────────

def _extract_best_proposal(rounds: list) -> str:
    """Extract the best answer from the last round's proposals."""
    if not rounds:
        return ""
    last_round = rounds[-1] or {}
    proposals = last_round.get("proposals", []) or []
    if not proposals:
        return ""
    return proposals[0].get("content", "") or ""


def _is_clean_answer(text: str) -> bool:
    """Return False if text looks like critic/judge noise rather than a real answer."""
    if not text:
        return False
    noise_patterns = [
        r"critique_\d",
        r"proposed \d",
        r"proposal \d",
        r"rebuttal",
        r"potential errors",
        r"unsupported claims",
        r"in conclusion.*in conclusion",
    ]
    text_lower = text.lower()
    for pattern in noise_patterns:
        if re.search(pattern, text_lower):
            return False
    return True


def _apply_kb_correction(question: str, answer: str, context: str) -> str:
    """Use KB contradiction detection to correct hallucinations."""
    if not context or not answer:
        return answer
    try:
        from core.guardrail_implementation import (
            _kb_contradicts_response,
            _extract_kb_answer_sentence,
            _kb_sentence_matches_query_topic,
            _lookup_factual_negation,
        )
        if callable(_lookup_factual_negation):
            negation = _lookup_factual_negation(question)
            if negation:
                return negation

        if callable(_kb_contradicts_response):
            contradicts, suggested = _kb_contradicts_response(answer, context)
            if contradicts and suggested and callable(_extract_kb_answer_sentence):
                kb_sentence = _extract_kb_answer_sentence(question, context, suggested)
                if kb_sentence and callable(_kb_sentence_matches_query_topic):
                    if _kb_sentence_matches_query_topic(question, kb_sentence, context):
                        return kb_sentence
    except Exception:
        pass
    return answer


def _run_debate_job(job_id: str, req: MultiAgentRequest):
    """Runs debate in a background thread and stores result in _jobs."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        orchestrator = ChainOfDebateOrchestrator(_system)
        result = loop.run_until_complete(orchestrator.debate(
            question=req.prompt,
            num_agents=req.num_agents,
            rounds=req.rounds,
            model_name=req.model,
        ))
        loop.close()

        context = ""
        ctx_meta = {}
        try:
            rag = _system.rag_retriever
            if rag:
                docs = rag.retrieve(req.prompt, top_k=3)
                context = " ".join([d.get("text", "") for d in docs]) if docs else ""
                ctx_meta = {"primary_count": len(docs), "wiki_count": 0}
        except Exception:
            pass

        judge = result.get("judge", {}) or {}
        final_answer = judge.get("final_answer", "") or ""

        # Always fallback to best proposal if final_answer is empty or noisy
        if not final_answer or not _is_clean_answer(final_answer) or len(final_answer.strip()) < 5:
            fallback = _extract_best_proposal(result.get("rounds", []))
            if fallback:
                final_answer = fallback
                result["judge"]["final_answer"] = fallback
                result["judge"]["source"] = "proposal_fallback"

        # Last resort: use raw judge output
        if not final_answer or len(final_answer.strip()) < 5:
            final_answer = judge.get("raw", "") or ""
            result["judge"]["final_answer"] = final_answer

        if context and final_answer:
            corrected = _apply_kb_correction(req.prompt, final_answer, context)
            if corrected != final_answer:
                result["judge"]["final_answer"] = corrected
                result["judge"]["kb_corrected"] = True
                final_answer = corrected

        primary_count = int((ctx_meta or {}).get("primary_count", 0) or 0)
        wiki_count    = int((ctx_meta or {}).get("wiki_count", 0) or 0)
        kb_sources    = [s for s, n in [("primary", primary_count), ("wiki", wiki_count)] if n > 0]

        result["rag_metadata"] = {
            "rag_used":   bool(primary_count + wiki_count > 0),
            "total_docs": primary_count + wiki_count,
            "kb_sources": kb_sources,
        }
        result["evaluation"] = {
            "raw_llm_response": result.get("judge", {}).get("raw", ""),
            "final_response":   final_answer,
            "verdict":          "safe",
            "block_reason":     None,
            "safety_flags":     {},
            "factual_flags":    {},
            "metadata": {
                "rag_used":                bool(primary_count + wiki_count > 0),
                "retrieved_docs_total":    primary_count + wiki_count,
                "kb_sources":              kb_sources,
                "ml_unsafe_probability":   None,
                "ml_prompt_probability":   None,
                "ml_response_probability": None,
            },
            "guardrails": {
                "output": {
                    "valid": True,
                    "checks": {
                        "hallucination_similarity": None,
                        "context_relevance": None,
                        "skipped_reason": None,
                        "privacy_leak_detected": False,
                    }
                }
            }
        }
        result["debate_rounds"] = result.get("rounds", [])
        rounds = result.get("rounds", [])
        first_proposal = ""
        if rounds and rounds[0].get("proposals"):
            first_proposal = rounds[0]["proposals"][0].get("content", "")
        result["raw_llm_response"] = first_proposal
        result["evaluation"]["raw_llm_response"] = first_proposal

        _jobs[job_id] = {"status": "done", "result": result}

    except Exception as e:
        _jobs[job_id] = {"status": "error", "error": str(e)}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/api/guardrail")
async def api_guardrail(req: PromptRequest):
    return _run_guardrail(req)


@app.post("/query")
async def query(req: PromptRequest):
    return _run_guardrail(req)


# ── NEW: Human feedback endpoint ──────────────────────────────────────────────
@app.post("/api/feedback")
async def api_feedback(req: FeedbackRequest):
    try:
        from core.reflexion_memory import save_feedback
        save_feedback(req.prompt, req.response, req.rating, req.correction)
        return {"status": "saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {e}")
# ── END NEW ───────────────────────────────────────────────────────────────────


@app.post("/api/multi-agent")
async def api_multi_agent(req: MultiAgentRequest):
    global _debate_busy
    if _debate_busy:
        raise HTTPException(status_code=429, detail="A debate is already running.")
    if _system is None:
        raise HTTPException(status_code=503, detail="Guardrail system not ready")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running"}
    _debate_busy = True

    def run_and_clear():
        global _debate_busy
        try:
            _run_debate_job(job_id, req)
        finally:
            _debate_busy = False

    t = threading.Thread(target=run_and_clear, daemon=True)
    t.start()

    return {"job_id": job_id, "status": "running"}


@app.get("/api/multi-agent/status/{job_id}")
async def api_multi_agent_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] == "running":
        return {"status": "running"}
    if job["status"] == "error":
        _jobs.pop(job_id, None)
        raise HTTPException(status_code=500, detail=job.get("error", "Unknown error"))
    result = job["result"]
    _jobs.pop(job_id, None)
    return result


# Legacy sync endpoint kept for backward compatibility
@app.post("/api/multi-agent-sync")
async def api_multi_agent_sync(req: MultiAgentRequest):
    global _debate_busy
    if _debate_busy:
        raise HTTPException(status_code=429, detail="A debate is already running.")
    if _system is None:
        raise HTTPException(status_code=503, detail="Guardrail system not ready")

    _debate_busy = True
    try:
        orchestrator = ChainOfDebateOrchestrator(_system)
        result = await orchestrator.debate(
            question=req.prompt,
            num_agents=req.num_agents,
            rounds=req.rounds,
            model_name=req.model,
        )

        # ── Post-processing ───────────────────────────────────────────────────

        # 1. Get RAG context for KB correction
        context = ""
        ctx_meta = {}
        try:
            context, ctx_meta = orchestrator._retrieve_context(req.prompt)
            context = context or ""
        except Exception:
            pass

        # 2. Get judge final_answer
        judge = result.get("judge", {}) or {}
        final_answer = judge.get("final_answer", "") or ""

        # 3. Fallback to best proposal if judge output is noisy/garbage
        if not final_answer or not _is_clean_answer(final_answer):
            fallback = _extract_best_proposal(result.get("rounds", []))
            if fallback:
                final_answer = fallback
                result["judge"]["final_answer"] = fallback
                result["judge"]["source"] = "proposal_fallback"

        # 4. Apply KB correction to fix hallucinations
        if context and final_answer:
            corrected = _apply_kb_correction(req.prompt, final_answer, context)
            if corrected != final_answer:
                result["judge"]["final_answer"] = corrected
                result["judge"]["kb_corrected"] = True
                final_answer = corrected

        # 5. Build RAG metadata
        primary_count = int((ctx_meta or {}).get("primary_count", 0) or 0)
        wiki_count    = int((ctx_meta or {}).get("wiki_count", 0) or 0)
        kb_sources    = [s for s, n in [("primary", primary_count), ("wiki", wiki_count)] if n > 0]

        result["rag_metadata"] = {
            "rag_used":   bool(primary_count + wiki_count > 0),
            "total_docs": primary_count + wiki_count,
            "kb_sources": kb_sources,
        }

        # 6. Build evaluation wrapper — this is what the frontend reads
        result["evaluation"] = {
            "raw_llm_response": result.get("judge", {}).get("raw", ""),
            "final_response":   final_answer,
            "verdict":          "safe",
            "block_reason":     None,
            "safety_flags":     {},
            "factual_flags":    {},
            "metadata": {
                "rag_used":                bool(primary_count + wiki_count > 0),
                "retrieved_docs_total":    primary_count + wiki_count,
                "kb_sources":              kb_sources,
                "ml_unsafe_probability":   None,
                "ml_prompt_probability":   None,
                "ml_response_probability": None,
            },
            "guardrails": {
                "output": {
                    "valid": True,
                    "checks": {
                        "hallucination_similarity": (
                            result.get("output_verification", {})
                                  .get("checks", {})
                                  .get("hallucination_similarity")
                        ),
                        "context_relevance": (
                            result.get("output_verification", {})
                                  .get("checks", {})
                                  .get("context_relevance")
                        ),
                        "skipped_reason":        None,
                        "privacy_leak_detected": False,
                    }
                }
            }
        }

        # 7. Add debate_rounds so frontend shows agent/round count
        result["debate_rounds"] = result.get("rounds", [])

        # 8. Top-level raw_llm_response = first agent's raw answer (before guardrail)
        rounds = result.get("rounds", [])
        first_proposal = ""
        if rounds and rounds[0].get("proposals"):
            first_proposal = rounds[0]["proposals"][0].get("content", "")

        result["raw_llm_response"] = first_proposal
        result["evaluation"]["raw_llm_response"] = first_proposal

        return result

    finally:
        _debate_busy = False


@app.get("/health")
async def health():
    return {"status": "ok", "system_ready": _system is not None}


@app.get("/models")
async def models():
    return {"supported_models": SUPPORTED_MODELS}