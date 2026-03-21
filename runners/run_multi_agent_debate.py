"""
Interactive runner for the Chain-of-Debate multi-agent orchestrator.

What this runner does
---------------------
- Prompts the user for a query.
- Runs the multi-agent debate.
- Normalizes judge output to strict JSON when needed.
- Evaluates the selected debate answer using the SAME guardrail-style structure
  as GuardrailSystem.generate_with_guardrails():
    * raw_llm_response
    * final_response
    * verdict
    * block_reason
    * safety_flags
    * factual_flags
    * metadata
- Prints a clean structured report.
- Saves the full run record to eval_outputs/multi_agent_runs/<timestamp>.json

Usage:
  python3 runners/run_multi_agent_debate.py
  or
  python -m runners.run_multi_agent_debate
"""

import os
import re
import json
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from core.guardrail_implementation import GuardrailConfig, GuardrailSystem
from core.multi_agent import ChainOfDebateOrchestrator, LLMAgent


OUT_DIR = "eval_outputs/multi_agent_runs"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Optional imports from guardrail_implementation.
# The runner works even if some helper functions are absent.
# ---------------------------------------------------------------------
try:
    from core.guardrail_implementation import (
        _kb_contradicts_response,
        _kb_sentence_matches_query_topic,
        _lookup_factual_negation,
        _response_is_self_contradictory,
        _response_claims_action_for_query_subject,
        _extract_kb_answer_sentence,
        _extract_named_entities,
    )
except Exception:
    _kb_contradicts_response = None
    _kb_sentence_matches_query_topic = None
    _lookup_factual_negation = None
    _response_is_self_contradictory = None
    _response_claims_action_for_query_subject = None
    _extract_kb_answer_sentence = None
    _extract_named_entities = None


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def check_ml_model(cfg: GuardrailConfig) -> bool:
    path = getattr(cfg, "ml_guardrail_model_path", "")
    return os.path.exists(path) if path else False


def save_result(record: dict) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"multi_agent_run_{ts}.json"
    path = os.path.join(OUT_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return path


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def re_prompt_judge(raw: str, model_name: str = "phi3", timeout: int = 30) -> dict:
    """
    Ask a fresh LLM agent to reformat malformed judge output into strict JSON.
    """
    system = (
        "You are an impartial judge formatter.\n"
        "The previous judge output may contain markdown or malformed text.\n"
        "Return ONLY valid JSON matching this schema:\n"
        "{\"final_answer\": string, \"rationale\": string, \"confidence\": number}\n"
        "No markdown. No code fences. No extra commentary."
    )
    instr = (
        f"Previous judge output:\n\n{raw}\n\n"
        "Return only valid JSON conforming to the schema."
    )
    agent = LLMAgent(agent_id="judge_formatter", system_prompt=system)
    try:
        parsed = asyncio.run(agent.run(instr, context=None, model_name=model_name, timeout=timeout))
        content = parsed.get("content", "")
        obj = safe_json_loads(content)
        if obj is not None:
            return obj
        return {"parse_error": True, "rationale_fallback": content}
    except Exception as e:
        return {"parse_error": True, "rationale_fallback": str(e)}


def extract_json_or_repair(record: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    Ensure record['judge'] is normalized to a parsed JSON-like structure.
    """
    judge = record.get("judge", {}) or {}
    if not judge:
        record["judge"] = {
            "parse_error": True,
            "raw": "",
            "final_answer": "",
            "rationale": "Judge output missing.",
            "confidence": 0.0,
        }
        return record

    if not judge.get("parse_error"):
        return record

    raw = judge.get("rationale_fallback") or judge.get("raw") or ""
    fixed = re_prompt_judge(raw, model_name=model_name, timeout=40)
    if not fixed.get("parse_error"):
        merged = {**judge, **fixed}
        record["judge"] = merged
    else:
        record["judge"]["reformat_attempt"] = fixed

    # If judge returned "Proposal X", replace it with actual proposal text.
    try:
        fa = str(record["judge"].get("final_answer", "")).strip()
        m = re.match(r"^\s*proposal\s*(\d+)\s*$", fa, re.IGNORECASE)
        if m and record.get("rounds"):
            idx = int(m.group(1))
            last_round = record["rounds"][-1] or {}
            props = last_round.get("proposals", []) or []
            if 1 <= idx <= len(props):
                chosen = props[idx - 1]
                record["judge"]["chosen_proposal"] = {
                    "index": idx,
                    "agent_id": chosen.get("agent_id"),
                }
                record["judge"]["final_answer"] = chosen.get("content", "")
    except Exception:
        pass

    return record


def get_context(orchestrator: ChainOfDebateOrchestrator, query: str) -> Tuple[str, Dict[str, Any]]:
    """
    Retrieve RAG context using orchestrator helper if available.
    """
    try:
        context_text, ctx_meta = orchestrator._retrieve_context(query)
        return context_text or "", ctx_meta or {}
    except Exception:
        return "", {}


def compute_proposal_scores(gs: GuardrailSystem, proposals: List[Dict[str, Any]], context_text: str) -> List[Dict[str, Any]]:
    scores_out: List[Dict[str, Any]] = []

    if not proposals:
        return scores_out

    try:
        from sentence_transformers import util

        embedder = None
        if getattr(gs, "rag_retriever_primary", None) is not None and getattr(gs.rag_retriever_primary, "embedding_model", None) is not None:
            embedder = gs.rag_retriever_primary.embedding_model
        elif getattr(gs, "rag_retriever_wiki", None) is not None and getattr(gs.rag_retriever_wiki, "embedding_model", None) is not None:
            embedder = gs.rag_retriever_wiki.embedding_model

        if embedder is not None and context_text:
            ctx_emb = embedder.encode(context_text, convert_to_tensor=True)
            for p in proposals:
                text = p.get("content", "") or ""
                p_emb = embedder.encode(text, convert_to_tensor=True)
                score = float(util.cos_sim(p_emb, ctx_emb).item())
                scores_out.append({
                    "agent_id": p.get("agent_id"),
                    "score": round(score, 4),
                    "content": text,
                    "elapsed": p.get("elapsed"),
                })
            return scores_out
    except Exception:
        pass

    for p in proposals:
        scores_out.append({
            "agent_id": p.get("agent_id"),
            "score": None,
            "content": p.get("content", "") or "",
            "elapsed": p.get("elapsed"),
        })
    return scores_out


def select_candidate_answer(record: Dict[str, Any], proposal_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Select the representative raw answer to judge with guardrails.
    Priority:
      1. judge chosen proposal
      2. judge final_answer
      3. highest similarity score
      4. first proposal
    """
    rounds = record.get("rounds", []) or []
    last_round = rounds[-1] if rounds else {}
    proposals = last_round.get("proposals", []) or []

    # 1. Judge chosen proposal
    cp = (record.get("judge", {}) or {}).get("chosen_proposal")
    if cp and isinstance(cp, dict):
        idx = (cp.get("index") or 0) - 1
        if 0 <= idx < len(proposals):
            p = proposals[idx]
            return {
                "source": "judge_chosen_proposal",
                "agent_id": p.get("agent_id"),
                "content": p.get("content", "") or "",
                "score": next((x.get("score") for x in proposal_scores if x.get("agent_id") == p.get("agent_id")), None),
            }

    # 2. Judge final answer
    judge_final = (record.get("judge", {}) or {}).get("final_answer")
    if judge_final:
        return {
            "source": "judge_final_answer",
            "agent_id": "judge",
            "content": judge_final,
            "score": None,
        }

    # 3. Best score
    scored = [x for x in proposal_scores if x.get("score") is not None]
    if scored:
        top = max(scored, key=lambda x: x.get("score", -1))
        return {
            "source": "top_scored_proposal",
            "agent_id": top.get("agent_id"),
            "content": top.get("content", "") or "",
            "score": top.get("score"),
        }

    # 4. First proposal
    if proposals:
        p = proposals[0]
        return {
            "source": "first_proposal_fallback",
            "agent_id": p.get("agent_id"),
            "content": p.get("content", "") or "",
            "score": None,
        }

    return {
        "source": "empty",
        "agent_id": None,
        "content": "",
        "score": None,
    }


# ---------------------------------------------------------------------
# Debate output -> guardrail-style evaluation
# ---------------------------------------------------------------------
def evaluate_multi_agent_answer(
    gs: GuardrailSystem,
    query: str,
    raw_response: str,
    context_text: str,
    ctx_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert a debate-selected response into the same structured format as
    GuardrailSystem.generate_with_guardrails().
    """
    result: Dict[str, Any] = {
        "prompt": query,
        "timestamp": datetime.now().isoformat(),
        "raw_llm_response": raw_response or "",
        "final_response": raw_response or "",
        "verdict": "safe",
        "block_reason": None,
        "safety_flags": {},
        "factual_flags": {},
        "metadata": {},
        "response": "",
        "guardrails": {},
    }

    BLOCK_MESSAGES = {
        "hate":             "This response has been blocked: it contains hateful or harassing content targeting people based on identity.",
        "bias":             "This response has been blocked: it contains stereotyping or demeaning generalizations about people or groups.",
        "violence_illegal": "This response has been blocked: it contains violence or illegal-activity content.",
        "self_harm":        "This response has been blocked: it contains self-harm content. If this is personal, please call or text 988 if you are in the US.",
        "drug_synthesis":   "This response has been blocked: it contains controlled-substance synthesis guidance.",
        "financial_fraud":  "This response has been blocked: it contains fraud or forgery guidance.",
        "misinformation":   "This response has been blocked: it contains misinformation or conspiracy content.",
        "privacy":          "This response has been blocked: it contains private personal information.",
        "prompt_injection": "This response has been blocked: it attempted to override safety or system instructions.",
        "hallucination":    "This response has been blocked: it could not be verified against the knowledge base.",
        "ml_unsafe":        "This response has been blocked by the ML guardrail.",
    }

    ig = gs.input_guardrail

    # ----------------------------
    # Safety scan on RAW response
    # ----------------------------
    safety_flags = {
        "hate": ig.detect_hate(raw_response),
        "bias": ig.detect_bias(raw_response),
        "violence_illegal": ig.detect_violence_or_illegal(raw_response),
        "self_harm": ig.detect_self_harm(raw_response),
        "drug_synthesis": ig.detect_drug_synthesis(raw_response),
        "financial_fraud": ig.detect_financial_fraud(raw_response),
        "misinformation": ig.detect_misinformation(raw_response),
        "prompt_injection": ig.detect_injection(raw_response),
        "privacy_pii": gs.output_guardrail._privacy_leak_in_response(raw_response),
    }

    ml_result = {"valid": True, "unsafe_probability": None}
    try:
        ml_result = gs.ml_input_guardrail.validate(raw_response)
        safety_flags["ml_unsafe"] = not ml_result.get("valid", True)
    except Exception:
        safety_flags["ml_unsafe"] = None

    result["safety_flags"] = safety_flags

    priority_order = [
        "self_harm",
        "privacy_pii",
        "violence_illegal",
        "drug_synthesis",
        "financial_fraud",
        "hate",
        "bias",
        "misinformation",
        "prompt_injection",
        "ml_unsafe",
    ]

    triggered = next((k for k in priority_order if safety_flags.get(k) is True), None)
    if triggered:
        block_key = "privacy" if triggered == "privacy_pii" else triggered
        result["verdict"] = "blocked"
        result["block_reason"] = triggered
        result["final_response"] = BLOCK_MESSAGES.get(block_key, "This response has been blocked.")
    else:
        # ----------------------------
        # Factual / KB verification
        # ----------------------------
        out_check = {"valid": True, "checks": {}}
        try:
            out_check = gs.output_guardrail.verify(
                query=query,
                response=raw_response,
                context=context_text,
            )
        except Exception:
            out_check = {"valid": True, "checks": {}}

        checks = out_check.get("checks", {}) or {}
        factual_flags: Dict[str, Any] = {
            "kb_verified": out_check.get("valid", True),
            "hallucination_sim": checks.get("hallucination_similarity"),
            "context_relevance": checks.get("context_relevance"),
            "skip_reason": checks.get("skipped_reason"),
            "privacy_leak_detected": checks.get("privacy_leak_detected", False),
        }

        skip_reason = factual_flags.get("skip_reason")

        if skip_reason not in ("no_context", "context_irrelevant") and context_text:
            negation_answer = None
            if callable(_lookup_factual_negation):
                try:
                    negation_answer = _lookup_factual_negation(query)
                except Exception:
                    negation_answer = None

            if negation_answer:
                result["final_response"] = negation_answer
                factual_flags["factual_verdict"] = "negation_kb_answer"
                factual_flags["hallucination_detected"] = True
            else:
                kb_contradicts_fn = _kb_contradicts_response
                extract_kb_sentence_fn = _extract_kb_answer_sentence
                topic_match_fn = _kb_sentence_matches_query_topic

                kb_contradicts = False
                kb_suggested = ""

                if callable(kb_contradicts_fn):
                    try:
                        kb_contradicts, kb_suggested = kb_contradicts_fn(raw_response, context_text)
                    except Exception:
                        kb_contradicts, kb_suggested = False, ""

                factual_flags["kb_contradicts"] = kb_contradicts
                factual_flags["kb_suggested"] = kb_suggested

                entities_fn = _extract_named_entities if callable(_extract_named_entities) else None
                raw_entities = entities_fn(raw_response) if entities_fn else []
                context_lower = (context_text or "").lower()
                raw_entity_in_kb = bool(raw_entities) and any(e.lower() in context_lower for e in raw_entities)
                factual_flags["raw_entity_in_kb"] = raw_entity_in_kb

                if out_check.get("valid", True) and not kb_contradicts:
                    factual_flags["factual_verdict"] = "kb_verified"

                elif kb_contradicts and callable(extract_kb_sentence_fn) and callable(topic_match_fn):
                    kb_sentence = extract_kb_sentence_fn(query, context_text, kb_suggested)
                    kb_on_topic = bool(kb_sentence) and topic_match_fn(query, kb_sentence, context_text)
                    if kb_on_topic:
                        result["final_response"] = kb_sentence
                        factual_flags["factual_verdict"] = "kb_corrected"
                        factual_flags["correction_source"] = kb_suggested
                        factual_flags["hallucination_detected"] = True
                    else:
                        factual_flags["factual_verdict"] = "kb_contradiction_off_topic"

                elif raw_entity_in_kb:
                    is_contradictory = False
                    is_negated_claim = False

                    if callable(_response_is_self_contradictory):
                        try:
                            is_contradictory = _response_is_self_contradictory(raw_response)
                        except Exception:
                            is_contradictory = False

                    if callable(_response_claims_action_for_query_subject):
                        try:
                            is_negated_claim = _response_claims_action_for_query_subject(query, raw_response, context_text)
                        except Exception:
                            is_negated_claim = False

                    if is_contradictory or is_negated_claim:
                        if callable(extract_kb_sentence_fn) and callable(topic_match_fn):
                            kb_sentence = extract_kb_sentence_fn(query, context_text, "")
                            kb_on_topic = bool(kb_sentence) and topic_match_fn(query, kb_sentence, context_text)
                            if kb_on_topic:
                                result["final_response"] = kb_sentence
                                factual_flags["factual_verdict"] = "kb_corrected_contradiction"
                            else:
                                factual_flags["factual_verdict"] = "contradiction_unverifiable"
                        else:
                            factual_flags["factual_verdict"] = "contradiction_unverifiable"
                        factual_flags["hallucination_detected"] = True
                    else:
                        factual_flags["factual_verdict"] = "raw_entity_in_kb_trusted"

                else:
                    factual_flags["factual_verdict"] = "raw_llm_unverified"
        else:
            factual_flags["factual_verdict"] = (
                "kb_irrelevant" if skip_reason == "context_irrelevant" else "no_kb"
            )

        result["factual_flags"] = factual_flags

    primary_count = int((ctx_meta or {}).get("primary_count", 0) or 0)
    wiki_count = int((ctx_meta or {}).get("wiki_count", 0) or 0)
    result["metadata"] = {
        "rag_used": bool(primary_count + wiki_count > 0),
        "retrieved_docs_total": primary_count + wiki_count,
        "rag_primary_docs": primary_count,
        "rag_wiki_docs": wiki_count,
        "kb_sources": [s for s, n in [("primary", primary_count), ("wiki", wiki_count)] if n > 0],
        "ml_unsafe_probability": ml_result.get("unsafe_probability"),
    }

    result["response"] = result["final_response"]
    result["guardrails"] = {
        "input": {
            "rule_based": {
                "valid": result["block_reason"] is None,
                "block_category": result["block_reason"],
                "checks": {k: {"passed": not bool(v)} for k, v in result["safety_flags"].items()},
            },
            "ml_based": {
                "valid": not bool(result["safety_flags"].get("ml_unsafe")),
                "unsafe_probability": result["metadata"].get("ml_unsafe_probability"),
            },
        },
        "output": {
            "valid": result.get("factual_flags", {}).get("kb_verified", True),
            "checks": {
                "hallucination_similarity": result.get("factual_flags", {}).get("hallucination_sim"),
                "context_relevance": result.get("factual_flags", {}).get("context_relevance"),
                "skipped_reason": result.get("factual_flags", {}).get("skip_reason"),
                "privacy_leak_detected": result.get("factual_flags", {}).get("privacy_leak_detected", False),
            },
        },
    }

    return result


# ---------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------
def print_status_line(name: str, value: Any) -> None:
    print(f"  {name:<18}: {value}")


def print_output(
    cfg: GuardrailConfig,
    evaluation: Dict[str, Any],
) -> None:
    print(f"\n{'─'*60}")
    print(f"  MODEL : {cfg.ollama_model_name}")

    print(f"\n📤 RAW LLM RESPONSE")
    print(f"  {evaluation.get('raw_llm_response', '(none)')}")

    verdict = evaluation.get("verdict", "unknown")
    if verdict == "blocked":
        print(f"\n{'='*60}")
        print(f"  🚫 BLOCKED  —  reason: {evaluation.get('block_reason')}")
        print(f"{'='*60}")
    elif verdict == "safe":
        print(f"\n  ✅ VERDICT: SAFE")
    else:
        print(f"\n  ⚠  VERDICT: {str(verdict).upper()}")

    print(f"\n💬 FINAL GUARDRAIL RESPONSE")
    print(f"  {evaluation.get('final_response', '(none)')}")

    print(f"\n🛡  SAFETY CHECKS (run on LLM output)")
    safety_flags = evaluation.get("safety_flags", {})
    ordered_keys = [
        "hate",
        "bias",
        "violence_illegal",
        "self_harm",
        "drug_synthesis",
        "financial_fraud",
        "misinformation",
        "prompt_injection",
        "privacy_pii",
        "ml_unsafe",
    ]
    triggered = [k for k in ordered_keys if safety_flags.get(k) is True]
    clean = [k for k in ordered_keys if safety_flags.get(k) is False]
    unknown = [k for k in ordered_keys if safety_flags.get(k) is None]

    if triggered:
        for flag in triggered:
            print(f"  ❌  {flag}")
    for flag in clean:
        print(f"  ✓   {flag}")
    for flag in unknown:
        print(f"  ?   {flag}")

    ml_prob = evaluation.get("metadata", {}).get("ml_unsafe_probability")
    if ml_prob is not None:
        print(f"  ML unsafe probability : {ml_prob}")

    factual_flags = evaluation.get("factual_flags", {})
    if factual_flags:
        print(f"\n🔍 FACTUAL / KB CHECKS")
        print_status_line("kb_verified", factual_flags.get("kb_verified"))
        print_status_line("hallucination_sim", factual_flags.get("hallucination_sim"))
        print_status_line("context_relevance", factual_flags.get("context_relevance"))
        print_status_line("factual_verdict", factual_flags.get("factual_verdict"))
        if factual_flags.get("kb_contradicts"):
            print(f"  ⚠  KB contradiction — suggested: {factual_flags.get('kb_suggested')}")
        if factual_flags.get("hallucination_detected"):
            print(f"  ⚠  Hallucination detected in raw response")

    meta = evaluation.get("metadata", {})
    print(f"\n📚 KB METADATA")
    print_status_line("rag_used", meta.get("rag_used"))
    print_status_line("total_docs", meta.get("retrieved_docs_total"))
    print_status_line("kb_sources", meta.get("kb_sources"))

    print(f"\n{'─'*60}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("Multi-Agent Chain-of-Debate interactive runner")

    query = input("Enter your query: ").strip()
    if not query:
        print("No query provided — exiting.")
        return

    try:
        num_agents = int(input("Number of proponent agents (default 3): ") or 3)
    except Exception:
        num_agents = 3

    try:
        rounds = int(input("Number of rebuttal rounds (default 2): ") or 2)
    except Exception:
        rounds = 2

    cfg = GuardrailConfig()
    cfg.ollama_model_name = "phi3"

    ml_ok = check_ml_model(cfg)
    if not ml_ok:
        print("WARNING: ML-based input guardrail model not found.")
        print("Rule-based checks will still run, but ML checks may be disabled.")
        print("If needed, train the model first to produce the configured .joblib file.")

    print("Initializing GuardrailSystem (this may take a moment)...")
    gs = GuardrailSystem(cfg)
    orchestrator = ChainOfDebateOrchestrator(gs)

    print(f"Running debate with {num_agents} proponents and {rounds} rebuttal rounds...")
    record = orchestrator.debate(
        query,
        num_agents=num_agents,
        rounds=rounds,
        model_name=cfg.ollama_model_name,
    )

    # Normalize judge
    record = extract_json_or_repair(record, cfg.ollama_model_name)

    # Retrieve context
    context_text, ctx_meta = get_context(orchestrator, query)

    # Score proposals
    last_round = (record.get("rounds", []) or [])[-1] if record.get("rounds") else {}
    proposals = (last_round or {}).get("proposals", []) or []
    proposal_scores = compute_proposal_scores(gs, proposals, context_text)

    # Select candidate answer
    candidate = select_candidate_answer(record, proposal_scores)

    # Evaluate selected answer in guardrail-style format
    evaluation = evaluate_multi_agent_answer(
        gs=gs,
        query=query,
        raw_response=candidate.get("content", "") or "",
        context_text=context_text,
        ctx_meta=ctx_meta,
    )

    # Attach debate-specific provenance
    full_record = {
        "prompt": query,
        "timestamp": datetime.now().isoformat(),
        "runner": "run_multi_agent_debate.py",
        "model": cfg.ollama_model_name,
        "num_agents": num_agents,
        "rounds_requested": rounds,
        "candidate": candidate,
        "evaluation": evaluation,
        "proposal_scores": proposal_scores,
        "judge": record.get("judge", {}),
        "debate_rounds": record.get("rounds", []),
        "context_metadata": ctx_meta,
        "context_text": context_text,
        "raw_record": record,
    }

    print_output(cfg, evaluation)

    path = save_result(full_record)
    print(f"\nFull run saved to: {path}")


if __name__ == "__main__":
    main()