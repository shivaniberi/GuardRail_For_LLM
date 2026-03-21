"""
Interactive Guardrail Runner (Ollama-backed)
============================================

Architecture: LLM always responds first. Guardrail judges the OUTPUT.

Result structure:
  raw_llm_response  — what the LLM actually said (always shown)
  final_response    — safe/KB-corrected answer, or block message
  verdict           — "safe" | "blocked" | "error"
  block_reason      — which category triggered the block (if any)
  safety_flags      — every safety check and whether it fired
  factual_flags     — KB verification details
"""

import os
from core.guardrail_implementation import GuardrailConfig, GuardrailSystem

_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.join(os.path.dirname(_HERE), "core")


def print_verdict_banner(verdict: str, block_reason: str | None):
    if verdict == "blocked":
        print(f"\n{'='*60}")
        print(f"  🚫 BLOCKED  —  reason: {block_reason}")
        print(f"{'='*60}")
    elif verdict == "safe":
        print(f"\n  ✅ VERDICT: SAFE")
    else:
        print(f"\n  ⚠  VERDICT: {verdict.upper()}")


def main():
    model_to_test = "phi3"   # swap to "qwen0.5" or "qwen2.5" as eded

    config = GuardrailConfig(
        enable_input_validation=True,
        enable_output_verification=True,
        enable_rag=True,
        enable_logging=True,

        rag_dataset_path=(
            "s3://guardrail-group-bucket/processed/train.parquet"
        ),
        wiki_rag_dataset_path=(
            "s3://guardrail-group-bucket/knowledge_base/wikipedia/"
            "2026/02/18/233610/simplewiki_articles.parquet"
        ),

        hallucination_threshold=0.72,
        context_relevance_threshold=0.35,

        ml_guardrail_model_path=os.path.join(_CORE, "input_safety_all7.joblib"),
        ml_guardrail_threshold=0.5,

        ollama_model_name=model_to_test,
        always_return_raw_llm=False,
    )

    system = GuardrailSystem(config)
    print("\nGuardrail system ready.")
    print("Type your prompt below (type 'exit' to quit)\n")

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit"):
            print("Exiting...")
            break

        result = system.generate_with_guardrails(
            prompt=prompt,
            max_new_tokens=256,
            use_rag=True,
        )

        verdict      = result.get("verdict", "unknown")
        block_reason = result.get("block_reason")
        safety_flags = result.get("safety_flags", {})
        factual_flags= result.get("factual_flags", {})
        metadata     = result.get("metadata", {})

        # ── MODEL ─────────────────────────────────────────────────────────────
        print(f"\n{'─'*60}")
        print(f"  MODEL : {model_to_test}")

        # ── RAW LLM RESPONSE ──────────────────────────────────────────────────
        print(f"\n📤 RAW LLM RESPONSE")
        print(f"  {result.get('raw_llm_response', '(none)')}")

        # ── VERDICT BANNER ────────────────────────────────────────────────────
        print_verdict_banner(verdict, block_reason)

        # ── FINAL RESPONSE ────────────────────────────────────────────────────
        print(f"\n💬 FINAL GUARDRAIL RESPONSE")
        print(f"  {result.get('final_response', '(none)')}")

        # ── SAFETY FLAGS ──────────────────────────────────────────────────────
        print(f"\n🛡  SAFETY CHECKS (run on LLM output)")
        triggered = [k for k, v in safety_flags.items() if v]
        clean     = [k for k, v in safety_flags.items() if not v]
        if triggered:
            for flag in triggered:
                print(f"  ❌  {flag}")
        for flag in clean:
            print(f"  ✓   {flag}")

        ml_prob = metadata.get("ml_unsafe_probability")
        if ml_prob is not None:
            print(f"  ML unsafe probability : {ml_prob}")

        # ── FACTUAL FLAGS ─────────────────────────────────────────────────────
        if factual_flags:
            print(f"\n🔍 FACTUAL / KB CHECKS")
            print(f"  kb_verified       : {factual_flags.get('kb_verified')}")
            print(f"  hallucination_sim : {factual_flags.get('hallucination_sim')}")
            print(f"  context_relevance : {factual_flags.get('context_relevance')}")
            print(f"  factual_verdict   : {factual_flags.get('factual_verdict')}")
            if factual_flags.get("kb_contradicts"):
                print(f"  ⚠  KB contradiction — suggested: {factual_flags.get('kb_suggested')}")
            if factual_flags.get("hallucination_detected"):
                print(f"  ⚠  Hallucination detected in raw response")

        # ── RAG METADATA ──────────────────────────────────────────────────────
        print(f"\n📚 KB METADATA")
        print(f"  rag_used     : {metadata.get('rag_used')}")
        print(f"  total_docs   : {metadata.get('retrieved_docs_total')}")
        print(f"  kb_sources   : {metadata.get('kb_sources')}")

        print(f"\n{'─'*60}\n")


if __name__ == "__main__":
    main()