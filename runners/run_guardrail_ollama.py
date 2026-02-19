"""
Interactive Guardrail Runner (Ollama-backed)
===========================================

This script runs the GuardrailSystem interactively in the terminal.

What it does:
- Uses an Ollama LLM (qwen / llama / mistral / phi3, etc.)
- Applies:
    • Rule-based input guardrails
    • ML-based safety classifier
    • Dual RAG knowledge bases:
        1) Primary dataset (original KB)
        2) Wikipedia dataset (knowledge_base/wikipedia in S3)
    • Output hallucination verification
- Prints:
    1) Raw LLM response (no RAG, no guardrail influence)
    2) Final guarded response (after safety + RAG)
    3) Input guardrail results
    4) Output guardrail results
    5) RAG metadata (retrieved documents, KB sources)

This allows comparison between the model’s original answer
and the safety-grounded answer.
"""

from core.guardrail_implementation import GuardrailConfig, GuardrailSystem


def main():
    # Model options:
    # "qwen0.5", "qwen2.5", "llama3", "mistral", "phi3"
    model_to_test = "phi3"

    config = GuardrailConfig(
        enable_input_validation=True,
        enable_output_verification=True,
        enable_rag=True,
        enable_logging=True,

        # Primary knowledge base
        rag_dataset_path="s3://guardrail-group-bucket/processed/train.parquet",

        # Wikipedia knowledge base
        wiki_rag_dataset_path="s3://guardrail-group-bucket/knowledge_base/wikipedia/latest/simplewiki_articles.parquet",

        ollama_model_name=model_to_test,
        always_return_raw_llm=True,
    )

    system = GuardrailSystem(config)

    print("\nGuardrail system ready.")
    print("Type your prompt below (type 'exit' to quit)\n")

    while True:
        prompt = input(">>> ").strip()

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

        print("\n=== MODEL ===")
        print(model_to_test)

        print("\n=== RAW LLM RESPONSE (no guardrail / no RAG) ===")
        print(result.get("metadata", {}).get("raw_llm_response"))

        print("\n=== GUARDED FINAL RESPONSE (guardrails + RAG) ===")
        print(result.get("response"))

        print("\n=== INPUT GUARDRAIL ===")
        print(result.get("guardrails", {}).get("input"))

        print("\n=== OUTPUT GUARDRAIL ===")
        print(result.get("guardrails", {}).get("output"))

        meta = result.get("metadata", {}) or {}
        print("\n=== RAG METADATA ===")
        print(
            {
                "rag_used": meta.get("rag_used"),
                "retrieved_docs_total": meta.get("retrieved_docs_total"),
                "kb_sources": meta.get("kb_sources"),
                "rag_primary_docs": meta.get("rag_primary_docs"),
                "rag_wiki_docs": meta.get("rag_wiki_docs"),
            }
        )

        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()