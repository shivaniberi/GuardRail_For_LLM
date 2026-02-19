# runners/run_guardrail_ollama.py
# Full runnable script to use:
# - Ollama model (qwen0.5 / qwen2.5 / llama3.1 / mistral / phi3)
# - your guardrail pipeline
# - interactive: type prompt manually in terminal
# - prints TWO replies:
#     1) raw LLM (no guardrail / no RAG)
#     2) final guardrailed response (may be blocked, may use RAG)
#
# IMPORTANT:
# This assumes your core GuardrailConfig supports BOTH:
#   - rag_dataset_path (your original KB)
#   - wiki_rag_dataset_path (your Wikipedia KB)
#
# Wikipedia parquet you uploaded:
#   s3://guardrail-group-bucket/knowledge_base/wikipedia/latest/simplewiki_articles.parquet

from core.phi3_guardrail_implementation import GuardrailConfig, Phi3GuardrailSystem


def main():
    # CHANGE model here if needed:
    # "qwen0.5", "qwen2.5", "llama3", "mistral", "phi3"
    model_to_test = "llama3"

    config = GuardrailConfig(
        enable_input_validation=True,
        enable_output_verification=True,
        enable_rag=True,
        enable_logging=True,

        # Primary (your original KB)
        rag_dataset_path="s3://guardrail-group-bucket/processed/train.parquet",

        # Wikipedia KB (NEW)
        wiki_rag_dataset_path="s3://guardrail-group-bucket/knowledge_base/wikipedia/latest/simplewiki_articles.parquet",

        # Model
        ollama_model_name=model_to_test,

        # Show raw model output for comparison
        always_return_raw_llm=True,
    )

    system = Phi3GuardrailSystem(config)

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

        # Helpful debug: show whether RAG was used
        meta = result.get("metadata", {}) or {}
        print("\n=== RAG METADATA ===")
        print(
            {
                "rag_used": meta.get("rag_used"),
                "retrieved_docs_total": meta.get("retrieved_docs"),
                "kb_sources": meta.get("kb_sources"),
            }
        )

        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
