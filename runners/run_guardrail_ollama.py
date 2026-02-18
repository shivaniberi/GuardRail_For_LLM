# runners/run_guardrail_ollama.py
# Full runnable script to use:
# - your S3 parquet as RAG dataset
# - Ollama model (qwen0.5 / qwen2.5 / llama3.1 / mistral / phi3)
# - your guardrail pipeline
# - prints TWO replies:
#     1) raw LLM (no guardrail / no RAG)
#     2) final guardrailed response (may be blocked, may use RAG)

# runners/run_guardrail_ollama.py
# Interactive version: enter prompt manually in terminal

from core.phi3_guardrail_implementation import GuardrailConfig, Phi3GuardrailSystem


def main():
    # CHANGE model here if needed
    # "qwen0.5", "qwen2.5", "llama3.1", "mistral", "phi3"
    model_to_test = "qwen0.5"

    config = GuardrailConfig(
        enable_input_validation=True,
        enable_output_verification=True,
        enable_rag=True,
        enable_logging=True,
        rag_dataset_path="s3://guardrail-group-bucket/processed/train.parquet",
        ollama_model_name=model_to_test,
        always_return_raw_llm=True,
    )

    system = Phi3GuardrailSystem(config)

    print("\nGuardrail system ready.")
    print("Type your prompt below (type 'exit' to quit)\n")

    while True:
        prompt = input(">>> ")

        if prompt.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        result = system.generate_with_guardrails(
            prompt=prompt,
            max_new_tokens=256,
            use_rag=True,
        )

        print("\n=== MODEL ===")
        print(model_to_test)

        print("\n=== RAW LLM RESPONSE (no guardrail) ===")
        print(result.get("metadata", {}).get("raw_llm_response"))

        print("\n=== GUARDED FINAL RESPONSE ===")
        print(result["response"])

        print("\n=== INPUT GUARDRAIL ===")
        print(result.get("guardrails", {}).get("input"))

        print("\n=== OUTPUT GUARDRAIL ===")
        print(result.get("guardrails", {}).get("output"))
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
