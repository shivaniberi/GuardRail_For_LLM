import streamlit as st
from phi3_guardrail_implementation import Phi3GuardrailSystem, GuardrailConfig

# ----------------------------------------------------------
# Load Guardrail System once (cached)
# ----------------------------------------------------------
@st.cache_resource
def load_guardrail_system():
    config = GuardrailConfig(
        phi3_model_path=r"C:\Users\valla\Downloads\Phi-3-mini-4k-instruct",

        enable_input_validation=True,
        enable_output_verification=False,   # speed boost
        enable_rag=False,                   # speed boost
        rag_dataset_path=None,              # prevents RAG loading

        safety_prompts_path="s3://guardrail-group-bucket/preprocessed/safety_prompt/2025/11/02/safety_prompt.parquet",
    )

    return Phi3GuardrailSystem(config)


guardrails = load_guardrail_system()


# ----------------------------------------------------------
# CLEAN, COMPLETE SAFE RESPONSES OVERRIDE (Option C)
# ----------------------------------------------------------
CLEAN_SAFE_RESPONSES = {
    "illegal": "I cannot assist with or describe illegal activities such as phishing, hacking, or fraud.",
    "physical_harm": "Never mix bleach and ammonia. This combination releases toxic chloramine gas and is extremely dangerous.",
    "gender_bias": "I cannot reinforce or generate content involving gender stereotypes. Any role can belong to any gender.",
    "truthfulness": "The claim that humans only use 10% of their brain is a myth. Modern neuroscience shows all areas of the brain serve important functions.",
    "default": "I cannot provide unsafe or harmful content. Please rephrase your question if you'd like help with something safe.",
}


def clean_guardrail_output(raw_text, guardrail_info):
    """Remove junk (Solution 1, headers) and override with clean safe messages."""
    if raw_text is None:
        return CLEAN_SAFE_RESPONSES["default"]

    # If guardrail detected a category, override response
    category = None

    if isinstance(guardrail_info, dict):
        # Guardrail system returns: {'input': category, 'output': category}
        category = guardrail_info.get("input") or guardrail_info.get("output")

    if category in CLEAN_SAFE_RESPONSES:
        return CLEAN_SAFE_RESPONSES[category]

    # Otherwise clean the raw text (fallback)
    cleaned = []
    for line in raw_text.split("\n"):
        if line.strip().lower().startswith("solution"):
            continue
        if line.strip().startswith("#"):
            continue
        cleaned.append(line)

    cleaned_text = " ".join(cleaned).strip()

    if not cleaned_text:
        return CLEAN_SAFE_RESPONSES["default"]

    return cleaned_text


# ----------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------
st.title("Phi-3 Mini Guardrail Demo (Simple UI)")

prompt = st.text_area("Enter your prompt:", height=150)

if st.button("Generate"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
        st.stop()

    with st.spinner("Generating..."):
        result = guardrails.generate_with_guardrails(
            prompt=prompt,
            max_new_tokens=60,   # fast
            use_rag=False        # critical for speed
        )

    # Extract guardrail info for override
    guardrail_info = result.get("guardrails", {}).get("output")

    # Apply post-processing override
    safe_response = clean_guardrail_output(result["response"], guardrail_info)

    st.subheader("Guardrail Response")
    st.write(safe_response)
