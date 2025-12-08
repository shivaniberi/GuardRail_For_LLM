import streamlit as st
from phi3_guardrail_implementation import Phi3GuardrailSystem, GuardrailConfig

# ----------------------------------------------------------
# Load Guardrail System once (cached for speed)
# ----------------------------------------------------------
@st.cache_resource
def load_guardrail_system():
    config = GuardrailConfig(
        phi3_model_path=r"C:\Users\valla\Downloads\Phi-3-mini-4k-instruct",
        enable_input_validation=True,
        enable_output_verification=True,
        enable_rag=True,
        safety_prompts_path="s3://guardrail-group-bucket/preprocessed/safety_prompt/2025/11/02/safety_prompt.parquet",
        rag_dataset_path="s3://guardrail-group-bucket/preprocessed/squad_qa/2025/11/02/squad_qa.parquet",
    )
    return Phi3GuardrailSystem(config)

guardrails = load_guardrail_system()

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
            max_new_tokens=80,
            temperature=0.0,
            use_rag=True
        )

    st.subheader("Guardrail Response:")
    st.write(result["response"])
