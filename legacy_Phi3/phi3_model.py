import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------
# LOAD MODEL + TOKENIZER
# ---------------------------------------------------------

MODEL_PATH = r"C:\Users\valla\Downloads\Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,  
)

model.to("cpu")


# ---------------------------------------------------------
# GENERATE FUNCTION
# ---------------------------------------------------------

def generate_phi3(prompt: str, max_new_tokens=200):
    # Build proper chat messages
    messages = [
        {"role": "user", "content": prompt}
    ]

    # Encode with chat template
    encoded = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    )

    # Build attention mask to avoid warnings
    attention_mask = torch.ones_like(encoded)

    # Generate output
    output = model.generate(
        encoded,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode response
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # ---------------------------------------------------------
    # Extract ONLY assistant answer
    # ---------------------------------------------------------

    if "<|assistant|>" in decoded:
        return decoded.split("<|assistant|>")[-1].strip()

    # fallback: remove the prompt if echoed
    if prompt in decoded:
        return decoded.replace(prompt, "").strip()

    return decoded.strip()

