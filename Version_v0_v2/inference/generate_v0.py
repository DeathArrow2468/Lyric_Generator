# inference/generate_v0.py

import argparse

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

from config import (
    BASE_MODEL,
    MODEL_OUTPUT_DIR,
    DEFAULT_MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    TOP_K,
    REPETITION_PENALTY,
)


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompt",
    type=str,
    required=True,
)

parser.add_argument(
    "--max-new-tokens",
    type=int,
    default=DEFAULT_MAX_NEW_TOKENS,
)

parser.add_argument(
    "--temperature",
    type=float,
    default=TEMPERATURE,
)

parser.add_argument(
    "--top-p",
    type=float,
    default=TOP_P,
)

parser.add_argument(
    "--top-k",
    type=int,
    default=TOP_K,
)

args = parser.parse_args()


# ============================================================
# Hardware
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required."
    )


# ============================================================
# Quantization
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)


# ============================================================
# Base model
# ============================================================

print(
    f"Loading base model: {BASE_MODEL}"
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)


# ============================================================
# Load LoRA
# ============================================================

print(
    f"Loading adapter: {MODEL_OUTPUT_DIR}"
)

model = PeftModel.from_pretrained(
    base_model,
    MODEL_OUTPUT_DIR,
)

model.eval()


# ============================================================
# Prompt
# ============================================================

messages = [
    {
        "role": "user",
        "content": (
            "Write song lyrics based on the "
            "following request:\n\n"
            + args.prompt
        ),
    }
]


# ============================================================
# Chat template
# ============================================================

inputs = tokenizer.apply_chat_template(
    messages,

    tokenize=True,

    add_generation_prompt=True,

    return_dict=True,

    return_tensors="pt",
)


inputs = {
    key: value.to(model.device)
    for key, value in inputs.items()
}


# ============================================================
# Generate
# ============================================================

print("\nGenerating...\n")

with torch.inference_mode():

    outputs = model.generate(

        **inputs,

        max_new_tokens=args.max_new_tokens,

        do_sample=True,

        temperature=args.temperature,

        top_p=args.top_p,

        top_k=args.top_k,

        repetition_penalty=REPETITION_PENALTY,

        pad_token_id=tokenizer.eos_token_id,
    )


# ============================================================
# Decode only newly generated tokens
# ============================================================

input_length = (
    inputs["input_ids"].shape[-1]
)

generated_ids = outputs[0][input_length:]

lyrics = tokenizer.decode(
    generated_ids,
    skip_special_tokens=True,
)


# ============================================================
# Output
# ============================================================

print("=" * 70)
print("GENERATED LYRICS")
print("=" * 70)
print()

print(lyrics.strip())

print()
print("=" * 70)