# generate_v0.py

import argparse
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


# ============================================================
# Configuration
# ============================================================

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

ADAPTER_PATH = Path("models/v0_qwen3_4b")

MAX_INPUT_LENGTH = 2048

DEFAULT_MAX_NEW_TOKENS = 512

# Generation parameters.
TEMPERATURE = 0.8
TOP_P = 0.9
TOP_K = 40
REPETITION_PENALTY = 1.05


# ============================================================
# Argument parsing
# ============================================================

parser = argparse.ArgumentParser(
    description="Generate lyrics using V0."
)

parser.add_argument(
    "--prompt",
    type=str,
    required=True,
    help="Natural-language lyric generation prompt.",
)

parser.add_argument(
    "--max-new-tokens",
    type=int,
    default=DEFAULT_MAX_NEW_TOKENS,
    help="Maximum number of tokens to generate.",
)

args = parser.parse_args()


# ============================================================
# Hardware
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required for V0 generation."
    )

print("=" * 60)
print("V0 Lyric Generator")
print("=" * 60)

print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Check adapter
# ============================================================

if not ADAPTER_PATH.exists():
    raise FileNotFoundError(
        f"Could not find trained adapter at: {ADAPTER_PATH}"
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
# Load tokenizer
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
)


# ============================================================
# Load base model
# ============================================================

print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)


# ============================================================
# Load LoRA adapter
# ============================================================

print("Loading V0 LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)

model.eval()


# ============================================================
# Build prompt
# ============================================================

user_prompt = (
    "Write song lyrics based on the following request:\n\n"
    + args.prompt
)

messages = [
    {
        "role": "user",
        "content": user_prompt,
    }
]


# ============================================================
# Apply Qwen chat template
# ============================================================

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)


# ============================================================
# Tokenize
# ============================================================

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=MAX_INPUT_LENGTH,
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

        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,

        repetition_penalty=REPETITION_PENALTY,

        # Don't generate padding.
        pad_token_id=tokenizer.eos_token_id,
    )


# ============================================================
# Extract generated portion
# ============================================================

input_length = inputs["input_ids"].shape[1]

generated_tokens = outputs[0][input_length:]

generated_text = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True,
)


# ============================================================
# Output
# ============================================================

print("=" * 60)
print("GENERATED LYRICS")
print("=" * 60)
print()

print(generated_text.strip())

print()
print("=" * 60)