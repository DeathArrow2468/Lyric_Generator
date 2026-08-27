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

# ------------------------------------------------------------
# IMPORTANT:
#
# False = test the original Qwen3-4B model
# True  = test Qwen3-4B + our V0 LoRA adapter
# ------------------------------------------------------------

USE_ADAPTER = True


# Maximum tokens allowed in the INPUT prompt.
MAX_INPUT_LENGTH = 2048

# Maximum number of NEW tokens generated.
DEFAULT_MAX_NEW_TOKENS = 512


# Generation parameters.
TEMPERATURE = 0.8
TOP_P = 0.9
TOP_K = 40
REPETITION_PENALTY = 1.1


# ============================================================
# Argument parsing
# ============================================================

parser = argparse.ArgumentParser(
    description="Generate lyrics using Qwen3-4B or V0."
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

print("=" * 60)
print("V0 Lyric Generator")
print("=" * 60)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required for generation."
    )

print("GPU:", torch.cuda.get_device_name(0))

gpu_memory = (
    torch.cuda.get_device_properties(0).total_memory
    / 1024**3
)

print(f"GPU memory: {gpu_memory:.2f} GB")

print(
    "Mode:",
    "Qwen3-4B + V0 LoRA"
    if USE_ADAPTER
    else "Base Qwen3-4B"
)


# ============================================================
# Check adapter
# ============================================================

if USE_ADAPTER:

    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(
            f"Could not find trained adapter at:\n"
            f"{ADAPTER_PATH.resolve()}"
        )

    print(
        "Adapter path:",
        ADAPTER_PATH.resolve()
    )


# ============================================================
# Quantization
# ============================================================

print("\nConfiguring 4-bit quantization...")

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

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# Load base model
# ============================================================

print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,

    quantization_config=bnb_config,

    device_map="auto",

    # Transformers 5.x prefers dtype.
    dtype=torch.bfloat16,
)


# ============================================================
# Load V0 LoRA adapter
# ============================================================

if USE_ADAPTER:

    print("Loading V0 LoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )

else:

    print("Using base Qwen3-4B without LoRA.")

    model = base_model


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


# Move inputs to the model's device.

inputs = {
    key: value.to(model.device)
    for key, value in inputs.items()
}


# ============================================================
# Show input
# ============================================================

print()
print("=" * 60)
print("INPUT")
print("=" * 60)

print(args.prompt)

print()
print(
    "Input tokens:",
    inputs["input_ids"].shape[1]
)


# ============================================================
# Generate
# ============================================================

print()
print("=" * 60)
print("Generating...")
print("=" * 60)
print()


with torch.inference_mode():

    outputs = model.generate(

        **inputs,

        max_new_tokens=args.max_new_tokens,

        do_sample=True,

        temperature=TEMPERATURE,

        top_p=TOP_P,

        top_k=TOP_K,

        repetition_penalty=REPETITION_PENALTY,

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


# ============================================================
# GPU memory statistics
# ============================================================

allocated = (
    torch.cuda.memory_allocated(0)
    / 1024**3
)

peak = (
    torch.cuda.max_memory_allocated(0)
    / 1024**3
)

print(
    f"GPU memory allocated: {allocated:.2f} GB"
)

print(
    f"Peak GPU memory:      {peak:.2f} GB"
)