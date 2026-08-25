# train_v0.py

from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

DATA_DIR = Path("dataset_v0")
OUTPUT_DIR = Path("models/v0_qwen3_4b")

MAX_SEQ_LENGTH = 2048

# First pipeline test
NUM_EPOCHS = 1

LEARNING_RATE = 2e-4

BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8

SEED = 42


# ============================================================
# Hardware check
# ============================================================

print("=" * 60)
print("Hardware")
print("=" * 60)

print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required for this training run.")

print("GPU:", torch.cuda.get_device_name(0))

gpu_memory = (
    torch.cuda.get_device_properties(0).total_memory
    / 1024**3
)

print(f"GPU memory: {gpu_memory:.2f} GB")


# ============================================================
# Load dataset
# ============================================================

dataset = load_dataset(
    "json",
    data_files={
        "train": str(
            DATA_DIR / "train_output_transcription.jsonl"
        ),
        "validation": str(
            DATA_DIR / "val_output_transcription.jsonl"
        ),
    },
)

print("\nDataset:")
print(dataset)


# ============================================================
# Format examples as conversations
# ============================================================

def format_example(example):

    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write song lyrics based on the following "
                    "request:\n\n"
                    + example["prompt"]
                ),
            },
            {
                "role": "assistant",
                "content": example["lyrics"],
            },
        ]
    }


dataset = dataset.map(
    format_example,
    remove_columns=dataset["train"].column_names,
)

# ============================================================
# Tiny smoke test
# ============================================================

dataset["train"] = dataset["train"].select(
    range(min(100, len(dataset["train"])))
)

dataset["validation"] = dataset["validation"].select(
    range(min(20, len(dataset["validation"])))
)

print("\nSmoke-test dataset:")
print(dataset)

print("\nFormatted example:")
print(dataset["train"][0])


# ============================================================
# Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# 4-bit QLoRA configuration
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# Load base model
# ============================================================

print("\nLoading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

model.config.use_cache = False


# ============================================================
# LoRA configuration
# ============================================================

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],

    bias="none",
    task_type="CAUSAL_LM",
)


# ============================================================
# Training configuration
# ============================================================

training_args = SFTConfig(
    output_dir=str(OUTPUT_DIR),

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    gradient_accumulation_steps=GRADIENT_ACCUMULATION,

    learning_rate=LEARNING_RATE,

    logging_steps=5,

    eval_strategy="steps",
    eval_steps=50,

    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,

    bf16=True,

    lr_scheduler_type="cosine",

    weight_decay=0.01,

    seed=SEED,

    report_to="none",

    max_length=MAX_SEQ_LENGTH,

    packing=False,

    gradient_checkpointing=True,
)


# ============================================================
# Trainer
# ============================================================

trainer = SFTTrainer(
    model=model,

    args=training_args,

    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],

    processing_class=tokenizer,

    peft_config=peft_config,
)


# ============================================================
# Train
# ============================================================

print("\n")
print("=" * 60)
print("Starting V0 training")
print("=" * 60)

trainer.train()


# ============================================================
# Save
# ============================================================

print("\nSaving V0 adapter...")

trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

print("\n")
print("=" * 60)
print("Training complete")
print("=" * 60)

print(f"Model adapter saved to: {OUTPUT_DIR}")