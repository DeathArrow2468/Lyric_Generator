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

# Tiny first run.
NUM_EPOCHS = 1

LEARNING_RATE = 2e-4

BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8

SEED = 42


# ============================================================
# Load dataset
# ============================================================

dataset = load_dataset(
    "json",
    data_files={
        "train": str(DATA_DIR / "train_output_transcription.jsonl"),
        "validation": str(DATA_DIR / "val_output_transcription.jsonl"),
    },
)

print(dataset)


# ============================================================
# Convert our records into conversational examples
# ============================================================

def format_example(example):

    messages = [
        {
            "role": "user",
            "content": (
                "Write song lyrics based on the following request:\n\n"
                + example["prompt"]
            ),
        },
        {
            "role": "assistant",
            "content": example["lyrics"],
        },
    ]

    return {
        "messages": messages
    }


dataset = dataset.map(
    format_example,
    remove_columns=dataset["train"].column_names,
)


# ============================================================
# Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


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
# Base model
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

model.config.use_cache = False


# ============================================================
# LoRA
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

    warmup_ratio=0.05,

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

print("\nStarting V0 training...\n")

trainer.train()


# ============================================================
# Save
# ============================================================

print("\nSaving V0 adapter...")

trainer.save_model(str(OUTPUT_DIR))

tokenizer.save_pretrained(str(OUTPUT_DIR))

print("\nTraining complete.")
print(f"Model saved to: {OUTPUT_DIR}")