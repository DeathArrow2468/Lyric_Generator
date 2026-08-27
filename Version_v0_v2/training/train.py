# training/train_v0.py

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

from Version_v0_v2.config import (
    BASE_MODEL,
    V0_DATA_DIR,
    MODEL_OUTPUT_DIR,
    MAX_SEQ_LENGTH,
    NUM_EPOCHS,
    LEARNING_RATE,
    TRAIN_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    SEED,
)


# ============================================================
# Hardware
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required."
    )

print("GPU:", torch.cuda.get_device_name(0))

print(
    "VRAM:",
    round(
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3,
        2
    ),
    "GB"
)


# ============================================================
# Load data
# ============================================================

dataset = load_dataset(
    "json",
    data_files={
        "train": str(
            V0_DATA_DIR / "train.jsonl"
        ),
        "validation": str(
            V0_DATA_DIR / "val.jsonl"
        ),
    },
)

print(dataset)


# ============================================================
# Convert to conversational format
# ============================================================

def format_example(example):

    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write song lyrics based on the "
                    "following request:\n\n"
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


print("\nExample:")
print(dataset["train"][0])


# ============================================================
# Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# 4-bit quantization
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

print("\nLoading base model:")
print(BASE_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

model.config.use_cache = False


# ============================================================
# LoRA
# ============================================================

peft_config = LoraConfig(
    r=LORA_R,

    lora_alpha=LORA_ALPHA,

    lora_dropout=LORA_DROPOUT,

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

    output_dir=str(
        MODEL_OUTPUT_DIR
    ),

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=TRAIN_BATCH_SIZE,

    per_device_eval_batch_size=TRAIN_BATCH_SIZE,

    gradient_accumulation_steps=(
        GRADIENT_ACCUMULATION_STEPS
    ),

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

    # IMPORTANT:
    # Only train on the assistant/lyrics portion.
    assistant_only_loss=True,
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

print()
print("=" * 70)
print("STARTING V0 TRAINING")
print("=" * 70)

trainer.train()


# ============================================================
# Save adapter
# ============================================================

print("\nSaving LoRA adapter...")

trainer.save_model(
    str(MODEL_OUTPUT_DIR)
)

tokenizer.save_pretrained(
    str(MODEL_OUTPUT_DIR)
)

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    "Adapter:",
    MODEL_OUTPUT_DIR
)