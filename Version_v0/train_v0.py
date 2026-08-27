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

TRAIN_FILE = DATA_DIR / "train_output_transcription.jsonl"
VAL_FILE = DATA_DIR / "val_output_transcription.jsonl"

# Keep 2048-token context as requested.
MAX_SEQ_LENGTH = 2048

# ------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------

# First run only.
# Once the pipeline works, set these to None.
SMOKE_TRAIN_SIZE = None
SMOKE_VAL_SIZE = None

NUM_EPOCHS = 5

LEARNING_RATE = 1e-4

# RTX 4060 8 GB
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8

SEED = 42


# ============================================================
# Hardware check
# ============================================================

print("=" * 60)
print("Hardware")
print("=" * 60)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required for this training run."
    )

gpu_name = torch.cuda.get_device_name(0)

gpu_memory = (
    torch.cuda.get_device_properties(0).total_memory
    / 1024**3
)

print("CUDA available:", torch.cuda.is_available())
print("GPU:", gpu_name)
print(f"GPU memory: {gpu_memory:.2f} GB")


# ============================================================
# Dataset paths
# ============================================================

if not TRAIN_FILE.exists():
    raise FileNotFoundError(
        f"Training file not found:\n{TRAIN_FILE}"
    )

if not VAL_FILE.exists():
    raise FileNotFoundError(
        f"Validation file not found:\n{VAL_FILE}"
    )


# ============================================================
# Load dataset
# ============================================================

print()
print("=" * 60)
print("Loading dataset")
print("=" * 60)

dataset = load_dataset(
    "json",
    data_files={
        "train": str(TRAIN_FILE),
        "validation": str(VAL_FILE),
    },
)

print(dataset)


# ============================================================
# Validate dataset
# ============================================================

required_fields = {
    "prompt",
    "lyrics",
}

for split in ["train", "validation"]:

    columns = set(dataset[split].column_names)

    missing = required_fields - columns

    if missing:
        raise ValueError(
            f"{split} is missing fields: {missing}"
        )


# ============================================================
# Tokenizer
# ============================================================

print()
print("=" * 60)
print("Loading tokenizer")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# Format examples using Qwen chat template
# ============================================================

def format_example(example):

    messages = [
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

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {
        "text": text
    }


print("\nFormatting dataset...")

dataset = dataset.map(
    format_example,
    remove_columns=dataset["train"].column_names,
)


# ============================================================
# Smoke-test subset
# ============================================================

if SMOKE_TRAIN_SIZE is not None:

    dataset["train"] = dataset["train"].select(
        range(
            min(
                SMOKE_TRAIN_SIZE,
                len(dataset["train"])
            )
        )
    )

if SMOKE_VAL_SIZE is not None:

    dataset["validation"] = dataset["validation"].select(
        range(
            min(
                SMOKE_VAL_SIZE,
                len(dataset["validation"])
            )
        )
    )


print()
print("=" * 60)
print("Training dataset")
print("=" * 60)

print(dataset)

print("\nExample:")
print(dataset["train"][0]["text"])


# ============================================================
# 4-bit QLoRA configuration
# ============================================================

print()
print("=" * 60)
print("Configuring 4-bit QLoRA")
print("=" * 60)

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.bfloat16,

    bnb_4bit_use_double_quant=True,
)


# ============================================================
# Load base model
# ============================================================

print()
print("=" * 60)
print("Loading model")
print("=" * 60)

print(MODEL_NAME)

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

    # 8 GB VRAM → keep physical batch size at 1.
    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    # Effective batch size = 1 × 8 = 8.
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,

    learning_rate=LEARNING_RATE,

    logging_steps=1,

    eval_strategy="steps",

    eval_steps=5,

    save_strategy="steps",

    save_steps=5,

    save_total_limit=2,

    # RTX 4060 supports BF16 on Ampere+.
    bf16=True,

    lr_scheduler_type="cosine",

    #warmup_ratio=0.05,

    weight_decay=0.01,

    seed=SEED,

    report_to="none",

    # Requested context length.
    max_length=MAX_SEQ_LENGTH,

    packing=False,

    # Important for fitting 2048 tokens into 8 GB.
    gradient_checkpointing=True,

    gradient_checkpointing_kwargs={
        "use_reentrant": False,    },
     dataset_text_field="text",
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

    #dataset_text_field="text",
)


# ============================================================
# Train
# ============================================================

print()
print("=" * 60)
print("Starting V0 QLoRA training")
print("=" * 60)

trainer.train()


# ============================================================
# Save
# ============================================================

print()
print("=" * 60)
print("Saving V0 adapter")
print("=" * 60)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

trainer.save_model(
    str(OUTPUT_DIR)
)

tokenizer.save_pretrained(
    str(OUTPUT_DIR)
)


# ============================================================
# GPU memory report
# ============================================================

allocated = (
    torch.cuda.memory_allocated(0)
    / 1024**3
)

reserved = (
    torch.cuda.memory_reserved(0)
    / 1024**3
)

peak = (
    torch.cuda.max_memory_allocated(0)
    / 1024**3
)

print()
print("=" * 60)
print("Training complete")
print("=" * 60)

print(f"Model adapter:          {OUTPUT_DIR}")
print(f"GPU memory allocated:   {allocated:.2f} GB")
print(f"GPU memory reserved:    {reserved:.2f} GB")
print(f"Peak GPU memory:        {peak:.2f} GB")