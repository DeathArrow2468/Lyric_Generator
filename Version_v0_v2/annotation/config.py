# config.py

from pathlib import Path


# ============================================================
# Annotation model
# ============================================================

ANNOTATION_MODEL = "qwen3:14b"


# ============================================================
# Generation / training model
# ============================================================

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


# ============================================================
# Data
# ============================================================

MASTER_CSV = Path("Master_csv/master_songs.csv")

ANNOTATED_JSONL = Path(
    "data/annotated/annotated_songs.jsonl"
)

V0_DATA_DIR = Path("data/v0")


# ============================================================
# Model output
# ============================================================

MODEL_OUTPUT_DIR = Path(
    "models/v0_qwen3_4b"
)


# ============================================================
# Training
# ============================================================

MAX_SEQ_LENGTH = 2048

NUM_EPOCHS = 2

LEARNING_RATE = 1e-4

TRAIN_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 8

LORA_R = 16

LORA_ALPHA = 32

LORA_DROPOUT = 0.05

SEED = 42


# ============================================================
# Generation
# ============================================================

DEFAULT_MAX_NEW_TOKENS = 512

TEMPERATURE = 0.8

TOP_P = 0.9

TOP_K = 40

REPETITION_PENALTY = 1.05