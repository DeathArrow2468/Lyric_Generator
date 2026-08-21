from pathlib import Path
import json
import random


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path("fma_annotations.jsonl")

OUTPUT_DIR = Path("dataset")

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

SEED = 42


# ============================================================
# Load JSONL
# ============================================================

records = []

with INPUT_FILE.open("r", encoding="utf-8") as f:
    for line_number, line in enumerate(f, start=1):
        line = line.strip()

        if not line:
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON on line {line_number}: {e}")
            continue

        records.append(record)


print(f"Loaded {len(records)} records.")


# ============================================================
# Assign unique IDs
# ============================================================

for index, record in enumerate(records, start=1):

    # Five-digit ID: 00001, 00002, ...
    record["example_id"] = f"{index:05d}"


# ============================================================
# Shuffle deterministically
# ============================================================

random.seed(SEED)
random.shuffle(records)


# ============================================================
# Split
# ============================================================

n = len(records)

train_end = int(n * TRAIN_RATIO)
val_end = train_end + int(n * VAL_RATIO)

train_records = records[:train_end]
val_records = records[train_end:val_end]
test_records = records[val_end:]


print(f"Train:      {len(train_records)}")
print(f"Validation: {len(val_records)}")
print(f"Test:       {len(test_records)}")


# ============================================================
# Save JSONL
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_jsonl(records, path):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )


save_jsonl(train_records, OUTPUT_DIR / "train.jsonl")
save_jsonl(val_records, OUTPUT_DIR / "val.jsonl")
save_jsonl(test_records, OUTPUT_DIR / "test.jsonl")


print("\nSaved:")
print(OUTPUT_DIR / "train.jsonl")
print(OUTPUT_DIR / "val.jsonl")
print(OUTPUT_DIR / "test.jsonl")