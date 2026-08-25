# prepare_data_v0.py

from pathlib import Path
import json
import re
import statistics


# ============================================================
# Configuration
# ============================================================

INPUT_DIR = Path(r"C:\Users\Manav\OneDrive\Desktop\LyricGenerator\audio_metadata_transcription")
OUTPUT_DIR = Path(r"C:\Users\Manav\OneDrive\Desktop\LyricGenerator\Version_v0\dataset_v0")

SPLITS = ["val_output_transcription", "train_output_transcription", "test_output_transcription"]

# Minimum number of words required in a lyric target.
MIN_WORDS = 10

# Minimum number of non-whitespace characters.
MIN_CHARS = 30


# ============================================================
# Helpers
# ============================================================

def load_jsonl(path: Path):
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(
                    f"[WARNING] Invalid JSON in {path}, "
                    f"line {line_number}: {e}"
                )

    return records


def normalize_lyrics(lyrics):
    """
    Minimal cleaning.

    We deliberately do NOT aggressively modify the transcript.
    V0 should use approximately the raw Whisper output.
    """

    if lyrics is None:
        return ""

    if not isinstance(lyrics, str):
        lyrics = str(lyrics)

    lyrics = lyrics.strip()

    # Normalize Windows line endings.
    lyrics = lyrics.replace("\r\n", "\n")
    lyrics = lyrics.replace("\r", "\n")

    # Remove excessive blank lines.
    lyrics = re.sub(r"\n{3,}", "\n\n", lyrics)

    # Remove trailing whitespace from lines.
    lyrics = "\n".join(
        line.rstrip()
        for line in lyrics.split("\n")
    )

    return lyrics.strip()


def count_words(text):
    return len(
        re.findall(r"\b[\w'-]+\b", text)
    )


def is_instrumental_or_non_lyrical(lyrics):
    """
    Detect a few obvious non-lyric cases.

    This is intentionally conservative.
    """

    normalized = lyrics.strip().lower()

    obvious_labels = {
        "instrumental",
        "[instrumental]",
        "(instrumental)",
        "music",
        "[music]",
        "(music)",
        "no lyrics",
        "no vocals",
    }

    if normalized in obvious_labels:
        return True

    return False


def extract_v0_example(record):
    """
    Convert a raw annotation record into the minimal V0 example.
    """

    lyrics = normalize_lyrics(
        record.get("lyrics")
    )

    analysis = record.get("analysis") or {}

    prompt = analysis.get(
        "lyric_generation_prompt"
    )

    if not isinstance(prompt, str):
        prompt = ""

    prompt = prompt.strip()

    # --------------------------------------------------------
    # Validate lyrics
    # --------------------------------------------------------

    if not lyrics:
        return None, "missing_lyrics"

    if len(lyrics) < MIN_CHARS:
        return None, "lyrics_too_short"

    word_count = count_words(lyrics)

    if word_count < MIN_WORDS:
        return None, "lyrics_too_short"

    if is_instrumental_or_non_lyrical(lyrics):
        return None, "non_lyrical"

    # --------------------------------------------------------
    # Validate prompt
    # --------------------------------------------------------

    if not prompt:
        return None, "missing_prompt"

    # --------------------------------------------------------
    # Construct V0 example
    # --------------------------------------------------------

    example = {
        "example_id": record.get("example_id"),
        "track_id": record.get("track_id"),
        "prompt": prompt,
        "lyrics": lyrics,
    }

    return example, None


def save_jsonl(records, path: Path):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )


# ============================================================
# Process a split
# ============================================================

def process_split(split):
    input_path = INPUT_DIR / f"{split}.jsonl"
    output_path = OUTPUT_DIR / f"{split}.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find {input_path}"
        )

    records = load_jsonl(input_path)

    valid = []
    rejected = {}

    for record in records:

        example, reason = extract_v0_example(record)

        if example is not None:
            valid.append(example)
        else:
            rejected[reason] = rejected.get(reason, 0) + 1

    save_jsonl(valid, output_path)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    lyric_lengths = [
        count_words(example["lyrics"])
        for example in valid
    ]

    prompt_lengths = [
        count_words(example["prompt"])
        for example in valid
    ]

    stats = {
        "split": split,
        "input_records": len(records),
        "valid_records": len(valid),
        "rejected_records": len(records) - len(valid),
        "rejection_reasons": rejected,
        "lyric_word_count": {
            "min": min(lyric_lengths) if lyric_lengths else 0,
            "max": max(lyric_lengths) if lyric_lengths else 0,
            "mean": (
                round(statistics.mean(lyric_lengths), 2)
                if lyric_lengths
                else 0
            ),
            "median": (
                round(statistics.median(lyric_lengths), 2)
                if lyric_lengths
                else 0
            ),
        },
        "prompt_word_count": {
            "min": min(prompt_lengths) if prompt_lengths else 0,
            "max": max(prompt_lengths) if prompt_lengths else 0,
            "mean": (
                round(statistics.mean(prompt_lengths), 2)
                if prompt_lengths
                else 0
            ),
            "median": (
                round(statistics.median(prompt_lengths), 2)
                if prompt_lengths
                else 0
            ),
        },
    }

    stats_path = OUTPUT_DIR / f"{split}_stats.json"

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(
            stats,
            f,
            indent=2,
            ensure_ascii=False
        )

    return stats


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_stats = {}

    for split in SPLITS:

        print()
        print("=" * 60)
        print(f"Processing {split}")
        print("=" * 60)

        stats = process_split(split)

        all_stats[split] = stats

        print(f"Input:    {stats['input_records']}")
        print(f"Valid:    {stats['valid_records']}")
        print(f"Rejected: {stats['rejected_records']}")

        if stats["rejection_reasons"]:
            print("Rejected because:")

            for reason, count in sorted(stats["rejection_reasons"].items()):
                print(f"  {reason}: {count}")

        print(
            f"Lyric words — "
            f"mean: {stats['lyric_word_count']['mean']}, "
            f"median: {stats['lyric_word_count']['median']}"
        )

    # --------------------------------------------------------
    # Combined statistics
    # --------------------------------------------------------

    combined_path = OUTPUT_DIR / "dataset_stats.json"

    with combined_path.open("w", encoding="utf-8") as f:
        json.dump(
            all_stats,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Statistics:       {combined_path}")


if __name__ == "__main__":
    main()