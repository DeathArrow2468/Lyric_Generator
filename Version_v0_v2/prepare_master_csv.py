# prepare_master_csv.py

from pathlib import Path
import pandas as pd
import re


# ============================================================
# Configuration
# ============================================================

INPUT_DIR = Path(r"C:\Users\Manav\OneDrive\Desktop\LyricGenerator\Version_v0_v2\lyrics_by_artists\csv")
OUTPUT_FILE = Path(r"C:\Users\Manav\OneDrive\Desktop\LyricGenerator\Version_v0_v2\Master_csvs\master_songs.csv")

MIN_LYRIC_WORDS = 5
MIN_LYRIC_CHARS = 10


# ============================================================
# Helpers
# ============================================================

EXPECTED_COLUMNS = [
    "Artists",
    "title",
    "Album",
    "Year",
    "Date",
    "Lyric",
]


def clean_text(value):
    """
    Basic text cleaning.

    We intentionally keep this conservative.
    We don't want to alter the actual lyrics unnecessarily.
    """

    if pd.isna(value):
        return ""

    value = str(value)

    # Normalize common whitespace.
    value = value.replace("\r\n", "\n")
    value = value.replace("\r", "\n")

    # Remove null characters.
    value = value.replace("\x00", "")

    # Remove excessive spaces while preserving line breaks.
    value = re.sub(r"[ \t]+", " ", value)

    # Collapse excessive blank lines.
    value = re.sub(r"\n{3,}", "\n\n", value)

    return value.strip()


def count_words(text):
    return len(
        re.findall(
            r"\b[\w'-]+\b",
            text
        )
    )


def normalize_columns(df):
    """
    Normalize column names so small differences between files
    don't break the merge.
    """

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # Handle common capitalization/spacing differences.
    rename_map = {}

    for column in df.columns:

        normalized = column.lower().strip()

        if normalized == "artists":
            rename_map[column] = "Artists"

        elif normalized == "artist":
            rename_map[column] = "Artists"

        elif normalized == "title":
            rename_map[column] = "title"

        elif normalized == "album":
            rename_map[column] = "Album"

        elif normalized == "year":
            rename_map[column] = "Year"

        elif normalized == "date":
            rename_map[column] = "Date"

        elif normalized == "lyric":
            rename_map[column] = "Lyric"

    df = df.rename(columns=rename_map)

    return df


# ============================================================
# Find CSV files
# ============================================================

csv_files = sorted(
    INPUT_DIR.glob("*.csv")
)

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in {INPUT_DIR.resolve()}"
    )

print("=" * 70)
print("MASTER CSV CREATION")
print("=" * 70)

print(f"\nFound {len(csv_files)} CSV files.")


# ============================================================
# Read all CSV files
# ============================================================

dataframes = []

for csv_file in csv_files:

    print(f"\nReading: {csv_file.name}")

    try:

        df = pd.read_csv(
            csv_file,
            encoding="utf-8",
            dtype=str,
            keep_default_na=False,
        )

    except UnicodeDecodeError:

        print("  UTF-8 failed, trying latin-1...")

        df = pd.read_csv(
            csv_file,
            encoding="latin-1",
            dtype=str,
            keep_default_na=False,
        )

    df = normalize_columns(df)

    # Check required columns.
    missing = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        print(
            f"  WARNING: Missing columns: {missing}"
        )

        # Add missing columns as empty.
        for column in missing:
            df[column] = ""

    # Keep only our expected columns.
    df = df[EXPECTED_COLUMNS]

    # Add source file for traceability.
    df["_source_file"] = csv_file.name

    dataframes.append(df)

    print(f"  Rows: {len(df)}")


# ============================================================
# Merge
# ============================================================

print("\nMerging files...")

master = pd.concat(
    dataframes,
    ignore_index=True
)

print(
    f"Rows after merge: {len(master)}"
)


# ============================================================
# Clean text fields
# ============================================================

print("\nCleaning text...")

text_columns = [
    "Artists",
    "title",
    "Album",
    "Year",
    "Date",
    "Lyric",
]

for column in text_columns:

    master[column] = master[column].apply(
        clean_text
    )


# ============================================================
# Remove missing lyrics
# ============================================================

before = len(master)

master = master[
    master["Lyric"].str.strip() != ""
].copy()

print(
    f"Removed missing lyrics: "
    f"{before - len(master)}"
)


# ============================================================
# Remove extremely short lyrics
# ============================================================

before = len(master)

master["_lyric_word_count"] = master["Lyric"].apply(
    count_words
)

master = master[
    (master["_lyric_word_count"] >= MIN_LYRIC_WORDS)
    &
    (master["Lyric"].str.len() >= MIN_LYRIC_CHARS)
].copy()

print(
    f"Removed very short lyrics: "
    f"{before - len(master)}"
)


# ============================================================
# Remove exact duplicate songs
# ============================================================

before = len(master)

master["_lyrics_normalized"] = (
    master["Lyric"]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

master = master.drop_duplicates(
    subset=[
        "Artists",
        "title",
        "_lyrics_normalized",
    ],
    keep="first",
).copy()

print(
    f"Removed duplicate songs: "
    f"{before - len(master)}"
)


# ============================================================
# Assign unique song IDs
# ============================================================

master = master.reset_index(drop=True)

master.insert(
    0,
    "song_id",
    [
        f"SONG_{i:07d}"
        for i in range(1, len(master) + 1)
    ]
)


# ============================================================
# Remove helper columns
# ============================================================

master = master.drop(
    columns=[
        "_lyric_word_count",
        "_lyrics_normalized",
        "_source_file",
    ]
)


# ============================================================
# Save
# ============================================================

master.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8",
)


# ============================================================
# Statistics
# ============================================================

lyric_lengths = master["Lyric"].apply(
    count_words
)

print("\n")
print("=" * 70)
print("FINAL DATASET")
print("=" * 70)

print(
    f"Total songs:       {len(master):,}"
)

print(
    f"Unique artists:    "
    f"{master['Artists'].nunique():,}"
)

print(
    f"Average lyric words: "
    f"{lyric_lengths.mean():.1f}"
)

print(
    f"Median lyric words:  "
    f"{lyric_lengths.median():.1f}"
)

print(
    f"Minimum lyric words:  "
    f"{lyric_lengths.min()}"
)

print(
    f"Maximum lyric words:  "
    f"{lyric_lengths.max()}"
)

print(
    f"\nSaved to:\n"
    f"{OUTPUT_FILE.resolve()}"
)

print("=" * 70)