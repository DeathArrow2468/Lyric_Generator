from __future__ import annotations

import argparse
import ast
import base64
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import ollama


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "gemma4:e4b"


# ============================================================
# LLM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a music analysis assistant helping create a dataset for
text-to-lyrics generation research.

You will receive:
1. audio for one song
2. metadata associated with the song

Analyze the song using BOTH sources.

Your goal is to produce a structured semantic description that can later
be used as conditioning data for a text-only lyric generation model.

Return ONLY valid JSON.

Required JSON schema:

{
  "genre": [],
  "mood": [],
  "themes": [],
  "emotional_character": "",
  "style_attributes": [],
  "narrative_concepts": [],
  "keywords": [],
  "song_structure": [],
  "description": "",
  "lyric_generation_prompt": ""
}

Rules:

- Do not invent concrete facts that cannot reasonably be inferred.
- Prefer semantic concepts over exact wording.
- Do not reproduce lyrics or distinctive lyric phrases.
- "genre" should describe the musical genre/style.
- "mood" should describe emotional qualities.
- "themes" should describe conceptual topics.
- "style_attributes" should describe characteristics useful for
  generating lyrics, such as introspective, conversational,
  repetitive, narrative, imagery-heavy, etc.
- "narrative_concepts" should describe possible narrative situations,
  relationships, conflicts, or perspectives suggested by the song.
- "keywords" should be short semantic concepts, not quotations.
- "song_structure" should contain likely section labels when
  reasonably inferable.
- "description" should be a concise overall description.
- "lyric_generation_prompt" should sound like a realistic user request
  for a lyric-generation system.

The generated lyric-generation prompt should describe the semantic
and stylistic characteristics of the song without copying lyrics.
"""


# ============================================================
# AUDIO
# ============================================================

def audio_to_base64(audio_path: Path) -> str:
    """
    Read an audio file and convert it to base64.

    Ollama's multimodal API receives the audio data encoded as base64.
    """

    with audio_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ============================================================
# METADATA
# ============================================================

def parse_list(value: Any) -> list:
    """
    FMA stores some metadata fields such as genres/tags as strings
    representing Python lists.

    Example:
        "['Hip-Hop', 'Electronic']"

    This converts them into actual Python lists.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, float) and pd.isna(value):
        return []

    if not isinstance(value, str):
        return [value]

    value = value.strip()

    if not value:
        return []

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return parsed

        return [parsed]

    except (ValueError, SyntaxError):
        return [value]

def make_json_serializable(value: Any) -> Any:
    """
    Convert Pandas/Numpy values into normal Python values that
    json.dumps() can serialize.

    This recursively handles dictionaries and lists.
    """

    # Handle dictionaries
    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(val)
            for key, val in value.items()
        }

    # Handle lists / tuples
    if isinstance(value, (list, tuple)):
        return [
            make_json_serializable(item)
            for item in value
        ]

    # Handle Pandas/Numpy scalar types
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass

    # Handle NaN / None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value

def load_fma_metadata(metadata_path: Path) -> pd.DataFrame:
    """
    Load the official FMA tracks.csv file.

    FMA tracks.csv uses a two-level column structure.
    """

    print(f"Loading metadata from: {metadata_path}")

    tracks = pd.read_csv(
        metadata_path,
        index_col=0,
        header=[0, 1],
    )

    print(f"Loaded {len(tracks)} metadata rows.")

    return tracks


def extract_track_metadata(
    tracks: pd.DataFrame,
    track_id: int,
) -> dict[str, Any]:
    """
    Extract only the metadata that is useful to our LLM.

    We deliberately do NOT send every FMA column to the model.
    """

    row = tracks.loc[track_id]

    metadata = {}

    # --------------------------------------------------------
    # Basic track information
    # --------------------------------------------------------

    metadata["track_id"] = int(track_id)

    for field in [
        "title",
        "genre_top",
        "genres",
        "genres_all",
        "tags",
        "duration",
        "language_code",
    ]:

        key = ("track", field)

        if key in tracks.columns:

            value = row[key]

            if field in ["genres", "genres_all", "tags"]:
                value = parse_list(value)

            elif pd.isna(value):
                value = None

            metadata[field] = value

    # --------------------------------------------------------
    # Artist information
    # --------------------------------------------------------

    for field in [
        "name",
        "location",
        "bio",
        "tags",
    ]:

        key = ("artist", field)

        if key in tracks.columns:

            value = row[key]

            if field == "tags":
                value = parse_list(value)

            elif pd.isna(value):
                value = None

            metadata[f"artist_{field}"] = value

    # --------------------------------------------------------
    # Album information
    # --------------------------------------------------------

    for field in [
        "title",
        "tags",
    ]:

        key = ("album", field)

        if key in tracks.columns:

            value = row[key]

            if field == "tags":
                value = parse_list(value)

            elif pd.isna(value):
                value = None

            metadata[f"album_{field}"] = value

    return make_json_serializable(metadata)


# ============================================================
# AUDIO PATH
# ============================================================

def find_audio_file(
    audio_root: Path,
    track_id: int,
) -> Path | None:
    """
    Locate an FMA audio file.

    FMA normally stores track 123 as:

        000/000123.mp3

    and track 12345 as:

        012/012345.mp3
    """

    track_string = str(track_id).zfill(6)

    directory = track_string[:3]

    for extension in [
        ".mp3",
        ".wav",
        ".flac",
        ".m4a",
        ".ogg",
    ]:

        candidate = (
            audio_root
            / directory
            / f"{track_string}{extension}"
        )

        if candidate.exists():
            return candidate

    return None


# ============================================================
# LLM
# ============================================================

def build_user_prompt(
    metadata: dict[str, Any],
) -> str:

    return f"""
Analyze this song using BOTH the supplied audio and metadata.

Song metadata:

{json.dumps(
    metadata,
    indent=2,
    ensure_ascii=False,
)}

Return ONLY the JSON object described in the system instructions.
"""


def clean_json_response(raw: str) -> str:
    """
    Remove Markdown code fences if the model adds them.
    """

    raw = raw.strip()

    if raw.startswith("```json"):
        raw = raw[len("```json"):].strip()

    elif raw.startswith("```"):
        raw = raw[len("```"):].strip()

    if raw.endswith("```"):
        raw = raw[:-3].strip()

    return raw


def analyze_song(
    audio_path: Path,
    metadata: dict[str, Any],
    retries: int = 2,
) -> dict[str, Any]:

    print("    Encoding audio...")

    audio_b64 = audio_to_base64(audio_path)

    user_prompt = build_user_prompt(metadata)

    last_error = None

    for attempt in range(retries + 1):

        try:

            print(
                f"    Sending to {MODEL} "
                f"(attempt {attempt + 1})..."
            )

            response = ollama.chat(

                model=MODEL,

                messages=[

                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },

                    {
                        "role": "user",
                        "content": user_prompt,

                        # Audio is supplied here.
                        "images": [audio_b64],
                    },
                ],

                # Important for Gemma 4 audio experiments.
                think=False,

                options={
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "num_predict": 1200,
                },
            )

            raw = response.message.content.strip()

            raw = clean_json_response(raw)

            result = json.loads(raw)

            if not isinstance(result, dict):
                raise ValueError(
                    "Model returned JSON but not a JSON object."
                )

            return result

        except Exception as exc:

            last_error = exc

            print(
                f"    ERROR: {exc}",
                file=sys.stderr,
            )

            if attempt < retries:

                wait_seconds = 2 ** attempt

                print(
                    f"    Retrying in {wait_seconds}s..."
                )

                time.sleep(wait_seconds)

    raise RuntimeError(
        f"Failed to analyze {audio_path}: {last_error}"
    )


# ============================================================
# RESUME SUPPORT
# ============================================================

def load_existing_ids(
    output_path: Path,
) -> set[int]:
    """
    Read already processed track IDs.

    This allows the program to resume if it is interrupted.
    """

    completed = set()

    if not output_path.exists():
        return completed

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(line)

                track_id = record.get("track_id")

                if track_id is not None:
                    completed.add(int(track_id))

            except (
                json.JSONDecodeError,
                ValueError,
            ):

                # Ignore malformed lines.
                continue

    return completed


# ============================================================
# DATASET PROCESSING
# ============================================================

def process_dataset(
    audio_root: Path,
    metadata_path: Path,
    output_path: Path,
    limit: int | None = None,
) -> None:

    # --------------------------------------------------------
    # Load FMA metadata
    # --------------------------------------------------------

    tracks = load_fma_metadata(metadata_path)

    # --------------------------------------------------------
    # Resume support
    # --------------------------------------------------------

    completed_ids = load_existing_ids(output_path)

    print(
        f"Already processed: {len(completed_ids)} tracks."
    )

    # --------------------------------------------------------
    # Prepare output
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed = 0

    # --------------------------------------------------------
    # Process tracks
    # --------------------------------------------------------

    with output_path.open(
        "a",
        encoding="utf-8",
    ) as output_file:

        for track_id in tracks.index:

            track_id = int(track_id)

            # Skip tracks already processed.
            if track_id in completed_ids:
                continue

            # Respect test limit.
            if (
                limit is not None
                and processed >= limit
            ):
                break

            # ------------------------------------------------
            # Find audio
            # ------------------------------------------------

            audio_path = find_audio_file(
                audio_root,
                track_id,
            )

            if audio_path is None:

                print(
                    f"[SKIP] Audio not found for "
                    f"track {track_id}"
                )

                continue

            # ------------------------------------------------
            # Extract metadata
            # ------------------------------------------------

            metadata = extract_track_metadata(
                tracks,
                track_id,
            )

            print()
            print("=" * 70)
            print(
                f"[{processed + 1}] "
                f"Track {track_id}"
            )
            print(
                f"    Audio: {audio_path}"
            )
            print(
                f"    Title: "
                f"{metadata.get('title')}"
            )
            print(
                f"    Genre: "
                f"{metadata.get('genre_top')}"
            )

            # ------------------------------------------------
            # Ask LLM
            # ------------------------------------------------

            try:

                analysis = analyze_song(
                    audio_path=audio_path,
                    metadata=metadata,
                )

                # ------------------------------------------------
                # Construct output record
                # ------------------------------------------------

                record = {
                    "track_id": track_id,

                    "metadata": metadata,

                    "analysis": analysis,
                }

                # ------------------------------------------------
                # Write immediately
                # ------------------------------------------------

                output_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                output_file.flush()

                completed_ids.add(track_id)

                processed += 1

                print("    SUCCESS")

            except Exception as exc:

                print(
                    f"    FAILED: {exc}",
                    file=sys.stderr,
                )

    print()
    print("=" * 70)
    print(
        f"Finished. Processed {processed} new tracks."
    )


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Annotate FMA tracks using "
            "Ollama Gemma 4 E4B."
        )
    )

    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help=(
            "Root directory containing "
            "FMA audio files."
        ),
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help=(
            "Path to FMA tracks.csv."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "llm_outputs/fma_annotations.jsonl"
        ),
        help=(
            "Output JSONL file."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of NEW tracks "
            "to process."
        ),
    )

    args = parser.parse_args()

    process_dataset(
        audio_root=args.audio_dir,
        metadata_path=args.metadata,
        output_path=args.output,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()