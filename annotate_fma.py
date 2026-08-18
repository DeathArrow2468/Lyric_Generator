from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import ollama


MODEL = "gemma4:e4b"


SYSTEM_PROMPT = """
You are a music analysis assistant helping create a dataset for
text-to-lyrics generation research.

You will receive:
1. audio for one song
2. metadata associated with the song

Analyze the song using BOTH sources.

Your job is to produce a structured semantic description that can later
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
- Do not copy distinctive lyric phrases.
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
  reasonably inferable, e.g. ["verse", "chorus", "verse", "chorus"].
- "description" should be a concise overall description.
- "lyric_generation_prompt" should sound like a realistic user request
  for a lyric-generation system. It should describe the semantic and
  stylistic characteristics of the song without copying lyrics.
"""


def audio_to_base64(audio_path: Path) -> str:
    with audio_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_user_prompt(metadata: dict[str, Any]) -> str:
    return f"""
Analyze this song using the supplied audio and metadata.

Song metadata:

{json.dumps(metadata, indent=2, ensure_ascii=False)}

Return the JSON object described in the system instructions.
"""


def analyze_song(
    audio_path: Path,
    metadata: dict[str, Any],
    retries: int = 2,
) -> dict[str, Any]:

    audio_b64 = audio_to_base64(audio_path)
    user_prompt = build_user_prompt(metadata)

    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
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
                        "images": [audio_b64],
                    },
                ],
                think=False,
                options={
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "num_predict": 1200,
                },
            )

            raw = response.message.content.strip()

            # Occasionally models wrap JSON in markdown fences.
            if raw.startswith("```"):
                raw = raw.replace("```json", "", 1)
                raw = raw.replace("```", "", 1)
                raw = raw.strip()

            result = json.loads(raw)

            if not isinstance(result, dict):
                raise ValueError("Model returned JSON but not an object.")

            return result

        except Exception as exc:
            last_error = exc

            if attempt < retries:
                wait_seconds = 2 ** attempt
                print(
                    f"  retry {attempt + 1}/{retries} "
                    f"after error: {exc}",
                    file=sys.stderr,
                )
                time.sleep(wait_seconds)

    raise RuntimeError(
        f"Failed to analyze {audio_path}: {last_error}"
    )


def load_existing_ids(output_path: Path) -> set[str]:
    """
    Read completed track IDs so the script can resume after interruption.
    """
    completed: set[str] = set()

    if not output_path.exists():
        return completed

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
                track_id = record.get("track_id")

                if track_id is not None:
                    completed.add(str(track_id))

            except json.JSONDecodeError:
                # Ignore a damaged final line rather than killing the run.
                continue

    return completed


def find_audio_file(
    audio_root: Path,
    track_id: str,
) -> Path | None:
    """
    FMA commonly stores tracks in three-digit directories.

    Example:
        track 2 -> 000/000002.mp3

    We first try that exact layout, then fall back to a recursive search.
    """

    numeric_id = str(track_id).zfill(6)

    candidates = [
        audio_root / numeric_id[:3] / f"{numeric_id}.mp3",
        audio_root / numeric_id[:3] / f"{numeric_id}.wav",
        audio_root / numeric_id[:3] / f"{numeric_id}.flac",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback for custom directory layouts.
    for extension in ("mp3", "wav", "flac", "m4a", "ogg"):
        matches = list(audio_root.rglob(f"{numeric_id}.{extension}"))

        if matches:
            return matches[0]

    return None


def load_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    """
    Load a CSV metadata file.

    This assumes the first row is a header and that one column contains
    the track ID. We'll make this more FMA-specific once we see your
    actual file.
    """

    rows: list[dict[str, Any]] = []

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("Metadata CSV has no header.")

        # Try common track ID names.
        possible_id_columns = [
            "track_id",
            "track",
            "id",
            "trackid",
        ]

        id_column = next(
            (
                column
                for column in possible_id_columns
                if column in reader.fieldnames
            ),
            None,
        )

        if id_column is None:
            raise ValueError(
                "Could not identify track ID column. "
                f"Available columns: {reader.fieldnames}"
            )

        for row in reader:
            track_id = row.get(id_column)

            if not track_id:
                continue

            row["_track_id"] = str(track_id).strip()
            rows.append(row)

    return rows


def process_dataset(
    audio_root: Path,
    metadata_path: Path,
    output_path: Path,
    limit: int | None = None,
) -> None:

    metadata_rows = load_metadata(metadata_path)
    completed_ids = load_existing_ids(output_path)

    print(f"Metadata rows: {len(metadata_rows)}")
    print(f"Already completed: {len(completed_ids)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_this_run = 0

    with output_path.open("a", encoding="utf-8") as out:

        for row in metadata_rows:

            track_id = str(row["_track_id"])

            if track_id in completed_ids:
                continue

            if limit is not None and processed_this_run >= limit:
                break

            audio_path = find_audio_file(audio_root, track_id)

            if audio_path is None:
                print(
                    f"[SKIP] No audio found for track {track_id}",
                    file=sys.stderr,
                )
                continue

            metadata = {
                key: value
                for key, value in row.items()
                if key != "_track_id"
            }

            print(
                f"[{processed_this_run + 1}] "
                f"Processing track {track_id}: {audio_path}"
            )

            try:
                analysis = analyze_song(
                    audio_path=audio_path,
                    metadata=metadata,
                )

                record = {
                    "track_id": track_id,
                    "metadata": metadata,
                    "analysis": analysis,
                }

                out.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                out.flush()

                completed_ids.add(track_id)
                processed_this_run += 1

                print("    OK")

            except Exception as exc:
                print(
                    f"    ERROR: {exc}",
                    file=sys.stderr,
                )

    print(
        f"Finished. Processed {processed_this_run} new tracks."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate FMA tracks using Ollama Gemma 4."
    )

    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Root directory containing FMA audio files.",
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="CSV metadata file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fma_annotations.jsonl"),
        help="Output JSONL file.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of new tracks to process.",
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