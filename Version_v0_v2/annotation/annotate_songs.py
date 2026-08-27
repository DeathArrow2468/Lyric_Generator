# annotation/annotate_songs.py

from pathlib import Path
import argparse
import csv
import json
import time

import ollama

from config import (
    ANNOTATION_MODEL,
    MASTER_CSV,
    ANNOTATED_JSONL,
)


# ============================================================
# Annotation prompt
# ============================================================

SYSTEM_PROMPT = """
You are creating structured training data for a research project
on controllable song-lyric generation.

You will receive metadata and the lyrics of one song.

Analyze the song and create a semantic description that could
be used to construct a natural-language prompt for a
lyric-generation model.

Return ONLY valid JSON.

Required schema:

{
  "genre": [],
  "mood": [],
  "themes": [],
  "emotional_tone": "",
  "narrative_perspective": "",
  "style_attributes": [],
  "imagery": [],
  "song_structure": [],
  "keywords": [],
  "description": "",
  "generation_prompt": ""
}

Rules:

1. Do not reproduce or quote the lyrics.

2. Do not mention the artist's name in generation_prompt.

3. Do not say "write a song like [artist]".

4. Describe the song using concepts such as:
   - genre
   - mood
   - themes
   - emotional tone
   - narrative perspective
   - lyrical style
   - imagery
   - structure
   - subject matter

5. generation_prompt must sound like a realistic user request
   to a lyric-generation system.

6. generation_prompt should describe WHAT the user wants,
   not describe the source song as an object.

7. Avoid overly generic prompts. Include concrete semantic
   characteristics when they can be inferred.

8. Do not invent facts that are not reasonably supported by
   the metadata or lyrics.

9. song_structure should contain likely sections if they can
   reasonably be inferred, for example:
   ["Verse", "Chorus", "Verse", "Chorus", "Bridge", "Chorus"].

10. If a field cannot reasonably be determined, return an empty
    list or a cautious description.

11. The purpose of generation_prompt is to create a plausible
    user request that could have resulted in the original
    song's semantic characteristics.

Return JSON only.
"""


# ============================================================
# Load CSV
# ============================================================

def load_master_csv(path: Path):

    if not path.exists():
        raise FileNotFoundError(
            f"Master CSV not found: {path}"
        )

    rows = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


# ============================================================
# Build annotation request
# ============================================================

def build_user_prompt(row):

    return f"""
Analyze the following song.

Metadata:

Artist: {row.get("Artists", "")}
Title: {row.get("title", "")}
Album: {row.get("Album", "")}
Year: {row.get("Year", "")}
Date: {row.get("Date", "")}

Lyrics:

{row.get("Lyric", "")}

Create the structured JSON analysis described in the system
instructions.
"""


# ============================================================
# Parse model JSON
# ============================================================

def parse_json(text):

    text = text.strip()

    # Remove accidental markdown fences.
    if text.startswith("```"):

        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    return json.loads(text)


# ============================================================
# Annotate one song
# ============================================================

def annotate_song(row, retries=3):

    prompt = build_user_prompt(row)

    for attempt in range(retries):

        try:

            response = ollama.chat(
                model=ANNOTATION_MODEL,

                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],

                options={
                    "temperature": 0.2,
                    "top_p": 0.9,
                },
            )

            return parse_json(
                response.message.content
            )

        except Exception as e:

            print(
                f"  Annotation error "
                f"(attempt {attempt + 1}/{retries}): {e}"
            )

            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    return None


# ============================================================
# Existing processed IDs
# ============================================================

def load_processed_ids(output_path):

    processed = set()

    if not output_path.exists():
        return processed

    with output_path.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            try:

                record = json.loads(line)

                song_id = record.get("song_id")

                if song_id is not None:
                    processed.add(str(song_id))

            except json.JSONDecodeError:
                continue

    return processed


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=MASTER_CSV,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=ANNOTATED_JSONL,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    rows = load_master_csv(args.input)

    print(f"Loaded {len(rows):,} songs.")
    print(f"Annotation model: {ANNOTATION_MODEL}")

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    processed_ids = load_processed_ids(
        args.output
    )

    print(
        f"Already annotated: "
        f"{len(processed_ids):,}"
    )

    processed_this_run = 0

    with args.output.open(
        "a",
        encoding="utf-8"
    ) as out:

        for index, row in enumerate(rows):

            if (
                args.limit is not None
                and processed_this_run >= args.limit
            ):
                break

            song_id = str(
                row.get(
                    "song_id",
                    index
                )
            )

            if song_id in processed_ids:
                continue

            lyrics = (
                row.get("Lyric") or ""
            ).strip()

            if not lyrics:
                print(
                    f"[SKIP] {song_id}: empty lyrics"
                )
                continue

            print(
                f"[{processed_this_run + 1}] "
                f"{song_id} | "
                f"{row.get('Artists', '')} | "
                f"{row.get('title', '')}"
            )

            analysis = annotate_song(row)

            if analysis is None:

                print(
                    f"  [FAILED] {song_id}"
                )

                continue

            record = {
                "song_id": song_id,

                "metadata": {
                    "artist": row.get("Artists", ""),
                    "title": row.get("title", ""),
                    "album": row.get("Album", ""),
                    "year": row.get("Year", ""),
                    "date": row.get("Date", ""),
                },

                "analysis": analysis,

                "lyrics": lyrics,
            }

            out.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

            out.flush()

            processed_ids.add(song_id)

            processed_this_run += 1

            print("  OK")

    print()
    print(
        f"Finished. New annotations: "
        f"{processed_this_run}"
    )


if __name__ == "__main__":
    main()