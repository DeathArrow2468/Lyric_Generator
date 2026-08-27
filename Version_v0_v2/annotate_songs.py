# annotate_songs.py

from pathlib import Path
import argparse
import csv
import json
import time

import ollama


# ============================================================
# Configuration
# ============================================================

MODEL = "qwen3:8b"


SYSTEM_PROMPT = """
You are creating structured training data for a research project
on controllable song-lyric generation.

You will receive metadata and the lyrics of one song.

Analyze the song and create a semantic description that could be
used to construct a natural-language prompt for a lyric-generation
model.

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

Instructions:

- Focus on semantic and stylistic characteristics.
- Do NOT reproduce lyrics.
- Do NOT quote distinctive phrases from the lyrics.
- Do NOT mention the artist's name in generation_prompt.
- Do NOT make the prompt say "write something similar to Artist X".
- Describe genre, mood, themes, narrative situation, imagery,
  emotional qualities, and useful lyrical characteristics.
- generation_prompt should read like a realistic request from a
  user who wants a new song.
- The generated prompt should contain enough information to guide
  a lyric-generation model but should NOT identify or reproduce
  the source song.
- song_structure should contain likely sections when inferable,
  for example:
  ["Verse", "Chorus", "Verse", "Chorus", "Bridge", "Chorus"].
- If something cannot reasonably be inferred, use an empty list or
  a cautious description rather than inventing information.
"""


# ============================================================
# Helpers
# ============================================================

def load_csv(path):
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


def build_prompt(row):

    artist = row.get("Artists", "")
    title = row.get("title", "")
    album = row.get("Album", "")
    year = row.get("Year", "")
    date = row.get("Date", "")
    lyrics = row.get("Lyrics", "")

    return f"""
Analyze the following song.

Metadata:

Artist: {artist}
Title: {title}
Album: {album}
Year: {year}
Date: {date}

Lyrics:

{lyrics}

Create the structured JSON analysis specified in your instructions.
"""


def parse_json(response_text):

    text = response_text.strip()

    # Handle accidental markdown fences.
    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    return json.loads(text)


def annotate_song(row, retries=3):

    prompt = build_prompt(row)

    for attempt in range(retries):

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
                f"Annotation error "
                f"(attempt {attempt + 1}/{retries}): {e}"
            )

            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    return None


# ============================================================
# Main processing
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    rows = load_csv(args.input)

    print(f"Loaded {len(rows)} songs.")

    # --------------------------------------------------------
    # Resume support
    # --------------------------------------------------------

    processed_ids = set()

    if args.output.exists():

        with args.output.open(
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                try:

                    record = json.loads(line)

                    if "song_id" in record:
                        processed_ids.add(
                            str(record["song_id"])
                        )

                except json.JSONDecodeError:
                    continue

    print(
        f"Already processed: "
        f"{len(processed_ids)}"
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
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

            # Use an existing ID if the CSV has one.
            song_id = (
                row.get("song_id")
                or row.get("track_id")
                or str(index)
            )

            song_id = str(song_id)

            if song_id in processed_ids:
                continue

            lyrics = (
                row.get("Lyrics") or ""
            ).strip()

            if not lyrics:
                print(
                    f"[SKIP] {song_id}: no lyrics"
                )
                continue

            print(
                f"[{processed_this_run + 1}] "
                f"Annotating {song_id}: "
                f"{row.get('title', '')}"
            )

            analysis = annotate_song(row)

            if analysis is None:

                print(
                    f"[FAILED] {song_id}"
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

            print("    OK")

    print()
    print(
        f"Finished. New annotations: "
        f"{processed_this_run}"
    )


if __name__ == "__main__":
    main()