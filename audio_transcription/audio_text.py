#!/usr/bin/env python3
"""
End-to-end pipeline: transcribe .mp3 songs with WhisperX and merge the
resulting lyrics into your existing per-track metadata (keyed by track_id).

------------------------------------------------------------------------
SETUP (run once, from bottom of the stack up)
------------------------------------------------------------------------
1) System dependency - ffmpeg (WhisperX/torchaudio need it to decode mp3):
     sudo apt update && sudo apt install -y ffmpeg

2) A CUDA-capable environment is strongly recommended (Whisper on CPU is
   very slow for large batches). Check:
     python -c "import torch; print(torch.cuda.is_available())"

3) Install WhisperX (this pulls in torch, torchaudio, faster-whisper,
   pyannote-audio, etc. as dependencies):
     pip install whisperx

   If you hit resolver issues, install torch matching your CUDA version
   first (see https://pytorch.org/get-started/locally/), then:
     pip install whisperx --no-deps
     pip install faster-whisper pyannote.audio nltk pandas

4) (Optional, only if you want speaker diarization - not needed for
   single-vocalist songs, skip it) you'd need a HuggingFace token and
   pyannote model acceptance. This script does NOT do diarization -
   just transcription + word-level alignment, which is what you need
   to line lyrics up with audio.

------------------------------------------------------------------------
INPUT ASSUMPTIONS
------------------------------------------------------------------------
- A folder of .mp3 files.
- A metadata file (.json or .csv) that is a list/table of records, each
  record has a "track_id" field uniquely identifying a song.
- By default the script matches audio files to metadata by assuming the
  mp3 filename stem equals the track_id, e.g. track_id "abc123" ->
  "abc123.mp3". If your metadata instead has an explicit filename
  column, pass --filename-col to use that instead.

------------------------------------------------------------------------
OUTPUT
------------------------------------------------------------------------
- A merged metadata file (same format as input: json or csv) with two
  new fields per record:
    "lyrics"          -> full transcribed text (str)
    "lyrics_segments" -> list of {start, end, text} segment dicts (json
                          only - flattened to a JSON string in csv mode)
- A cache directory of raw per-track WhisperX outputs (json), so re-runs
  don't re-transcribe tracks that already succeeded.

------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------
python transcribe_lyrics.py \
    --audio-dir ./mp3s \
    --metadata ./metadata.json \
    --output ./metadata_with_lyrics.json \
    --model large-v3 \
    --device cuda \
    --compute-type float16 \
    --language en
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("transcribe_lyrics")


# --------------------------------------------------------------------------
# Metadata I/O
# --------------------------------------------------------------------------

def load_metadata(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # allow a {track_id: {...}} mapping as well as a list of records
            records = []
            for tid, rec in data.items():
                rec = dict(rec)
                rec.setdefault("track_id", tid)
                records.append(rec)
            return records
        return data
    elif path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    else:
        raise ValueError(f"Unsupported metadata format: {path.suffix}")


def save_metadata(records: list[dict], path: Path) -> None:
    if path.suffix.lower() == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    elif path.suffix.lower() == ".csv":
        # flatten segment lists to a JSON string so they survive CSV
        flat_records = []
        for rec in records:
            rec = dict(rec)
            if "lyrics_segments" in rec and not isinstance(rec["lyrics_segments"], str):
                rec["lyrics_segments"] = json.dumps(rec["lyrics_segments"], ensure_ascii=False)
            flat_records.append(rec)
        fieldnames = sorted({k for rec in flat_records for k in rec.keys()})
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_records)
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}")


def resolve_audio_path(record: dict, audio_dir: Path, filename_col: str | None) -> Path | None:
    if filename_col and record.get(filename_col):
        p = audio_dir / record[filename_col]
    else:
        p = audio_dir / f"{record['track_id']}.mp3"
    return p if p.exists() else None


# --------------------------------------------------------------------------
# WhisperX transcription
# --------------------------------------------------------------------------

class Transcriber:
    """Loads the WhisperX model + aligner once, reuses across all tracks."""

    def __init__(self, model_name: str, device: str, compute_type: str,
                 language: str | None, batch_size: int):
        import whisperx  # imported lazily so --help works without it installed

        self.whisperx = whisperx
        self.device = device
        self.language = language
        self.batch_size = batch_size

        log.info(f"Loading WhisperX model '{model_name}' on {device} ({compute_type})...")
        self.model = whisperx.load_model(
            model_name, device, compute_type=compute_type,
            language=language,  # None -> auto-detect per file
        )

        # Alignment model is language-specific; loaded lazily on first use
        # per detected/declared language, then cached.
        self._align_models: dict[str, tuple] = {}

    def _get_align_model(self, language_code: str):
        if language_code not in self._align_models:
            log.info(f"Loading alignment model for language '{language_code}'...")
            model_a, metadata = self.whisperx.load_align_model(
                language_code=language_code, device=self.device
            )
            self._align_models[language_code] = (model_a, metadata)
        return self._align_models[language_code]

    def transcribe(self, audio_path: Path) -> dict:
        """Returns {"text": str, "segments": [{"start","end","text"}, ...], "language": str}"""
        audio = self.whisperx.load_audio(str(audio_path))
        result = self.model.transcribe(audio, batch_size=self.batch_size, language=self.language)
        lang = result["language"]

        # Word-level alignment gives cleaner, tighter segment timing than
        # raw Whisper output - useful later for lining lyrics up with audio.
        try:
            model_a, metadata = self._get_align_model(lang)
            result = self.whisperx.align(
                result["segments"], model_a, metadata, audio, self.device,
                return_char_alignments=False,
            )
        except Exception as e:
            log.warning(f"Alignment failed for {audio_path.name} ({e}); keeping unaligned segments.")

        segments = [
            {"start": seg.get("start"), "end": seg.get("end"), "text": seg.get("text", "").strip()}
            for seg in result["segments"]
        ]
        full_text = " ".join(s["text"] for s in segments if s["text"])
        return {"text": full_text, "segments": segments, "language": lang}


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-dir", required=True, type=Path, help="Folder containing .mp3 files")
    parser.add_argument("--metadata", required=True, type=Path, help="Input metadata .json or .csv")
    parser.add_argument("--output", required=True, type=Path, help="Output metadata .json or .csv (with lyrics merged in)")
    parser.add_argument("--cache-dir", type=Path, default=Path("./whisperx_cache"),
                         help="Per-track raw transcription cache, to skip already-done tracks on re-run")
    parser.add_argument("--filename-col", default=None,
                         help="Metadata column holding the mp3 filename, if it's not just '<track_id>.mp3'")
    parser.add_argument("--model", default="large-v3", help="WhisperX/Whisper model size, e.g. tiny/base/small/medium/large-v3")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--compute-type", default="float16", help="float16/int8 for cuda, int8/float32 for cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--language", default=None, help="Force a language code (e.g. 'en'); default auto-detect per track")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N tracks (for testing)")
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    records = load_metadata(args.metadata)
    log.info(f"Loaded {len(records)} metadata records")
    if args.limit:
        records = records[: args.limit]

    transcriber = None  # lazy-init only once we know we have work to do
    n_done = n_skipped_missing_audio = n_cached = n_failed = 0

    for i, record in enumerate(records, 1):
        track_id = record["track_id"]
        cache_path = args.cache_dir / f"{track_id}.json"

        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                result = json.load(f)
            n_cached += 1
        else:
            audio_path = resolve_audio_path(record, args.audio_dir, args.filename_col)
            if audio_path is None:
                log.warning(f"[{i}/{len(records)}] No audio file found for track_id={track_id}, skipping")
                n_skipped_missing_audio += 1
                continue

            if transcriber is None:
                transcriber = Transcriber(args.model, args.device, args.compute_type,
                                           args.language, args.batch_size)

            try:
                log.info(f"[{i}/{len(records)}] Transcribing {audio_path.name}")
                result = transcriber.transcribe(audio_path)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                n_done += 1
            except Exception as e:
                log.error(f"[{i}/{len(records)}] Failed on track_id={track_id}: {e}")
                n_failed += 1
                continue

        record["lyrics"] = result["text"]
        record["lyrics_segments"] = result["segments"]

    save_metadata(records, args.output)

    log.info(
        f"Done. newly_transcribed={n_done} from_cache={n_cached} "
        f"missing_audio={n_skipped_missing_audio} failed={n_failed}"
    )
    log.info(f"Merged metadata written to {args.output}")


if __name__ == "__main__":
    main()