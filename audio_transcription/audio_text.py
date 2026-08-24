#!/usr/bin/env python3

import argparse
import csv
import json
import logging
import multiprocessing as mp
import os
import queue
import time
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger("transcribe_lyrics")


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------

def load_metadata(path: Path) -> list[dict]:
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            records = []
            for tid, rec in data.items():
                rec = dict(rec)
                rec.setdefault("track_id", tid)
                records.append(rec)
            return records

        return data

    elif suffix == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    raise ValueError(f"Unsupported metadata format: {path.suffix}")


def save_metadata(records: list[dict], path: Path) -> None:
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    elif suffix == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    elif suffix == ".csv":
        flat_records = []

        for rec in records:
            rec = dict(rec)

            if "lyrics_segments" in rec and not isinstance(
                rec["lyrics_segments"], str
            ):
                rec["lyrics_segments"] = json.dumps(
                    rec["lyrics_segments"],
                    ensure_ascii=False,
                )

            flat_records.append(rec)

        fieldnames = sorted(
            {k for rec in flat_records for k in rec.keys()}
        )

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_records)

    else:
        raise ValueError(f"Unsupported output format: {path.suffix}")


# --------------------------------------------------------------------------
# FMA audio path
# --------------------------------------------------------------------------

def resolve_audio_path(
    record: dict,
    audio_dir: Path,
    filename_col: str | None,
) -> Path | None:

    if filename_col and record.get(filename_col):
        p = audio_dir / record[filename_col]
        return p if p.exists() else None

    track_id = record["track_id"]

    # Flat layout
    p = audio_dir / f"{track_id}.mp3"

    if p.exists():
        return p

    # FMA nested layout
    padded = str(track_id).zfill(6)

    p = audio_dir / padded[:3] / f"{padded}.mp3"

    return p if p.exists() else None


# --------------------------------------------------------------------------
# CUDA worker
# --------------------------------------------------------------------------

def worker_process(
    task_queue,
    result_queue,
    model_name,
    device,
    compute_type,
    language,
    batch_size,
):
    """
    Long-lived worker.

    Loads WhisperX once.

    If a CUDA OOM or invalid-device error occurs, the worker reports it
    and exits. The parent process then starts a completely fresh worker,
    creating a fresh CUDA context.
    """

    try:
        import torch
        import whisperx

        log.info(
            f"Worker PID={os.getpid()} loading "
            f"WhisperX '{model_name}' on {device} "
            f"({compute_type})..."
        )

        model = whisperx.load_model(
            model_name,
            device,
            compute_type=compute_type,
            language=language,
        )

        align_models = {}

        log.info(
            f"Worker PID={os.getpid()} ready"
        )

        while True:

            task = task_queue.get()

            if task is None:
                break

            track_id, audio_path = task

            try:
                log.info(
                    f"[Worker {os.getpid()}] "
                    f"Transcribing track_id={track_id}"
                )

                audio = whisperx.load_audio(str(audio_path))

                result = model.transcribe(
                    audio,
                    batch_size=batch_size,
                    language=language,
                )

                detected_language = result["language"]

                # Alignment
                try:
                    if detected_language not in align_models:
                        log.info(
                            f"[Worker {os.getpid()}] "
                            f"Loading alignment model for "
                            f"'{detected_language}'"
                        )

                        model_a, metadata = whisperx.load_align_model(
                            language_code=detected_language,
                            device=device,
                        )

                        align_models[detected_language] = (
                            model_a,
                            metadata,
                        )

                    model_a, metadata = align_models[
                        detected_language
                    ]

                    result = whisperx.align(
                        result["segments"],
                        model_a,
                        metadata,
                        audio,
                        device,
                        return_char_alignments=False,
                    )

                except Exception as e:
                    log.warning(
                        f"Alignment failed for "
                        f"track_id={track_id}: {e}"
                    )

                segments = [
                    {
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "text": seg.get("text", "").strip(),
                    }
                    for seg in result["segments"]
                ]

                full_text = " ".join(
                    s["text"]
                    for s in segments
                    if s["text"]
                )

                output = {
                    "text": full_text,
                    "segments": segments,
                    "language": detected_language,
                }

                result_queue.put(
                    (
                        "success",
                        track_id,
                        output,
                    )
                )

            except RuntimeError as e:

                error_text = str(e)

                if (
                    "out of memory" in error_text.lower()
                    or "invalid device ordinal" in error_text.lower()
                    or "cudaerror" in error_text.lower()
                ):

                    result_queue.put(
                        (
                            "cuda_failure",
                            track_id,
                            error_text,
                        )
                    )

                    log.error(
                        f"Worker PID={os.getpid()} encountered "
                        f"CUDA failure on track_id={track_id}: "
                        f"{error_text}"
                    )

                    # IMPORTANT:
                    # Do NOT continue using this CUDA process.
                    # Exit and let the parent create a new worker.
                    os._exit(2)

                result_queue.put(
                    (
                        "failure",
                        track_id,
                        error_text,
                    )
                )

            except Exception as e:

                result_queue.put(
                    (
                        "failure",
                        track_id,
                        str(e),
                    )
                )

    except Exception as e:

        log.exception(
            f"Worker PID={os.getpid()} failed during startup: {e}"
        )

        os._exit(3)


# --------------------------------------------------------------------------
# Worker manager
# --------------------------------------------------------------------------

class WorkerManager:

    def __init__(
        self,
        model_name,
        device,
        compute_type,
        language,
        batch_size,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.batch_size = batch_size

        self.task_queue = None
        self.result_queue = None
        self.process = None

    def start(self):

        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()

        self.process = mp.Process(
            target=worker_process,
            args=(
                self.task_queue,
                self.result_queue,
                self.model_name,
                self.device,
                self.compute_type,
                self.language,
                self.batch_size,
            ),
        )

        self.process.start()

        log.info(
            f"Started WhisperX worker PID={self.process.pid}"
        )

    def stop(self):

        if self.process is None:
            return

        if self.process.is_alive():

            try:
                self.task_queue.put(None)
            except Exception:
                pass

            self.process.join(timeout=10)

        if self.process.is_alive():
            self.process.terminate()
            self.process.join()

        self.process = None

    def restart(self):

        log.warning(
            "Restarting WhisperX worker with a fresh CUDA context..."
        )

        self.stop()

        time.sleep(2)

        self.start()

    def submit(self, track_id, audio_path):

        self.task_queue.put(
            (
                track_id,
                str(audio_path),
            )
        )

    def get_result(self, timeout=3600):

        return self.result_queue.get(
            timeout=timeout
        )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--metadata",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./whisperx_cache"),
    )

    parser.add_argument(
        "--filename-col",
        default=None,
    )

    parser.add_argument(
        "--model",
        default="medium",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
    )

    parser.add_argument(
        "--compute-type",
        default="float16",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--language",
        default=None,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    args.cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = load_metadata(args.metadata)

    log.info(
        f"Loaded {len(records)} metadata records"
    )

    if args.limit:
        records = records[:args.limit]

    n_done = 0
    n_cached = 0
    n_missing = 0
    n_failed = 0
    n_cuda_failures = 0

    # ------------------------------------------------------------------
    # First build list of work
    # ------------------------------------------------------------------

    work = []

    for record in records:

        track_id = record["track_id"]

        cache_path = (
            args.cache_dir /
            f"{track_id}.json"
        )

        if cache_path.exists():

            try:
                with open(
                    cache_path,
                    "r",
                    encoding="utf-8",
                ) as f:
                    result = json.load(f)

                record["lyrics"] = result["text"]
                record["lyrics_segments"] = result["segments"]

                n_cached += 1

                continue

            except Exception as e:

                log.warning(
                    f"Invalid cache for track_id={track_id}: "
                    f"{e}. Reprocessing."
                )

        audio_path = resolve_audio_path(
            record,
            args.audio_dir,
            args.filename_col,
        )

        if audio_path is None:

            log.warning(
                f"No audio found for track_id={track_id}"
            )

            n_missing += 1
            continue

        work.append(
            (
                record,
                audio_path,
                cache_path,
            )
        )

    log.info(
        f"Work remaining: {len(work)} "
        f"(cached={n_cached}, missing={n_missing})"
    )

    if not work:

        save_metadata(
            records,
            args.output,
        )

        log.info(
            f"Nothing to transcribe. "
            f"Output written to {args.output}"
        )

        return

    # ------------------------------------------------------------------
    # Start worker
    # ------------------------------------------------------------------

    mp.set_start_method(
        "spawn",
        force=True,
    )

    manager = WorkerManager(
        args.model,
        args.device,
        args.compute_type,
        args.language,
        args.batch_size,
    )

    manager.start()

    # ------------------------------------------------------------------
    # Process tracks
    # ------------------------------------------------------------------

    for index, (record, audio_path, cache_path) in enumerate(
        work,
        1,
    ):

        track_id = record["track_id"]

        log.info(
            f"[{index}/{len(work)}] "
            f"Submitting track_id={track_id} "
            f"({audio_path.name})"
        )

        # --------------------------------------------------------------
        # Send task
        # --------------------------------------------------------------

        manager.submit(
            track_id,
            audio_path,
        )

        # --------------------------------------------------------------
        # Wait for worker
        # --------------------------------------------------------------

        try:

            status, result_track_id, payload = (
                manager.get_result()
            )

        except queue.Empty:

            log.error(
                f"Worker timed out on track_id={track_id}"
            )

            n_failed += 1

            manager.restart()

            continue

        # --------------------------------------------------------------
        # Success
        # --------------------------------------------------------------

        if status == "success":

            result = payload

            with open(
                cache_path,
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    result,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            record["lyrics"] = result["text"]
            record["lyrics_segments"] = result["segments"]

            n_done += 1

            continue

        # --------------------------------------------------------------
        # CUDA failure
        # --------------------------------------------------------------

        if status == "cuda_failure":

            log.error(
                f"CUDA failure on track_id={track_id}: "
                f"{payload}"
            )

            n_cuda_failures += 1

            # Worker already exited.
            # Start a completely fresh CUDA process.
            manager.restart()

            # Do NOT immediately retry the same pathological file.
            # It is recorded as failed for this run.
            n_failed += 1

            continue

        # --------------------------------------------------------------
        # Normal failure
        # --------------------------------------------------------------

        log.error(
            f"Failed on track_id={track_id}: {payload}"
        )

        n_failed += 1

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    manager.stop()

    # ------------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------------

    save_metadata(
        records,
        args.output,
    )

    log.info("=" * 70)

    log.info(
        f"Done. "
        f"newly_transcribed={n_done} "
        f"from_cache={n_cached} "
        f"missing_audio={n_missing} "
        f"failed={n_failed} "
        f"cuda_failures={n_cuda_failures}"
    )

    log.info(
        f"Merged metadata written to {args.output}"
    )


if __name__ == "__main__":
    main()