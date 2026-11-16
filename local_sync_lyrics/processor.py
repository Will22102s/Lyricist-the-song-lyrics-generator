"""Batch runner: discover tracks, resolve lyrics, write ``.lrc`` files.

Guarantees
----------
* **Per-file isolation.** Every track is processed inside its own ``try/except``.
  A corrupt file, a provider bug or a failed transcription is logged, cached as
  ``failed`` and the loop moves on to the next track.
* **Cancellation** is honoured between files, between providers and between
  Whisper segments (``OperationCancelled``).
* **Idempotence.** Finished tracks are recorded in SQLite and skipped next time
  (unless the .lrc was deleted, "overwrite" is on, or a retry rule applies).
* The runner talks to the outside world only through the ``emit`` callback, so it
  works unchanged under the Flet GUI, the CLI and the tests.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from .api_providers import LyricsFetcher
from .config import Settings
from .database import LyricsDatabase, TrackRecord, fingerprint_file
from .lrc_utils import validate_synced, write_text_atomic
from .metadata import read_track_info
from .models import BatchSummary, Event, Status, TrackInfo
from .transcriber import (
    AudioDecodeError, NoSpeechError, OperationCancelled, Transcriber, TranscriptionError,
)

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = frozenset(
    {".mp3", ".flac", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wav", ".wma", ".aiff", ".aif", ".ape", ".wv", ".mka"}
)
RETRY_AFTER_DAYS = 7.0
Emit = Callable[[Event], None]

_LEVELS = {"info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}


class BatchProcessor:
    def __init__(
        self,
        settings: Settings,
        db: LyricsDatabase,
        emit: Emit,
        cancel: Optional[threading.Event] = None,
        *,
        fetcher: Optional[LyricsFetcher] = None,
        transcriber: Optional[Transcriber] = None,
    ) -> None:
        self.settings = settings
        self.db = db
        self.emit = emit
        self.cancel = cancel or threading.Event()
        self._fetcher = fetcher
        self._transcriber = transcriber
        self.summary = BatchSummary()

    # ---------------------------------------------------------------- logging
    def log(self, tag: str, message: str, level: str = "info") -> None:
        logger.log(_LEVELS.get(level, logging.INFO), "[%s] %s", tag, message)
        try:
            self.emit(Event("log", message=message, tag=tag, level=level))
        except Exception:  # a broken UI callback must never kill the batch
            logger.exception("emit failed")

    def _progress(self, current: int, total: int, name: str = "") -> None:
        try:
            self.emit(Event("progress", message=name, current=current, total=total))
            self.emit(Event("stats", data=self.summary.as_dict(), current=current, total=total))
        except Exception:
            logger.exception("emit failed")

    # -------------------------------------------------------------- lazy deps
    def _get_fetcher(self) -> LyricsFetcher:
        if self._fetcher is None:
            self._fetcher = LyricsFetcher(self.settings, self.log, self.cancel)
        return self._fetcher

    def _get_transcriber(self) -> Transcriber:
        if self._transcriber is None:
            self._transcriber = Transcriber(self.settings, log=self.log, cancel=self.cancel)
        return self._transcriber

    # -------------------------------------------------------------- discovery
    def discover(self, root: Path) -> list[Path]:
        files: list[Path] = []

        def onerror(err: OSError) -> None:
            self.log("WARN", f"Cannot read folder {err.filename}: {err.strerror}", "warning")

        if self.settings.recursive:
            for dirpath, dirnames, filenames in os.walk(root, onerror=onerror):
                dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
                for name in sorted(filenames):
                    if not name.startswith(".") and Path(name).suffix.lower() in AUDIO_EXTENSIONS:
                        files.append(Path(dirpath) / name)
        else:
            try:
                for entry in sorted(root.iterdir()):
                    if entry.is_file() and not entry.name.startswith(".") \
                            and entry.suffix.lower() in AUDIO_EXTENSIONS:
                        files.append(entry)
            except OSError as exc:
                onerror(exc)
        return files

    # --------------------------------------------------------------- main loop
    def run(self, root: Path) -> BatchSummary:
        started = time.monotonic()
        try:
            self.log("SYS", f"Scanning {root} …")
            files = self.discover(root)
            self.summary.total = len(files)
            if not files:
                self.log("WARN", "No audio files found in that folder.", "warning")
                return self.summary
            self.log("SYS", f"Found {len(files)} audio file(s).")
            self._progress(0, len(files))

            for index, path in enumerate(files, 1):
                if self.cancel.is_set():
                    break
                self.log("INFO", f"[{index}/{len(files)}] {path.name}")
                self._progress(index - 1, len(files), path.name)
                try:
                    status = self.process_track(path)
                except OperationCancelled:
                    break
                except Exception as exc:  # strict per-file boundary
                    logger.exception("Unhandled error on %s", path)
                    self.log("ERR", f"{path.name}: {exc.__class__.__name__}: {exc}", "error")
                    status = Status.FAILED
                    self._record_failure(path, str(exc))
                self.summary.bump(status)
                self._progress(index, len(files), path.name)
        finally:
            self.summary.cancelled = self.cancel.is_set()
            self._shutdown()
            elapsed = time.monotonic() - started
            self.emit(Event("done", message=f"{elapsed:.0f}s", data=self.summary.as_dict(),
                            current=self.summary.processed, total=self.summary.total))
        return self.summary

    def _shutdown(self) -> None:
        for resource in (self._fetcher, self._transcriber):
            try:
                if resource is not None:
                    resource.close()
            except Exception:
                logger.debug("close failed", exc_info=True)

    # ----------------------------------------------------------- single track
    def _cache_skip_reason(self, record: Optional[TrackRecord], lrc_path: Path) -> Optional[str]:
        if record is None or self.settings.overwrite_existing:
            return None
        status = record.status
        if status in (Status.SYNCED_ONLINE.value, Status.SYNCED_AI.value):
            return "already synced (cached)" if lrc_path.exists() else None
        if status == Status.INSTRUMENTAL.value:
            return "marked instrumental (cached)"
        if status in (Status.NOT_FOUND.value, Status.FAILED.value, Status.PLAIN_ONLY.value):
            if self.settings.retry_failed:
                return None
            if self.settings.use_ai_fallback and not record.ai_tried:
                return None  # AI is newly enabled: worth another attempt
            age_days = (time.time() - record.updated_at) / 86400.0
            if age_days >= RETRY_AFTER_DAYS:
                return None
            return f"previously {status.replace('_', ' ')} {age_days:.0f}d ago (enable “Retry failed” to force)"
        return None

    def _record(self, fingerprint: str, track: TrackInfo, status: Status, *, source: str = "",
                lrc_path: Optional[Path] = None, ai_tried: bool = False, error: str = "") -> None:
        self.db.upsert(TrackRecord(
            fingerprint=fingerprint, path=str(track.path), status=status.value,
            artist=track.artist, title=track.title, duration=track.duration, source=source,
            lrc_path=str(lrc_path) if lrc_path else "", ai_tried=ai_tried, error=error,
        ))

    def _record_failure(self, path: Path, error: str) -> None:
        try:
            track = TrackInfo(path=path, title=path.stem)
            self._record(fingerprint_file(path), track, Status.FAILED, error=error)
        except Exception:
            logger.debug("could not record failure", exc_info=True)

    def process_track(self, path: Path) -> Status:
        lrc_path = path.with_suffix(".lrc")
        fingerprint = fingerprint_file(path)
        record = self.db.get(fingerprint)

        reason = self._cache_skip_reason(record, lrc_path)
        if reason:
            self.log("SKIP", reason)
            return Status.SKIPPED

        if lrc_path.exists() and not self.settings.overwrite_existing:
            self.log("SKIP", "A .lrc file already exists next to this track")
            return Status.SKIPPED

        track = read_track_info(path)
        dur = f"{track.duration:.0f}s" if track.duration else "unknown length"
        self.log("INFO", f"Identified as “{track.display_name}” ({dur})")

        # ---- 1) online providers ------------------------------------------
        outcome = self._get_fetcher().fetch(track)
        if self.cancel.is_set():
            raise OperationCancelled()

        if outcome.instrumental:
            self.log("OK", "Instrumental track - no lyrics needed")
            self._record(fingerprint, track, Status.INSTRUMENTAL, source="online")
            return Status.INSTRUMENTAL

        if outcome.synced and outcome.synced.synced_lrc:
            return self._write_synced(
                path, lrc_path, fingerprint, track, outcome.synced.synced_lrc,
                Status.SYNCED_ONLINE, outcome.synced.provider, ai_tried=False,
            )

        if outcome.all_providers_failed:
            self.log("WARN", "Every online provider failed - are you offline?", "warning")

        # ---- 2) local AI fallback -----------------------------------------
        ai_tried = False
        ai_error = ""
        ai_crashed = False
        if self.settings.use_ai_fallback:
            ai_tried = True
            lrc_text, ai_error, ai_crashed = self._try_ai(track)
            if lrc_text:
                return self._write_synced(
                    path, lrc_path, fingerprint, track, lrc_text, Status.SYNCED_AI, "Whisper",
                    ai_tried=True,
                )
        else:
            self.log("SKIP", "AI fallback is disabled in Settings")

        # ---- 3) optional unsynced text ------------------------------------
        if outcome.plain and outcome.plain.plain and self.settings.save_plain_fallback:
            txt_path = path.with_suffix(".txt")
            if not txt_path.exists() or self.settings.overwrite_existing:
                write_text_atomic(txt_path, outcome.plain.plain.strip() + "\n")
                self.log("OK", f"Saved unsynced lyrics to {txt_path.name} (from {outcome.plain.provider})")
                self._record(fingerprint, track, Status.PLAIN_ONLY, source=outcome.plain.provider,
                             lrc_path=txt_path, ai_tried=ai_tried, error=ai_error)
                return Status.PLAIN_ONLY

        self.log("WARN", "No lyrics found for this track", "warning")
        status = Status.FAILED if ai_crashed else Status.NOT_FOUND
        self._record(fingerprint, track, status, ai_tried=ai_tried, error=ai_error)
        return status

    def _try_ai(self, track: TrackInfo) -> tuple[Optional[str], str, bool]:
        """Run Whisper. Returns ``(lrc_text, error_message, crashed)``."""
        transcriber = self._get_transcriber()
        last_logged = {"p": -1}

        def progress(fraction: float) -> None:
            pct = int(fraction * 100)
            if pct != last_logged["p"]:
                last_logged["p"] = pct
                self.log("AI", f"Transcribing… {pct}%")

        try:
            result = transcriber.transcribe(track, progress=progress)
        except OperationCancelled:
            raise
        except NoSpeechError as exc:
            self.log("AI", f"{exc}", "warning")
            return None, str(exc), False
        except (AudioDecodeError, TranscriptionError) as exc:
            self.log("AI", f"Transcription failed: {exc}", "error")
            return None, str(exc), True
        self.log("AI", f"Transcribed {result.line_count} lines in {result.seconds:.0f}s")
        return result.lrc_text, "", False

    def _write_synced(self, path: Path, lrc_path: Path, fingerprint: str, track: TrackInfo,
                      lrc_text: str, status: Status, source: str, *, ai_tried: bool) -> Status:
        ok, reason, _ = validate_synced(lrc_text, track.duration, min_lines=1)
        if not ok:
            raise ValueError(f"refusing to write invalid LRC ({reason})")
        try:
            write_text_atomic(lrc_path, lrc_text)
        except OSError as exc:
            raise OSError(f"cannot write {lrc_path.name}: {exc.strerror or exc}") from exc
        self.log("OK", f"Saved {lrc_path.name} ({source})")
        self._record(fingerprint, track, status, source=source, lrc_path=lrc_path, ai_tried=ai_tried)
        return status
