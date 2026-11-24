"""Batch runner: discover tracks, resolve lyrics, write ``.lrc`` files / embed tags.

Per-track pipeline (search first, listen second, guess last)
------------------------------------------------------------
1. **[API: Tier 1]**  Ready-made synced lyrics (LRCLIB, BetterLyrics, RCLyricsBand,
   Megalobiz, NetEase, QQ Music) for the full title. Hit -> save, done.
2. **[Mashup Detected]**  ``"A x B"``, ``"A / B"``, ``"A & B (Mashup)"``, ``"A vs B"``:
   split into component songs and fetch each one's official lyrics.
3. **[API: Plain Text]**  Official words from Genius / Lyrics.ovh / Musixmatch.
4. **[AI: Identify]**  Still no text, or a component is missing? Blind-listen to a few
   slices, search lyric databases with what was sung, and VERIFY each candidate
   against the sung words before trusting it.
5. **[AI: Forced Alignment]**  Snap the official words onto the audio with Whisper word
   timestamps. Output keeps the official spelling; mishearings never reach the file.
6. **[AI: Blind]**  Only when no lyrics text exists anywhere (or alignment is too weak
   to trust): plain Whisper transcription.

Phase 2 additions
-----------------
* **Output mode** (``settings.output_mode``): a separate ``.lrc`` (default), lyrics
  embedded in the audio file's own tags (``metadata.embed_lyrics``), or both. A format
  that cannot be tagged falls back to a ``.lrc`` so a result is never lost; an existing
  ``.lrc`` is embedded without any new search.
* **Language** - every Whisper step runs with the language locked on a 30 s sample
  (see ``transcriber.AudioSession.lock_language``); the GUI is told through a
  ``language`` event.
* **Vocal isolation** - optional, applied inside the audio session (see
  ``separation.py``); nothing here needs to know about it.
* **Your Own Lyrics** - :meth:`BatchProcessor.run_custom` syncs ONE file with lyrics the
  user pasted: a quick speech-to-text pass is compared with the text first
  (``lrc_utils.match_lyrics``), a mismatch is put to the user through the ``ask``
  callback (Cancel / Force Sync Anyway), a match goes straight to forced alignment.

Guarantees (unchanged)
----------------------
* Every track runs inside its own ``try/except``: one bad file never stops the batch.
* Cancellation is honoured between files, providers, windows and Whisper segments.
* Finished tracks are cached in SQLite and skipped next time.
* The runner talks to the outside world only through ``emit`` (a Qt signal in the
  GUI), so it works unchanged under the Qt front-end, the CLI and the tests.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from .api_providers import TAG_TAGS, LyricsFetcher, MetadataClient
from .config import CUSTOM_LYRICS_MIN_MATCH, Settings
from .database import LyricsDatabase, TrackRecord, fingerprint_file
from .language import LanguageGuess, detect_text_language
from .lrc_utils import (
    MatchReport, match_lyrics, plain_to_lines, strip_to_plain, validate_synced, write_text_atomic,
)
from .metadata import (
    EmbedResult, embed_lyrics, has_embedded_lyrics, make_thumbnail, read_artwork,
    read_extra_tags, read_track_info, write_tags,
)
from .models import (
    BatchSummary, Event, ReferenceComponent, ReferenceScript, Status, TrackInfo,
)
from .text_utils import (
    MASHUP_CONFIDENT, MASHUP_POSSIBLE, MashupInfo, best_query_phrase, detect_mashup,
    snippet_match_score,
)
from .transcriber import (
    AudioDecodeError, AudioSession, NoSpeechError, OperationCancelled, Transcriber,
    TranscriptionError,
)

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = frozenset(
    {".mp3", ".flac", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wav", ".wma", ".aiff", ".aif", ".ape", ".wv", ".mka"}
)
RETRY_AFTER_DAYS = 7.0
VERIFY_SCORE = 70            # sung snippet must appear this well in a candidate's lyrics
EXPLAINED_SCORE = 80         # ... and this well to count as "already explained"
Emit = Callable[[Event], None]

_LEVELS = {"info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}

TAG_MASHUP = "Mashup Detected"
TAG_ISOLATE = "Audio: Isolate"
TAG_ALIGN = "AI: Forced Alignment"
TAG_IDENT = "AI: Identify"
TAG_BLIND = "AI: Blind"
TAG_PLAIN = "API: Plain Text"
TAG_OUTPUT = "Output"
TAG_VERIFY = "AI: Verify"
CUSTOM_SNIPPET_FRACTIONS = (0.12, 0.35, 0.60, 0.85)   # where the quick speech-to-text pass listens
CUSTOM_SNIPPET_SEC = 25.0

Ask = Callable[[dict], bool]   # blocking "may I continue?" question to the user (worker thread)


def discover_audio(root: Path, recursive: bool = True,
                   on_error: Optional[Callable[[OSError], None]] = None) -> list[Path]:
    """Audio files under *root* (or *root* itself when it is a single file).

    Shared by the GUI queue (run in a background job) and :meth:`BatchProcessor.discover`.
    """
    root = Path(root)
    if root.is_file():
        return [root] if root.suffix.lower() in AUDIO_EXTENSIONS else []
    files: list[Path] = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root, onerror=on_error):
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
            if on_error is not None:
                on_error(exc)
    return files


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
        ask: Optional[Ask] = None,
    ) -> None:
        self.settings = settings
        self.db = db
        self.emit = emit
        self.cancel = cancel or threading.Event()
        self.ask = ask          # headless callers leave this None: questions are answered "no"
        self._fetcher = fetcher
        self._transcriber = transcriber
        self._metadata_client: Optional[MetadataClient] = None
        self.summary = BatchSummary()
        self._ai_down: Optional[str] = None   # set once the Whisper model is unusable
        self._language_code = ""
        self._language_name = ""

    # ------------------------------------------------------ AI availability
    def _ai_enabled(self) -> bool:
        return self.settings.use_ai_fallback and self._ai_down is None

    def _note_ai_error(self, error: str) -> None:
        """If Whisper itself cannot load, switch every AI step off for the whole batch."""
        if self._ai_down is None and "could not load model" in error:
            self._ai_down = error
            self.log("AI", "Local AI is unavailable - skipping every AI step for the rest of "
                     "this batch (online lyrics still work).", "warning")

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
            self._transcriber = Transcriber(self.settings, log=self.log, cancel=self.cancel,
                                            on_language=self._on_language)
        return self._transcriber

    def _get_metadata_client(self) -> MetadataClient:
        if self._metadata_client is None:
            self._metadata_client = MetadataClient(timeout=self.settings.request_timeout)
        return self._metadata_client

    def _on_language(self, guess: LanguageGuess) -> None:
        """Tell the UI which language Whisper is now locked to (``data={}`` clears the badge)."""
        self._language_code = guess.code or ""
        self._language_name = guess.name or guess.code or ""
        try:
            self.emit(Event("language", message=guess.describe(), data={
                "code": guess.code, "name": guess.name, "probability": guess.probability,
                "source": guess.source}))
        except Exception:
            logger.exception("emit failed")

    def _clear_language(self) -> None:
        self._language_code = ""
        self._language_name = ""
        try:
            self.emit(Event("language", message="", data={}))
        except Exception:
            logger.exception("emit failed")

    def _track_event(self, kind: str, path: Path, status: str = "") -> None:
        """Queue feedback for the GUI (``track_start`` / ``track_done``)."""
        try:
            self.emit(Event(kind, message=path.name, data={
                "path": str(path), "status": status,
                "language": self._language_code, "language_name": self._language_name}))
        except Exception:
            logger.exception("emit failed")

    # -------------------------------------------------------------- discovery
    def discover(self, root: Path) -> list[Path]:
        def onerror(err: OSError) -> None:
            self.log("WARN", f"Cannot read folder {err.filename}: {err.strerror}", "warning")

        return discover_audio(root, self.settings.recursive, onerror)

    def _prepare_isolation(self) -> None:
        """Disable vocal isolation gracefully when its optional stack is incomplete.

        This is the last line of defence: the GUI already shows a banner, but the CLI
        (and a stale settings file) can still ask for isolation on a machine without
        FFmpeg / audioread / a separation backend.  Falling back here means a batch
        never dies halfway through because of a missing module.
        """
        if not self.settings.isolate_vocals:
            return
        try:
            from .hardware_profiler import run_preflight

            report = run_preflight(refresh=True)
            if report.isolation_ready:
                return
            reason = report.isolation_reason()
        except Exception as exc:  # pragma: no cover - defensive
            reason = f"the audio pre-flight check failed ({exc.__class__.__name__})"
        self.log(TAG_ISOLATE,
                 f"Vocal isolation disabled for this batch: {reason}. "
                 "Using the original audio.", "warning")
        self.settings.isolate_vocals = False

    # --------------------------------------------------------------- main loop
    def run(self, root: Path | list[Path] | tuple[Path, ...]) -> BatchSummary:
        started = time.monotonic()
        try:
            self._prepare_isolation()
            if isinstance(root, (list, tuple, set)):
                files = [Path(p) for p in root
                         if Path(p).is_file() and Path(p).suffix.lower() in AUDIO_EXTENSIONS]
                self.log("SYS", f"Processing {len(files)} selected track(s) …")
            else:
                self.log("SYS", f"Scanning {root} …")
                files = self.discover(Path(root))
            self.summary.total = len(files)
            if not files:
                self.log("WARN", "No audio files found in that folder.", "warning")
                return self.summary
            self.log("SYS", f"Found {len(files)} audio file(s).")
            self._log_output_mode()
            self._progress(0, len(files))

            for index, path in enumerate(files, 1):
                if self.cancel.is_set():
                    break
                self.log("INFO", f"[{index}/{len(files)}] {path.name}")
                self._clear_language()
                self._track_event("track_start", path,
                                  "isolating" if self.settings.isolate_vocals else "syncing")
                self._progress(index - 1, len(files), path.name)
                try:
                    status = self.process_track(path)
                except OperationCancelled:
                    self._track_event("track_done", path, Status.SKIPPED.value)
                    break
                except Exception as exc:  # strict per-file boundary
                    logger.exception("Unhandled error on %s", path)
                    self.log("ERR", f"{path.name}: {exc.__class__.__name__}: {exc}", "error")
                    status = Status.FAILED
                    self._record_failure(path, str(exc))
                self.summary.bump(status)
                self._track_event("track_done", path, status.value)
                self._progress(index, len(files), path.name)
        finally:
            self.summary.cancelled = self.cancel.is_set()
            self._shutdown()
            elapsed = time.monotonic() - started
            self.emit(Event("done", message=f"{elapsed:.0f}s", data=self.summary.as_dict(),
                            current=self.summary.processed, total=self.summary.total))
        return self.summary

    def _shutdown(self) -> None:
        for resource in (self._fetcher, self._transcriber, self._metadata_client):
            try:
                if resource is not None:
                    resource.close()
            except Exception:
                logger.debug("close failed", exc_info=True)

    # ----------------------------------------------------------- cache helpers
    def _log_output_mode(self) -> None:
        from .config import OUTPUT_MODE_LABELS, TAG_MODE_LABELS

        s = self.settings
        extras = []
        if s.isolate_vocals:
            extras.append("vocal isolation ON")
        if not s.detect_language:
            extras.append("language detection OFF")
        if s.auto_tag:
            extras.append(f"auto-tag {TAG_MODE_LABELS.get(s.tag_mode, s.tag_mode)}")
        self.log(TAG_OUTPUT, f"{OUTPUT_MODE_LABELS.get(s.output_mode, s.output_mode)}"
                 + (f" · {', '.join(extras)}" if extras else ""))

    def _outputs_needed(self, path: Path, lrc_path: Path) -> tuple[bool, bool]:
        """``(need_lrc_file, need_embedded)`` for the selected output mode."""
        s = self.settings
        need_lrc = s.wants_lrc_file and (s.overwrite_existing or not lrc_path.exists())
        need_embed = s.wants_embedded and (s.overwrite_existing or not has_embedded_lyrics(path))
        return need_lrc, need_embed

    def _cache_skip_reason(self, record: Optional[TrackRecord], lrc_path: Path,
                           outputs_present: Optional[bool] = None) -> Optional[str]:
        if record is None or self.settings.overwrite_existing:
            return None
        status = record.status
        if status in (Status.SYNCED_ONLINE.value, Status.SYNCED_ALIGNED.value, Status.SYNCED_AI.value):
            present = lrc_path.exists() if outputs_present is None else outputs_present
            return "already synced (cached)" if present else None
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
                lrc_path: Optional[Path] = None, ai_tried: bool = False, error: str = "",
                language: str = "") -> None:
        extra: dict[str, str] = {}
        thumbnail, thumbnail_mime = b"", ""
        if status.value in ("synced_online", "synced_aligned", "synced_ai"):
            try:
                extra = read_extra_tags(track.path)
            except Exception:
                logger.debug("extra tags unavailable for %s", track.path, exc_info=True)
            try:
                raw, mime = read_artwork(track.path)
                if raw:
                    thumbnail, thumbnail_mime = make_thumbnail(raw, mime)
            except Exception:
                logger.debug("artwork unavailable for %s", track.path, exc_info=True)
        self.db.upsert(TrackRecord(
            fingerprint=fingerprint, path=str(track.path), status=status.value,
            artist=track.artist, title=track.title, duration=track.duration, source=source,
            lrc_path=str(lrc_path) if lrc_path else "", ai_tried=ai_tried, error=error,
            album=track.album or extra.get("album", ""),
            genre=extra.get("genre", ""), year=extra.get("year", ""),
            language=language or extra.get("language", ""),
            output_mode=self.settings.output_mode, provider_used=source,
            synced_at=time.time() if status.value in (
                "synced_online", "synced_aligned", "synced_ai") else 0.0,
            thumbnail=thumbnail, thumbnail_mime=thumbnail_mime,
        ))

    def _cache_artwork(self, path: Path) -> None:
        """Refresh the stored thumbnail after the auto-tagger embedded new cover art."""
        if self.db is None:
            return
        try:
            raw, mime = read_artwork(path)
            if not raw:
                return
            thumbnail, thumbnail_mime = make_thumbnail(raw, mime)
            self.db.set_thumbnail(fingerprint_file(path), thumbnail, thumbnail_mime)
        except Exception:
            logger.debug("thumbnail refresh failed for %s", path, exc_info=True)

    def _record_failure(self, path: Path, error: str) -> None:
        try:
            track = TrackInfo(path=path, title=path.stem)
            self._record(fingerprint_file(path), track, Status.FAILED, error=error)
        except Exception:
            logger.debug("could not record failure", exc_info=True)

    # ------------------------------------------------------------ one track
    def process_track(self, path: Path) -> Status:
        """Resolve lyrics, then (Phase 3) optionally complete the file's tags."""
        fingerprint = fingerprint_file(path)
        status = self._process_track(path, fingerprint)
        if self.settings.auto_tag:
            try:
                track = read_track_info(path)
                # Lyrics embedding may already have moved the cache key; re-key from the
                # file's *current* fingerprint so tag writing cannot leave a stale entry.
                current = fingerprint_file(path)
                if self._auto_tag(path, track):
                    self.db.rekey(current, fingerprint_file(path))
            except Exception as exc:  # tagging must never change the lyrics outcome
                logger.debug("auto-tag failed for %s", path, exc_info=True)
                self.log(TAG_TAGS, f"{path.name}: {exc.__class__.__name__}: {exc}", "warning")
        return status

    def _process_track(self, path: Path, fingerprint: str) -> Status:
        lrc_path = path.with_suffix(".lrc")
        record = self.db.get(fingerprint)

        need_lrc, need_embed = self._outputs_needed(path, lrc_path)
        reason = self._cache_skip_reason(record, lrc_path, outputs_present=not (need_lrc or need_embed))
        if reason:
            self.log("SKIP", reason)
            return Status.SKIPPED
        if not (need_lrc or need_embed):
            self.log("SKIP", {
                "lrc": "A .lrc file already exists next to this track",
                "embed": "Lyrics are already embedded in this file",
            }.get(self.settings.output_mode, "Lyrics already exist as .lrc and embedded"))
            return Status.SKIPPED

        track = read_track_info(path)
        if need_embed and lrc_path.exists() and not self.settings.overwrite_existing:
            return self._embed_existing(path, lrc_path)
        dur = f"{track.duration:.0f}s" if track.duration else "unknown length"
        self.log("INFO", f"Identified as “{track.display_name}” ({dur})")

        holder: dict = {}   # shared, lazily-created audio session for THIS track
        try:
            return self._resolve(path, lrc_path, fingerprint, track, holder)
        finally:
            session = holder.get("session")
            if session is not None:
                session.close()   # frees the decoded audio before the next track

    # ------------------------------------------------------------- auto-tagging
    def _auto_tag(self, path: Path, track: TrackInfo) -> bool:
        """Look up canonical tags + cover art and write them. Returns True when changed."""
        client = self._get_metadata_client()
        self.log(TAG_TAGS, f"Looking up tags for “{track.display_name}”…")
        try:
            meta = client.lookup(track.artist, track.title, track.album, track.duration)
        except Exception as exc:  # network/parse failures are never fatal
            logger.debug("metadata lookup failed for %s", path, exc_info=True)
            self.log(TAG_TAGS, f"No tag match ({exc.__class__.__name__}: {exc})", "warning")
            return False
        if meta is None or meta.is_empty():
            self.log(TAG_TAGS, "No tag match found", "warning")
            return False
        if self.cancel.is_set():
            return False
        before = fingerprint_file(path)
        try:
            result = write_tags(path, meta, mode=self.settings.tag_mode)
        except Exception as exc:  # write_tags already guards; belt and braces
            logger.debug("tag write failed for %s", path, exc_info=True)
            self.log(TAG_TAGS, f"Could not write tags: {exc.__class__.__name__}", "warning")
            return False
        if not result.ok:
            self.log(TAG_TAGS, f"Could not write tags to {path.name}: {result.message}", "warning")
            return False
        if result.fields or result.cover_written:
            after = fingerprint_file(path)
            if after != before and self.db is not None:
                self.db.rekey(before, after)
            mode = "overwritten" if self.settings.tag_mode == "overwrite_all" else "completed"
            self.log("OK", f"Tags {mode} for {path.name} via {meta.source}: {result.describe()}")
            try:
                self.emit(Event("tagged", message=path.name, data={
                    "path": str(path), "fields": result.fields,
                    "cover": result.cover_written, "source": meta.source}))
            except Exception:
                logger.exception("emit failed")
            if result.cover_written:
                self._cache_artwork(path)
            return True
        self.log(TAG_TAGS, f"{path.name}: tags were already complete")
        return False

    # ------------------------------------------------------------- the pipeline
    def _resolve(self, path: Path, lrc_path: Path, fingerprint: str, track: TrackInfo,
                 holder: dict) -> Status:
        settings = self.settings
        fetcher = self._get_fetcher()

        def session() -> AudioSession:
            """One shared, lazily-decoded audio session per track."""
            if "session" not in holder:
                holder["session"] = self._get_transcriber().open_session(track)
            return holder["session"]

        mashup = MashupInfo()
        if settings.detect_mashups:
            mashup = detect_mashup(track.title, track.artist, track.album, path.stem)

        # ---- 1) Tier 1: ready-made synced lyrics for the full title -------------
        outcome = fetcher.fetch(track, tier2=False)
        if self.cancel.is_set():
            raise OperationCancelled()
        if outcome.instrumental:
            self.log("OK", "Instrumental track - no lyrics needed")
            self._record(fingerprint, track, Status.INSTRUMENTAL, source="online")
            return Status.INSTRUMENTAL
        if outcome.synced and outcome.synced.synced_lrc:
            return self._write_synced(
                path, lrc_path, fingerprint, track, outcome.synced.synced_lrc,
                Status.SYNCED_ONLINE, outcome.synced.provider, ai_tried=False)
        if outcome.all_providers_failed:
            self.log("WARN", "Every Tier 1 provider failed - are you offline?", "warning")

        # ---- 2/3) official TEXT: mashup components or one full-title text -------
        reference: Optional[ReferenceScript] = None
        if mashup.is_mashup:
            self.log(TAG_MASHUP, mashup.describe())

        if mashup.is_mashup and mashup.confidence >= MASHUP_CONFIDENT and mashup.has_named_components:
            reference = self._mashup_reference(track, mashup, session)
        else:
            fetcher.fetch_tier2(track, outcome)
            if self.cancel.is_set():
                raise OperationCancelled()
            if outcome.plain and plain_to_lines(outcome.plain.plain or ""):
                reference = ReferenceScript([ReferenceComponent(
                    outcome.plain.artist or track.artist, outcome.plain.title or track.title,
                    plain_to_lines(outcome.plain.plain or ""), outcome.plain.provider,
                    partial=outcome.plain.partial)])
            elif mashup.is_mashup and mashup.confidence >= MASHUP_POSSIBLE and mashup.has_named_components:
                self.log(TAG_MASHUP, "Full-title search failed; treating it as a mashup")
                reference = self._mashup_reference(track, mashup, session)

        can_align = self._ai_enabled() and settings.use_forced_alignment
        ai_tried = False
        ai_error = ""
        ai_crashed = False

        # ---- 4) name the songs from what is actually sung -----------------------
        if reference is None and self._ai_enabled() and can_align and settings.identify_by_audio:
            ai_tried = True
            found = self._identify_components(session(), [])
            if found:
                reference = ReferenceScript(found, is_mashup=len(found) > 1)

        # ---- 5) forced alignment -----------------------------------------------
        can_align = self._ai_enabled() and settings.use_forced_alignment   # may have changed above
        if reference is not None and can_align:
            ai_tried = True
            lrc_text, source, error = self._align(session(), track, reference, allow_identify=True)
            if lrc_text:
                return self._write_synced(path, lrc_path, fingerprint, track, lrc_text,
                                          Status.SYNCED_ALIGNED, source, ai_tried=True,
                                          language=self._session_language(holder))
            ai_error = error
            ai_crashed = bool(error) and not error.startswith("weak alignment")
        elif reference is not None:
            self.log(TAG_PLAIN, "Official lyrics found, but AI alignment is disabled in Settings", "warning")

        # ---- 6) last resort: blind transcription -------------------------------
        if self._ai_enabled() and not (ai_crashed and self._blind_pointless(ai_error)):
            ai_tried = True
            if reference is not None:
                self.log(TAG_BLIND, "Falling back to blind transcription (last resort)", "warning")
            lrc_text, error, crashed = self._blind(session())
            if lrc_text:
                return self._write_synced(path, lrc_path, fingerprint, track, lrc_text,
                                          Status.SYNCED_AI, "Whisper", ai_tried=True,
                                          language=self._session_language(holder))
            ai_error, ai_crashed = error or ai_error, crashed or ai_crashed
        elif not settings.use_ai_fallback:
            self.log("SKIP", "AI fallback is disabled in Settings")

        # ---- 7) unsynced text, or give up --------------------------------------
        plain = outcome.plain
        if plain and plain.plain and settings.save_plain_fallback:
            txt_path = path.with_suffix(".txt")
            embedded: Optional[EmbedResult] = None
            if settings.wants_embedded:
                embedded = embed_lyrics(path, None, plain_text=plain.plain,
                                        language=self._text_language(plain.plain))
                self._log_embed(path, embedded, synced=False)
            wrote_txt = False
            if (settings.wants_lrc_file or embedded is None or not embedded.ok) and \
                    (not txt_path.exists() or settings.overwrite_existing):
                write_text_atomic(txt_path, plain.plain.strip() + "\n")
                self.log("OK", f"Saved unsynced lyrics to {txt_path.name} (from {plain.provider})")
                wrote_txt = True
            if wrote_txt or (embedded is not None and embedded.ok):
                self._record(self._fingerprint_after(path, fingerprint, embedded), track,
                             Status.PLAIN_ONLY, source=plain.provider,
                             lrc_path=txt_path if wrote_txt else None,
                             ai_tried=ai_tried and self._ai_down is None, error=ai_error)
                return Status.PLAIN_ONLY

        self.log("WARN", "No lyrics found for this track", "warning")
        status = Status.FAILED if ai_crashed else Status.NOT_FOUND
        # If Whisper itself was unusable (offline first run...) the AI never really
        # looked at this track: leave ai_tried False so the next run tries again.
        self._record(fingerprint, track, status, ai_tried=ai_tried and self._ai_down is None,
                     error=ai_error)
        return status

    def _blind_pointless(self, error: str) -> bool:
        """Blind transcription would hit the same wall (bad audio / model unavailable)."""
        tr = self._transcriber
        return "could not load model" in error or "decod" in error.lower() or (
            tr is not None and bool(getattr(tr, "_fatal", None)))

    # --------------------------------------------------------- mashup handling
    def _fetch_component(self, artist: str, title: str, fallback_artist: str) -> Optional[ReferenceComponent]:
        """Look up ONE mashup component; any provider/parse failure means 'not found'."""
        fetcher = self._get_fetcher()
        try:
            outcome = fetcher.fetch_for(artist, title)
            if not outcome.reference_text and not artist and fallback_artist:
                outcome = fetcher.fetch_for(fallback_artist, title)   # tag artist as a second chance
            lines = plain_to_lines(outcome.reference_text)
        except OperationCancelled:
            raise
        except Exception as exc:  # a broken provider must never abort the batch
            logger.warning("component lookup failed for %r - %r: %s", artist, title, exc)
            return None
        if not lines:
            return None
        hit = outcome.plain or outcome.synced
        return ReferenceComponent(
            artist=(hit.artist if hit else "") or artist, title=(hit.title if hit else "") or title,
            lines=lines, source=hit.provider if hit else "?",
            partial=bool(outcome.plain and outcome.plain.partial and not outcome.synced))

    def _mashup_reference(self, track: TrackInfo, mashup: MashupInfo,
                          session: Callable[[], AudioSession]) -> Optional[ReferenceScript]:
        """Fetch official lyrics for every named component and assemble the script."""
        components: list[ReferenceComponent] = []
        for artist, title in mashup.components:
            if self.cancel.is_set():
                raise OperationCancelled()
            self.log(TAG_MASHUP, f"Looking up component “{title}”" + (f" by {artist}" if artist else ""))
            try:
                comp = self._fetch_component(artist, title, track.artist)
            except OperationCancelled:
                raise
            except Exception as exc:  # split/parse fallback must stay best-effort
                logger.warning("mashup component %r failed: %s", title, exc)
                self.log(TAG_MASHUP, f"Component “{title}” failed: {exc.__class__.__name__}", "warning")
                comp = None
            if comp:
                components.append(comp)
                self.log(TAG_MASHUP, f"“{comp.label}”: {len(comp.lines)} official lines ({comp.source})")
            else:
                self.log(TAG_MASHUP, f"No lyrics found for component “{title}”", "warning")

        wanted = len(mashup.components)
        if len(components) < wanted and self.settings.identify_by_audio and self._ai_enabled():
            self.log(TAG_MASHUP, f"Only {len(components)} of {wanted} components resolved - "
                     "listening to the audio to find the rest")
            components.extend(self._identify_components(session(), components))
        if not components:
            return None
        return ReferenceScript(components, is_mashup=True)

    # ---------------------------------------------------------- identification
    def _identify_components(self, session: AudioSession,
                             known: list[ReferenceComponent]) -> list[ReferenceComponent]:
        """Blind-listen to slices, search lyric databases with what was sung, verify, collect."""
        fetcher = self._get_fetcher()
        try:
            snippets = session.sample_snippets()
        except OperationCancelled:
            raise
        except (AudioDecodeError, TranscriptionError, NoSpeechError) as exc:
            self.log(TAG_IDENT, f"Could not listen to the audio: {exc}", "warning")
            self._note_ai_error(str(exc))
            return []

        found: list[ReferenceComponent] = []
        for snip in snippets:
            if self.cancel.is_set():
                raise OperationCancelled()
            phrase = best_query_phrase(snip.text, words=9)
            if not phrase:
                continue
            everything = known + found
            if any(snippet_match_score(phrase, "\n".join(c.lines)) >= EXPLAINED_SCORE for c in everything):
                self.log(TAG_IDENT, f"{int(snip.start)}s: already explained by a known song")
                continue
            self.log(TAG_IDENT, f"{int(snip.start)}s: searching lyric databases for “{phrase}”")
            for cand in fetcher.identify_by_lyrics(phrase)[:4]:
                outcome = fetcher.fetch_for(cand.artist, cand.title)
                lines = plain_to_lines(outcome.reference_text)
                text = "\n".join(lines)
                if lines and snippet_match_score(phrase, text) >= VERIFY_SCORE:
                    hit = outcome.plain or outcome.synced
                    comp = ReferenceComponent(cand.artist, cand.title, lines,
                                              hit.provider if hit else cand.source,
                                              partial=bool(outcome.plain and outcome.plain.partial),
                                              first_seen=snip.start)
                    found.append(comp)
                    self.log(TAG_IDENT, f"Identified “{comp.label}” (verified against sung words)")
                    break
            else:
                self.log(TAG_IDENT, f"{int(snip.start)}s: no verified match", "info")
        return found

    # --------------------------------------------------------------- alignment
    def _align(self, session: AudioSession, track: TrackInfo, reference: ReferenceScript,
               *, allow_identify: bool, force: bool = False) -> tuple[Optional[str], str, str]:
        """Forced alignment. Returns ``(lrc_text, source_label, error)``."""
        last_logged = {"p": -1}

        def progress(fraction: float) -> None:
            pct = int(fraction * 100)
            if pct // 20 != last_logged["p"] // 20:
                last_logged["p"] = pct
                self.log(TAG_ALIGN, f"{pct}%")

        label = " + ".join(c.label for c in reference.components)
        self.log(TAG_ALIGN, f"Reference: {label}" + (" [mashup]" if reference.is_mashup else ""))
        try:
            # force=True ("Force Sync Anyway"): accept whatever the alignment produced
            guided = session.align(reference, progress, **({"min_match": 0.0} if force else {}))
        except OperationCancelled:
            raise
        except (AudioDecodeError, TranscriptionError, NoSpeechError) as exc:
            self.log(TAG_ALIGN, f"Alignment failed: {exc}", "error")
            self._note_ai_error(str(exc))
            return None, "", str(exc)

        self.log(TAG_ALIGN, guided.alignment.summary())
        if guided.alignment.switches:
            self.log(TAG_ALIGN, f"{guided.alignment.switches} song transition(s) located")
        if guided.ok:
            if force and guided.alignment.match_ratio < self.settings.min_alignment_match / 100.0:
                self.log(TAG_ALIGN, f"Forced: writing the result although only "
                         f"{guided.alignment.match_ratio:.0%} of the sung words match", "warning")
            self.log("OK", f"Aligned {guided.alignment.emitted_lines} official lines in "
                     f"{guided.seconds:.0f}s ({guided.model}, {guided.device.upper()})")
            sources = sorted({c.source for c in reference.components if c.source})
            return guided.lrc_text, "+".join(sources) + "+Whisper-aligned", ""

        self.log(TAG_ALIGN,
                 f"Only {guided.alignment.match_ratio:.0%} of the sung words match the lyrics "
                 f"(need {guided.min_match:.0%}) - wrong version, or another song is mixed in",
                 "warning")
        if allow_identify and self.settings.identify_by_audio and not reference.any_partial:
            extra = self._identify_components(session, list(reference.components))
            if extra:
                reference = ReferenceScript(list(reference.components) + extra, is_mashup=True)
                self.log(TAG_ALIGN, "Re-aligning with the newly identified song(s)")
                return self._align(session, track, reference, allow_identify=False)
        return None, "", "weak alignment"

    # ------------------------------------------------------------------- blind
    def _blind(self, session: AudioSession) -> tuple[Optional[str], str, bool]:
        """Plain Whisper transcription. Returns ``(lrc_text, error, crashed)``."""
        last_logged = {"p": -1}

        def progress(fraction: float) -> None:
            pct = int(fraction * 100)
            if pct != last_logged["p"]:
                last_logged["p"] = pct
                self.log(TAG_BLIND, f"Transcribing… {pct}%")

        try:
            result = session.blind(progress)
        except OperationCancelled:
            raise
        except NoSpeechError as exc:
            self.log(TAG_BLIND, f"{exc}", "warning")
            return None, str(exc), False
        except (AudioDecodeError, TranscriptionError) as exc:
            self.log(TAG_BLIND, f"Transcription failed: {exc}", "error")
            self._note_ai_error(str(exc))
            return None, str(exc), True
        self.log(TAG_BLIND, f"Transcribed {result.line_count} lines in {result.seconds:.0f}s")
        return result.lrc_text, "", False

    # ------------------------------------------------------------------- write
    @staticmethod
    def _session_language(holder: dict) -> Optional[str]:
        session = holder.get("session")
        info = getattr(session, "language_info", None) if session is not None else None
        return getattr(info, "code", None)

    @staticmethod
    def _text_language(text: str) -> Optional[str]:
        guess = detect_text_language(text)
        return guess.code if guess else None

    def _log_embed(self, path: Path, result: EmbedResult, *, synced: bool = True) -> None:
        if result.ok:
            kind = "synced" if synced else "unsynced"
            self.log("OK", f"Embedded {kind} lyrics into {path.name} ({result.describe()})")
        else:
            self.log("WARN", f"Could not embed lyrics into {path.name}: {result.message}", "warning")

    def _announce_saved(self, path: Path, lrc_text: str, status: Status, source: str) -> None:
        """Tell the GUI which file now has editable lyrics (Phase 3 completion card)."""
        try:
            self.emit(Event("saved", message=path.name, data={
                "path": str(path), "lrc": lrc_text,
                "status": status.value if isinstance(status, Status) else str(status),
                "source": source}))
        except Exception:  # a broken UI callback must never kill the batch
            logger.exception("emit failed")

    @staticmethod
    def _fingerprint_after(path: Path, before: str, embedded: Optional[EmbedResult]) -> str:
        """Tagging rewrites the head of the file, so the cache key must be recomputed."""
        if embedded is not None and embedded.ok:
            try:
                return fingerprint_file(path)
            except OSError:
                pass
        return before

    def _embed_existing(self, path: Path, lrc_path: Path) -> Status:
        """An ``.lrc`` is already there: put it into the tags, no search needed."""
        try:
            text = lrc_path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError as exc:
            self.log("WARN", f"Cannot read {lrc_path.name}: {exc.strerror or exc}", "warning")
            return Status.SKIPPED
        ok, reason, _ = validate_synced(text, None, min_lines=1)
        if not ok:
            self.log("WARN", f"{lrc_path.name} is not usable for embedding ({reason})", "warning")
            return Status.SKIPPED
        result = embed_lyrics(path, text, language=self._text_language(strip_to_plain(text)))
        if result.ok:
            self.log("OK", f"Embedded the existing {lrc_path.name} into {path.name} ({result.describe()})")
            self._announce_saved(path, text, Status.SKIPPED, f"embedded {lrc_path.name}")
        else:
            self._log_embed(path, result)
        return Status.SKIPPED

    def _write_synced(self, path: Path, lrc_path: Path, fingerprint: str, track: TrackInfo,
                      lrc_text: str, status: Status, source: str, *, ai_tried: bool,
                      language: Optional[str] = None) -> Status:
        ok, reason, _ = validate_synced(lrc_text, track.duration, min_lines=1)
        if not ok:
            raise ValueError(f"refusing to write invalid LRC ({reason})")
        settings = self.settings
        resolved_language = language or self._text_language(strip_to_plain(lrc_text))
        wrote_lrc = False
        if settings.wants_lrc_file:
            self._write_lrc_file(lrc_path, lrc_text, source)
            wrote_lrc = True
        embedded: Optional[EmbedResult] = None
        if settings.wants_embedded:
            embedded = embed_lyrics(path, lrc_text, language=resolved_language)
            self._log_embed(path, embedded)
            if not embedded.ok and not wrote_lrc:
                # embed-only mode, but this file cannot be tagged: never lose the result
                self.log(TAG_OUTPUT, f"Saving a separate {lrc_path.name} instead", "warning")
                self._write_lrc_file(lrc_path, lrc_text, source)
                wrote_lrc = True
        self._record(self._fingerprint_after(path, fingerprint, embedded), track, status,
                     source=source, lrc_path=lrc_path if wrote_lrc else None, ai_tried=ai_tried,
                     language=resolved_language or "")
        self._announce_saved(path, lrc_text, status, source)
        return status

    def _write_lrc_file(self, lrc_path: Path, lrc_text: str, source: str) -> None:
        try:
            write_text_atomic(lrc_path, lrc_text)
        except OSError as exc:
            raise OSError(f"cannot write {lrc_path.name}: {exc.strerror or exc}") from exc
        self.log("OK", f"Saved {lrc_path.name} ({source})")

    # ============================================================ "Your Own Lyrics"
    def _confirm(self, **payload) -> bool:
        """Ask the user a yes/no question (worker thread blocks). No ``ask`` -> no."""
        if self.ask is None:
            return False
        try:
            return bool(self.ask(payload))
        except OperationCancelled:
            raise
        except Exception:
            logger.exception("confirmation callback failed")
            return False

    def run_custom(self, audio_path: Path, lyrics_text: str) -> BatchSummary:
        """Sync ONE audio file with lyrics the user pasted.

        1. language: the lyrics' language and the sung language are determined and compared;
        2. validation: a quick speech-to-text pass over a few slices is matched against the
           lyrics. Below ``CUSTOM_LYRICS_MIN_MATCH`` the user decides (Cancel / Force);
        3. text-guided forced alignment maps real timestamps onto the pasted lines;
        4. the result goes out through the normal output mode (.lrc / embedded / both).
        """
        started = time.monotonic()
        self.summary.total = 1
        status = Status.FAILED
        path = Path(audio_path)
        try:
            self._prepare_isolation()
            self._progress(0, 1, path.name)
            self._clear_language()
            self._track_event("track_start", path,
                              "isolating" if self.settings.isolate_vocals else "syncing")
            status = self._custom_track(path, lyrics_text)
        except OperationCancelled:
            status = Status.SKIPPED
        except Exception as exc:  # same per-file boundary as the batch
            logger.exception("custom sync failed for %s", path)
            self.log("ERR", f"{path.name}: {exc.__class__.__name__}: {exc}", "error")
            status = Status.FAILED
        finally:
            self.summary.bump(status)
            self.summary.cancelled = self.cancel.is_set()
            self._track_event("track_done", path, status.value)
            self._progress(1, 1, path.name)
            self._shutdown()
            self.emit(Event("done", message=f"{time.monotonic() - started:.0f}s",
                            data=self.summary.as_dict(), current=self.summary.processed, total=1))
        return self.summary

    def _custom_track(self, path: Path, lyrics_text: str) -> Status:
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            self.log("ERR", "Choose an existing audio file first.", "error")
            return Status.FAILED
        lines = plain_to_lines(lyrics_text or "")
        if len(lines) < 2:
            self.log("ERR", "Paste at least two lines of lyrics first.", "error")
            return Status.FAILED
        self._log_output_mode()
        track = read_track_info(path)
        self.log("INFO", f"Your own lyrics for “{track.display_name}” ({len(lines)} lines)")
        lrc_path = path.with_suffix(".lrc")
        fingerprint = fingerprint_file(path)
        session = self._get_transcriber().open_session(track)
        try:
            return self._custom_sync(session, path, lrc_path, fingerprint, track, lines)
        finally:
            session.close()

    def _custom_sync(self, session: AudioSession, path: Path, lrc_path: Path, fingerprint: str,
                     track: TrackInfo, lines: list[str]) -> Status:
        lyrics_blob = "\n".join(lines)
        text_guess = detect_text_language(lyrics_blob)
        try:
            session.lock_language(lyrics_blob)
            report = self._validate_custom(session, lines)
        except OperationCancelled:
            raise
        except (AudioDecodeError, TranscriptionError, NoSpeechError) as exc:
            self.log(TAG_VERIFY, f"Could not listen to the audio: {exc}", "error")
            self._note_ai_error(str(exc))
            return Status.FAILED

        force = False
        if report.verdict == "mismatch":
            audio_lang = session.language_info
            force = self._confirm(
                kind="mismatch", title=track.display_name, ratio=report.ratio,
                percent=report.percent, threshold=CUSTOM_LYRICS_MIN_MATCH,
                audio_language=audio_lang.name if audio_lang else "",
                lyrics_language=text_guess.name if text_guess else "")
            if not force:
                self.log(TAG_VERIFY, "Stopped: the pasted lyrics do not match this audio.", "warning")
                return Status.SKIPPED
            self.log(TAG_VERIFY, "Continuing anyway (forced by you).", "warning")
        elif report.verdict == "inconclusive":
            self.log(TAG_VERIFY, f"Could not verify the lyrics ({report.summary()}); "
                     "the alignment check will still protect the result.", "warning")
        else:
            self.log(TAG_VERIFY, f"The lyrics match the audio ({report.percent}%).")

        reference = ReferenceScript([ReferenceComponent(track.artist, track.title, lines, "Pasted lyrics")])
        lrc_text, source, error = self._align(session, track, reference, allow_identify=False, force=force)
        if not lrc_text and error == "weak alignment" and not force:
            # The up-front check passed (or could not judge) but the full pass disagrees.
            if self._confirm(kind="weak", title=track.display_name, ratio=report.ratio or 0.0,
                             percent=report.percent, threshold=self.settings.min_alignment_match / 100.0,
                             audio_language=session.language_info.name if session.language_info else "",
                             lyrics_language=text_guess.name if text_guess else ""):
                lrc_text, source, error = self._align(session, track, reference, allow_identify=False,
                                                      force=True)
            else:
                self.log(TAG_ALIGN, "Stopped: the alignment was too weak to trust.", "warning")
                return Status.SKIPPED
        if not lrc_text:
            self.log("ERR", f"Could not time the pasted lyrics ({error or 'no result'})", "error")
            self._record(fingerprint, track, Status.FAILED, ai_tried=self._ai_down is None, error=error)
            return Status.FAILED
        return self._write_synced(path, lrc_path, fingerprint, track, lrc_text, Status.SYNCED_ALIGNED,
                                  "Pasted lyrics + Whisper-aligned", ai_tried=True,
                                  language=session.language_info.code if session.language_info else None)

    def _validate_custom(self, session: AudioSession, lines: list[str]) -> MatchReport:
        """Quick speech-to-text on a few slices, compared with the pasted lyrics."""
        self.log(TAG_VERIFY, "Listening to a few slices to check the lyrics belong to this audio…")
        snippets = session.sample_snippets(CUSTOM_SNIPPET_FRACTIONS, CUSTOM_SNIPPET_SEC,
                                           language=session._language(), tag=TAG_VERIFY)
        report = match_lyrics([sn.text for sn in snippets], lines, CUSTOM_LYRICS_MIN_MATCH)
        self.log(TAG_VERIFY, f"Audio vs. pasted lyrics: {report.summary()}")
        return report
