"""Batch runner: discover tracks, resolve lyrics, write ``.lrc`` files.

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

Guarantees (unchanged)
----------------------
* Every track runs inside its own ``try/except``: one bad file never stops the batch.
* Cancellation is honoured between files, providers, windows and Whisper segments.
* Finished tracks are cached in SQLite and skipped next time.
* The runner talks to the outside world only through ``emit`` (thread-safe queue in
  the GUI), so it works unchanged under Flet, the CLI and the tests.
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
from .lrc_utils import plain_to_lines, validate_synced, write_text_atomic
from .metadata import read_track_info
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
TAG_ALIGN = "AI: Forced Alignment"
TAG_IDENT = "AI: Identify"
TAG_BLIND = "AI: Blind"
TAG_PLAIN = "API: Plain Text"


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
        self._ai_down: Optional[str] = None   # set once the Whisper model is unusable

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

    # ----------------------------------------------------------- cache helpers
    def _cache_skip_reason(self, record: Optional[TrackRecord], lrc_path: Path) -> Optional[str]:
        if record is None or self.settings.overwrite_existing:
            return None
        status = record.status
        if status in (Status.SYNCED_ONLINE.value, Status.SYNCED_ALIGNED.value, Status.SYNCED_AI.value):
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

    # ------------------------------------------------------------ one track
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

        holder: dict = {}   # shared, lazily-created audio session for THIS track
        try:
            return self._resolve(path, lrc_path, fingerprint, track, holder)
        finally:
            session = holder.get("session")
            if session is not None:
                session.close()   # frees the decoded audio before the next track

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
                                          Status.SYNCED_ALIGNED, source, ai_tried=True)
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
                                          Status.SYNCED_AI, "Whisper", ai_tried=True)
            ai_error, ai_crashed = error or ai_error, crashed or ai_crashed
        elif not settings.use_ai_fallback:
            self.log("SKIP", "AI fallback is disabled in Settings")

        # ---- 7) unsynced text, or give up --------------------------------------
        plain = outcome.plain
        if plain and plain.plain and settings.save_plain_fallback:
            txt_path = path.with_suffix(".txt")
            if not txt_path.exists() or settings.overwrite_existing:
                write_text_atomic(txt_path, plain.plain.strip() + "\n")
                self.log("OK", f"Saved unsynced lyrics to {txt_path.name} (from {plain.provider})")
                self._record(fingerprint, track, Status.PLAIN_ONLY, source=plain.provider,
                             lrc_path=txt_path, ai_tried=ai_tried and self._ai_down is None,
                             error=ai_error)
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
        fetcher = self._get_fetcher()
        outcome = fetcher.fetch_for(artist, title)
        if not outcome.reference_text and not artist and fallback_artist:
            outcome = fetcher.fetch_for(fallback_artist, title)   # tag artist as a second chance
        lines = plain_to_lines(outcome.reference_text)
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
            comp = self._fetch_component(artist, title, track.artist)
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
               *, allow_identify: bool) -> tuple[Optional[str], str, str]:
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
            guided = session.align(reference, progress)
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
