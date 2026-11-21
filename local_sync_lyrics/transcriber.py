"""Local AI fallback: transcribe a song to timed lyrics with faster-whisper.

Two robustness problems this module solves
------------------------------------------
1. **PyAV version drift.** ``faster_whisper.audio.decode_audio`` calls
   ``av.open(..., metadata_errors="ignore")``, which raises ``TypeError`` on some
   PyAV releases (and silently differs between others). We therefore decode with
   our own PyAV code that probes the available ``av.open`` signature, tolerates
   both list/single-frame resampler return types, and keeps partial audio from a
   truncated file. faster-whisper's own decoder is only a last-resort fallback.

2. **GPU libraries that are missing or broken.** CTranslate2 often loads fine and
   then dies during the *first inference* with ``RuntimeError: Library
   cublas64_12.dll is not found``. We catch that (and OOM / unsupported compute
   type), drop to CPU ``int8`` and retry the same track transparently.
"""

from __future__ import annotations

import gc
import logging
import re
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TypeVar

import numpy as np

from .app_paths import models_dir
from .config import Settings
from .hardware_profiler import HardwareProfile, get_profile, register_pip_cuda_libraries
from .lrc_utils import AlignmentResult, ForcedAligner, LrcLine, RecWord, build_lrc
from .models import ReferenceScript, Snippet, TrackInfo, TranscriptionResult

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
LogFn = Callable[[str, str, str], None]  # (tag, message, level)

# --- guided alignment tuning -------------------------------------------------
CHUNK_SEC = 30.0            # == one Whisper window, so initial_prompt applies to every call
OVERLAP_SEC = 3.0           # windows overlap so words on a seam are never cut in half
PROMPT_TOKEN_BUDGET = 200   # Whisper keeps only the last ~223 prompt tokens
MAX_GAP_REFINEMENTS = 6     # re-listens per track (bounded cost)
SNIPPET_FRACTIONS = (0.02, 0.30, 0.55, 0.80)   # where to listen when naming mashup songs
SNIPPET_SEC = 30.0
# Silero VAD tuned for singing: lower threshold so soft vocals are kept.
VAD_SINGING = {"threshold": 0.30, "min_silence_duration_ms": 500, "speech_pad_ms": 300}


class OperationCancelled(Exception):
    """Raised when the user pressed Stop."""


class AudioDecodeError(Exception):
    pass


class TranscriptionError(Exception):
    pass


class NoSpeechError(TranscriptionError):
    """Audio decoded fine but no usable lyrics were recognised."""


# ---------------------------------------------------------------- audio decode
def _as_frames(result) -> list:
    """PyAV's ``resample`` returns a list (>=9), a frame (older) or None."""
    if result is None:
        return []
    if isinstance(result, (list, tuple)):
        return list(result)
    return [result]


def _open_container(stack: ExitStack, path: Path):
    """Open *path* with PyAV, surviving signature differences between versions."""
    import av

    attempts = (
        ("path+metadata_errors", lambda: av.open(str(path), mode="r", metadata_errors="ignore")),
        ("path", lambda: av.open(str(path), mode="r")),
        # Some builds mishandle non-ASCII Windows paths: read through a file object.
        ("file-object", lambda: av.open(stack.enter_context(open(path, "rb")), mode="r")),
    )
    errors: list[str] = []
    for label, opener in attempts:
        try:
            container = opener()
            stack.callback(container.close)
            return container
        except TypeError as exc:  # unknown keyword on this PyAV version
            errors.append(f"{label}: {exc}")
        except Exception as exc:
            errors.append(f"{label}: {exc.__class__.__name__}: {exc}")
    raise AudioDecodeError("PyAV could not open the file (" + "; ".join(errors) + ")")


def _decode_with_pyav(
    path: Path, max_seconds: Optional[float], cancel: Optional[threading.Event]
) -> np.ndarray:
    from av.audio.resampler import AudioResampler

    chunks: list[np.ndarray] = []
    total = 0
    limit = int(max_seconds * SAMPLE_RATE) if max_seconds else None
    truncated_by_error: Optional[Exception] = None

    with ExitStack() as stack:
        container = _open_container(stack, path)
        stream = next((s for s in container.streams if getattr(s, "type", "") == "audio"), None)
        if stream is None:
            raise AudioDecodeError("file has no audio stream")
        resampler = AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)

        def take(frames) -> None:
            nonlocal total
            for out in _as_frames(frames):
                arr = np.asarray(out.to_ndarray()).reshape(-1)
                if arr.size:
                    chunks.append(arr.astype(np.int16, copy=False))
                    total += arr.size

        try:
            for frame in container.decode(stream):
                if cancel is not None and cancel.is_set():
                    raise OperationCancelled()
                take(resampler.resample(frame))
                if limit is not None and total >= limit:
                    break
        except OperationCancelled:
            raise
        except Exception as exc:  # corrupt/truncated stream: keep what we have
            truncated_by_error = exc
        try:
            take(resampler.resample(None))  # flush
        except Exception:
            pass

    if truncated_by_error is not None:
        if total < SAMPLE_RATE * 8:
            raise AudioDecodeError(f"decoding failed early: {truncated_by_error}")
        logger.warning("%s: decoding stopped early (%s); using %.0fs of audio",
                       path.name, truncated_by_error, total / SAMPLE_RATE)
    if not chunks:
        raise AudioDecodeError("no audio samples decoded")
    audio = np.concatenate(chunks)
    if limit is not None:
        audio = audio[:limit]
    return audio.astype(np.float32) / 32768.0


def decode_audio_safe(
    path: Path, max_seconds: Optional[float] = None, cancel: Optional[threading.Event] = None
) -> np.ndarray:
    """Decode any audio file to mono 16 kHz float32. Raises AudioDecodeError."""
    errors: list[str] = []
    try:
        audio = _decode_with_pyav(Path(path), max_seconds, cancel)
    except OperationCancelled:
        raise
    except Exception as exc:
        errors.append(f"own decoder: {exc}")
        audio = None
    if audio is None:
        try:  # last resort: faster-whisper's decoder (may TypeError on odd PyAV)
            from faster_whisper.audio import decode_audio

            audio = np.asarray(decode_audio(str(path), sampling_rate=SAMPLE_RATE), dtype=np.float32)
            if max_seconds:
                audio = audio[: int(max_seconds * SAMPLE_RATE)]
        except Exception as exc:
            errors.append(f"faster-whisper decoder: {exc.__class__.__name__}: {exc}")
            raise AudioDecodeError("; ".join(errors)) from exc
    if audio.size < SAMPLE_RATE:  # < 1 second
        raise AudioDecodeError("audio is shorter than one second")
    if float(np.max(np.abs(audio))) < 1e-4:
        raise NoSpeechError("audio is silent")
    return audio


# ------------------------------------------------------------ post-processing
_HALLUCINATION_RE = re.compile(
    r"(thanks?\s+(you\s+)?for\s+watching|please\s+subscribe|like\s+and\s+subscribe|"
    r"subscribe\s+to|subtitles?\s+by|captions?\s+by|amara\.org|transcribed\s+by|"
    r"www\.\w+\.\w+|©|all\s+rights\s+reserved)",
    re.IGNORECASE,
)
_NON_LYRIC_RE = re.compile(r"^[\s♪♫\[\]\(\)\*\-–—.…]*(music|applause|laughter|instrumental|singing)?[\s♪♫\[\]\(\)\*\-–—.…]*$", re.I)


@dataclass
class _Segment:
    start: float
    end: float
    text: str
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0


def clean_segments(segments: list[_Segment], max_repeats: int = 4) -> list[LrcLine]:
    """Drop typical Whisper junk on music (hallucinated credits, endless loops)."""
    lines: list[LrcLine] = []
    repeat = 0
    previous = ""
    for seg in segments:
        text = " ".join(seg.text.split())
        if not text or _NON_LYRIC_RE.match(text) or _HALLUCINATION_RE.search(text):
            continue
        if seg.no_speech_prob > 0.85 and seg.avg_logprob < -1.0:
            continue
        key = text.casefold()
        repeat = repeat + 1 if key == previous else 1
        previous = key
        if repeat > max_repeats:
            continue
        lines.append(LrcLine(seg.start, text))
    return lines


_GPU_ERROR_HINTS = (
    "cublas", "cudnn", "cuda", "cudart", "nvrtc", "libcu", "out of memory",
    "no kernel image", "compute type", "float16", "device",
)


def is_gpu_failure(exc: BaseException) -> bool:
    if not isinstance(exc, (RuntimeError, ValueError, OSError)):
        return False
    text = str(exc).lower()
    return any(hint in text for hint in _GPU_ERROR_HINTS)


# ------------------------------------------------------------------ transcriber
class Transcriber:
    """Lazy-loading wrapper around ``faster_whisper.WhisperModel``."""

    def __init__(
        self,
        settings: Settings,
        profile: Optional[HardwareProfile] = None,
        log: Optional[LogFn] = None,
        cancel: Optional[threading.Event] = None,
    ) -> None:
        self.settings = settings
        self.profile = profile or get_profile()
        self._log = log or (lambda tag, msg, level: None)
        self._cancel = cancel or threading.Event()
        self._model = None
        self.model_name = ""
        self.device = ""
        self.compute_type = ""
        self._force_cpu = False
        self._fatal: Optional[str] = None  # model could not be loaded: don't retry per track

    # ------------------------------------------------------------- planning
    def _plan(self) -> tuple[str, str, list[str]]:
        """Return (model_name, device, [compute types to try in order])."""
        rec = self.profile.recommendation
        use_gpu = (
            self.settings.prefer_gpu and self.profile.cuda_usable and not self._force_cpu
        )
        if use_gpu:
            if self.settings.whisper_model != "auto":
                model = self.settings.whisper_model
            elif rec.device == "cuda":
                model = rec.model
            else:
                model = "small"
            first = rec.compute_type if rec.device == "cuda" else "float16"
            order = [first, "int8_float16", "int8", "float32"]
            return model, "cuda", list(dict.fromkeys(order))
        model = (
            self.profile.cpu_recommendation.model
            if self.settings.whisper_model == "auto" else self.settings.whisper_model
        )
        return model, "cpu", ["int8", "float32"]

    # -------------------------------------------------------------- loading
    @staticmethod
    def _make_model(name: str, device: str, compute_type: str, threads: int, local_only: bool):
        from faster_whisper import WhisperModel

        return WhisperModel(
            name,
            device=device,
            compute_type=compute_type,
            cpu_threads=threads if device == "cpu" else 0,
            num_workers=1,
            download_root=str(models_dir()),
            local_files_only=local_only,
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        if self._fatal:
            raise TranscriptionError(self._fatal)
        try:
            self._load_uncached()
        except TranscriptionError as exc:
            self._fatal = str(exc)  # e.g. offline on first run: fail fast for the rest of the batch
            raise

    def _load_uncached(self) -> None:
        self.settings.apply_environment()  # injects HF_TOKEN
        register_pip_cuda_libraries()
        name, device, compute_types = self._plan()
        threads = max(1, min(self.profile.physical_cores, 8))
        last_error: Optional[BaseException] = None

        for ctype in compute_types:
            try:
                try:  # cached copy first: no network, instant start
                    model = self._make_model(name, device, ctype, threads, local_only=True)
                except Exception as exc:
                    if device == "cuda" and is_gpu_failure(exc):
                        raise
                    self._log("AI", f"Downloading Whisper model “{name}” "
                              "(first run only; this can take a few minutes)…", "info")
                    model = self._make_model(name, device, ctype, threads, local_only=False)
                self._model, self.model_name = model, name
                self.device, self.compute_type = device, ctype
                self._log("AI", f"Model “{name}” ready on {device.upper()} ({ctype})", "info")
                if device == "cpu" and name in ("medium", "large-v3", "large-v3-turbo"):
                    self._log("AI", f"“{name}” on CPU is slow - consider a smaller model "
                              "in Settings for big libraries", "warning")
                return
            except Exception as exc:
                last_error = exc
                if device == "cuda" and is_gpu_failure(exc):
                    self._log("AI", f"GPU init failed with {ctype}: {_short(exc)}", "warning")
                    continue
                raise TranscriptionError(f"could not load model “{name}”: {_short(exc)}") from exc

        if device == "cuda":
            self._log("AI", "GPU unavailable - falling back to CPU (int8)", "warning")
            self._force_cpu = True
            self._model = None
            self._load_uncached()
            return
        raise TranscriptionError(f"could not load model “{name}”: {_short(last_error)}")

    def close(self) -> None:
        self._model = None
        gc.collect()

    # ------------------------------------------------------------- inference
    def count_tokens(self, text: str) -> int:
        """Whisper-tokenizer length of *text* (estimated if the tokenizer is unavailable)."""
        tokenizer = getattr(self._model, "hf_tokenizer", None)
        if tokenizer is not None:
            try:
                return len(tokenizer.encode(" " + text, add_special_tokens=False).ids)
            except Exception:
                pass
        cjk = sum(1 for ch in text if "\u3040" <= ch <= "\ud7af")
        return int(max(len(text) / 3.0, float(cjk))) + 1

    def _guard_gpu(self, work: Callable[[], T]) -> T:
        """Run *work*; on a GPU failure reload the model on CPU int8 and run it once more.

        CTranslate2 loads fine and then dies lazily during the first inference
        ("Library cublas64_12.dll is not found"), so the retry has to wrap the
        inference itself, not just the model construction.
        """
        try:
            return work()
        except OperationCancelled:
            raise
        except Exception as exc:
            if self.device == "cuda" and is_gpu_failure(exc):
                self._log("AI", f"GPU error during inference ({_short(exc)}); "
                          "retrying on CPU int8", "warning")
                self._force_cpu = True
                self._model = None
                gc.collect()
                self._load()
                try:
                    return work()
                except OperationCancelled:
                    raise
                except Exception as exc2:
                    raise TranscriptionError(f"CPU retry failed: {_short(exc2)}") from exc2
            raise TranscriptionError(_short(exc)) from exc

    def _run_blind(self, audio: np.ndarray, language: Optional[str],
                   progress: Optional[Callable[[float], None]]) -> tuple[list[_Segment], object]:
        assert self._model is not None
        on_gpu = self.device == "cuda"
        segments_iter, info = self._model.transcribe(
            audio,
            language=language,
            task="transcribe",
            beam_size=5 if on_gpu else 2,
            condition_on_previous_text=False,   # avoids repetition loops on songs
            vad_filter=self.settings.vad_filter,
            no_speech_threshold=0.6,
            word_timestamps=False,
        )
        total = float(getattr(info, "duration", 0.0) or len(audio) / SAMPLE_RATE)
        out: list[_Segment] = []
        last_reported = 0.0
        # The generator does the real work lazily: GPU failures surface *here*.
        for seg in segments_iter:
            if self._cancel.is_set():
                raise OperationCancelled()
            out.append(_Segment(float(seg.start), float(seg.end), str(seg.text),
                                float(getattr(seg, "no_speech_prob", 0.0) or 0.0),
                                float(getattr(seg, "avg_logprob", 0.0) or 0.0)))
            if progress and total > 0:
                frac = min(1.0, seg.end / total)
                if frac - last_reported >= 0.25:
                    last_reported = frac
                    progress(frac)
        return out, info

    def _recognize(self, audio_slice: np.ndarray, offset: float, *, prompt: str,
                   vad: bool, language: Optional[str]) -> "_Window":
        """Word-level recognition of ONE <=30 s slice (absolute timestamps).

        One slice == one Whisper window on purpose: with
        ``condition_on_previous_text=False`` faster-whisper applies ``initial_prompt``
        only to the first window of a call, so a longer slice would lose the
        reference lyrics after 30 s.
        """
        self._load()

        def work() -> _Window:
            assert self._model is not None
            on_gpu = self.device == "cuda"
            segments_iter, info = self._model.transcribe(
                audio_slice,
                language=language,
                task="transcribe",
                beam_size=5 if on_gpu else 2,
                condition_on_previous_text=False,
                initial_prompt=prompt or None,
                word_timestamps=True,
                vad_filter=vad,
                vad_parameters=dict(VAD_SINGING) if vad else None,
                no_speech_threshold=0.6,
            )
            phrases: list[list[RecWord]] = []
            for seg in segments_iter:
                if self._cancel.is_set():
                    raise OperationCancelled()
                text = " ".join(str(seg.text).split())
                if not text or _NON_LYRIC_RE.match(text) or _HALLUCINATION_RE.search(text):
                    continue
                if (float(getattr(seg, "no_speech_prob", 0.0) or 0.0) > 0.85
                        and float(getattr(seg, "avg_logprob", 0.0) or 0.0) < -1.0):
                    continue
                words = [RecWord(str(w.word), offset + float(w.start), offset + float(w.end),
                                 float(getattr(w, "probability", 1.0) or 0.0))
                         for w in (getattr(seg, "words", None) or [])]
                if words:
                    phrases.append(words)
            return _Window(phrases, str(getattr(info, "language", "") or ""),
                           float(getattr(info, "language_probability", 0.0) or 0.0))

        return self._guard_gpu(work)

    # ---------------------------------------------------------------- sessions
    def open_session(self, track: TrackInfo) -> "AudioSession":
        return AudioSession(self, track)

    def transcribe(self, track: TrackInfo,
                   progress: Optional[Callable[[float], None]] = None) -> TranscriptionResult:
        """Blind transcription (the last-resort path). Kept for backward compatibility."""
        with self.open_session(track) as session:
            return session.blind(progress)


T = TypeVar("T")


@dataclass
class _Window:
    phrases: list[list[RecWord]]
    language: str
    probability: float

    @property
    def text(self) -> str:
        return "\n".join(" ".join(w.text.strip() for w in p) for p in self.phrases)


@dataclass
class GuidedResult:
    """Outcome of text-guided forced alignment."""

    lrc_text: str
    alignment: AlignmentResult
    model: str
    device: str
    language: str
    seconds: float
    used_vad: bool
    refined_gaps: int
    min_match: float

    @property
    def ok(self) -> bool:
        return self.alignment.ok(self.min_match)


def plan_windows(duration: float, chunk: float = CHUNK_SEC,
                 overlap: float = OVERLAP_SEC) -> list[tuple[float, float]]:
    """Overlapping <=30 s windows covering the whole track."""
    if duration <= chunk + 2.0:
        return [(0.0, duration)]
    hop = chunk - overlap
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < duration - overlap:
        end = min(start + chunk, duration)
        windows.append((start, end))
        if end >= duration:
            break
        start += hop
    return windows


class AudioSession:
    """One track's decoded audio plus the three things we do with it:

    * :meth:`sample_snippets`  - quick blind listens, to NAME the songs in a mashup
    * :meth:`align`            - text-guided forced alignment (the preferred path)
    * :meth:`blind`            - plain transcription (last resort)

    The audio is decoded once and shared; ``close()`` frees it.
    """

    def __init__(self, transcriber: Transcriber, track: TrackInfo) -> None:
        self.tr = transcriber
        self.track = track
        self._audio: Optional[np.ndarray] = None

    def __enter__(self) -> "AudioSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._audio = None
        gc.collect()

    @property
    def audio(self) -> np.ndarray:
        if self._audio is None:
            limit = self.tr.settings.max_audio_minutes * 60
            if self.track.duration and self.track.duration > limit:
                raise TranscriptionError(
                    f"track is {self.track.duration / 60:.0f} min long (limit "
                    f"{self.tr.settings.max_audio_minutes} min) - raise the limit in Settings")
            self.tr._load()  # fail fast (once per batch) if the model is unavailable
            self.tr._log("AI", "Decoding audio…", "info")
            self._audio = decode_audio_safe(self.track.path, max_seconds=limit, cancel=self.tr._cancel)
        return self._audio

    @property
    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE

    def _language(self) -> Optional[str]:
        lang = self.tr.settings.language
        return None if lang in ("", "auto") else lang

    # ------------------------------------------------------------------ blind
    def blind(self, progress: Optional[Callable[[float], None]] = None) -> TranscriptionResult:
        """Whisper-only transcription. Spelling and timing both come from the model."""
        tr, track = self.tr, self.track
        started = time.monotonic()
        audio = self.audio
        language = self._language()
        tr._log("AI: Blind", f"Transcribing with “{tr.model_name}” on {tr.device.upper()} "
                "(no reference lyrics)…", "info")
        segments, info = tr._guard_gpu(lambda: tr._run_blind(audio, language, progress))
        lines = clean_segments(segments)
        if not lines:
            raise NoSpeechError("no lyrics recognised (instrumental or unintelligible)")
        detected = str(getattr(info, "language", "") or language or "?")
        probability = float(getattr(info, "language_probability", 0.0) or 0.0)
        tr._log("AI: Blind", f"Language: {detected} ({probability:.0%}) · {len(lines)} lines", "info")
        lrc = build_lrc(lines, artist=track.artist, title=track.title, album=track.album,
                        duration=track.duration, source=f"Whisper {tr.model_name}")
        return TranscriptionResult(
            lrc_text=lrc, language=detected, language_probability=probability,
            line_count=len(lines), model=tr.model_name, device=tr.device,
            seconds=time.monotonic() - started,
        )

    # --------------------------------------------------------------- snippets
    def sample_snippets(self, fractions: tuple[float, ...] = SNIPPET_FRACTIONS,
                        seconds: float = SNIPPET_SEC) -> list[Snippet]:
        """Blind-listen to a few slices (start + mid-track) to harvest lyric snippets.

        In a mashup the songs change over time, so slices spread across the track
        each catch a different song; the text is later used to look the songs up.
        """
        audio, duration = self.audio, self.duration
        starts = sorted({round(min(max(0.0, duration * f), max(0.0, duration - seconds)), 1)
                         for f in fractions})
        snippets: list[Snippet] = []
        for start in starts:
            if self.tr._cancel.is_set():
                raise OperationCancelled()
            chunk = audio[int(start * SAMPLE_RATE): int((start + seconds) * SAMPLE_RATE)]
            if len(chunk) < 6 * SAMPLE_RATE:
                continue
            window = self.tr._recognize(chunk, start, prompt="", vad=True, language=None)
            text = window.text
            if len(text.split()) >= 5:
                snippets.append(Snippet(start, text))
                preview = " ".join(text.split())[:70]
                self.tr._log("AI: Identify", f"{int(start // 60)}:{int(start % 60):02d} heard “{preview}”", "info")
        return snippets

    # ----------------------------------------------------------------- guided
    def _align_pass(self, aligner: ForcedAligner, windows: list[tuple[float, float]],
                    use_vad: bool, multi: bool,
                    progress: Optional[Callable[[float], None]]) -> None:
        tr, audio = self.tr, self.audio
        language = self._language()
        last = len(windows) - 1
        for i, (w0, w1) in enumerate(windows):
            if tr._cancel.is_set():
                raise OperationCancelled()
            # Whisper reads only ~220 prompt tokens: send the lines expected NEXT.
            prompt = aligner.prompt_hint(PROMPT_TOKEN_BUDGET, tr.count_tokens)
            chunk = audio[int(w0 * SAMPLE_RATE): int(w1 * SAMPLE_RATE)]
            window = tr._recognize(chunk, w0, prompt=prompt, vad=use_vad, language=language)
            if language is None and not multi and window.language and window.probability >= 0.5:
                language = window.language   # lock: stops per-window language flip-flops
                tr._log("AI: Forced Alignment", f"Language locked to {language} "
                        f"({window.probability:.0%})", "info")
            # Overlap seams: each phrase is owned by exactly one window (by midpoint).
            lo = w0 + (OVERLAP_SEC / 2 if i > 0 else 0.0)
            hi = w1 - OVERLAP_SEC / 2 if i < last else float("inf")
            owned = [p for p in window.phrases if lo <= (p[0].start + p[-1].end) / 2 < hi]
            aligner.feed(owned)
            if progress:
                progress((i + 1) / len(windows))

    def _refine_gaps(self, aligner: ForcedAligner, result: AlignmentResult) -> int:
        """Re-listen to stretches whose words were never heard (timing was a guess).

        For each gap we run Whisper again on just that audio, WITHOUT VAD and with the
        missing lines as the prompt, and match only inside that known region.
        """
        tr, audio, duration = self.tr, self.audio, self.duration
        done = 0
        for gap in result.gaps[:MAX_GAP_REFINEMENTS]:
            if tr._cancel.is_set():
                raise OperationCancelled()
            comp = aligner.comps[gap.comp]
            first, last = comp.line_of[gap.tok_lo], comp.line_of[gap.tok_hi - 1]
            start = max(0.0, gap.t_start - 0.6)
            end = min(duration, gap.t_end + 0.6, start + CHUNK_SEC)
            if end - start < 1.5:
                continue
            prompt = "\n".join(comp.lines[first:last + 1])
            chunk = audio[int(start * SAMPLE_RATE): int(end * SAMPLE_RATE)]
            window = tr._recognize(chunk, start, prompt=prompt, vad=False, language=self._language())
            added = aligner.feed(window.phrases, focus=(gap.comp, gap.tok_lo, gap.tok_hi))
            done += 1
            tr._log("AI: Forced Alignment",
                    f"Re-listened to {start:.0f}s–{end:.0f}s for {gap.size} unheard words "
                    f"({added} phrase(s) placed)", "info")
        return done

    def align(self, reference: ReferenceScript,
              progress: Optional[Callable[[float], None]] = None) -> GuidedResult:
        """Snap the official lyrics onto the audio.

        Whisper supplies WHEN words are sung; the reference supplies WHAT they are.
        Output lines carry the official spelling, so mishearings never reach the file.
        """
        tr, track = self.tr, self.track
        started = time.monotonic()
        windows = plan_windows(self.duration)
        multi = len(reference.components) > 1
        min_match = tr.settings.min_alignment_match / 100.0
        best: Optional[tuple[ForcedAligner, AlignmentResult, bool]] = None

        # VAD keeps Whisper from "singing" over instrumentals, but Silero can also
        # swallow soft vocals; if the match is weak we listen again without it.
        for use_vad in (True, False):
            aligner = ForcedAligner([c.lines for c in reference.components])
            tr._log("AI: Forced Alignment",
                    f"Aligning {reference.line_count} official lines over {len(windows)} window(s)"
                    f"{' with VAD' if use_vad else ' without VAD'}…", "info")
            self._align_pass(aligner, windows, use_vad, multi, progress)
            result = aligner.finalize(self.duration)
            if best is None or result.match_ratio > best[1].match_ratio:
                best = (aligner, result, use_vad)
            if result.ok(min_match):
                break
            if use_vad:
                tr._log("AI: Forced Alignment",
                        f"Weak match with VAD ({result.match_ratio:.0%}); retrying without it", "warning")
        assert best is not None
        aligner, result, used_vad = best

        refined = 0
        if result.gaps and result.ok(min_match):
            refined = self._refine_gaps(aligner, result)
            if refined:
                result = aligner.finalize(self.duration)

        lrc = build_lrc(result.lines, artist=track.artist, title=track.title, album=track.album,
                        duration=track.duration or self.duration,
                        source=f"Whisper {tr.model_name} aligned")
        return GuidedResult(
            lrc_text=lrc, alignment=result, model=tr.model_name, device=tr.device,
            language=self._language() or "auto", seconds=time.monotonic() - started,
            used_vad=used_vad, refined_gaps=refined, min_match=min_match,
        )


def _short(exc: Optional[BaseException], limit: int = 170) -> str:
    if exc is None:
        return "unknown error"
    text = f"{exc.__class__.__name__}: {exc}".replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"
