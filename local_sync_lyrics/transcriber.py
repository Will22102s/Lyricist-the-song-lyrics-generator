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
from typing import Callable, Optional

import numpy as np

from .app_paths import models_dir
from .config import Settings
from .hardware_profiler import HardwareProfile, get_profile, register_pip_cuda_libraries
from .lrc_utils import LrcLine, build_lrc
from .models import TrackInfo, TranscriptionResult

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
LogFn = Callable[[str, str, str], None]  # (tag, message, level)


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
    def _run(self, audio: np.ndarray, language: Optional[str],
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

    def transcribe(
        self, track: TrackInfo, progress: Optional[Callable[[float], None]] = None
    ) -> TranscriptionResult:
        started = time.monotonic()
        max_seconds = self.settings.max_audio_minutes * 60
        if track.duration and track.duration > max_seconds:
            raise TranscriptionError(
                f"track is {track.duration / 60:.0f} min long (limit "
                f"{self.settings.max_audio_minutes} min) - raise the limit in Settings"
            )

        self._load()  # fail fast (and only once per batch) if the model is unavailable
        self._log("AI", "Decoding audio…", "info")
        audio = decode_audio_safe(track.path, max_seconds=max_seconds, cancel=self._cancel)
        language = None if self.settings.language in ("", "auto") else self.settings.language

        self._log("AI", f"Transcribing with “{self.model_name}” on {self.device.upper()}…", "info")
        try:
            segments, info = self._run(audio, language, progress)
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
                    segments, info = self._run(audio, language, progress)
                except OperationCancelled:
                    raise
                except Exception as exc2:
                    raise TranscriptionError(f"CPU retry failed: {_short(exc2)}") from exc2
            else:
                raise TranscriptionError(_short(exc)) from exc
        finally:
            del audio

        lines = clean_segments(segments)
        if not lines:
            raise NoSpeechError("no lyrics recognised (instrumental or unintelligible)")

        detected = str(getattr(info, "language", "") or language or "?")
        probability = float(getattr(info, "language_probability", 0.0) or 0.0)
        self._log("AI", f"Language: {detected} ({probability:.0%}) · {len(lines)} lines", "info")
        lrc = build_lrc(lines, artist=track.artist, title=track.title, album=track.album,
                        duration=track.duration, source=f"Whisper {self.model_name}")
        return TranscriptionResult(
            lrc_text=lrc, language=detected, language_probability=probability,
            line_count=len(lines), model=self.model_name, device=self.device,
            seconds=time.monotonic() - started,
        )


def _short(exc: Optional[BaseException], limit: int = 220) -> str:
    if exc is None:
        return "unknown error"
    text = f"{exc.__class__.__name__}: {exc}".replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"
