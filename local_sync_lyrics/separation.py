"""Vocal isolation: strip the band so Whisper hears the singer.

Dense mixes (distorted guitars, EDM, thick backing vocals) are where Whisper's word
timestamps drift and the forced alignment's match ratio drops. Running the track
through a source-separation model first and feeding Whisper only the vocal stem fixes
most of that, at the price of minutes of CPU per song - hence the switch is OFF by
default.

Backends (first one that is installed wins)
-------------------------------------------
1. ``audio-separator``  (``pip install "audio-separator[cpu]"``) with the ONNX model
   **UVR-MDX-NET-Inst_HQ_3** (~65 MB, downloaded once into the models folder).
2. ``demucs``           (``pip install demucs``) with the **mdx_extra_q** model.

Nothing here is ever fatal. A missing package, a model that cannot be downloaded
(offline first run), too little free RAM or any runtime error simply makes
:meth:`VocalIsolator.vocals` yield ``None`` and the caller carries on with the original
mix. Problems that would repeat on every track (no backend, model unavailable) switch
isolation off for the rest of the batch after one clear log line.

Disk hygiene: the stem is written as ``vocals.wav`` into a private temp directory that
is removed as soon as the caller leaves the ``with`` block - after it has decoded the
stem into memory - and again on :meth:`close`. Folders left behind by a crash are swept
on the next start.
"""

from __future__ import annotations

import gc
import importlib.util
import logging
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional

from .app_paths import models_dir

logger = logging.getLogger(__name__)

LogFn = Callable[[str, str, str], None]  # (tag, message, level)

TAG = "Audio: Isolate"
MODEL_FILENAME = "UVR-MDX-NET-Inst_HQ_3.onnx"
MODEL_LABEL = "UVR-MDX-NET-Inst_HQ_3"
DEMUCS_MODEL = "mdx_extra_q"
TEMP_PREFIX = "lyricist-vocals-"
STALE_AFTER_SECONDS = 6 * 3600
HEARTBEAT_SECONDS = 20.0
BASE_RAM_GB = 1.6            # model + runtime overhead
RAM_GB_PER_MINUTE = 0.08     # STFT buffers grow with the track length
MAX_TRACK_FAILURES = 3       # consecutive runtime failures before isolation is switched off

_OOM_HINTS = ("bad allocation", "out of memory", "cannot allocate", "memoryerror",
              "not enough memory", "std::bad_alloc")


class IsolationCancelled(Exception):
    """The user pressed Stop while a stem was being produced."""


class IsolationUnavailable(Exception):
    """No usable backend / model: don't try again this batch."""


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def available_backend() -> Optional[str]:
    """``"audio-separator"``, ``"demucs"`` or ``None`` (cheap: nothing is imported)."""
    if _has("audio_separator"):
        return "audio-separator"
    if _has("demucs"):
        return "demucs"
    return None


def isolation_preflight() -> tuple[bool, str]:
    """Verify the optional isolation stack before a batch starts.

    Returns ``(ready, reason)``.  The check is deliberately cheap and never raises;
    a missing FFmpeg or ``audioread`` is the usual cause of a mid-batch crash, so we
    refuse isolation up-front and let the caller use the original mix.
    """
    try:
        from .hardware_profiler import run_preflight

        report = run_preflight(refresh=True)
        return report.isolation_ready, report.isolation_reason()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("isolation pre-flight failed: %s", exc)
        return False, "the audio pre-flight check failed"


def backend_summary() -> str:
    """One human sentence for the Settings screen / tooltips."""
    backend = available_backend()
    if backend == "audio-separator":
        return f"Ready - audio-separator with {MODEL_LABEL} (ONNX, CPU)"
    if backend == "demucs":
        return f"Ready - demucs ({DEMUCS_MODEL})"
    return 'Not installed - run: pip install "audio-separator[cpu]"'


def free_ram_gb() -> Optional[float]:
    try:
        import psutil

        return psutil.virtual_memory().available / 2**30
    except Exception:
        return None


def required_ram_gb(duration: Optional[float]) -> float:
    minutes = (duration or 240.0) / 60.0
    return BASE_RAM_GB + RAM_GB_PER_MINUTE * minutes


def sweep_stale_temp_dirs(max_age: float = STALE_AFTER_SECONDS) -> int:
    """Delete ``lyricist-vocals-*`` folders a crashed run left behind. Returns the count."""
    removed = 0
    cutoff = time.time() - max_age
    try:
        for entry in Path(tempfile.gettempdir()).glob(f"{TEMP_PREFIX}*"):
            try:
                if entry.is_dir() and entry.stat().st_mtime < cutoff:
                    shutil.rmtree(entry, ignore_errors=True)
                    removed += 1
            except OSError:
                continue
    except OSError:
        pass
    return removed


class VocalIsolator:
    """Produces ``vocals.wav`` stems; owns every temp file it creates."""

    def __init__(self, log: Optional[LogFn] = None, cancel: Optional[threading.Event] = None,
                 *, models_root: Optional[Path] = None,
                 free_ram: Callable[[], Optional[float]] = free_ram_gb,
                 preflight: Callable[[], tuple[bool, str]] = isolation_preflight) -> None:
        self._log = log or (lambda tag, msg, level: None)
        self._cancel = cancel or threading.Event()
        self._models_root = models_root
        self._free_ram = free_ram
        self._preflight = preflight
        self._workdir: Optional[Path] = None
        self._separator = None            # audio_separator.Separator once loaded
        self._backend: Optional[str] = None
        self._disabled: Optional[str] = None
        self._failures = 0
        self._counter = 0
        self.last_seconds = 0.0
        self.model_label = MODEL_LABEL

    # ------------------------------------------------------------------- state
    @property
    def disabled_reason(self) -> Optional[str]:
        return self._disabled

    def _disable(self, reason: str) -> None:
        if self._disabled is None:
            self._disabled = reason
            self._log(TAG, f"Vocal isolation unavailable - {reason}. Continuing with the original "
                      "audio for the rest of this batch.", "warning")

    def _ensure_workdir(self) -> Path:
        if self._workdir is None or not self._workdir.is_dir():
            sweep_stale_temp_dirs()
            self._workdir = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX))
        return self._workdir

    # ------------------------------------------------------------------ public
    @contextmanager
    def vocals(self, source: Path, *, duration: Optional[float] = None) -> Iterator[Optional[Path]]:
        """Yield the path of a temporary ``vocals.wav`` (or ``None`` -> use the original mix).

        The file exists only inside the ``with`` block; decode it there, do not keep the path.
        """
        stem: Optional[Path] = None
        track_dir: Optional[Path] = None
        try:
            if self._disabled is None:
                track_dir, stem = self._produce(Path(source), duration)
            yield stem
        finally:
            if track_dir is not None:
                shutil.rmtree(track_dir, ignore_errors=True)
            self._wipe_output()

    def close(self) -> None:
        self._separator = None
        if self._workdir is not None:
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None
        gc.collect()

    # --------------------------------------------------------------- internals
    def _wipe_output(self) -> None:
        if self._workdir is None:
            return
        out = self._workdir / "out"
        if out.is_dir():
            for child in out.iterdir():
                try:
                    child.unlink() if child.is_file() else shutil.rmtree(child, ignore_errors=True)
                except OSError:
                    pass

    def _pick_backend(self) -> str:
        backend = available_backend()
        if backend is None:
            raise IsolationUnavailable('no backend installed - run: pip install "audio-separator[cpu]"')
        return backend

    def _check_ram(self, duration: Optional[float]) -> Optional[str]:
        free = self._free_ram()
        if free is None:
            return None
        need = required_ram_gb(duration)
        if free < need:
            return f"only {free:.1f} GB RAM free (need about {need:.1f} GB)"
        return None

    def _produce(self, source: Path, duration: Optional[float]) -> tuple[Optional[Path], Optional[Path]]:
        """Run the separation. Returns ``(track_dir, stem)``; ``(None, None)`` = fall back."""
        if self._cancel.is_set():
            raise IsolationCancelled()
        if self._disabled is None:
            try:
                ready, reason = self._preflight()
            except Exception as exc:  # a broken probe must not crash the batch
                ready, reason = False, f"the audio pre-flight check failed ({exc.__class__.__name__})"
            if not ready:
                self._disable(reason or "required audio components are missing")
                return None, None
        try:
            self._backend = self._backend or self._pick_backend()
        except IsolationUnavailable as exc:
            self._disable(str(exc))
            return None, None

        shortage = self._check_ram(duration)
        if shortage:
            self._log(TAG, f"Skipping vocal isolation for this track: {shortage}.", "warning")
            return None, None

        workdir = self._ensure_workdir()
        self._counter += 1
        track_dir = workdir / f"t{self._counter}"
        track_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        stop = threading.Event()
        beat = threading.Thread(target=self._heartbeat, args=(stop, started), name="lsl-isolate-beat",
                                daemon=True)
        detail = (f"{MODEL_LABEL} · ONNX/CPU" if self._backend == "audio-separator"
                  else f"demucs {DEMUCS_MODEL} · CPU")
        self._log(TAG, f"Isolating vocals ({detail}) - this can take a few minutes on CPU…", "info")
        beat.start()
        try:
            if self._backend == "audio-separator":
                produced = self._run_audio_separator(source, workdir)
            else:
                produced = self._run_demucs(source, workdir)
            if self._cancel.is_set():
                raise IsolationCancelled()
            stem = track_dir / "vocals.wav"
            shutil.move(str(produced), str(stem))
        except IsolationCancelled:
            shutil.rmtree(track_dir, ignore_errors=True)
            raise
        except IsolationUnavailable as exc:
            shutil.rmtree(track_dir, ignore_errors=True)
            self._disable(str(exc))
            return None, None
        except BaseException as exc:                      # noqa: BLE001 - incl. SystemExit from demucs
            shutil.rmtree(track_dir, ignore_errors=True)
            if isinstance(exc, KeyboardInterrupt):
                raise
            self._on_failure(exc)
            return None, None
        finally:
            stop.set()
            beat.join(timeout=1.0)
        self._failures = 0
        self.last_seconds = time.monotonic() - started
        self._log(TAG, f"Vocals isolated in {self.last_seconds:.0f}s", "info")
        return track_dir, stem

    def _on_failure(self, exc: BaseException) -> None:
        text = f"{exc.__class__.__name__}: {exc}".replace("\n", " ")
        short = text if len(text) <= 160 else text[:159] + "…"
        low_ram = isinstance(exc, MemoryError) or any(h in text.lower() for h in _OOM_HINTS)
        self._failures += 1
        if low_ram:
            self._log(TAG, f"Not enough memory for vocal isolation ({short}); using the original "
                      "audio for this track.", "warning")
        else:
            self._log(TAG, f"Vocal isolation failed ({short}); using the original audio for this "
                      "track.", "warning")
        logger.debug("isolation failure", exc_info=exc)
        if self._failures >= MAX_TRACK_FAILURES:
            self._disable(f"it failed {self._failures} tracks in a row")

    def _heartbeat(self, stop: threading.Event, started: float) -> None:
        while not stop.wait(HEARTBEAT_SECONDS):
            self._log(TAG, f"Isolating vocals… {time.monotonic() - started:.0f}s elapsed", "info")

    # ---------------------------------------------------------------- backends
    def _run_audio_separator(self, source: Path, workdir: Path) -> Path:
        out = workdir / "out"
        out.mkdir(parents=True, exist_ok=True)
        if self._separator is None:
            try:
                from audio_separator.separator import Separator  # type: ignore
            except Exception as exc:
                raise IsolationUnavailable(f"audio-separator could not be imported ({exc})") from exc
            root = self._models_root or (models_dir() / "separator")
            root.mkdir(parents=True, exist_ok=True)
            try:
                separator = Separator(
                    output_dir=str(out), output_format="WAV", output_single_stem="Vocals",
                    model_file_dir=str(root), log_level=logging.WARNING,
                    # Gentle settings for modest machines: small segments, no batching.
                    mdx_params={"hop_length": 1024, "segment_size": 256, "overlap": 0.25,
                                "batch_size": 1, "enable_denoise": False},
                )
                separator.load_model(model_filename=MODEL_FILENAME)   # downloads on first use
            except Exception as exc:
                raise IsolationUnavailable(
                    f"the model {MODEL_LABEL} could not be loaded - is this the first run and "
                    f"are you offline? ({exc.__class__.__name__}: {exc})") from exc
            self._separator = separator
        names = self._separator.separate(str(source))
        return self._find_vocals(out, names)

    @staticmethod
    def _find_vocals(out: Path, names) -> Path:
        candidates: list[Path] = []
        for name in names or []:
            p = Path(str(name))
            candidates.append(p if p.is_absolute() and p.is_file() else out / p.name)
        candidates += sorted(out.glob("*.wav"))
        existing = [c for c in candidates if c.is_file()]
        for c in existing:
            if "(vocals)" in c.name.lower():
                return c
        if existing:
            return existing[0]        # single-stem mode produced exactly one file
        raise RuntimeError("the separator produced no output file")

    def _run_demucs(self, source: Path, workdir: Path) -> Path:
        out = workdir / "out"
        out.mkdir(parents=True, exist_ok=True)
        try:
            from demucs.separate import main as demucs_main  # type: ignore
        except Exception as exc:
            raise IsolationUnavailable(f"demucs could not be imported ({exc})") from exc
        try:
            demucs_main(["-n", DEMUCS_MODEL, "--two-stems", "vocals", "--device", "cpu",
                         "--jobs", "0", "-o", str(out), str(source)])
        except SystemExit as exc:                          # demucs reports errors via sys.exit
            raise RuntimeError(f"demucs exited: {exc}") from exc
        hits = sorted(out.rglob("vocals.wav"))
        if not hits:
            raise RuntimeError("demucs produced no vocals.wav")
        return hits[0]
