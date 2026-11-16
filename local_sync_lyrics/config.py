"""User settings: a typed dataclass persisted as JSON.

Resolution order for every value
--------------------------------
1. the value saved in ``settings.json`` (if present and non-empty)
2. a matching environment variable (``os.environ``)
3. the built-in default

Environment fallbacks
---------------------
``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` -> hf_token
``LSL_WHISPER_MODEL``                     -> whisper_model
``LSL_LANGUAGE``                          -> language

The Hugging Face token is injected into ``os.environ`` at runtime by
:meth:`Settings.apply_environment` so ``huggingface_hub`` picks it up (needed for
rate-limit-free / gated model downloads). It is never logged.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .app_paths import settings_path

logger = logging.getLogger(__name__)

PROVIDER_IDS = ("lrclib", "betterlyrics", "megalobiz")
WHISPER_MODELS = (
    "auto", "tiny", "base", "small", "medium", "large-v3-turbo", "large-v3",
)

# value in file  ->  environment variables consulted when the file has no value
_ENV_FALLBACKS: dict[str, tuple[str, ...]] = {
    "hf_token": ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
    "whisper_model": ("LSL_WHISPER_MODEL",),
    "language": ("LSL_LANGUAGE",),
}


@dataclass
class Settings:
    # --- AI transcription -------------------------------------------------
    whisper_model: str = "auto"          # "auto" = pick from hardware profile
    language: str = "auto"               # "auto" = let Whisper detect
    hf_token: str = ""                   # optional Hugging Face token
    use_ai_fallback: bool = True
    prefer_gpu: bool = True
    vad_filter: bool = False             # skip long instrumental passages
    max_audio_minutes: int = 20          # refuse to transcribe longer tracks

    # --- Lyrics search ----------------------------------------------------
    enabled_providers: list[str] = field(default_factory=lambda: list(PROVIDER_IDS))
    request_timeout: float = 12.0
    min_match_score: int = 72            # 0-100 fuzzy score for accepting a hit

    # --- Batch behaviour --------------------------------------------------
    recursive: bool = True
    overwrite_existing: bool = False     # replace existing .lrc files
    retry_failed: bool = False           # retry tracks cached as not found/failed
    save_plain_fallback: bool = False    # write unsynced lyrics to .txt if nothing synced

    # --- UI ---------------------------------------------------------------
    last_folder: str = ""
    theme_mode: str = "dark"             # dark | light | system

    # not persisted: which fields were filled from the environment
    _env_fields: set[str] = field(default_factory=set, repr=False, compare=False)

    # ------------------------------------------------------------------ helpers
    def effective_hf_token(self) -> str:
        if self.hf_token.strip():
            return self.hf_token.strip()
        for name in _ENV_FALLBACKS["hf_token"]:
            value = os.environ.get(name, "").strip()
            if value:
                return value
        return ""

    def token_from_environment(self) -> bool:
        return not self.hf_token.strip() and bool(self.effective_hf_token())

    def apply_environment(self) -> None:
        """Inject runtime environment variables (HF token, quiet hub output)."""
        token = self.effective_hf_token()
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    def copy(self) -> "Settings":
        data = self.to_dict()
        clone = Settings.from_dict(data)
        clone._env_fields = set(self._env_fields)
        return clone

    # ------------------------------------------------------------ (de)serialise
    def to_dict(self) -> dict[str, Any]:
        data = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        return data

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Settings":
        """Build Settings from arbitrary JSON, coercing/validating every field."""
        settings = cls()
        for f in fields(cls):
            if f.name.startswith("_") or f.name not in raw:
                continue
            default = (
                f.default if f.default is not MISSING else f.default_factory()  # type: ignore[misc]
            )
            setattr(settings, f.name, _coerce(raw[f.name], default))
        settings._normalise()
        return settings

    def _normalise(self) -> None:
        """Clamp values to sane ranges / allowed sets."""
        self.whisper_model = (self.whisper_model or "auto").strip() or "auto"
        self.language = (self.language or "auto").strip().lower() or "auto"
        self.enabled_providers = [
            p for p in self.enabled_providers if p in PROVIDER_IDS
        ] or list(PROVIDER_IDS)
        self.request_timeout = min(max(float(self.request_timeout), 3.0), 60.0)
        self.min_match_score = min(max(int(self.min_match_score), 40), 100)
        self.max_audio_minutes = min(max(int(self.max_audio_minutes), 1), 240)
        if self.theme_mode not in ("dark", "light", "system"):
            self.theme_mode = "dark"


def _coerce(value: Any, default: Any) -> Any:
    """Coerce *value* to the type of *default*; fall back to default on failure."""
    try:
        if isinstance(default, bool):
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if isinstance(default, int):
            return int(value)
        if isinstance(default, float):
            return float(value)
        if isinstance(default, str):
            return "" if value is None else str(value)
        if isinstance(default, list):
            if isinstance(value, (list, tuple)):
                return [str(v) for v in value]
            return list(default)
    except (TypeError, ValueError):
        pass
    return default


class SettingsStore:
    """Thread-safe loader/saver for :class:`Settings`."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else settings_path()
        self._lock = threading.Lock()

    def exists(self) -> bool:
        return self.path.is_file()

    # ---------------------------------------------------------------- loading
    def load(self) -> Settings:
        raw: dict[str, Any] = {}
        with self._lock:
            if self.path.is_file():
                try:
                    parsed = json.loads(self.path.read_text(encoding="utf-8-sig"))
                    if isinstance(parsed, dict):
                        raw = parsed
                    else:
                        raise ValueError("settings root must be an object")
                except (OSError, ValueError) as exc:
                    self._quarantine(exc)
                    raw = {}
        settings = Settings.from_dict(raw)

        # Environment fallbacks for values the file does not provide.
        for name, env_names in _ENV_FALLBACKS.items():
            if name == "hf_token":  # handled lazily by effective_hf_token()
                continue
            if raw.get(name):
                continue
            for env_name in env_names:
                value = os.environ.get(env_name, "").strip()
                if value:
                    setattr(settings, name, value)
                    settings._env_fields.add(name)
                    break
        settings._normalise()
        settings.apply_environment()
        return settings

    def _quarantine(self, exc: Exception) -> None:
        """Move an unreadable settings file aside so we never crash on it."""
        logger.warning("Settings file unreadable (%s); using defaults", exc)
        try:
            backup = self.path.with_suffix(f".corrupt-{int(time.time())}.json")
            self.path.replace(backup)
        except OSError:
            pass

    # ----------------------------------------------------------------- saving
    def save(self, settings: Settings) -> bool:
        """Atomically write settings. Returns False (never raises) on failure."""
        data = settings.to_dict()
        # Do not copy an environment-provided model/language into the file
        # unless the user changed it.
        for name in settings._env_fields:
            env_value = next(
                (os.environ.get(e, "").strip() for e in _ENV_FALLBACKS[name]
                 if os.environ.get(e, "").strip()),
                "",
            )
            if data.get(name) == env_value:
                data[name] = Settings.__dataclass_fields__[name].default
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp_name = tempfile.mkstemp(
                    prefix=".settings-", suffix=".tmp", dir=str(self.path.parent)
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as handle:
                        json.dump(data, handle, indent=2, ensure_ascii=False)
                    os.replace(tmp_name, self.path)
                finally:
                    if os.path.exists(tmp_name):
                        os.unlink(tmp_name)
                # The file may contain an API token: keep it private (POSIX).
                try:
                    os.chmod(self.path, 0o600)
                except OSError:
                    pass
                return True
            except OSError as exc:
                logger.error("Could not save settings: %s", exc)
                return False
