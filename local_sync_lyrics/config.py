"""User settings: a typed dataclass persisted as JSON.

Resolution order for every value
--------------------------------
1. the value saved in ``settings.json`` (if present and non-empty)
2. a matching environment variable (``os.environ``)
3. the built-in default

Environment fallbacks
---------------------
``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN``   -> hf_token
``GENIUS_ACCESS_TOKEN`` / ``GENIUS_TOKEN``  -> genius_token (official Genius API)
``MUSIXMATCH_API_KEY``                      -> musixmatch_key
``LYRICIST_WHISPER_MODEL`` / ``LSL_WHISPER_MODEL`` -> whisper_model
``LYRICIST_LANGUAGE`` / ``LSL_LANGUAGE``            -> language

Appearance
----------
``theme_mode`` (dark | light | system), ``accent_preset`` (one of :data:`ACCENT_PRESETS`
or ``custom``) and ``custom_accent`` (``#RRGGBB``) are plain settings; the GUI saves
them the moment the user changes them. :meth:`Settings.accent_color` resolves the
colour actually in use. ``settings_version`` is deliberately NOT bumped: files written
before the appearance options simply fall back to the defaults.

Audio pipeline & output (Phase 2)
---------------------------------
``isolate_vocals`` (default off - it costs minutes per song), ``detect_language``
(default on) and ``output_mode`` (``lrc`` | ``embed`` | ``both``). All three are plain
settings; the GUI saves them the moment they are toggled.

Secrets (the three above) are resolved lazily by ``effective_*()`` and are never
copied from the environment into the settings file. The Hugging Face token is
injected into ``os.environ`` at runtime by :meth:`Settings.apply_environment` so
``huggingface_hub`` picks it up. No secret is ever logged.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .app_paths import settings_path
from .theme import DEFAULT_ACCENT, normalise_hex

logger = logging.getLogger(__name__)

# The single source of truth for the application version.  ``local_sync_lyrics``
# re-exports it, ``build.py`` stamps it into the bundle and the updater compares
# GitHub release tags against it.
__version__ = "2.3.2"

# --- updates & support ---------------------------------------------------------------
# Change these two constants (or the setting in Settings) to point the in-app
# updater at your own fork / release feed.
DEFAULT_UPDATE_REPO = "Will22102s/Lyricist"
GITHUB_RELEASES_URL = f"https://github.com/{DEFAULT_UPDATE_REPO}/releases"
SUPPORT_LINKS: tuple[tuple[str, str], ...] = (
    ("GitHub Sponsors", "https://github.com/sponsors/Will22102s"),
    ("Ko-fi", "https://ko-fi.com/lyricist"),
    ("Buy Me a Coffee", "https://www.buymeacoffee.com/lyricist"),
)
CREDITS = (
    ("faster-whisper", "local speech-to-text and forced alignment"),
    ("CTranslate2", "fast CPU / GPU Whisper inference"),
    ("PySide6 (Qt 6)", "native desktop interface"),
    ("mutagen", "audio tags, artwork and embedded lyrics"),
    ("MusicBrainz + Cover Art Archive", "canonical tags and cover art"),
    ("LRCLIB & the other lyric providers", "synced and plain lyrics"),
)

# Search order = tier order. Tier 1 returns ready-made synced .lrc files; Tier 2 returns
# plain official text that must be timestamped by forced alignment.
TIER1_IDS = ("lrclib", "betterlyrics", "rclyricsband", "megalobiz", "netease", "qqmusic")
TIER2_IDS = ("genius", "lyricsovh", "musixmatch")
PROVIDER_IDS = TIER1_IDS + TIER2_IDS
_LEGACY_PROVIDER_IDS = ("lrclib", "betterlyrics", "megalobiz")   # settings_version 1
SETTINGS_VERSION = 2
WHISPER_MODELS = (
    "auto", "tiny", "base", "small", "medium", "large-v3-turbo", "large-v3",
)

# --- custom providers ----------------------------------------------------------------
CUSTOM_PROVIDER_TYPES = ("lrc", "json")
MAX_CUSTOM_PROVIDERS = 50


@dataclass
class CustomProvider:
    """A user-defined lyrics endpoint (Phase 3).

    ``url_template`` may contain ``{artist}``, ``{title}`` and ``{album}``; the values are
    URL-quoted before substitution. ``response_type`` is ``lrc`` (the response body *is*
    an LRC document) or ``json`` (extract the lyrics with ``json_lyrics_path``, a
    dot-separated path such as ``data.lyrics``).
    """

    name: str = ""
    enabled: bool = True
    url_template: str = ""
    response_type: str = "lrc"
    json_lyrics_path: str = ""

    @property
    def slug(self) -> str:
        return f"custom:{self.name.strip().casefold()}"

    def valid(self) -> bool:
        """True when the entry could actually be queried."""
        url = self.url_template.strip()
        return bool(self.name.strip()) and url.startswith(("http://", "https://")) and (
            "{title}" in url or "{artist}" in url)

    def normalised(self) -> "CustomProvider":
        return CustomProvider(
            name=self.name.strip()[:60],
            enabled=bool(self.enabled),
            url_template=self.url_template.strip(),
            response_type=(self.response_type or "lrc").strip().lower()
            if (self.response_type or "").strip().lower() in CUSTOM_PROVIDER_TYPES else "lrc",
            json_lyrics_path=self.json_lyrics_path.strip(),
        )

    @classmethod
    def from_dict(cls, raw: Any) -> "CustomProvider":
        if not isinstance(raw, dict):
            return cls()
        return cls(
            name=str(raw.get("name", "") or ""),
            enabled=bool(raw.get("enabled", True)),
            url_template=str(raw.get("url_template", "") or ""),
            response_type=str(raw.get("response_type", "lrc") or "lrc"),
            json_lyrics_path=str(raw.get("json_lyrics_path", "") or ""),
        ).normalised()

    def to_dict(self) -> dict[str, Any]:
        provider = self.normalised()
        return {
            "name": provider.name,
            "enabled": provider.enabled,
            "url_template": provider.url_template,
            "response_type": provider.response_type,
            "json_lyrics_path": provider.json_lyrics_path,
        }


# --- tagging --------------------------------------------------------------------------
TAG_MODES = ("fill_missing", "overwrite_all")
TAG_MODE_LABELS = {
    "fill_missing": "Fill Missing",
    "overwrite_all": "Overwrite",
}
DEFAULT_TAG_MODE = "fill_missing"


# --- output ---------------------------------------------------------------------------
OUTPUT_MODES = ("lrc", "embed", "both")
OUTPUT_MODE_LABELS = {
    "lrc": "Save Separate .lrc File",
    "embed": "Embed Into Audio File",
    "both": "Both (.lrc + Embedded)",
}
DEFAULT_OUTPUT_MODE = "lrc"
# "Your Own Lyrics": below this audio-vs-text similarity the user is asked before syncing.
CUSTOM_LYRICS_MIN_MATCH = 0.35


# --- appearance ----------------------------------------------------------------------
THEME_MODES = ("dark", "light", "system")
THEME_MODE_LABELS = {
    "dark": "Dark (Liquid Obsidian)",
    "light": "Light (Frosted Pearl)",
    "system": "System",
}
CUSTOM_ACCENT_ID = "custom"


@dataclass(frozen=True)
class AccentPreset:
    id: str
    label: str
    color: str
    caption: str


ACCENT_PRESETS: tuple[AccentPreset, ...] = (
    AccentPreset("apple_blue", "Apple Blue", "#0A84FF", "Clear, calm, system-native"),
    AccentPreset("spotify_green", "Spotify Green", "#1DB954", "Deep green glass"),
    AccentPreset("apple_music", "Apple Music Pink/Red", "#FA243C", "Warm rose-red glow"),
    AccentPreset("midnight_purple", "Midnight Purple", "#BF5AF2", "Violet after dark"),
    AccentPreset("sunset_amber", "Sunset Amber", "#FF9F0A", "Golden-hour amber"),
)
ACCENT_PRESET_IDS = tuple(p.id for p in ACCENT_PRESETS)
DEFAULT_ACCENT_PRESET = ACCENT_PRESETS[0].id


def find_accent_preset(preset_id: str) -> AccentPreset | None:
    return next((p for p in ACCENT_PRESETS if p.id == preset_id), None)


# value in file  ->  environment variables consulted when the file has no value
_ENV_FALLBACKS: dict[str, tuple[str, ...]] = {
    "hf_token": ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
    "genius_token": ("GENIUS_ACCESS_TOKEN", "GENIUS_TOKEN"),
    "musixmatch_key": ("MUSIXMATCH_API_KEY",),
    "whisper_model": ("LYRICIST_WHISPER_MODEL", "LSL_WHISPER_MODEL"),
    "language": ("LYRICIST_LANGUAGE", "LSL_LANGUAGE"),
}
_SECRET_FIELDS = ("hf_token", "genius_token", "musixmatch_key")


@dataclass
class Settings:
    # --- AI transcription -------------------------------------------------
    whisper_model: str = "auto"          # "auto" = pick from hardware profile
    language: str = "auto"               # "auto" = let Whisper detect
    hf_token: str = ""                   # optional Hugging Face token
    use_ai_fallback: bool = True         # master switch for everything that needs Whisper
    prefer_gpu: bool = True
    vad_filter: bool = False             # blind transcription: skip instrumental passages
    max_audio_minutes: int = 20          # refuse to process longer tracks

    # --- Smart search-first pipeline --------------------------------------
    use_forced_alignment: bool = True    # snap official plain lyrics onto the audio
    detect_mashups: bool = True          # split "A x B" / merged tracks into songs
    identify_by_audio: bool = True       # name unknown songs from sung snippets
    min_alignment_match: int = 40        # % of sung words the lyrics must explain
    detect_language: bool = True         # lock Whisper to the language detected on a 30 s sample
    isolate_vocals: bool = False         # strip the band first (slow; for dense / noisy mixes)

    # --- Lyrics search ----------------------------------------------------
    enabled_providers: list[str] = field(default_factory=lambda: list(PROVIDER_IDS))
    genius_token: str = ""               # optional: official Genius API (else public search)
    musixmatch_key: str = ""             # optional: enables Musixmatch (free tier = partial text)
    request_timeout: float = 12.0
    min_match_score: int = 72            # 0-100 fuzzy score for accepting a hit

    # --- Batch behaviour --------------------------------------------------
    recursive: bool = True
    overwrite_existing: bool = False     # replace existing .lrc files
    retry_failed: bool = False           # retry tracks cached as not found/failed
    save_plain_fallback: bool = False    # write unsynced lyrics to .txt if nothing synced
    output_mode: str = DEFAULT_OUTPUT_MODE   # lrc | embed | both

    # --- Phase 3: custom providers + auto-tagging --------------------------
    custom_providers: list[CustomProvider] = field(default_factory=list)
    auto_tag: bool = False               # look up MusicBrainz/iTunes and write tags + cover
    tag_mode: str = DEFAULT_TAG_MODE     # fill_missing | overwrite_all

    # --- UI / bookkeeping --------------------------------------------------
    last_folder: str = ""
    theme_mode: str = "dark"             # dark | light | system
    accent_preset: str = DEFAULT_ACCENT_PRESET   # a preset id or "custom"
    custom_accent: str = DEFAULT_ACCENT          # #RRGGBB, used when accent_preset == "custom"

    # --- Phase 4: updates & support ---------------------------------------
    auto_check_updates: bool = True
    update_repo: str = DEFAULT_UPDATE_REPO       # owner/name on GitHub
    last_update_check: float = 0.0               # unix time of the last successful check

    settings_version: int = SETTINGS_VERSION

    # not persisted: which fields were filled from the environment
    _env_fields: set[str] = field(default_factory=set, repr=False, compare=False)

    # ------------------------------------------------------------------ secrets
    def _secret(self, name: str) -> str:
        value = str(getattr(self, name, "") or "").strip()
        if value:
            return value
        for env_name in _ENV_FALLBACKS.get(name, ()):
            env_value = os.environ.get(env_name, "").strip()
            if env_value:
                return env_value
        return ""

    def effective_hf_token(self) -> str:
        return self._secret("hf_token")

    def effective_genius_token(self) -> str:
        return self._secret("genius_token")

    def effective_musixmatch_key(self) -> str:
        return self._secret("musixmatch_key")

    def token_from_environment(self) -> bool:
        return not self.hf_token.strip() and bool(self.effective_hf_token())

    @property
    def wants_lrc_file(self) -> bool:
        return self.output_mode in ("lrc", "both")

    @property
    def wants_embedded(self) -> bool:
        return self.output_mode in ("embed", "both")

    def accent_color(self) -> str:
        """The accent colour in use (``#RRGGBB``): a preset's colour, or the custom one."""
        preset = find_accent_preset(self.accent_preset)
        if preset is not None:
            return preset.color
        return normalise_hex(self.custom_accent, DEFAULT_ACCENT)

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
        clone = Settings.from_dict(self.to_dict())
        clone._env_fields = set(self._env_fields)
        return clone

    # ------------------------------------------------------------ (de)serialise
    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Settings":
        """Build Settings from arbitrary JSON, coercing/validating every field."""
        settings = cls()
        for f in fields(cls):
            if f.name.startswith("_") or f.name not in raw:
                continue
            if f.name == "custom_providers":
                entries = raw[f.name] if isinstance(raw[f.name], (list, tuple)) else []
                settings.custom_providers = [
                    CustomProvider.from_dict(entry) for entry in entries
                    if isinstance(entry, dict)
                ]
                continue
            default = (
                f.default if f.default is not MISSING else f.default_factory()  # type: ignore[misc]
            )
            setattr(settings, f.name, _coerce(raw[f.name], default))

        # Upgrade files written before the multi-provider pipeline: they list only the
        # original three providers, which would silently exclude every newer one.
        if _coerce(raw.get("settings_version", 1), 1) < SETTINGS_VERSION:
            chosen = {p for p in settings.enabled_providers if p in PROVIDER_IDS}
            if "enabled_providers" in raw and chosen:
                settings.enabled_providers = [
                    pid for pid in PROVIDER_IDS
                    if pid in chosen or pid not in _LEGACY_PROVIDER_IDS
                ]
            else:
                settings.enabled_providers = list(PROVIDER_IDS)
        settings._normalise()
        return settings

    def _normalise(self) -> None:
        """Clamp values to sane ranges / allowed sets."""
        self.whisper_model = (self.whisper_model or "auto").strip() or "auto"
        self.language = (self.language or "auto").strip().lower() or "auto"
        self.enabled_providers = [
            p for p in PROVIDER_IDS if p in self.enabled_providers   # canonical order
        ] or list(PROVIDER_IDS)
        self.request_timeout = min(max(float(self.request_timeout), 3.0), 60.0)
        self.min_match_score = min(max(int(self.min_match_score), 40), 100)
        self.max_audio_minutes = min(max(int(self.max_audio_minutes), 1), 240)
        self.min_alignment_match = min(max(int(self.min_alignment_match), 20), 90)
        self.genius_token = self.genius_token.strip()
        self.musixmatch_key = self.musixmatch_key.strip()
        if self.output_mode not in OUTPUT_MODES:
            self.output_mode = DEFAULT_OUTPUT_MODE
        if self.tag_mode not in TAG_MODES:
            self.tag_mode = DEFAULT_TAG_MODE
        providers: list[CustomProvider] = []
        for entry in self.custom_providers or []:
            provider = entry if isinstance(entry, CustomProvider) else CustomProvider.from_dict(entry)
            if provider.valid():
                providers.append(provider.normalised())
        self.custom_providers = providers[:MAX_CUSTOM_PROVIDERS]
        if self.theme_mode not in THEME_MODES:
            self.theme_mode = "dark"
        self.custom_accent = normalise_hex(self.custom_accent, DEFAULT_ACCENT)
        if self.accent_preset not in ACCENT_PRESET_IDS and self.accent_preset != CUSTOM_ACCENT_ID:
            self.accent_preset = DEFAULT_ACCENT_PRESET
        repo = (self.update_repo or DEFAULT_UPDATE_REPO).strip().strip("/")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            repo = DEFAULT_UPDATE_REPO
        self.update_repo = repo
        try:
            self.last_update_check = max(0.0, float(self.last_update_check or 0.0))
        except (TypeError, ValueError):
            self.last_update_check = 0.0
        self.settings_version = SETTINGS_VERSION


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

        # Environment fallbacks for plain values the file does not provide.
        for name, env_names in _ENV_FALLBACKS.items():
            if name in _SECRET_FIELDS:
                continue  # resolved lazily by effective_*(); never copied to disk
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
                # The file may contain API tokens: keep it private (POSIX).
                try:
                    os.chmod(self.path, 0o600)
                except OSError:
                    pass
                return True
            except OSError as exc:
                logger.error("Could not save settings: %s", exc)
                return False
