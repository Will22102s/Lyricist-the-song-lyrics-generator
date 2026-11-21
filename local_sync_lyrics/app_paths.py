"""Cross-platform locations for settings, cache, models and logs.

Layout
------
Windows : %APPDATA%/LocalSyncLyrics          (settings.json)
          %LOCALAPPDATA%/LocalSyncLyrics     (cache.db, models/, logs/)
Linux   : $XDG_CONFIG_HOME or ~/.config/LocalSyncLyrics   (settings.json)
          $XDG_DATA_HOME   or ~/.local/share/LocalSyncLyrics (cache.db, models/, logs/)
macOS   : ~/Library/Application Support/LocalSyncLyrics   (everything)

Set ``LSL_HOME`` to force *everything* into one folder (portable mode).

Every helper is defensive: if the preferred directory cannot be created
(read-only home, locked-down profile, ...) we fall back to a folder inside the
system temp directory so the app can still start instead of crashing.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

APP_NAME = "LocalSyncLyrics"
logger = logging.getLogger(__name__)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


def _ensure(path: Path) -> Path:
    """Create *path* (and parents). Fall back to a temp dir if that fails."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        # Make sure we can actually write there.
        probe = path / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return path
    except OSError as exc:
        fallback = Path(tempfile.gettempdir()) / APP_NAME / path.name
        logger.warning("Cannot use %s (%s); falling back to %s", path, exc, fallback)
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _portable_home() -> Path | None:
    return _env_path("LSL_HOME")


def config_dir() -> Path:
    home = _portable_home()
    if home:
        return _ensure(home)
    if sys.platform == "win32":
        base = _env_path("APPDATA") or Path.home() / "AppData" / "Roaming"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = _env_path("XDG_CONFIG_HOME") or Path.home() / ".config"
    return _ensure(base / APP_NAME)


def data_dir() -> Path:
    home = _portable_home()
    if home:
        return _ensure(home)
    if sys.platform == "win32":
        base = _env_path("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = _env_path("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return _ensure(base / APP_NAME)


def settings_path() -> Path:
    return config_dir() / "settings.json"


def database_path() -> Path:
    return data_dir() / "cache.db"


def models_dir() -> Path:
    return _ensure(data_dir() / "models")


def logs_dir() -> Path:
    return _ensure(data_dir() / "logs")


def is_frozen() -> bool:
    """True when running from a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False))
