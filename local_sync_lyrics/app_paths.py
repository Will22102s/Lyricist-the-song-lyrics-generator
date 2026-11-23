"""Cross-platform locations for settings, cache, models and logs.

Layout
------
Windows : %APPDATA%/Lyricist          (settings.json)
          %LOCALAPPDATA%/Lyricist     (cache.db, models/, logs/)
Linux   : $XDG_CONFIG_HOME or ~/.config/Lyricist   (settings.json)
          $XDG_DATA_HOME   or ~/.local/share/Lyricist (cache.db, models/, logs/)
macOS   : ~/Library/Application Support/Lyricist   (everything)

Set ``LYRICIST_HOME`` (or the older ``LSL_HOME``) to force *everything* into one folder
(portable mode).

The app used to be called "Local Sync Lyrics" and stored its data in ``LocalSyncLyrics``.
On the first start of the renamed app that folder is moved to ``Lyricist`` (a plain
rename on the same drive - instant, even with gigabytes of models), so settings, cache
and downloaded models carry over.

Bundled read-only resources (``assets/logo.*``) live next to the project in a source
checkout and inside PyInstaller's data folder (``sys._MEIPASS``) in a frozen build -
see :func:`assets_dir`.

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

APP_NAME = "Lyricist"
LEGACY_APP_NAMES = ("LocalSyncLyrics",)
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
    return _env_path("LYRICIST_HOME") or _env_path("LSL_HOME")


def _app_folder(base: Path) -> Path:
    """``base/Lyricist``, first adopting a folder left by the old app name."""
    target = base / APP_NAME
    if not target.exists():
        for legacy in LEGACY_APP_NAMES:
            old = base / legacy
            if old.is_dir():
                try:
                    old.rename(target)
                    logger.info("Moved %s to %s", old, target)
                except OSError as exc:      # locked / other drive: start fresh, keep the old copy
                    logger.warning("Could not migrate %s (%s)", old, exc)
                break
    return target


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
    return _ensure(_app_folder(base))


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
    return _ensure(_app_folder(base))


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


# ------------------------------------------------------------ bundled resources
def resource_dir() -> Path:
    """Root that holds bundled, read-only files (``assets/``)."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent.parent


def assets_dir() -> Path:
    """Folder with the logo / icons (``<project>/assets``; bundled via ``--add-data``)."""
    return resource_dir() / "assets"


def asset_path(name: str) -> Path | None:
    """Absolute path of a bundled asset, or ``None`` if it is not there."""
    path = assets_dir() / name
    return path if path.is_file() else None
