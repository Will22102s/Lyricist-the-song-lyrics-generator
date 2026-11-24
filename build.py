"""Build a standalone desktop app with PyInstaller.

    pip install -r requirements-dev.txt
    python build.py                  # one-folder build (instant start, recommended)
    python build.py --onefile        # single executable (slower first launch)
    python build.py --clean          # wipe PyInstaller's cache first
    python build.py --icons          # force-regenerate the icons first
    python build.py --dry-run        # print the PyInstaller command and exit

Under the hood a normal build is the same as::

    pyinstaller --noconfirm --clean --windowed --onedir --name "Lyricist" \\
                --icon assets/icon.ico --add-data "assets:assets" \\
                --collect-all faster_whisper --collect-all ctranslate2 \\
                --hidden-import mutagen --hidden-import audioread \\
                --exclude-module PySide6.QtWebEngineCore \\
                --collect-submodules local_sync_lyrics main.py

``--icon`` stamps the executable (Explorer, taskbar, title bar); ``--add-data`` ships
the ``assets/`` folder so the running app can find its logo and window icon. Output
goes to ``dist/Lyricist/`` (onedir) or ``dist/Lyricist`` (onefile). Build on the OS
you are targeting (Windows -> .exe, Linux -> ELF binary); PyInstaller cannot
cross-compile.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path

from make_icons import ASSETS, ensure_assets

ROOT = Path(__file__).resolve().parent
APP_NAME = "Lyricist"

# Packages that ship native libraries or data files PyInstaller cannot discover
# by static analysis (CTranslate2 DLLs/.so, Silero VAD .onnx, FFmpeg inside PyAV,
# Qt's platform plugins are handled by PyInstaller's own PySide6 hooks).
COLLECT_ALL = ("faster_whisper", "ctranslate2", "onnxruntime", "av")
# Optional extras are bundled only when they are installed in the build environment.
# (audio_separator drags in PyTorch - expect a much larger package when it is present.)
OPTIONAL_COLLECT = ("audio_separator", "langdetect", "soundfile", "librosa")

# Modules imported dynamically / lazily by the app or its dependencies.
HIDDEN_IMPORTS = (
    "mutagen",
    "mutagen.id3",
    "mutagen.mp3",
    "mutagen.mp4",
    "mutagen.flac",
    "mutagen.oggvorbis",
    "mutagen.asf",
    "audioread",
    "PySide6",
)

# Qt modules this app never uses. Excluding them keeps the bundle ~150-300 MB smaller
# and avoids shipping the Chromium (WebEngine) payload.
EXCLUDES = (
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtWebChannel",
    "PySide6.QtWebSockets",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic",
    "PySide6.Qt3DAnimation",
    "PySide6.Qt3DExtras",
    "PySide6.QtQuick",
    "PySide6.QtQuickWidgets",
    "PySide6.QtQml",
    "PySide6.QtQuickControls2",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtDesigner",
    "PySide6.QtTest",
)


def app_version() -> str:
    """Read the single source of truth: ``local_sync_lyrics/config.py``."""
    namespace: dict[str, str] = {}
    config = ROOT / "local_sync_lyrics" / "config.py"
    for line in config.read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__"):
            exec(line, namespace)  # noqa: S102 - a literal assignment in our own package
    return namespace.get("__version__", "1.0.0")


def icon_path() -> Path:
    """The icon format each platform's bundler expects."""
    if sys.platform == "win32":
        return ASSETS / "icon.ico"
    if sys.platform == "darwin":
        return ASSETS / "icon.icns"
    return ASSETS / "icon.png"


def _installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def build_command(*, onefile: bool = False, clean: bool = False, windowed: bool = True,
                  icons: bool = False) -> list[str]:
    """Compose the PyInstaller command line."""
    data_arg = f"{ASSETS.relative_to(ROOT)}{os.pathsep}assets"
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--windowed" if windowed else "--console",
        "--onefile" if onefile else "--onedir",
        "--name", APP_NAME,
        "--add-data", data_arg,
        "--collect-submodules", "local_sync_lyrics",
    ]
    if clean:
        cmd.append("--clean")
    icon = icon_path()
    if sys.platform in ("win32", "darwin") and icon.is_file():
        cmd += ["--icon", str(icon.relative_to(ROOT))]
    for module in HIDDEN_IMPORTS:
        cmd += ["--hidden-import", module]
    for module in EXCLUDES:
        cmd += ["--exclude-module", module]

    extras = [name for name in OPTIONAL_COLLECT if _installed(name)]
    if "audio_separator" in extras:
        print("audio-separator found: vocal isolation will be bundled (large build).")
    for package in (*COLLECT_ALL, *extras):
        cmd += ["--collect-all", package]
    cmd.append("main.py")
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Lyricist desktop app.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--onedir", dest="onefile", action="store_false",
                      help="one-folder build with instant startup (default)")
    mode.add_argument("--onefile", dest="onefile", action="store_true",
                      help="pack everything into a single executable (slower start)")
    parser.set_defaults(onefile=False)
    parser.add_argument("--clean", action="store_true",
                        help="wipe PyInstaller's cache and previous build directory")
    parser.add_argument("--icons", action="store_true", help="force-regenerate assets first")
    parser.add_argument("--console", action="store_true",
                        help="keep a console window (useful for debugging)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the PyInstaller command without building")
    args = parser.parse_args(argv)

    print(f"Lyricist {app_version()} - {'onefile' if args.onefile else 'onedir'} build")
    if args.dry_run:
        if not ASSETS.is_dir():
            print(f"(assets would be generated in {ASSETS})")
        cmd = build_command(onefile=args.onefile, clean=args.clean,
                            windowed=not args.console, icons=args.icons)
        print(" ".join(shlex.quote(part) for part in cmd))
        return 0

    if not _installed("PyInstaller"):
        print("PyInstaller not found. Run: pip install -r requirements-dev.txt", file=sys.stderr)
        return 2

    ensure_assets(force=args.icons)
    cmd = build_command(onefile=args.onefile, clean=args.clean,
                        windowed=not args.console, icons=args.icons)
    print("Running:", " ".join(shlex.quote(part) for part in cmd))
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
