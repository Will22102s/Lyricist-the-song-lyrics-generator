"""Build a standalone desktop app with PyInstaller.

    pip install -r requirements-dev.txt
    python build.py                 # regenerates assets/logo.* if needed, then packs
    python build.py --icons         # force-regenerate the icons first

Under the hood this is the same as::

    pyinstaller --noconfirm --clean --windowed --onedir --name "Lyricist" \\
                --icon assets/logo.ico --add-data "assets:assets" \\
                --collect-all faster_whisper --collect-all ctranslate2 \\
                --collect-all onnxruntime --collect-all av \\
                --collect-submodules local_sync_lyrics main.py

``--icon`` stamps the executable (Explorer, taskbar, title bar); ``--add-data`` ships
the ``assets/`` folder so the running app can find its logo and window icon. Output goes
to ``dist/Lyricist/``. Build on the OS you are targeting (Windows -> .exe, Linux -> ELF
binary); PyInstaller cannot cross-compile.
"""

from __future__ import annotations

import importlib.util
import os
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


def app_version() -> str:
    namespace: dict[str, str] = {}
    init = ROOT / "local_sync_lyrics" / "__init__.py"
    for line in init.read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__"):
            exec(line, namespace)  # noqa: S102 - a literal assignment in our own package
    return namespace.get("__version__", "1.0.0")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if importlib.util.find_spec("PyInstaller") is None:
        print("PyInstaller not found. Run: pip install -r requirements-dev.txt", file=sys.stderr)
        return 2

    ensure_assets(force="--icons" in argv)
    icon = ASSETS / ("logo.ico" if sys.platform == "win32" else "logo.png")

    # PyInstaller wants "SRC<os.pathsep>DEST" (';' on Windows, ':' elsewhere).
    data_arg = f"{ASSETS.relative_to(ROOT)}{os.pathsep}assets"
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean",
        "--windowed",                 # no console window on Windows
        "--onedir",                   # one-folder: instant start for the ML payload
        "--name", APP_NAME,
        "--add-data", data_arg,
        "--collect-submodules", "local_sync_lyrics",
    ]
    if sys.platform in ("win32", "darwin"):
        cmd += ["--icon", str(icon.relative_to(ROOT))]

    extras = [name for name in OPTIONAL_COLLECT
              if importlib.util.find_spec(name) is not None]
    if "audio_separator" in extras:
        print("audio-separator found: vocal isolation will be bundled (large build).")
    for package in (*COLLECT_ALL, *extras):
        cmd += ["--collect-all", package]
    cmd.append("main.py")

    print("Running:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
