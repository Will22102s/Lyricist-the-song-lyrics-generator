"""Build a standalone desktop app with `flet pack` (a PyInstaller wrapper).

    pip install -r requirements-dev.txt
    python build.py                 # regenerates assets/logo.* if needed, then packs
    python build.py --icons         # force-regenerate the icons first

Under the hood this is the same as:

    flet pack main.py --name "Local Sync Lyrics" --icon assets/logo.ico \\
              --add-data "assets;assets" --yes -- --collect-all faster_whisper ...

``--icon`` stamps the .exe (Explorer, taskbar, title bar); ``--add-data`` ships the
``assets/`` folder so the running app can find its logo and window icon. Output goes
to ./dist. Build on the OS you are targeting (Windows -> .exe, Linux -> ELF binary);
PyInstaller cannot cross-compile.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from make_icons import ASSETS, ensure_assets

ROOT = Path(__file__).resolve().parent
APP_NAME = "Local Sync Lyrics"

# Packages that ship native libraries or data files PyInstaller cannot discover
# by static analysis (CTranslate2 DLLs/.so, Silero VAD .onnx, FFmpeg inside PyAV).
COLLECT_ALL = ("faster_whisper", "ctranslate2", "onnxruntime", "av")


def app_version() -> str:
    namespace: dict[str, str] = {}
    init = ROOT / "local_sync_lyrics" / "__init__.py"
    for line in init.read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__"):
            exec(line, namespace)  # noqa: S102 - a literal assignment in our own package
    return namespace.get("__version__", "1.0.0")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    flet = shutil.which("flet")
    if not flet:
        print("`flet` not found. Run: pip install -r requirements-dev.txt", file=sys.stderr)
        return 2

    ensure_assets(force="--icons" in argv)
    icon = ASSETS / ("logo.ico" if sys.platform == "win32" else "logo.png")

    cmd = [flet, "pack", "main.py", "--name", APP_NAME, "--product-name", APP_NAME,
           "--file-description", APP_NAME, "--product-version", app_version(),
           "--icon", str(icon.relative_to(ROOT)),
           # PyInstaller wants "SRC<os.pathsep>DEST" (';' on Windows, ':' elsewhere)
           "--add-data", f"{ASSETS.relative_to(ROOT)}{os.pathsep}assets",
           "--yes"]
    if sys.platform == "win32":
        # One-folder build starts instantly; a one-file build would re-extract
        # several hundred MB of ML libraries on every launch.
        cmd.append("--onedir")
    cmd.append("--")
    for package in COLLECT_ALL:
        cmd += ["--collect-all", package]
    cmd += ["--collect-submodules", "local_sync_lyrics"]
    print("Running:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
