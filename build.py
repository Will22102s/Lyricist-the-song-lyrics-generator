"""Build a standalone desktop app with `flet pack` (a PyInstaller wrapper).

    pip install -r requirements-dev.txt
    python build.py

Output goes to ./dist. Build on the OS you are targeting (Windows -> .exe,
Linux -> ELF binary); PyInstaller cannot cross-compile.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

APP_NAME = "Local Sync Lyrics"

# Packages that ship native libraries or data files PyInstaller cannot discover
# by static analysis (CTranslate2 DLLs/.so, Silero VAD .onnx, FFmpeg inside PyAV).
COLLECT_ALL = ("faster_whisper", "ctranslate2", "onnxruntime", "av")


def main() -> int:
    flet = shutil.which("flet")
    if not flet:
        print("`flet` not found. Run: pip install -r requirements-dev.txt", file=sys.stderr)
        return 2
    cmd = [flet, "pack", "main.py", "--name", APP_NAME, "--product-name", APP_NAME, "--yes"]
    if sys.platform == "win32":
        # One-folder build starts instantly; a one-file build would re-extract
        # several hundred MB of ML libraries on every launch.
        cmd.append("--onedir")
    cmd.append("--")
    for package in COLLECT_ALL:
        cmd += ["--collect-all", package]
    cmd += ["--collect-submodules", "local_sync_lyrics"]
    print("Running:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
