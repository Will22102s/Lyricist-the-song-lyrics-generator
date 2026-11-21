"""Entry point.

    python main.py                       # desktop GUI
    python main.py --cli /path/to/music  # headless, with rich progress
    python main.py --make-icons          # (re)generate assets/logo.svg|png|ico (needs Pillow)

Package as a desktop app (see README) - `python build.py` does all of this for you:
    flet pack main.py --name "Local Sync Lyrics" --icon assets/logo.ico \\
              --add-data "assets;assets" --yes -- --collect-all faster_whisper ...
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import sys


def main() -> int:
    multiprocessing.freeze_support()  # required for frozen (PyInstaller) builds

    # Must run before anything prints: --noconsole builds have sys.stdout = None.
    from local_sync_lyrics.logging_setup import ensure_std_streams, setup_logging

    ensure_std_streams()
    log = setup_logging()
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args = sys.argv[1:]
    if args and args[0] == "--make-icons":
        from make_icons import main as make_icons

        return make_icons(args[1:])
    if args and args[0] in ("--cli", "cli"):
        from local_sync_lyrics.cli import run_cli

        return run_cli(args[1:])

    try:
        from local_sync_lyrics.gui import run

        log.info("Starting GUI")
        run()
    except Exception:
        log.exception("GUI crashed")
        return 1
    finally:
        logging.shutdown()
    return 0


if __name__ == "__main__":
    code = main()
    # Skip interpreter teardown: native inference threads (CTranslate2/ONNX) can
    # otherwise delay or crash the exit after the window has already closed.
    sys.stdout and sys.stdout.flush()
    os._exit(code)
