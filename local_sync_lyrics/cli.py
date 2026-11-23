"""Headless mode:  ``python main.py --cli /path/to/music [options]``.

Shares 100% of the engine with the GUI (same cache, settings and providers) and
uses ``rich`` for a live progress display.

Your own lyrics for ONE file (the same validation + forced alignment as the GUI dialog)::

    python main.py --cli song.mp3 --lyrics my_lyrics.txt [--force] [--output both]
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
)

from .app_paths import database_path
from .config import OUTPUT_MODE_LABELS, OUTPUT_MODES, PROVIDER_IDS, SettingsStore
from .database import LyricsDatabase
from .models import Event
from .processor import BatchProcessor

_COLORS = {"API": "cyan", "AI": "magenta", "OK": "green", "WARN": "yellow",
           "ERR": "red", "SKIP": "bright_black", "SYS": "blue", "INFO": "white",
           "Audio": "bright_cyan", "Output": "bright_blue"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lyricist --cli",
                                description="Fetch or generate synced lyrics for a music folder.")
    p.add_argument("folder", help="music folder to process (an audio file when --lyrics is used)")
    p.add_argument("--no-ai", action="store_true", help="disable the Whisper fallback")
    p.add_argument("--overwrite", action="store_true", help="replace existing .lrc files")
    p.add_argument("--retry-failed", action="store_true", help="retry tracks cached as failed")
    p.add_argument("--no-recursive", action="store_true", help="do not descend into sub-folders")
    p.add_argument("--model", help="Whisper model (auto, tiny, base, small, medium, large-v3...)")
    p.add_argument("--language", help="lyrics language code, or 'auto'")
    p.add_argument("--providers", help=f"comma list from: {', '.join(PROVIDER_IDS)}")
    p.add_argument("--output", choices=OUTPUT_MODES,
                   help="where results go: " + "; ".join(f"{k} = {v}" for k, v in OUTPUT_MODE_LABELS.items()))
    p.add_argument("--isolate-vocals", action="store_true",
                   help="strip background music before transcribing (slow; needs audio-separator)")
    p.add_argument("--no-detect-language", action="store_true",
                   help="do not lock Whisper to a language detected on a 30 s sample")
    p.add_argument("--lyrics", metavar="FILE",
                   help="sync ONE audio file (the 'folder' argument) with your own lyrics from this text file")
    p.add_argument("--force", action="store_true",
                   help="with --lyrics: sync even if the lyrics do not seem to match the audio")
    return p


def run_cli(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    folder = Path(args.folder).expanduser()
    console = Console()
    lyrics_text = ""
    if args.lyrics:
        try:
            lyrics_text = Path(args.lyrics).expanduser().read_text(encoding="utf-8-sig", errors="replace")
        except OSError as exc:
            console.print(f"[red]Cannot read the lyrics file:[/] {escape(str(exc))}")
            return 2
        if not folder.is_file():
            console.print(f"[red]With --lyrics the first argument must be an audio file:[/] {escape(str(folder))}")
            return 2
    elif not folder.is_dir():
        console.print(f"[red]Not a folder:[/] {escape(str(folder))}")
        return 2

    settings = SettingsStore().load()
    if args.no_ai:
        settings.use_ai_fallback = False
    if args.overwrite:
        settings.overwrite_existing = True
    if args.retry_failed:
        settings.retry_failed = True
    if args.no_recursive:
        settings.recursive = False
    if args.model:
        settings.whisper_model = args.model
    if args.language:
        settings.language = args.language
    if args.providers:
        settings.enabled_providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    if args.output:
        settings.output_mode = args.output
    if args.isolate_vocals:
        settings.isolate_vocals = True
    if args.no_detect_language:
        settings.detect_language = False
    settings._normalise()
    settings.apply_environment()

    cancel = threading.Event()
    with Progress(SpinnerColumn(), TextColumn("[bold]{task.description}"), BarColumn(),
                  MofNCompleteColumn(), TimeElapsedColumn(), console=console) as progress, \
            LyricsDatabase(database_path()) as db:
        task = progress.add_task("Syncing", total=None)

        def emit(event: Event) -> None:
            if event.kind == "log":
                color = _COLORS.get(event.tag.split(":")[0], "white")
                progress.console.print(f"[{color}]\\[{event.tag}][/] {escape(event.message)}")
            elif event.kind == "progress":
                progress.update(task, total=event.total, completed=event.current)

        def ask(payload: dict) -> bool:
            """Mismatch question for --lyrics runs (worker thread = this thread)."""
            if args.force:
                return True
            if not sys.stdin.isatty():
                return False
            what = ("only {p}% of the sung words match" if payload.get("kind") == "weak"
                    else "the lyrics are only {p}% similar to the audio").format(p=payload.get("percent", 0))
            progress.stop()
            try:
                answer = console.input(f"[yellow]Warning:[/] {what}. Sync anyway? [y/N] ")
            finally:
                progress.start()
            return answer.strip().lower() in ("y", "yes")

        try:
            processor = BatchProcessor(settings, db, emit, cancel, ask=ask)
            summary = processor.run_custom(folder, lyrics_text) if args.lyrics else processor.run(folder)
        except KeyboardInterrupt:
            cancel.set()
            console.print("[yellow]Interrupted.[/]")
            return 130

    console.print(
        f"\n[bold]Done:[/] {summary.synced_online} online, {summary.synced_aligned} aligned, "
        f"{summary.synced_ai} blind AI, "
        f"{summary.skipped} skipped, {summary.not_found} not found, {summary.failed} failed."
    )
    return 1 if summary.failed and not (summary.synced_online or summary.synced_aligned or summary.synced_ai) else 0
