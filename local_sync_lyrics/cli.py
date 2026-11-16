"""Headless mode:  ``python main.py --cli /path/to/music [options]``.

Shares 100% of the engine with the GUI (same cache, settings and providers) and
uses ``rich`` for a live progress display.
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
)

from .app_paths import database_path
from .config import PROVIDER_IDS, SettingsStore
from .database import LyricsDatabase
from .models import Event
from .processor import BatchProcessor

_COLORS = {"API": "cyan", "AI": "magenta", "OK": "green", "WARN": "yellow",
           "ERR": "red", "SKIP": "bright_black", "SYS": "blue", "INFO": "white"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="local-sync-lyrics --cli",
                                description="Fetch or generate synced lyrics for a music folder.")
    p.add_argument("folder", help="music folder to process")
    p.add_argument("--no-ai", action="store_true", help="disable the Whisper fallback")
    p.add_argument("--overwrite", action="store_true", help="replace existing .lrc files")
    p.add_argument("--retry-failed", action="store_true", help="retry tracks cached as failed")
    p.add_argument("--no-recursive", action="store_true", help="do not descend into sub-folders")
    p.add_argument("--model", help="Whisper model (auto, tiny, base, small, medium, large-v3...)")
    p.add_argument("--language", help="lyrics language code, or 'auto'")
    p.add_argument("--providers", help=f"comma list from: {', '.join(PROVIDER_IDS)}")
    return p


def run_cli(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    folder = Path(args.folder).expanduser()
    console = Console()
    if not folder.is_dir():
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
    settings._normalise()
    settings.apply_environment()

    cancel = threading.Event()
    with Progress(SpinnerColumn(), TextColumn("[bold]{task.description}"), BarColumn(),
                  MofNCompleteColumn(), TimeElapsedColumn(), console=console) as progress, \
            LyricsDatabase(database_path()) as db:
        task = progress.add_task("Syncing", total=None)

        def emit(event: Event) -> None:
            if event.kind == "log":
                color = _COLORS.get(event.tag, "white")
                progress.console.print(f"[{color}]\\[{event.tag}][/] {escape(event.message)}")
            elif event.kind == "progress":
                progress.update(task, total=event.total, completed=event.current)

        try:
            summary = BatchProcessor(settings, db, emit, cancel).run(folder)
        except KeyboardInterrupt:
            cancel.set()
            console.print("[yellow]Interrupted.[/]")
            return 130

    console.print(
        f"\n[bold]Done:[/] {summary.synced_online} online, {summary.synced_ai} AI, "
        f"{summary.skipped} skipped, {summary.not_found} not found, {summary.failed} failed."
    )
    return 1 if summary.failed and not (summary.synced_online or summary.synced_ai) else 0
