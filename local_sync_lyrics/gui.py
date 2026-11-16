"""Flet (1.x) desktop front-end.

Threading model (important for Flet 1.x)
----------------------------------------
Flet 1.x runs *all* sync event handlers on the single asyncio event loop that also
paints the UI, so blocking work inside a handler freezes the window. Therefore:

* The heavy work (network + Whisper) runs in a plain ``threading.Thread``.
* That thread never touches a Flet control. It only pushes ``Event`` objects into
  a thread-safe ``queue.Queue``.
* One async task (``_pump``) on the Flet loop drains the queue ~8x/second,
  updates the controls and calls ``page.update()``. All UI mutation therefore
  happens on the UI loop - no locks, no races.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import queue
import sys
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import flet as ft

from .app_paths import data_dir, database_path, logs_dir
from .config import PROVIDER_IDS, Settings, SettingsStore
from .database import LyricsDatabase, clear_cache_file
from .hardware_profiler import HardwareProfile, get_profile
from .models import Event
from .processor import BatchProcessor

logger = logging.getLogger(__name__)

APP_TITLE = "Local Sync Lyrics"
MAX_LOG_LINES = 1500
TRIM_LOG_BY = 300
MONO_FONT = "Consolas" if sys.platform == "win32" else "monospace"

TAG_COLORS = {
    "API": "#58a6ff",
    "AI": "#bc8cff",
    "OK": "#3fb950",
    "WARN": "#d29922",
    "ERR": "#f85149",
    "SKIP": "#8b949e",
    "SYS": "#79c0ff",
    "INFO": "#e6edf3",
}
LEVEL_OVERRIDE = {"warning": "#d29922", "error": "#f85149"}

LANGUAGES = [
    ("auto", "Auto-detect"), ("en", "English"), ("es", "Spanish"), ("fr", "French"),
    ("de", "German"), ("it", "Italian"), ("pt", "Portuguese"), ("nl", "Dutch"),
    ("sv", "Swedish"), ("pl", "Polish"), ("tr", "Turkish"), ("ru", "Russian"),
    ("uk", "Ukrainian"), ("ar", "Arabic"), ("hi", "Hindi"), ("ja", "Japanese"),
    ("ko", "Korean"), ("zh", "Chinese"), ("id", "Indonesian"), ("vi", "Vietnamese"),
    ("th", "Thai"), ("he", "Hebrew"), ("fa", "Persian"),
]
MODEL_LABELS = {
    "auto": "Auto (recommended for this PC)",
    "tiny": "tiny - fastest, lowest accuracy",
    "base": "base - fast",
    "small": "small - balanced",
    "medium": "medium - accurate, heavy",
    "large-v3-turbo": "large-v3-turbo - fast & accurate (GPU)",
    "large-v3": "large-v3 - best accuracy (GPU)",
}
PROVIDER_LABELS = {"lrclib": "LRCLIB", "betterlyrics": "BetterLyrics", "megalobiz": "Megalobiz"}
THEME_MODES = {"dark": ft.ThemeMode.DARK, "light": ft.ThemeMode.LIGHT, "system": ft.ThemeMode.SYSTEM}


async def _maybe_await(value) -> None:
    if inspect.isawaitable(value):
        await value


class LyricsApp:
    """Builds the UI and wires it to the worker thread."""

    def __init__(self, page: ft.Page) -> None:
        self.page = page
        self.store = SettingsStore()
        self.settings: Settings = self.store.load()
        self.events: "queue.Queue[Event]" = queue.Queue()
        self.cancel = threading.Event()
        self.worker: Optional[threading.Thread] = None
        self.profile: Optional[HardwareProfile] = None
        self.log_buffer: deque[str] = deque(maxlen=5000)
        self.file_picker = ft.FilePicker()

        self._configure_page()
        self._build_controls()
        self.page.add(self._layout())
        try:  # services register themselves in Flet 1.x; explicit add is harmless
            self.page.services.append(self.file_picker)
        except Exception:
            pass
        self.page.run_task(self._pump)
        self.page.run_task(self._startup)

    # ------------------------------------------------------------------ page
    def _configure_page(self) -> None:
        page = self.page
        page.title = APP_TITLE
        page.padding = 0
        page.theme_mode = THEME_MODES.get(self.settings.theme_mode, ft.ThemeMode.DARK)
        page.theme = ft.Theme(color_scheme_seed=ft.Colors.INDIGO, use_material3=True)
        page.dark_theme = ft.Theme(color_scheme_seed=ft.Colors.INDIGO, use_material3=True)
        page.window.width = 1200
        page.window.height = 820
        page.window.min_width = 920
        page.window.min_height = 660
        page.window.prevent_close = True
        page.window.on_event = self._on_window_event

    # --------------------------------------------------------------- controls
    def _build_controls(self) -> None:
        s = self.settings

        # header ------------------------------------------------------------
        self.hw_badge = ft.Text("Analysing hardware…", size=12, color=ft.Colors.ON_SURFACE_VARIANT)
        self.theme_btn = ft.IconButton(
            icon=ft.Icons.LIGHT_MODE if s.theme_mode == "dark" else ft.Icons.DARK_MODE,
            tooltip="Toggle light / dark theme", on_click=self.on_toggle_theme,
        )
        self.settings_btn = ft.IconButton(
            icon=ft.Icons.TUNE, tooltip="Settings", on_click=self.on_toggle_settings,
        )

        # folder + actions ---------------------------------------------------
        self.folder_field = ft.TextField(
            label="Music folder", value=s.last_folder, expand=True, dense=True,
            prefix_icon=ft.Icons.LIBRARY_MUSIC,
            hint_text="Browse… or paste a folder path",
            on_submit=lambda e: self.on_start(e),
        )
        self.browse_btn = ft.Button(
            content="Browse…", icon=ft.Icons.FOLDER_OPEN, on_click=self.on_browse,
        )
        self.start_btn = ft.FilledButton(
            content="Start sync", icon=ft.Icons.PLAY_ARROW, on_click=self.on_start,
        )
        self.stop_btn = ft.OutlinedButton(
            content="Stop", icon=ft.Icons.STOP, on_click=self.on_stop, disabled=True,
        )

        # progress -----------------------------------------------------------
        self.progress_bar = ft.ProgressBar(value=0, bar_height=8, border_radius=ft.BorderRadius.all(4))
        self.progress_label = ft.Text("Idle", weight=ft.FontWeight.W_600)
        self.current_file = ft.Text("", size=12, color=ft.Colors.ON_SURFACE_VARIANT,
                                    no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS, expand=True)
        self.stat_values: dict[str, ft.Text] = {}
        self.stats_row = ft.Row(
            controls=[
                self._stat_chip("synced_online", "Online", "#3fb950"),
                self._stat_chip("synced_ai", "AI", "#bc8cff"),
                self._stat_chip("skipped", "Skipped", "#8b949e"),
                self._stat_chip("not_found", "Not found", "#d29922"),
                self._stat_chip("failed", "Failed", "#f85149"),
            ],
            spacing=10, wrap=True,
        )

        # log terminal ---------------------------------------------------------
        self.log_view = ft.ListView(expand=True, auto_scroll=True, spacing=1, padding=8)

        # settings panel -------------------------------------------------------
        self._build_settings_controls()

    def _stat_chip(self, key: str, label: str, color: str) -> ft.Control:
        value = ft.Text("0", size=20, weight=ft.FontWeight.BOLD, color=color)
        self.stat_values[key] = value
        return ft.Container(
            content=ft.Column([value, ft.Text(label, size=11, color=ft.Colors.ON_SURFACE_VARIANT)],
                              spacing=0, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=ft.Padding.symmetric(vertical=6, horizontal=16),
            border_radius=ft.BorderRadius.all(10),
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
        )

    def _build_settings_controls(self) -> None:
        s = self.settings
        model_keys = list(MODEL_LABELS)
        self.model_dd = ft.Dropdown(
            label="Whisper model",
            options=[ft.DropdownOption(key=k, text=MODEL_LABELS[k]) for k in model_keys],
            value=s.whisper_model if s.whisper_model in MODEL_LABELS else "auto",
            dense=True,
        )
        self.model_hint = ft.Text("", size=11, color=ft.Colors.ON_SURFACE_VARIANT)
        lang_keys = [k for k, _ in LANGUAGES]
        self.lang_dd = ft.Dropdown(
            label="Lyrics language",
            options=[ft.DropdownOption(key=k, text=f"{name} ({k})" if k != "auto" else name)
                     for k, name in LANGUAGES],
            value=s.language if s.language in lang_keys else "auto",
            dense=True,
        )
        helper = ("Using HF_TOKEN from the environment" if s.token_from_environment()
                  else "Optional. Speeds up / unlocks model downloads.")
        self.token_field = ft.TextField(
            label="Hugging Face token", value=s.hf_token, password=True,
            can_reveal_password=True, dense=True, helper=helper,
        )

        def switch(label: str, value: bool) -> ft.Switch:
            return ft.Switch(label=label, value=value)

        self.sw_ai = switch("Use local AI when no online match", s.use_ai_fallback)
        self.sw_gpu = switch("Prefer GPU when available", s.prefer_gpu)
        self.sw_vad = switch("Skip instrumental parts (VAD)", s.vad_filter)
        self.sw_recursive = switch("Include sub-folders", s.recursive)
        self.sw_overwrite = switch("Overwrite existing .lrc files", s.overwrite_existing)
        self.sw_retry = switch("Retry tracks that failed before", s.retry_failed)
        self.sw_plain = switch("Save unsynced lyrics (.txt) as last resort", s.save_plain_fallback)
        self.provider_switches = {
            pid: switch(PROVIDER_LABELS[pid], pid in s.enabled_providers) for pid in PROVIDER_IDS
        }

        self.apply_rec_btn = ft.TextButton(
            content="Use recommended model", icon=ft.Icons.AUTO_FIX_HIGH,
            on_click=self.on_apply_recommendation, disabled=True,
        )
        self.save_btn = ft.FilledButton(content="Save settings", icon=ft.Icons.SAVE, on_click=self.on_save)
        self.clear_btn = ft.TextButton(
            content="Clear cache", icon=ft.Icons.DELETE_SWEEP, on_click=self.on_clear_cache,
        )

    # ----------------------------------------------------------------- layout
    @staticmethod
    def _card(content: ft.Control, expand: bool = False) -> ft.Container:
        return ft.Container(
            content=content, expand=expand, padding=ft.Padding.all(16),
            border_radius=ft.BorderRadius.all(14),
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
            border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT),
        )

    @staticmethod
    def _section(title: str) -> ft.Control:
        return ft.Text(title, size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.PRIMARY)

    def _layout(self) -> ft.Control:
        header = ft.Row(
            controls=[
                ft.Icon(ft.Icons.LYRICS, size=30, color=ft.Colors.PRIMARY),
                ft.Column([
                    ft.Text(APP_TITLE, size=22, weight=ft.FontWeight.BOLD),
                    ft.Text("Synced lyrics for your local music library", size=12,
                            color=ft.Colors.ON_SURFACE_VARIANT),
                ], spacing=0),
                ft.Container(expand=True),
                ft.Icon(ft.Icons.MEMORY, size=16, color=ft.Colors.ON_SURFACE_VARIANT),
                self.hw_badge,
                self.theme_btn,
                self.settings_btn,
            ],
            spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        folder_card = self._card(ft.Column([
            ft.Row([self.folder_field, self.browse_btn], spacing=10,
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Row([self.start_btn, self.stop_btn], spacing=10),
        ], spacing=12))

        progress_card = self._card(ft.Column([
            ft.Row([self.progress_label, self.current_file], spacing=12),
            self.progress_bar,
            self.stats_row,
        ], spacing=10))

        log_header = ft.Row([
            ft.Icon(ft.Icons.TERMINAL, size=16, color="#8b949e"),
            ft.Text("Activity log", size=12, weight=ft.FontWeight.BOLD, color="#8b949e"),
            ft.Container(expand=True),
            ft.IconButton(icon=ft.Icons.CONTENT_COPY, icon_size=16, tooltip="Copy log",
                          icon_color="#8b949e", on_click=self.on_copy_log),
            ft.IconButton(icon=ft.Icons.CLEAR_ALL, icon_size=18, tooltip="Clear log",
                          icon_color="#8b949e", on_click=self.on_clear_log),
        ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        log_card = ft.Container(
            content=ft.Column([log_header, ft.Divider(height=1, color="#30363d"), self.log_view],
                              spacing=4, expand=True),
            expand=True, bgcolor="#0d1117", border_radius=ft.BorderRadius.all(14),
            padding=ft.Padding.only(left=10, right=6, top=6, bottom=6),
            border=ft.Border.all(1, "#30363d"),
        )

        main_column = ft.Column([header, folder_card, progress_card, log_card],
                                spacing=14, expand=True)

        self.settings_panel = ft.Container(
            width=390, visible=False, padding=ft.Padding.all(16),
            border_radius=ft.BorderRadius.all(14),
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
            border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT),
            content=ft.Column(
                scroll=ft.ScrollMode.AUTO, spacing=10,
                controls=[
                    ft.Row([ft.Icon(ft.Icons.TUNE, color=ft.Colors.PRIMARY),
                            ft.Text("Settings", size=18, weight=ft.FontWeight.BOLD),
                            ft.Container(expand=True),
                            ft.IconButton(icon=ft.Icons.CLOSE, on_click=self.on_toggle_settings)]),
                    self._section("LOCAL AI (WHISPER)"),
                    self.model_dd, self.model_hint, self.apply_rec_btn,
                    self.lang_dd, self.token_field,
                    self.sw_ai, self.sw_gpu, self.sw_vad,
                    ft.Divider(height=8),
                    self._section("LYRICS PROVIDERS (searched in this order)"),
                    *self.provider_switches.values(),
                    ft.Divider(height=8),
                    self._section("BATCH BEHAVIOUR"),
                    self.sw_recursive, self.sw_overwrite, self.sw_retry, self.sw_plain,
                    ft.Divider(height=8),
                    ft.Row([self.save_btn, self.clear_btn], spacing=8, wrap=True),
                    ft.Text(f"Data folder: {data_dir()}", size=11, selectable=True,
                            color=ft.Colors.ON_SURFACE_VARIANT),
                    ft.Text(f"Log files: {logs_dir()}", size=11, selectable=True,
                            color=ft.Colors.ON_SURFACE_VARIANT),
                ],
            ),
        )

        return ft.Container(
            padding=ft.Padding.all(18), expand=True,
            content=ft.Row([main_column, self.settings_panel], spacing=14, expand=True,
                           vertical_alignment=ft.CrossAxisAlignment.STRETCH),
        )

    # ------------------------------------------------------------ log helpers
    def _append_log(self, tag: str, message: str, level: str = "info") -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"{stamp}  [{tag:<4}] {message}"
        self.log_buffer.append(line)
        color = LEVEL_OVERRIDE.get(level) or TAG_COLORS.get(tag, TAG_COLORS["INFO"])
        self.log_view.controls.append(
            ft.Text(line, size=12, color=color, font_family=MONO_FONT, selectable=True)
        )
        if len(self.log_view.controls) > MAX_LOG_LINES:
            del self.log_view.controls[:TRIM_LOG_BY]

    def _snack(self, message: str) -> None:
        try:
            self.page.show_dialog(ft.SnackBar(content=ft.Text(message), duration=4500))
        except Exception:
            logger.debug("snackbar failed", exc_info=True)

    # ------------------------------------------------------- background tasks
    async def _startup(self) -> None:
        self._append_log("SYS", "Analysing hardware…")
        self.page.update()
        try:
            self.profile = await asyncio.to_thread(get_profile)
        except Exception as exc:  # the profiler is defensive, but belt and braces
            logger.exception("hardware profile failed")
            self._append_log("WARN", f"Hardware analysis failed: {exc}", "warning")
            self.page.update()
            return
        profile = self.profile
        self.hw_badge.value = profile.badge()
        for line in profile.summary_lines():
            self._append_log("WARN" if line.startswith("Note:") else "SYS", line,
                             "warning" if line.startswith("Note:") else "info")
        rec = profile.recommendation
        self.model_hint.value = f"Recommended for this PC: {rec.model} on {rec.device.upper()} ({rec.compute_type})"
        self.apply_rec_btn.disabled = False
        self.page.update()

    async def _pump(self) -> None:
        """Drain worker events on the UI loop (the only place controls change)."""
        while True:
            try:
                if self._drain_events():
                    self.page.update()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("UI update failed")
            await asyncio.sleep(0.12)

    def _drain_events(self) -> bool:
        changed = False
        for _ in range(400):  # bound work per tick so the UI stays responsive
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            changed = True
            self._handle_event(event)
        return changed

    def _handle_event(self, event: Event) -> None:
        if event.kind == "log":
            self._append_log(event.tag, event.message, event.level)
        elif event.kind == "progress":
            total = max(event.total, 1)
            self.progress_bar.value = min(1.0, event.current / total)
            pct = 100.0 * event.current / total
            self.progress_label.value = f"{event.current} / {event.total}  ·  {pct:.0f}%"
            self.current_file.value = event.message
        elif event.kind == "stats":
            for key, text in self.stat_values.items():
                text.value = str(event.data.get(key, 0))
        elif event.kind == "done":
            data = event.data
            synced = data.get("synced_online", 0) + data.get("synced_ai", 0)
            verb = "Stopped" if self.cancel.is_set() else "Finished"
            self._append_log(
                "OK", f"{verb} in {event.message}: {synced} synced, {data.get('skipped', 0)} skipped, "
                      f"{data.get('not_found', 0)} not found, {data.get('failed', 0)} failed.")
            self.current_file.value = ""
            self._snack(f"{verb}: {synced} track(s) synced.")
        elif event.kind == "finished":
            self._set_running(False)

    # -------------------------------------------------------------- run state
    def _set_running(self, running: bool) -> None:
        self.start_btn.disabled = running
        self.stop_btn.disabled = not running
        self.browse_btn.disabled = running
        self.folder_field.disabled = running
        self.clear_btn.disabled = running
        if not running:
            self.stop_btn.content = "Stop"

    # ---------------------------------------------------------------- handlers
    async def on_browse(self, e) -> None:
        try:
            initial = self.folder_field.value if os.path.isdir(self.folder_field.value or "") else None
            path = await self.file_picker.get_directory_path(
                dialog_title="Select your music folder", initial_directory=initial,
            )
        except Exception as exc:
            logger.warning("folder dialog failed: %s", exc)
            hint = " On Linux install 'zenity'." if sys.platform.startswith("linux") else ""
            self._snack(f"Folder dialog unavailable - paste the path instead.{hint}")
            return
        if path:
            self.folder_field.value = path
            self.page.update()

    def on_start(self, e) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        raw = (self.folder_field.value or "").strip().strip('"').strip("'")
        folder = Path(os.path.expanduser(raw)) if raw else None
        if folder is None or not folder.is_dir():
            self._snack("Please choose an existing music folder first.")
            return

        settings = self._collect_settings()
        settings.last_folder = str(folder)
        self.settings = settings
        self.store.save(settings)
        settings.apply_environment()

        self.cancel = threading.Event()
        self.progress_bar.value = 0
        self.progress_label.value = "Starting…"
        for text in self.stat_values.values():
            text.value = "0"
        self._set_running(True)
        self._append_log("SYS", f"Starting sync of {folder}")
        self.worker = threading.Thread(
            target=self._worker_main, args=(folder, settings, self.cancel),
            name="lsl-worker", daemon=True,
        )
        self.worker.start()

    def _worker_main(self, folder: Path, settings: Settings, cancel: threading.Event) -> None:
        """Runs on the worker thread. Never touches Flet controls."""
        db: Optional[LyricsDatabase] = None
        try:
            db = LyricsDatabase(database_path())
            BatchProcessor(settings, db, self.events.put, cancel).run(folder)
        except Exception as exc:
            logger.exception("worker crashed")
            self.events.put(Event("log", message=f"Fatal error: {exc.__class__.__name__}: {exc}",
                                  tag="ERR", level="error"))
        finally:
            if db is not None:
                db.close()
            self.events.put(Event("finished"))

    def on_stop(self, e) -> None:
        self.cancel.set()
        self.stop_btn.disabled = True
        self._append_log("SYS", "Stopping after the current step…")

    def on_toggle_settings(self, e) -> None:
        self.settings_panel.visible = not self.settings_panel.visible
        self.settings_btn.icon = ft.Icons.CLOSE if self.settings_panel.visible else ft.Icons.TUNE

    def on_toggle_theme(self, e) -> None:
        dark = self.page.theme_mode != ft.ThemeMode.LIGHT
        self.page.theme_mode = ft.ThemeMode.LIGHT if dark else ft.ThemeMode.DARK
        self.theme_btn.icon = ft.Icons.DARK_MODE if dark else ft.Icons.LIGHT_MODE
        self.settings.theme_mode = "light" if dark else "dark"
        self.store.save(self.settings)

    def _collect_settings(self) -> Settings:
        s = self.settings.copy()
        s.whisper_model = self.model_dd.value or "auto"
        s.language = self.lang_dd.value or "auto"
        s.hf_token = (self.token_field.value or "").strip()
        s.use_ai_fallback = bool(self.sw_ai.value)
        s.prefer_gpu = bool(self.sw_gpu.value)
        s.vad_filter = bool(self.sw_vad.value)
        s.recursive = bool(self.sw_recursive.value)
        s.overwrite_existing = bool(self.sw_overwrite.value)
        s.retry_failed = bool(self.sw_retry.value)
        s.save_plain_fallback = bool(self.sw_plain.value)
        s.enabled_providers = [pid for pid, sw in self.provider_switches.items() if sw.value]
        s.last_folder = (self.folder_field.value or "").strip()
        s._normalise()  # also restores all providers if none are selected
        return s

    def on_save(self, e) -> None:
        self.settings = self._collect_settings()
        for pid, sw in self.provider_switches.items():  # reflect the normalised value
            sw.value = pid in self.settings.enabled_providers
        ok = self.store.save(self.settings)
        self.settings.apply_environment()
        self._snack("Settings saved." if ok else "Could not write the settings file.")

    def on_apply_recommendation(self, e) -> None:
        if self.profile is not None:
            self.model_dd.value = self.profile.recommendation.model
            self._snack(f"Model set to {self.profile.recommendation.model} - press Save to keep it.")

    def on_clear_cache(self, e) -> None:
        def close_dialog(_e=None) -> None:
            self.page.pop_dialog()

        def confirm(_e=None) -> None:
            self.page.pop_dialog()
            removed = clear_cache_file(database_path())
            self._append_log("SYS", f"Cache cleared ({removed} entries).")
            self._snack(f"Cache cleared ({removed} entries).")

        self.page.show_dialog(ft.AlertDialog(
            modal=True,
            title=ft.Text("Clear the lyrics cache?"),
            content=ft.Text("Every track will be looked up again on the next run. "
                            "Existing .lrc files are not touched."),
            actions=[ft.TextButton(content="Cancel", on_click=close_dialog),
                     ft.FilledButton(content="Clear cache", on_click=confirm)],
            actions_alignment=ft.MainAxisAlignment.END,
        ))

    def on_clear_log(self, e) -> None:
        self.log_view.controls.clear()
        self.log_buffer.clear()

    async def on_copy_log(self, e) -> None:
        try:
            await ft.Clipboard().set("\n".join(self.log_buffer))
            self._snack("Log copied to clipboard.")
        except Exception:
            self._snack("Clipboard is not available on this system.")

    async def _on_window_event(self, e) -> None:
        if e.type != ft.WindowEventType.CLOSE:
            return
        self.cancel.set()
        worker = self.worker
        if worker is not None and worker.is_alive():
            await asyncio.to_thread(worker.join, 3.0)  # let the current file wrap up
        await _maybe_await(self.page.window.destroy())


def main(page: ft.Page) -> None:
    LyricsApp(page)


def run() -> None:
    ft.run(main)
