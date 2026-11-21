"""Flet (1.x) desktop front-end - *Liquid Glass* edition.

Look & feel
-----------
Layered frosted glass over an ambient mesh: a gradient plane, three blurred light orbs
and, above them, translucent ``ft.Container`` surfaces with a backdrop blur, a specular
1 px rim and soft rounded corners. Colours come from :mod:`.theme` (one accent + dark /
light), so the appearance mode and the accent can change on the fly: every themed
control is registered with :meth:`LyricsApp._paint` and repainted by
:meth:`LyricsApp._apply_theme` - no restart, nothing rebuilt.

Threading model (important for Flet 1.x)
----------------------------------------
Flet 1.x runs sync event handlers on the same asyncio loop that paints the UI, so
blocking work inside a handler freezes the window. Therefore:

* The heavy work (network + Whisper) runs in a plain ``threading.Thread``.
* Disk work started from a handler (saving settings, clearing the cache) also runs on a
  short-lived thread; nothing in this module reads or writes a file on the UI loop
  once the window is up.
* Worker threads never touch a Flet control. They push ``Event`` objects into a
  thread-safe ``queue.Queue``.
* One async task (``_pump``) on the Flet loop drains the queue ~8x/second, updates the
  controls and calls ``page.update()``. All UI mutation happens on the UI loop - no
  locks, no races.
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
from typing import Any, Callable, Optional, Union

import flet as ft

from . import __version__
from .app_paths import asset_path, assets_dir, data_dir, database_path, logs_dir
from .branding import logo_svg
from .config import (
    ACCENT_PRESETS, CUSTOM_ACCENT_ID, PROVIDER_IDS, THEME_MODE_LABELS, TIER1_IDS, TIER2_IDS,
    Settings, SettingsStore, find_accent_preset,
)
from .database import LyricsDatabase, clear_cache_file
from .hardware_profiler import HardwareProfile, get_profile
from .models import Event
from .processor import BatchProcessor
from .theme import (
    Palette, build_palette, clear, darken, lighten, normalise_hex, parse_hex, resolve_dark, with_alpha,
)

logger = logging.getLogger(__name__)

APP_TITLE = "Local Sync Lyrics"
MAX_LOG_LINES = 1500
TRIM_LOG_BY = 300
MONO_FONT = "Consolas" if sys.platform == "win32" else "monospace"

CLEAR = "#00000000"
ANIM = ft.Animation(300, ft.AnimationCurve.EASE_OUT_CUBIC)        # cards, buttons, pills
ANIM_FAST = ft.Animation(180, ft.AnimationCurve.EASE_OUT_CUBIC)   # hover feedback
ANIM_AMBIENT = ft.Animation(500, ft.AnimationCurve.EASE_OUT_CUBIC)  # accent cross-fade
GLASS_BLUR = ft.Blur(18, 18, ft.BlurTileMode.MIRROR)

SIDEBAR_WIDTH = 240
SECTION_GAP = 14
SWITCH_WIDTH = 316

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
PROVIDER_LABELS = {
    "lrclib": "LRCLIB", "betterlyrics": "BetterLyrics", "rclyricsband": "RCLyricsBand",
    "megalobiz": "Megalobiz", "netease": "NetEase", "qqmusic": "QQ Music",
    "genius": "Genius", "lyricsovh": "Lyrics.ovh", "musixmatch": "Musixmatch (needs API key)",
}
FLET_THEME_MODES = {"dark": ft.ThemeMode.DARK, "light": ft.ThemeMode.LIGHT,
                    "system": ft.ThemeMode.SYSTEM}
MODE_SEGMENTS = (
    ("dark", ft.Icons.DARK_MODE, "Dark", "Liquid Obsidian"),
    ("light", ft.Icons.LIGHT_MODE, "Light", "Frosted Pearl"),
    ("system", ft.Icons.BRIGHTNESS_AUTO, "System", "Follows your OS"),
)
STAT_LABELS = (("synced_online", "Online"), ("synced_aligned", "Aligned"), ("synced_ai", "Blind AI"),
               ("skipped", "Skipped"), ("not_found", "Not found"), ("failed", "Failed"))

Spec = Union[str, Callable[[Palette], Any]]  # a Palette field name or a function of the palette


async def _maybe_await(value) -> None:
    if inspect.isawaitable(value):
        await value


def _is_hovering(e) -> bool:
    return str(getattr(e, "data", "")).strip().lower() in ("true", "1")


class SmoothBar:
    """Determinate progress bar whose fill glides to each new value.

    Mirrors the one attribute the rest of the code uses from ``ft.ProgressBar``
    (``value`` in 0..1). The fill is a gradient container whose width is animated by
    Flet's implicit animation; ``control`` is what goes into the layout.
    """

    def __init__(self, height: int = 10) -> None:
        self._value = 0.0
        self._track_width = 0.0
        self.fill = ft.Container(width=0, height=height, border_radius=height / 2, animate=ANIM)
        self.track = ft.Container(
            height=height, border_radius=height / 2, alignment=ft.Alignment.CENTER_LEFT,
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS, content=self.fill, animate=ANIM,
            on_size_change=self._on_size,
        )

    @property
    def control(self) -> ft.Control:
        return self.track

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, value: float) -> None:
        self._value = max(0.0, min(1.0, float(value or 0.0)))
        self.fill.width = self._value * self._track_width

    def _on_size(self, e) -> None:
        width = float(getattr(e, "width", 0) or 0)
        if abs(width - self._track_width) < 0.5:
            return
        self._track_width = width
        self.fill.width = self._value * width
        try:
            self.fill.update()
        except Exception:  # not mounted yet
            pass


class LyricsApp:
    """Builds the UI and wires it to the worker thread."""

    def __init__(self, page: ft.Page, store: Optional[SettingsStore] = None,
                 settings: Optional[Settings] = None) -> None:
        self.page = page
        # ``main`` loads both off the UI loop; the fallback keeps ``LyricsApp(page)`` working.
        self.store = store if store is not None else SettingsStore()
        self.settings: Settings = settings if settings is not None else self.store.load()
        self.events: "queue.Queue[Event]" = queue.Queue()
        self.cancel = threading.Event()
        self.worker: Optional[threading.Thread] = None
        self.profile: Optional[HardwareProfile] = None
        self.log_buffer: deque[str] = deque(maxlen=5000)
        self.file_picker = ft.FilePicker()

        self.view_name = "workspace"
        self._status = "idle"
        self._status_after = "idle"
        self._running = False
        self._run_id = 0
        self._paints: list[tuple[Any, str, Callable[[Palette], Any]]] = []
        self._refreshers: list[Callable[[], None]] = []
        self._animated: list[tuple[Any, Any]] = []   # (control, its implicit animation)
        self._anim_gen = 0
        self.palette: Palette = self._compute_palette()

        self._configure_page()
        self._build_controls()
        self.page.add(self._layout())
        try:  # services register themselves in Flet 1.x; explicit add is harmless
            self.page.services.append(self.file_picker)
        except Exception:
            pass
        self.page.run_task(self._pump)
        self.page.run_task(self._startup)

    # ================================================================== theming
    def _system_is_dark(self) -> bool:
        brightness = getattr(self.page, "platform_brightness", None)
        return brightness != ft.Brightness.LIGHT   # unknown -> dark

    def _compute_palette(self) -> Palette:
        dark = resolve_dark(self.settings.theme_mode, self._system_is_dark())
        return build_palette(dark, self.settings.accent_color())

    def _paint(self, control: Any, attr: str, spec: Spec) -> None:
        """Register ``control.<attr>`` as palette-driven and apply it right away."""
        fn: Callable[[Palette], Any] = (
            (lambda p, name=spec: getattr(p, name)) if isinstance(spec, str) else spec
        )
        self._paints.append((control, attr, fn))
        setattr(control, attr, fn(self.palette))

    def _color_scheme(self, p: Palette) -> ft.ColorScheme:
        """Material colour roles for the stock controls (switch, dropdown, dialog...)."""
        return ft.ColorScheme(
            primary=p.accent, on_primary=p.on_accent, secondary=p.accent, on_secondary=p.on_accent,
            primary_container=p.accent_soft, on_primary_container=p.text,
            surface=p.surface, on_surface=p.text, on_surface_variant=p.text2,
            outline=p.field_border, outline_variant=p.border, surface_tint=CLEAR,
            surface_container_lowest=p.surface, surface_container_low=p.surface,
            surface_container=p.surface, surface_container_high=p.surface,
            surface_container_highest=p.surface, inverse_surface=p.surface,
            on_inverse_surface=p.text, error=p.err, scrim="#000000", shadow="#000000",
        )

    def _apply_page_theme(self) -> None:
        page = self.page
        accent = self.settings.accent_color()
        page.theme_mode = FLET_THEME_MODES.get(self.settings.theme_mode, ft.ThemeMode.DARK)
        page.theme = ft.Theme(color_scheme=self._color_scheme(build_palette(False, accent)),
                              use_material3=True)
        page.dark_theme = ft.Theme(color_scheme=self._color_scheme(build_palette(True, accent)),
                                   use_material3=True)
        page.bgcolor = self.palette.bg_bottom

    def _track(self, control: Any) -> Any:
        """Remember a control's implicit animation so a light/dark flip can pause it."""
        if getattr(control, "animate", None) is not None:
            self._animated.append((control, control.animate))
        return control

    def _set_animations(self, enabled: bool) -> None:
        for control, animation in self._animated:
            control.animate = animation if enabled else None

    async def _restore_animations(self) -> None:
        generation = self._anim_gen
        await asyncio.sleep(0.25)
        if generation == self._anim_gen:
            self._set_animations(True)
            self.page.update()

    def _apply_theme(self) -> None:
        """Re-resolve the palette and repaint everything (instant, animated, no restart).

        Accent changes cross-fade. A light <-> dark flip also flips every text colour
        at once, so the cross-fade is paused for that one update - otherwise white text
        would sit on a half-faded light card for a few hundred milliseconds.
        """
        was_dark = self.palette.dark
        self.palette = self._compute_palette()
        p = self.palette
        flip = p.dark != was_dark
        if flip:
            self._anim_gen += 1
            self._set_animations(False)
        self._apply_page_theme()
        for control, attr, fn in self._paints:
            setattr(control, attr, fn(p))
        for line in self.log_view.controls:           # old log lines keep their tag colour
            meta = getattr(line, "data", None)
            if isinstance(meta, tuple):
                line.color = p.tag_color(*meta)
        for refresh in self._refreshers:
            refresh()
        self.page.update()
        if flip:
            self.page.run_task(self._restore_animations)

    def _refresh_theme_btn(self) -> None:
        self.theme_btn.icon = ft.Icons.LIGHT_MODE if self.palette.dark else ft.Icons.DARK_MODE

    def _persist_appearance(self) -> None:
        self._save_bg(self.settings.copy())

    # ------------------------------------------------------------- style factories
    @staticmethod
    def _primary_style(p: Palette) -> ft.ButtonStyle:
        S = ft.ControlState
        return ft.ButtonStyle(
            bgcolor={S.DEFAULT: p.accent, S.HOVERED: p.accent_hi, S.PRESSED: p.accent_lo,
                     S.DISABLED: p.glass},
            color={S.DEFAULT: p.on_accent, S.DISABLED: p.text3},
            icon_color={S.DEFAULT: p.on_accent, S.DISABLED: p.text3},
            overlay_color=CLEAR, elevation=0, animation_duration=300,
            shape=ft.RoundedRectangleBorder(radius=14),
            padding=ft.Padding.symmetric(vertical=16, horizontal=24),
            text_style=ft.TextStyle(weight=ft.FontWeight.W_600, size=14),
        )

    @staticmethod
    def _glass_style(p: Palette, label: Optional[str] = None) -> ft.ButtonStyle:
        S = ft.ControlState
        fg = getattr(p, label) if label else p.text
        return ft.ButtonStyle(
            bgcolor={S.DEFAULT: p.glass, S.HOVERED: p.glass_hi, S.PRESSED: p.glass_strong,
                     S.DISABLED: p.field},
            color={S.DEFAULT: fg, S.DISABLED: p.text3},
            icon_color={S.DEFAULT: fg, S.DISABLED: p.text3},
            side={S.DEFAULT: ft.BorderSide(1, p.border), S.HOVERED: ft.BorderSide(1, p.border_hi),
                  S.DISABLED: ft.BorderSide(1, p.border)},
            overlay_color=CLEAR, elevation=0, animation_duration=300,
            shape=ft.RoundedRectangleBorder(radius=14),
            padding=ft.Padding.symmetric(vertical=16, horizontal=22),
            text_style=ft.TextStyle(weight=ft.FontWeight.W_500, size=14),
        )

    @staticmethod
    def _quiet_style(p: Palette) -> ft.ButtonStyle:
        S = ft.ControlState
        return ft.ButtonStyle(
            bgcolor={S.DEFAULT: CLEAR, S.HOVERED: p.glass, S.DISABLED: CLEAR},
            color={S.DEFAULT: p.accent_fg, S.DISABLED: p.text3},
            icon_color={S.DEFAULT: p.accent_fg, S.DISABLED: p.text3},
            overlay_color=CLEAR, elevation=0, animation_duration=300,
            shape=ft.RoundedRectangleBorder(radius=12),
            padding=ft.Padding.symmetric(vertical=12, horizontal=14),
        )

    # ------------------------------------------------------------ small builders
    def _text(self, value: str = "", size: float = 14, weight: Optional[ft.FontWeight] = None,
              role: str = "text", **kw: Any) -> ft.Text:
        t = ft.Text(value, size=size, weight=weight, **kw)
        self._paint(t, "color", role)
        return t

    def _title(self, value: str, size: float = 30) -> ft.Text:
        return self._text(value, size, ft.FontWeight.W_700,
                          style=ft.TextStyle(letter_spacing=-0.8))

    def _icon(self, name: Any, size: float = 20, role: str = "text2") -> ft.Icon:
        icon = ft.Icon(name, size=size)
        self._paint(icon, "color", role)
        return icon

    def _glass(self, content: ft.Control, *, padding: Any = 20, radius: float = 22,
               strong: bool = False, blur: bool = True, **kw: Any) -> ft.Container:
        """A frosted surface: backdrop blur + sheen gradient + specular rim."""
        box = ft.Container(
            content=content, border_radius=radius, animate=ANIM,
            padding=padding if isinstance(padding, ft.Padding) else ft.Padding.all(padding),
            **kw,
        )
        if blur:
            box.blur = GLASS_BLUR
        self._track(box)
        self._paint(box, "gradient", lambda p: ft.LinearGradient(
            begin=ft.Alignment.TOP_LEFT, end=ft.Alignment.BOTTOM_RIGHT,
            colors=[p.glass_hi, p.glass_strong] if strong else [p.sheen_top, p.sheen_bottom]))
        self._paint(box, "border", lambda p: ft.Border.all(1, p.border))
        self._paint(box, "shadow", lambda p: ft.BoxShadow(
            blur_radius=32, color=p.shadow, offset=ft.Offset(0, 12)))
        return box

    def _logo(self, size: float, radius: float) -> ft.Container:
        """The brand SVG (re-tinted with the accent) with an accent-coloured glow."""
        image = ft.Image(src=logo_svg(self.palette.accent), width=size, height=size,
                         fit=ft.BoxFit.CONTAIN)
        self._paint(image, "src", lambda p: logo_svg(p.accent))
        holder = self._track(ft.Container(content=image, width=size, height=size,
                                          border_radius=radius, animate=ANIM))
        self._paint(holder, "shadow", lambda p: ft.BoxShadow(
            blur_radius=size * 0.55, spread_radius=0, color=p.accent_glow,
            offset=ft.Offset(0, size * 0.10)))
        return holder

    def _style_field(self, field: Any) -> None:
        """Glass look for TextField / Dropdown."""
        field.filled = True
        field.border_radius = 12
        field.border_width = 1
        self._paint(field, "fill_color", "field")
        self._paint(field, "border_color", "field_border")
        self._paint(field, "focused_border_color", "accent")
        self._paint(field, "label_style", lambda p: ft.TextStyle(color=p.text2))
        if isinstance(field, ft.TextField):
            self._paint(field, "cursor_color", "accent")

    def _style_switch(self, sw: ft.Switch) -> ft.Switch:
        sw.active_color = "#FFFFFF"        # white thumb on every accent, like Apple's switches
        self._paint(sw, "active_track_color", "accent")
        self._paint(sw, "inactive_thumb_color", "text2")
        self._paint(sw, "inactive_track_color", "field")
        self._paint(sw, "track_outline_color", "field_border")
        return sw

    # ================================================================== page
    def _configure_page(self) -> None:
        page = self.page
        page.title = APP_TITLE
        page.padding = 0
        page.spacing = 0
        self._apply_page_theme()
        page.window.width = 1240
        page.window.height = 840
        page.window.min_width = 980
        page.window.min_height = 680
        page.window.prevent_close = True
        page.window.on_event = self._on_window_event
        if asset_path("logo.ico") is not None:
            page.window.icon = "logo.ico"      # relative to assets_dir; Windows only
        page.on_platform_brightness_change = self._on_system_brightness

    def _on_system_brightness(self, e) -> None:
        if self.settings.theme_mode == "system":
            self._apply_theme()

    # ================================================================ controls
    def _build_controls(self) -> None:
        s = self.settings

        # sidebar ---------------------------------------------------------------
        self.hw_badge = self._text("Analysing hardware…", 12, role="text2", max_lines=2)
        self.theme_btn = ft.IconButton(
            icon=ft.Icons.LIGHT_MODE if self.palette.dark else ft.Icons.DARK_MODE,
            icon_size=18, tooltip="Toggle light / dark", on_click=self.on_toggle_theme,
        )
        self._paint(self.theme_btn, "icon_color", "text2")
        self._refreshers.append(self._refresh_theme_btn)
        self.nav_items: dict[str, tuple[ft.Container, ft.Icon, ft.Text]] = {}

        # workspace ---------------------------------------------------------------
        self.folder_field = ft.TextField(
            label="Music folder", value=s.last_folder, expand=True,
            prefix_icon=ft.Icons.LIBRARY_MUSIC, hint_text="Browse… or paste a folder path",
            on_submit=lambda e: self.on_start(e),
        )
        self._style_field(self.folder_field)
        self.browse_btn = ft.Button(content="Browse…", icon=ft.Icons.FOLDER_OPEN,
                                    on_click=self.on_browse)
        self._paint(self.browse_btn, "style", self._glass_style)
        self.start_btn = ft.FilledButton(content="Start sync", icon=ft.Icons.PLAY_ARROW,
                                         on_click=self.on_start)
        self._paint(self.start_btn, "style", self._primary_style)
        self.stop_btn = ft.OutlinedButton(content="Stop", icon=ft.Icons.STOP,
                                          on_click=self.on_stop, disabled=True)
        self._paint(self.stop_btn, "style", lambda p: self._glass_style(p, "err"))

        # status pill -------------------------------------------------------------
        self.status_dot = ft.Container(width=8, height=8, border_radius=4, animate=ANIM)
        self.status_text = self._text("Ready", 12.5, ft.FontWeight.W_500, role="text2")
        self.status_pill = ft.Container(
            content=ft.Row([self.status_dot, self.status_text], spacing=8, tight=True),
            padding=ft.Padding.symmetric(vertical=7, horizontal=14), border_radius=20, animate=ANIM,
        )
        self._track(self.status_dot)
        self._track(self.status_pill)
        self._paint(self.status_pill, "bgcolor", "field")
        self._paint(self.status_pill, "border", lambda p: ft.Border.all(1, p.field_border))
        self._refreshers.append(self._refresh_status)
        self._refresh_status()

        # progress (floating card) ----------------------------------------------------
        self.progress_bar = SmoothBar(10)
        self._paint(self.progress_bar.track, "bgcolor", "field")
        self._paint(self.progress_bar.fill, "gradient", lambda p: ft.LinearGradient(
            begin=ft.Alignment.CENTER_LEFT, end=ft.Alignment.CENTER_RIGHT,
            colors=[p.accent_lo, p.accent, p.accent_hi]))
        self.progress_label = self._text("Idle", 16, ft.FontWeight.W_600)
        self.current_file = self._text("", 12.5, role="text2", no_wrap=True,
                                       overflow=ft.TextOverflow.ELLIPSIS, expand=True)
        self.progress_stop_btn = ft.OutlinedButton(content="Stop", icon=ft.Icons.STOP, disabled=True,
                                                   on_click=self.on_stop)
        self._paint(self.progress_stop_btn, "style", lambda p: self._glass_style(p, "err"))
        self.stat_values: dict[str, ft.Text] = {}

        # activity log -------------------------------------------------------------------
        self.log_view = ft.ListView(expand=True, auto_scroll=True, spacing=2, padding=ft.Padding.all(10))

        self._build_settings_controls()

    def _build_settings_controls(self) -> None:
        s = self.settings
        self.model_dd = ft.Dropdown(
            label="Whisper model", expand=True, options=[ft.DropdownOption(key=k, text=v) for k, v in MODEL_LABELS.items()],
            value=s.whisper_model if s.whisper_model in MODEL_LABELS else "auto", dense=True,
        )
        self.model_hint = self._text("", 12, role="text2")
        lang_keys = [k for k, _ in LANGUAGES]
        self.lang_dd = ft.Dropdown(
            label="Lyrics language", expand=True,
            options=[ft.DropdownOption(key=k, text=f"{name} ({k})" if k != "auto" else name)
                     for k, name in LANGUAGES],
            value=s.language if s.language in lang_keys else "auto", dense=True,
        )
        helper = ("Using HF_TOKEN from the environment" if s.token_from_environment()
                  else "Optional. Speeds up / unlocks model downloads.")
        self.token_field = ft.TextField(label="Hugging Face token", value=s.hf_token, password=True,
                                        can_reveal_password=True, dense=True, helper=helper)
        self.genius_field = ft.TextField(
            label="Genius API token (optional)", value=s.genius_token, password=True, expand=True,
            can_reveal_password=True, dense=True,
            helper="Without a token, public Genius search is used.")
        self.musixmatch_field = ft.TextField(
            label="Musixmatch API key (optional)", value=s.musixmatch_key, password=True, expand=True,
            can_reveal_password=True, dense=True,
            helper="Free plan returns excerpts only; used to identify songs.")
        for field in (self.model_dd, self.lang_dd, self.token_field, self.genius_field,
                      self.musixmatch_field):
            self._style_field(field)

        def switch(label: str, value: bool, tip: Optional[str] = None) -> ft.Switch:
            # NB: never set ``width`` on a Flet Switch - it wraps the whole control in a
            # FittedBox(fill) that stretches the label. Cells (see _switch_grid) size it instead.
            return self._style_switch(ft.Switch(
                label=label, value=value, tooltip=tip, label_text_style=ft.TextStyle(size=13.5)))

        self.sw_ai = switch("Use local AI when no online match", s.use_ai_fallback)
        self.sw_gpu = switch("Prefer GPU when available", s.prefer_gpu)
        self.sw_vad = switch("Skip instrumental parts (VAD)", s.vad_filter)
        self.sw_recursive = switch("Include sub-folders", s.recursive)
        self.sw_overwrite = switch("Overwrite existing .lrc files", s.overwrite_existing)
        self.sw_retry = switch("Retry tracks that failed before", s.retry_failed)
        self.sw_plain = switch("Save plain .txt as a last resort", s.save_plain_fallback,
                              "Save unsynced lyrics as a .txt file when nothing better is found")
        self.sw_align = switch("Align official lyrics to the audio", s.use_forced_alignment,
                           "Forced alignment of official lyrics (recommended)")
        self.sw_mashup = switch("Detect mashups / merged tracks", s.detect_mashups)
        self.sw_ident = switch("Name unknown songs by listening", s.identify_by_audio)
        self.provider_switches = {
            pid: switch(PROVIDER_LABELS[pid], pid in s.enabled_providers) for pid in PROVIDER_IDS
        }

        self.apply_rec_btn = ft.TextButton(content="Use recommended model", icon=ft.Icons.AUTO_FIX_HIGH,
                                           on_click=self.on_apply_recommendation, disabled=True)
        self._paint(self.apply_rec_btn, "style", self._quiet_style)
        self.save_btn = ft.FilledButton(content="Save settings", icon=ft.Icons.SAVE, on_click=self.on_save)
        self._paint(self.save_btn, "style", self._primary_style)
        self.clear_btn = ft.OutlinedButton(content="Clear cache", icon=ft.Icons.DELETE_SWEEP,
                                           on_click=self.on_clear_cache)
        self._paint(self.clear_btn, "style", self._glass_style)

        # appearance ------------------------------------------------------------------------
        self.accent_caption = self._text("", 12.5, role="text2")
        self.mode_segments: dict[str, ft.Container] = {}
        self.swatches: dict[str, tuple[ft.Container, ft.Container]] = {}
        self.custom_hex_field = ft.TextField(
            label="Custom accent", value=s.custom_accent.lstrip("#"), prefix="#", dense=True,
            width=190, hint_text="0A84FF",
            on_change=self.on_custom_accent_change, on_submit=self.on_custom_accent_submit,
        )
        self._style_field(self.custom_hex_field)
        self.data_path_text = self._text("Data folder: …", 12, role="text2", selectable=True)
        self.logs_path_text = self._text("Log files: …", 12, role="text2", selectable=True)

    # ================================================================== layout
    def _layout(self) -> ft.Control:
        self.workspace_view = self._build_workspace_view()
        self.settings_view = self._build_settings_view()
        self.view_switcher = ft.AnimatedSwitcher(
            content=self.workspace_view, expand=True,
            transition=ft.AnimatedSwitcherTransition.FADE, duration=320, reverse_duration=180,
            switch_in_curve=ft.AnimationCurve.EASE_OUT_CUBIC,
            switch_out_curve=ft.AnimationCurve.EASE_IN_CUBIC,
        )

        bg = self._track(ft.Container(expand=True, animate=ANIM_AMBIENT))
        self._paint(bg, "gradient", lambda p: ft.LinearGradient(
            begin=ft.Alignment.TOP_LEFT, end=ft.Alignment.BOTTOM_RIGHT,
            colors=[p.bg_top, p.bg_bottom]))

        def orb(role: str, size: float, **pos: Any) -> ft.Container:
            o = self._track(ft.Container(width=size, height=size, shape=ft.BoxShape.CIRCLE,
                                         animate=ANIM_AMBIENT, **pos))
            self._paint(o, "gradient", lambda p: ft.RadialGradient(
                colors=[getattr(p, role), clear(getattr(p, role))], radius=0.5))
            return o

        shell = ft.Container(
            expand=True, padding=ft.Padding.all(16),
            content=ft.Row([
                self._build_sidebar(),
                ft.Container(width=SECTION_GAP + 4),
                ft.Container(content=self.view_switcher, expand=True),
            ], spacing=0, expand=True, vertical_alignment=ft.CrossAxisAlignment.STRETCH),
        )
        self._refreshers.append(self._refresh_nav)
        self._refresh_nav()
        return ft.Stack(
            expand=True,
            controls=[
                bg,
                orb("orb_a", 760, left=-220, top=-260),
                orb("orb_b", 640, right=-200, bottom=-240),
                orb("orb_c", 460, right=260, top=90),
                shell,
            ],
        )

    # ---------------------------------------------------------------- sidebar
    def _build_sidebar(self) -> ft.Control:
        brand = ft.Row([
            self._logo(46, 13),
            ft.Column([
                self._text(APP_TITLE, 15.5, ft.FontWeight.W_700, style=ft.TextStyle(letter_spacing=-0.3)),
                self._text(f"Version {__version__}", 11.5, role="text3"),
            ], spacing=1, expand=True),
        ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER)

        nav = ft.Column([
            self._nav_item("workspace", ft.Icons.GRAPHIC_EQ, "Sync Workspace"),
            self._nav_item("settings", ft.Icons.TUNE, "Settings"),
        ], spacing=6)

        footer = self._glass(
            ft.Row([
                self._icon(ft.Icons.MEMORY, 18),
                ft.Column([self._text("This PC", 11, role="text3"), self.hw_badge],
                          spacing=1, expand=True),
                self.theme_btn,
            ], spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            padding=ft.Padding.only(left=12, right=4, top=8, bottom=8), radius=18, blur=False,
        )
        return self._glass(
            ft.Column([brand, ft.Container(height=22), nav, ft.Container(expand=True), footer],
                      spacing=0, expand=True),
            width=SIDEBAR_WIDTH, padding=ft.Padding.all(18), radius=28, strong=True,
        )

    def _nav_item(self, key: str, icon: Any, label: str) -> ft.Container:
        ic = self._icon(icon, 20)
        txt = self._text(label, 14.5, ft.FontWeight.W_500, role="text2")
        box = ft.Container(
            content=ft.Row([ic, txt], spacing=12), padding=ft.Padding.symmetric(vertical=12, horizontal=14),
            border_radius=14, animate=ANIM, data=key,
            on_click=lambda e, k=key: self._show_view(k), on_hover=self._on_nav_hover,
        )
        self._track(box)
        self.nav_items[key] = (box, ic, txt)
        return box

    def _on_nav_hover(self, e) -> None:
        box: ft.Container = e.control
        if box.data == self.view_name:
            return
        box.bgcolor = self.palette.glass if _is_hovering(e) else CLEAR
        box.update()

    def _refresh_nav(self) -> None:
        p = self.palette
        for key, (box, ic, txt) in self.nav_items.items():
            active = key == self.view_name
            box.bgcolor = p.accent_soft if active else CLEAR
            box.border = ft.Border.all(1, with_alpha(p.accent, 0.42) if active else CLEAR)
            ic.color = p.accent_fg if active else p.text2
            txt.color = p.text if active else p.text2
            txt.weight = ft.FontWeight.W_600 if active else ft.FontWeight.W_500

    # -------------------------------------------------------------- workspace
    def _page_header(self, title: str, caption: str, *trailing: ft.Control) -> ft.Control:
        return ft.Container(
            margin=ft.Margin.only(left=4, bottom=SECTION_GAP + 4),
            content=ft.Row([
                ft.Column([self._title(title), self._text(caption, 13.5, role="text2")], spacing=2),
                ft.Container(expand=True), *trailing,
            ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
        )

    def _stat_tile(self, key: str, label: str) -> ft.Container:
        value = ft.Text("0", size=24, weight=ft.FontWeight.W_700)
        self._paint(value, "color", lambda p, k=key: p.stats[k])
        self.stat_values[key] = value
        tile = self._glass(
            ft.Column([value, self._text(label, 12, role="text2")], spacing=0,
                      horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=ft.Padding.symmetric(vertical=14, horizontal=8), radius=20, blur=False, expand=True,
            on_hover=self._on_tile_hover, animate_scale=ANIM_FAST,
        )
        return tile

    def _on_tile_hover(self, e) -> None:
        tile: ft.Container = e.control
        hovering = _is_hovering(e)
        tile.scale = 1.035 if hovering else 1.0
        tile.border = ft.Border.all(1, self.palette.border_hi if hovering else self.palette.border)
        tile.update()

    def _build_workspace_view(self) -> ft.Control:
        library = self._glass(
            ft.Column([
                ft.Row([self.folder_field, self.browse_btn], spacing=12,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Row([self.start_btn, self.stop_btn], spacing=12),
            ], spacing=16),
            padding=20, radius=24, margin=ft.Margin.only(bottom=SECTION_GAP),
        )

        # floating progress card: revealed by an animated size change while a batch runs
        self.progress_card = self._glass(
            ft.Column([
                ft.Row([self.progress_label, self.current_file, self.progress_stop_btn], spacing=14,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                self.progress_bar.control,
            ], spacing=14),
            padding=ft.Padding.symmetric(vertical=18, horizontal=20), radius=24, strong=True,
            margin=ft.Margin.only(bottom=SECTION_GAP), visible=False,
        )
        # accent-tinted glass (a glow *shadow* would show through the translucent fill)
        self._paint(self.progress_card, "gradient", lambda p: ft.LinearGradient(
            begin=ft.Alignment.TOP_LEFT, end=ft.Alignment.BOTTOM_RIGHT,
            colors=[with_alpha(p.accent, 0.30 if p.dark else 0.22),
                    with_alpha(p.accent, 0.09 if p.dark else 0.07)]))
        self._paint(self.progress_card, "border", lambda p: ft.Border.all(1, with_alpha(p.accent, 0.60)))
        self.progress_holder = ft.Container(
            content=self.progress_card, animate_size=ANIM, clip_behavior=ft.ClipBehavior.NONE,
        )

        stats = ft.Container(
            margin=ft.Margin.only(bottom=SECTION_GAP),
            content=ft.Row([self._stat_tile(k, label) for k, label in STAT_LABELS], spacing=12),
        )

        log_header = ft.Row([
            self._icon(ft.Icons.TERMINAL, 17),
            self._text("Activity log", 13, ft.FontWeight.W_600, role="text2"),
            ft.Container(expand=True),
            self._log_icon_button(ft.Icons.CONTENT_COPY, "Copy log", self.on_copy_log, 16),
            self._log_icon_button(ft.Icons.CLEAR_ALL, "Clear log", self.on_clear_log, 18),
        ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        log_surface = ft.Container(content=self.log_view, expand=True, border_radius=16)
        self._paint(log_surface, "bgcolor", "log_bg")
        self._paint(log_surface, "border", lambda p: ft.Border.all(1, p.field_border))
        log_card = self._glass(
            ft.Column([log_header, log_surface], spacing=10, expand=True),
            padding=ft.Padding.only(left=18, right=12, top=10, bottom=14), radius=24, expand=True,
        )

        return ft.Container(
            key="workspace", expand=True,
            content=ft.Column([
                self._page_header("Sync Workspace", "Synced lyrics for your local music library",
                                  self.status_pill),
                library, self.progress_holder, stats, log_card,
            ], spacing=0, expand=True),
        )

    def _log_icon_button(self, icon: Any, tip: str, handler: Callable, size: float) -> ft.IconButton:
        btn = ft.IconButton(icon=icon, icon_size=size, tooltip=tip, on_click=handler)
        self._paint(btn, "icon_color", "text2")
        return btn

    # --------------------------------------------------------------- settings
    def _card_header(self, icon: Any, title: str, caption: str) -> ft.Control:
        chip = ft.Container(content=self._icon(icon, 20, "accent_fg"), width=40, height=40,
                            border_radius=12, alignment=ft.Alignment.CENTER, animate=ANIM)
        self._track(chip)
        self._paint(chip, "bgcolor", "accent_soft")
        return ft.Row([chip, ft.Column([self._text(title, 17, ft.FontWeight.W_600),
                                        self._text(caption, 12.5, role="text2")], spacing=1)],
                      spacing=14, vertical_alignment=ft.CrossAxisAlignment.CENTER)

    @staticmethod
    def _switch_grid(*switches: ft.Switch) -> ft.Row:
        """Switches in equal-width cells that wrap to the available width."""
        return ft.Row([ft.Container(content=sw, width=SWITCH_WIDTH) for sw in switches],
                      wrap=True, spacing=0, run_spacing=2)

    def _subhead(self, value: str) -> ft.Control:
        return ft.Container(margin=ft.Margin.only(top=6), content=self._text(value, 13, ft.FontWeight.W_600, role="text2"))

    def _settings_card(self, icon: Any, title: str, caption: str, *body: ft.Control) -> ft.Container:
        return self._glass(
            ft.Column([self._card_header(icon, title, caption), *body], spacing=12,
                      horizontal_alignment=ft.CrossAxisAlignment.STRETCH),
            padding=22, radius=24, margin=ft.Margin.only(bottom=SECTION_GAP),
        )

    def _build_settings_view(self) -> ft.Control:
        appearance = self._settings_card(
            ft.Icons.PALETTE, "Appearance & Theme", "Pick a surface and an accent - changes apply instantly.",
            self._subhead("Appearance"),
            ft.Row([self._mode_segment(*seg) for seg in MODE_SEGMENTS], spacing=10),
            self._subhead("Accent colour"),
            self._build_swatch_row(),
            self.accent_caption,
            ft.Row([self.custom_hex_field]),
        )
        local_ai = self._settings_card(
            ft.Icons.PSYCHOLOGY, "Local AI (Whisper)", "Used when no online lyrics match the track.",
            ft.Row([self.model_dd, self.lang_dd], spacing=12),
            ft.Row([self.model_hint, self.apply_rec_btn], wrap=True, spacing=8,
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            self.token_field,
            self._switch_grid(self.sw_ai, self.sw_gpu, self.sw_vad),
        )
        pipeline = self._settings_card(
            ft.Icons.AUTO_FIX_HIGH, "Smart pipeline", "Search first, listen second, guess last.",
            self._switch_grid(self.sw_align, self.sw_mashup, self.sw_ident),
        )
        providers = self._settings_card(
            ft.Icons.QUEUE_MUSIC, "Lyrics providers", "Tier 1 is searched first, in this order.",
            self._subhead("Tier 1 - synced lyrics"),
            self._switch_grid(*[self.provider_switches[pid] for pid in TIER1_IDS]),
            self._subhead("Tier 2 - plain text (aligned to the audio)"),
            self._switch_grid(*[self.provider_switches[pid] for pid in TIER2_IDS]),
            ft.Row([self.genius_field, self.musixmatch_field], spacing=12),
        )
        batch = self._settings_card(
            ft.Icons.LIBRARY_MUSIC, "Batch behaviour", "How a sync run treats your files.",
            self._switch_grid(self.sw_recursive, self.sw_overwrite, self.sw_retry, self.sw_plain),
        )
        about = self._glass(
            ft.Row([
                self._logo(72, 20),
                ft.Column([
                    self._text(APP_TITLE, 20, ft.FontWeight.W_700, style=ft.TextStyle(letter_spacing=-0.4)),
                    self._text(f"Version {__version__}  ·  Liquid Glass interface", 12.5, role="text2"),
                    self.data_path_text, self.logs_path_text,
                ], spacing=3, expand=True),
            ], spacing=18, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            padding=22, radius=24, margin=ft.Margin.only(bottom=4),
        )
        self._refresh_appearance()
        self._refreshers.append(self._refresh_appearance)

        return ft.Container(
            key="settings", expand=True,
            content=ft.Column([
                self._page_header("Settings", "Appearance, local AI and the lyrics pipeline",
                                  self.clear_btn, self.save_btn),
                ft.Column([appearance, local_ai, pipeline, providers, batch, about],
                          scroll=ft.ScrollMode.AUTO, expand=True, spacing=0),
            ], spacing=0, expand=True),
        )

    # ---------------------------------------------------- appearance controls
    def _mode_segment(self, key: str, icon: Any, title: str, caption: str) -> ft.Container:
        ic = self._icon(icon, 22)
        head = self._text(title, 14.5, ft.FontWeight.W_600)
        sub = self._text(caption, 11.5, role="text2")
        seg = ft.Container(
            content=ft.Row([ic, ft.Column([head, sub], spacing=0)], spacing=12),
            expand=True, padding=ft.Padding.symmetric(vertical=14, horizontal=16), border_radius=16,
            animate=ANIM, data=key, tooltip=THEME_MODE_LABELS[key],
            on_click=lambda e, k=key: self.on_select_mode(k), on_hover=self._on_segment_hover,
        )
        self._track(seg)
        self.mode_segments[key] = seg
        return seg

    def _on_segment_hover(self, e) -> None:
        seg: ft.Container = e.control
        if seg.data == self.settings.theme_mode:
            return
        seg.bgcolor = self.palette.glass_hi if _is_hovering(e) else self.palette.field
        seg.update()

    def _build_swatch_row(self) -> ft.Control:
        items = [self._swatch(preset.id, preset.label) for preset in ACCENT_PRESETS]
        items.append(self._swatch(CUSTOM_ACCENT_ID, "Custom colour"))
        return ft.Row(items, spacing=14, wrap=True, run_spacing=10)

    def _swatch(self, key: str, tooltip: str) -> ft.Container:
        """A colour dot inside a selection ring; hover grows it a little."""
        dot = ft.Container(width=32, height=32, shape=ft.BoxShape.CIRCLE, animate=ANIM)
        if key == CUSTOM_ACCENT_ID:
            self._paint(dot, "gradient", lambda p: self._custom_gradient())
        else:
            self._paint(dot, "gradient", lambda p, k=key: self._swatch_gradient(k))
        ring = ft.Container(
            content=dot, width=46, height=46, shape=ft.BoxShape.CIRCLE, alignment=ft.Alignment.CENTER,
            animate=ANIM, animate_scale=ANIM_FAST, data=key, tooltip=tooltip,
            on_click=lambda e, k=key: self.on_select_accent(k), on_hover=self._on_swatch_hover,
        )
        self._track(ring)
        self._track(dot)
        self.swatches[key] = (ring, dot)
        return ring

    def _swatch_color(self, key: str) -> str:
        if key == CUSTOM_ACCENT_ID:
            return normalise_hex(self.settings.custom_accent)
        preset = find_accent_preset(key)
        return preset.color if preset else self.settings.accent_color()

    def _swatch_gradient(self, key: str) -> ft.LinearGradient:
        base = self._swatch_color(key)
        return ft.LinearGradient(begin=ft.Alignment.TOP_LEFT, end=ft.Alignment.BOTTOM_RIGHT,
                                 colors=[lighten(base, 0.18), base, darken(base, 0.22)])

    def _custom_gradient(self) -> Any:
        """Rainbow ring-fill while no custom colour is chosen, the colour itself once it is."""
        if self.settings.accent_preset == CUSTOM_ACCENT_ID:
            return self._swatch_gradient(CUSTOM_ACCENT_ID)
        rainbow = ["#FF453A", "#FF9F0A", "#FFD60A", "#32D74B", "#0A84FF", "#BF5AF2", "#FF453A"]
        sweep = getattr(ft, "SweepGradient", None)
        if sweep is not None:
            return sweep(colors=rainbow)
        return ft.LinearGradient(colors=rainbow)

    def _on_swatch_hover(self, e) -> None:
        ring: ft.Container = e.control
        ring.scale = 1.12 if _is_hovering(e) else 1.0
        ring.update()

    def _refresh_appearance(self) -> None:
        """Selection state of the mode segments and the accent swatches."""
        p = self.palette
        mode = self.settings.theme_mode
        for key, seg in self.mode_segments.items():
            on = key == mode
            row = seg.content
            icon, column = row.controls[0], row.controls[1]
            seg.bgcolor = p.accent_soft if on else p.field
            seg.border = ft.Border.all(1.5 if on else 1, with_alpha(p.accent, 0.60) if on else p.field_border)
            icon.color = p.accent_fg if on else p.text2
            column.controls[0].color = p.text
            column.controls[1].color = p.text2
        chosen = self.settings.accent_preset
        for key, (ring, dot) in self.swatches.items():
            on = key == chosen
            ring.border = ft.Border.all(2.5, p.text if on else CLEAR)
            dot.shadow = (ft.BoxShadow(blur_radius=18, color=p.accent_glow, offset=ft.Offset(0, 4))
                          if on else None)
        preset = find_accent_preset(chosen)
        self.accent_caption.value = (f"{preset.label}  -  {preset.caption}" if preset
                                     else f"Custom colour  -  #{self.settings.custom_accent.lstrip('#')}")

    # ============================================================ log helpers
    def _append_log(self, tag: str, message: str, level: str = "info") -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"{stamp}  [{tag}] {message}"
        self.log_buffer.append(line)
        self.log_view.controls.append(
            ft.Text(line, size=12, color=self.palette.tag_color(tag, level), font_family=MONO_FONT,
                    selectable=True, data=(tag, level))
        )
        if len(self.log_view.controls) > MAX_LOG_LINES:
            del self.log_view.controls[:TRIM_LOG_BY]

    def _snack(self, message: str) -> None:
        p = self.palette
        try:
            self.page.show_dialog(ft.SnackBar(
                content=ft.Text(message, color=p.text), bgcolor=p.surface,
                behavior=ft.SnackBarBehavior.FLOATING, shape=ft.RoundedRectangleBorder(radius=14),
                duration=4500))
        except Exception:
            try:
                self.page.show_dialog(ft.SnackBar(content=ft.Text(message), duration=4500))
            except Exception:
                logger.debug("snackbar failed", exc_info=True)

    # ===================================================== background tasks
    async def _startup(self) -> None:
        if self.settings.theme_mode == "system":     # the OS brightness is known by now
            self._apply_theme()
        try:
            data, logs = await asyncio.to_thread(lambda: (str(data_dir()), str(logs_dir())))
            self.data_path_text.value = f"Data folder: {data}"
            self.logs_path_text.value = f"Log files: {logs}"
        except Exception:
            logger.debug("could not resolve data folders", exc_info=True)
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
            synced = (data.get("synced_online", 0) + data.get("synced_aligned", 0)
                      + data.get("synced_ai", 0))
            stopped = self.cancel.is_set()
            verb = "Stopped" if stopped else "Finished"
            self._append_log(
                "OK", f"{verb} in {event.message}: {synced} synced, {data.get('skipped', 0)} skipped, "
                      f"{data.get('not_found', 0)} not found, {data.get('failed', 0)} failed.")
            self.progress_label.value = f"{verb}  ·  {synced} synced"
            if not stopped:
                self.progress_bar.value = 1.0
            self.current_file.value = ""
            self._status_after = "idle" if stopped else "done"
            self._snack(f"{verb}: {synced} track(s) synced.")
        elif event.kind == "finished":
            self._set_running(False)
        elif event.kind == "toast":
            self._snack(event.message)

    # ----------------------------------------------------------- run state
    def _set_status(self, status: str) -> None:
        self._status = status
        self._refresh_status()

    def _refresh_status(self) -> None:
        p = self.palette
        dot, label = {
            "idle": (p.text3, "Ready"),
            "running": (p.accent, "Syncing"),
            "done": (p.ok, "Finished"),
        }[self._status]
        self.status_dot.bgcolor = dot
        self.status_dot.shadow = (ft.BoxShadow(blur_radius=10, color=p.accent_glow)
                                  if self._status == "running" else None)
        self.status_text.value = label

    def _show_progress(self, visible: bool) -> None:
        self.progress_card.visible = visible

    def _set_running(self, running: bool) -> None:
        self._running = running
        self.start_btn.disabled = running
        self.stop_btn.disabled = not running
        self.progress_stop_btn.disabled = not running
        self.browse_btn.disabled = running
        self.folder_field.disabled = running
        self.clear_btn.disabled = running
        if running:
            self._set_status("running")
            self._show_progress(True)
        else:
            self.stop_btn.content = "Stop"
            self._set_status(self._status_after)
            self._status_after = "idle"
            self.page.run_task(self._collapse_progress_later)   # keep the result visible a moment

    async def _collapse_progress_later(self) -> None:
        run_id = self._run_id
        await asyncio.sleep(7)
        if run_id == self._run_id and not self._running:
            self._show_progress(False)
            self.page.update()

    # ================================================================ handlers
    def _show_view(self, name: str) -> None:
        if name == self.view_name:
            return
        self.view_name = name
        self.view_switcher.content = self.workspace_view if name == "workspace" else self.settings_view
        self._refresh_nav()
        self.page.update()

    def on_toggle_settings(self, e) -> None:
        self._show_view("workspace" if self.view_name == "settings" else "settings")

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
        self._save_bg(settings.copy())
        settings.apply_environment()

        self.cancel = threading.Event()
        self._run_id += 1
        self._status_after = "idle"
        self.progress_bar.value = 0
        self.progress_label.value = "Starting…"
        self.current_file.value = ""
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

    def _save_bg(self, settings: Settings, *, toast: bool = False) -> None:
        """Write settings.json on a short-lived thread (never on the UI loop)."""
        def job() -> None:
            ok = self.store.save(settings)
            if toast:
                self.events.put(Event("toast", message="Settings saved." if ok
                                      else "Could not write the settings file."))

        threading.Thread(target=job, name="lsl-save", daemon=True).start()

    def on_stop(self, e) -> None:
        self.cancel.set()
        self.stop_btn.disabled = True
        self.progress_stop_btn.disabled = True
        self._append_log("SYS", "Stopping after the current step…")

    # --- appearance ------------------------------------------------------------
    def on_select_mode(self, mode: str) -> None:
        if mode == self.settings.theme_mode:
            return
        self.settings.theme_mode = mode
        self._after_appearance_change()

    def on_toggle_theme(self, e) -> None:
        """Quick light/dark flip from the sidebar (the full choice lives in Settings)."""
        self.settings.theme_mode = "light" if self.palette.dark else "dark"
        self._after_appearance_change()

    def on_select_accent(self, key: str) -> None:
        if key == CUSTOM_ACCENT_ID:
            raw = (self.custom_hex_field.value or "").strip().lstrip("#")
            self.settings.custom_accent = normalise_hex(raw, self.settings.custom_accent)
        if key == self.settings.accent_preset and key != CUSTOM_ACCENT_ID:
            return
        self.settings.accent_preset = key
        self._after_appearance_change()

    def on_custom_accent_change(self, e) -> None:
        raw = (self.custom_hex_field.value or "").strip().lstrip("#")
        if len(raw) in (3, 6) and parse_hex(raw) is not None:
            self.custom_hex_field.error = None
            self.settings.custom_accent = normalise_hex(raw)
            self.settings.accent_preset = CUSTOM_ACCENT_ID
            self._after_appearance_change()

    def on_custom_accent_submit(self, e) -> None:
        raw = (self.custom_hex_field.value or "").strip().lstrip("#")
        if len(raw) in (3, 6) and parse_hex(raw) is not None:
            self.on_custom_accent_change(e)
        else:
            self.custom_hex_field.error = "Enter 3 or 6 hex digits, e.g. 1DB954"
            self.custom_hex_field.update()

    def _after_appearance_change(self) -> None:
        self._apply_theme()          # repaints every control, then page.update()
        self._persist_appearance()

    # --- settings -----------------------------------------------------------------
    def _collect_settings(self) -> Settings:
        s = self.settings.copy()          # keeps theme_mode / accent_* chosen in the UI
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
        s.use_forced_alignment = bool(self.sw_align.value)
        s.detect_mashups = bool(self.sw_mashup.value)
        s.identify_by_audio = bool(self.sw_ident.value)
        s.genius_token = (self.genius_field.value or "").strip()
        s.musixmatch_key = (self.musixmatch_field.value or "").strip()
        s.enabled_providers = [pid for pid, sw in self.provider_switches.items() if sw.value]
        s.last_folder = (self.folder_field.value or "").strip()
        s._normalise()  # also restores all providers if none are selected
        return s

    def on_save(self, e) -> None:
        self.settings = self._collect_settings()
        for pid, sw in self.provider_switches.items():  # reflect the normalised value
            sw.value = pid in self.settings.enabled_providers
        self.settings.apply_environment()
        self._save_bg(self.settings.copy(), toast=True)

    def on_apply_recommendation(self, e) -> None:
        if self.profile is not None:
            self.model_dd.value = self.profile.recommendation.model
            self._snack(f"Model set to {self.profile.recommendation.model} - press Save to keep it.")

    def on_clear_cache(self, e) -> None:
        p = self.palette

        def close_dialog(_e=None) -> None:
            self.page.pop_dialog()

        def confirm(_e=None) -> None:
            self.page.pop_dialog()

            def job() -> None:
                removed = clear_cache_file(database_path())
                self.events.put(Event("log", message=f"Cache cleared ({removed} entries).", tag="SYS"))
                self.events.put(Event("toast", message=f"Cache cleared ({removed} entries)."))

            threading.Thread(target=job, name="lsl-clear-cache", daemon=True).start()

        self.page.show_dialog(ft.AlertDialog(
            modal=True, bgcolor=p.surface, shape=ft.RoundedRectangleBorder(radius=24),
            title=ft.Text("Clear the lyrics cache?", color=p.text),
            content=ft.Text("Every track will be looked up again on the next run. "
                            "Existing .lrc files are not touched.", color=p.text2),
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


async def main(page: ft.Page) -> None:
    # Settings live on disk: read them off the UI loop before building the window.
    def load() -> tuple[SettingsStore, Settings]:
        store = SettingsStore()
        return store, store.load()

    store, settings = await asyncio.to_thread(load)
    LyricsApp(page, store, settings)


def run() -> None:
    assets = assets_dir()
    ft.run(main, assets_dir=str(assets) if assets.is_dir() else None)
