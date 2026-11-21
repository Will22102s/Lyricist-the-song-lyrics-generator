"""Phase 1 - Liquid Glass UI: colour maths, palettes, appearance settings, logo, GUI wiring."""

from __future__ import annotations

import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_sync_lyrics import branding, theme  # noqa: E402
from local_sync_lyrics.config import (  # noqa: E402
    ACCENT_PRESETS, CUSTOM_ACCENT_ID, DEFAULT_ACCENT_PRESET, SETTINGS_VERSION, Settings, SettingsStore,
)


# ------------------------------------------------------------------ colour maths
def test_hex_parsing_and_normalising():
    assert theme.parse_hex("#0a84ff") == (10, 132, 255)
    assert theme.parse_hex("1DB954") == (29, 185, 84)
    assert theme.parse_hex("#08f") == (0, 136, 255)
    for bad in ("", None, "#12", "#GGGGGG", "#12345678", "blue"):
        assert theme.parse_hex(bad) is None
    assert theme.normalise_hex("fa243c") == "#FA243C"
    assert theme.normalise_hex("nope", "#123456") == "#123456"


def test_alpha_mix_and_clear():
    assert theme.with_alpha("#FFFFFF", 0.0) == "#00FFFFFF"
    assert theme.with_alpha("#0A84FF", 1.0) == "#FF0A84FF"
    assert theme.with_alpha("#000000", 0.5) == "#80000000"
    assert theme.mix("#000000", "#FFFFFF", 0.5) == "#808080"
    assert theme.mix("#112233", "#445566", 0.0) == "#112233"
    assert theme.clear("#B31DB954") == "#001DB954" and theme.clear("#1DB954") == "#001DB954"


def test_label_colour_is_readable_on_every_preset():
    # Apple's own saturated colours keep white labels; Spotify green / amber need dark ones.
    assert theme.on_color("#0A84FF") == "#FFFFFF"
    assert theme.on_color("#FA243C") == "#FFFFFF"
    assert theme.on_color("#BF5AF2") == "#FFFFFF"
    assert theme.on_color("#1DB954") != "#FFFFFF"
    assert theme.on_color("#FF9F0A") != "#FFFFFF"
    for preset in ACCENT_PRESETS:
        assert theme.contrast_ratio(preset.color, theme.on_color(preset.color)) >= 3.4, preset.id


def test_legible_accent_text_even_for_extreme_custom_colours():
    for accent in ("#000000", "#FFFFFF", "#101010", "#F5F5F5"):
        dark, light = theme.build_palette(True, accent), theme.build_palette(False, accent)
        assert theme.contrast_ratio(dark.accent_fg, "#14141A") >= 3.0, accent
        assert theme.contrast_ratio(light.accent_fg, "#EEF0F6") >= 3.0, accent


# --------------------------------------------------------------------- palettes
@pytest.mark.parametrize("preset", ACCENT_PRESETS, ids=lambda p: p.id)
def test_palettes_for_every_preset_and_mode(preset):
    for dark in (True, False):
        p = theme.build_palette(dark, preset.color)
        assert p.accent == preset.color and p.dark is dark
        # body text must be readable on the ambient background
        assert theme.contrast_ratio(p.text, p.bg_top) >= 7
        assert theme.contrast_ratio(p.text2, p.bg_bottom) >= 4.5
        # every log tag stays readable on the log surface
        surface = "#101014" if dark else "#F4F6FA"
        for tag, colour in p.tags.items():
            assert theme.contrast_ratio(colour, surface) >= 3.0, (preset.id, dark, tag)
        # alpha colours are #AARRGGBB
        assert len(p.glass) == 9 and len(p.border) == 9 and len(p.orb_a) == 9


def test_dark_and_light_surfaces_and_accent_tint():
    dark, light = theme.build_palette(True), theme.build_palette(False)
    assert theme.luminance(dark.bg_top) < 0.05 < 0.6 < theme.luminance(light.bg_top)
    assert dark.glass == "#18FFFFFF" and light.glass == "#88FFFFFF"     # the specified glass fills
    assert dark.border == "#33FFFFFF"
    green = theme.build_palette(True, "#1DB954")
    r, g, b = theme.parse_hex(green.bg_top)
    assert g > r and g > b            # Spotify Green really tints the obsidian green
    assert theme.build_palette(True, "#FA243C").bg_top != green.bg_top


def test_tag_colours_and_mode_resolution():
    p = theme.build_palette(True)
    assert p.tag_color("API: Tier 1") == p.tags["API"]
    assert p.tag_color("AI: Forced Alignment") == p.tags["AI"]
    assert p.tag_color("Mashup Detected") == p.tags["Mashup Detected"]
    assert p.tag_color("anything", "warning") == p.tags["WARN"]
    assert p.tag_color("anything", "error") == p.tags["ERR"]
    assert p.tag_color("???") == p.tags["INFO"]
    assert theme.resolve_dark("dark", False) and not theme.resolve_dark("light", True)
    assert theme.resolve_dark("system", True) and not theme.resolve_dark("system", False)


# ----------------------------------------------------------------------- config
def test_appearance_defaults_match_the_spec():
    s = Settings()
    assert s.theme_mode == "dark" and s.accent_preset == DEFAULT_ACCENT_PRESET == "apple_blue"
    assert s.accent_color() == "#0A84FF"
    assert {p.id: p.color for p in ACCENT_PRESETS} == {
        "apple_blue": "#0A84FF", "spotify_green": "#1DB954", "apple_music": "#FA243C",
        "midnight_purple": "#BF5AF2", "sunset_amber": "#FF9F0A"}


def test_appearance_settings_validate_and_resolve():
    s = Settings.from_dict({"theme_mode": "system", "accent_preset": "spotify_green"})
    assert s.theme_mode == "system" and s.accent_color() == "#1DB954"
    s = Settings.from_dict({"accent_preset": "custom", "custom_accent": "ff2d95"})
    assert s.accent_preset == CUSTOM_ACCENT_ID and s.accent_color() == "#FF2D95"
    s = Settings.from_dict({"accent_preset": "neon", "custom_accent": "not-a-colour", "theme_mode": "x"})
    assert (s.accent_preset, s.custom_accent, s.theme_mode) == ("apple_blue", "#0A84FF", "dark")


def test_appearance_survives_a_save_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("LSL_HOME", str(tmp_path))
    store = SettingsStore()
    s = Settings(theme_mode="light", accent_preset="custom", custom_accent="#12ABEF")
    assert store.save(s)
    loaded = store.load()
    assert (loaded.theme_mode, loaded.accent_preset, loaded.accent_color()) == ("light", "custom", "#12ABEF")


def test_old_settings_files_still_load_and_keep_provider_choices(tmp_path, monkeypatch):
    """No schema bump: a v2 file without appearance keys loads with defaults and is NOT
    pushed through the legacy provider migration (which would re-enable disabled providers)."""
    monkeypatch.setenv("LSL_HOME", str(tmp_path))
    store = SettingsStore()
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(json.dumps({"settings_version": SETTINGS_VERSION, "theme_mode": "light",
                                      "enabled_providers": ["lrclib", "genius"]}))
    loaded = store.load()
    assert loaded.theme_mode == "light" and loaded.accent_preset == "apple_blue"
    assert loaded.enabled_providers == ["lrclib", "genius"]


# ---------------------------------------------------------------------- branding
def test_logo_svg_is_valid_and_flutter_friendly():
    svg = branding.logo_svg("#1DB954")
    ET.fromstring(svg)                                   # well-formed XML
    assert ' xmlns="http://www.w3.org/2000/svg"' in svg  # Flet detects SVG strings by this
    for unsupported in ("<filter", "<style", "<mask", "@import", "class="):
        assert unsupported not in svg
    assert svg.count("<rect") == len(branding.BAR_HEIGHTS) + len(branding.LYRIC_LINES)


def test_logo_follows_the_accent():
    blue, green = branding.logo_svg("#0A84FF"), branding.logo_svg("#1DB954")
    assert blue != green and "#0A84FF" in blue and "#1DB954" in green and "#1DB954" not in blue
    assert 'fill="#FFFFFF"' in blue                      # white glyphs on blue ...
    assert 'fill="#0B0B10"' in green                     # ... dark glyphs on bright green
    assert branding.logo_svg("junk") == branding.logo_svg(theme.DEFAULT_ACCENT)


def test_squircle_geometry():
    pts = branding.squircle_points(count=360)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    assert min(xs) == pytest.approx(16, abs=0.5) and max(xs) == pytest.approx(496, abs=0.5)
    assert min(ys) == pytest.approx(16, abs=0.5) and max(ys) == pytest.approx(496, abs=0.5)
    path = branding.squircle_path()
    assert path.startswith("M") and path.endswith("Z") and path.count("C") == 48


def test_icon_assets_are_generated(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    paths = branding.write_assets(tmp_path / "assets")
    png = Image.open(paths["png"])
    assert png.size == (512, 512) and png.mode == "RGBA"
    assert png.getpixel((2, 2))[3] == 0 and png.getpixel((256, 256))[3] == 255   # transparent corners
    ico = Image.open(paths["ico"])
    assert set(ico.info["sizes"]) == {(s, s) for s in branding.ICO_SIZES}
    assert ET.parse(paths["svg"]).getroot().tag.endswith("svg")


def test_shipped_assets_match_the_generator():
    root = Path(__file__).resolve().parents[1] / "assets"
    for name in ("logo.svg", "logo.png", "logo.ico"):
        assert (root / name).is_file(), name
    assert (root / "logo.svg").read_text(encoding="utf-8") == branding.logo_svg(theme.DEFAULT_ACCENT)


# --------------------------------------------------------------------------- GUI
class FakePage:
    def __init__(self, brightness=None):
        self.window = SimpleNamespace()
        self.services, self.added, self.tasks, self.dialogs = [], [], [], []
        self.theme_mode = None
        self.updates = 0
        if brightness is not None:
            self.platform_brightness = brightness

    def add(self, *controls):
        self.added.extend(controls)

    def update(self):
        self.updates += 1

    def run_task(self, fn):
        self.tasks.append(fn)

    def show_dialog(self, d):
        self.dialogs.append(d)

    def pop_dialog(self):
        self.dialogs.pop()


def _wait_for(predicate, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            if predicate():
                return True
        except (FileNotFoundError, ValueError, KeyError):   # file not written yet
            pass
        time.sleep(0.02)
    return False


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("LSL_HOME", str(tmp_path / "home"))
    from local_sync_lyrics import gui

    return gui.LyricsApp(FakePage())


def _encodes(control):
    import flet as ft
    import msgpack
    from flet.messaging.protocol import configure_encode_object_for_msgpack

    encode = configure_encode_object_for_msgpack(ft.BaseControl)
    return len(msgpack.packb(control, default=encode, use_bin_type=True))


def test_gui_starts_dark_apple_blue_and_theme_is_applied_to_the_page(app):
    import flet as ft

    assert app.palette.dark and app.palette.accent == "#0A84FF"
    assert app.page.theme_mode == ft.ThemeMode.DARK
    assert app.page.theme.color_scheme.primary == "#0A84FF"
    assert app.page.dark_theme.color_scheme.primary == "#0A84FF"
    assert len(app.page.tasks) == 2                      # exactly the pump + the startup task


def test_accent_change_repaints_everything_and_is_persisted(app):
    import flet as ft

    blue_swatch = app.swatches["apple_blue"][1].gradient.colors[1]
    app.on_select_accent("spotify_green")
    p = app.palette
    assert p.accent == "#1DB954" and app.settings.accent_preset == "spotify_green"
    assert app.page.theme.color_scheme.primary == "#1DB954"
    assert app.page.dark_theme.color_scheme.primary == "#1DB954"
    assert app.page.bgcolor == p.bg_bottom
    assert app.start_btn.style.bgcolor[ft.ControlState.DEFAULT] == "#1DB954"
    assert app.progress_bar.fill.gradient.colors[1] == "#1DB954"
    assert app.swatches["apple_blue"][1].gradient.colors[1] == blue_swatch == "#0A84FF"   # swatches keep their own colour
    assert "Spotify Green" in app.accent_caption.value
    logos = [ctl for ctl, attr, _fn in app._paints if attr == "src"]
    assert logos and all("#1DB954" in ctl.src and "#0A84FF" not in ctl.src for ctl in logos)
    assert _wait_for(lambda: json.loads(app.store.path.read_text())["accent_preset"] == "spotify_green")
    assert _encodes(app.page.added[0]) > 1000            # tree still serialises


def test_mode_switch_dark_light_system(app):
    import flet as ft

    app.on_select_mode("light")
    assert not app.palette.dark and app.page.theme_mode == ft.ThemeMode.LIGHT
    assert app.palette.glass == "#88FFFFFF"
    assert app.progress_bar.fill.gradient is not None
    app.on_select_mode("system")
    assert app.page.theme_mode == ft.ThemeMode.SYSTEM and app.palette.dark   # unknown OS -> dark
    app.page.platform_brightness = ft.Brightness.LIGHT
    app._on_system_brightness(None)
    assert not app.palette.dark                                              # follows the OS
    app.on_toggle_theme(None)                                                # quick flip -> dark
    assert app.settings.theme_mode == "dark" and app.palette.dark
    assert _wait_for(lambda: json.loads(app.store.path.read_text())["theme_mode"] == "dark")


def test_light_dark_flip_pauses_then_restores_animations(app):
    assert app._animated, "animated controls must be tracked"
    app.on_select_mode("light")
    assert all(c.animate is None for c, _ in app._animated)       # no half-faded text
    assert len(app.page.tasks) == 3                               # restore task scheduled
    import asyncio
    asyncio.run(app._restore_animations())
    assert all(c.animate is not None for c, _ in app._animated)
    n = len(app.page.tasks)
    app.on_select_accent("apple_music")                           # accent-only change keeps fades
    assert all(c.animate is not None for c, _ in app._animated) and len(app.page.tasks) == n


def test_custom_accent_field(app):
    app.custom_hex_field.value = "ff2d95"
    app.on_custom_accent_change(None)
    assert app.settings.accent_preset == CUSTOM_ACCENT_ID and app.palette.accent == "#FF2D95"
    assert "FF2D95" in app.accent_caption.value
    app.custom_hex_field.value = "zz"
    app.on_custom_accent_change(None)                             # incomplete input is ignored
    assert app.palette.accent == "#FF2D95"
    app.on_select_accent("apple_blue")                            # pick a preset ...
    assert app.palette.accent == "#0A84FF"
    app.custom_hex_field.value = "FF2D95"
    app.on_select_accent("custom")                                # ... and come back to the custom colour
    assert app.palette.accent == "#FF2D95" and app.settings.accent_preset == CUSTOM_ACCENT_ID


def test_old_log_lines_are_recoloured_on_theme_change(app):
    app._append_log("OK", "done")
    app._append_log("ERR", "broken", "error")
    dark_ok = app.log_view.controls[-2].color
    app.on_select_mode("light")
    assert app.log_view.controls[-2].color != dark_ok
    assert app.log_view.controls[-2].color == app.palette.tags["OK"]
    assert app.log_view.controls[-1].color == app.palette.tags["ERR"]


def test_nav_and_segments_follow_selection_and_theme(app):
    app._show_view("settings")
    assert app.view_name == "settings" and app.view_switcher.content is app.settings_view
    active = app.nav_items["settings"][0].bgcolor
    assert active == app.palette.accent_soft
    app.on_select_accent("sunset_amber")                          # nav highlight re-tints with the accent
    assert app.nav_items["settings"][0].bgcolor == app.palette.accent_soft != active
    assert app.nav_items["workspace"][0].bgcolor == "#00000000"
    app._show_view("workspace")
    assert app.view_switcher.content is app.workspace_view
    assert _encodes(app.page.added[0]) > 1000


def test_progress_card_and_smooth_bar(app):
    from local_sync_lyrics.models import Event

    assert app.progress_card.visible is False
    app._set_running(True)
    assert app.progress_card.visible is True and app.status_text.value == "Syncing"
    app.events.put(Event("progress", message="x.flac", current=1, total=4))
    app._drain_events()
    assert app.progress_bar.value == 0.25 and "1 / 4" in app.progress_label.value
    app.progress_bar._on_size(SimpleNamespace(width=400))
    assert app.progress_bar.fill.width == pytest.approx(100.0)
    app.progress_bar.value = 1.0
    assert app.progress_bar.fill.width == pytest.approx(400.0)
    app.events.put(Event("done", message="1.0 s", data={"synced_online": 3}))
    app.events.put(Event("finished"))
    app._drain_events()
    assert app.status_text.value == "Finished" and not app.start_btn.disabled


def test_nothing_in_the_gui_touches_the_disk_on_the_ui_loop(app, monkeypatch):
    """Saving settings / clearing the cache must hop to a thread, not block the handler."""
    import threading

    names = []
    real_save = app.store.save

    def spy(settings):
        names.append(threading.current_thread().name)
        return real_save(settings)

    monkeypatch.setattr(app.store, "save", spy)
    app.on_select_accent("midnight_purple")
    app.on_save(None)
    assert _wait_for(lambda: len(names) >= 2)
    assert all(n != threading.main_thread().name for n in names)
