"""Phase 1 - Liquid Glass UI: colour maths, palettes, appearance settings, logo, GUI wiring."""

from __future__ import annotations

import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

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
def test_logo_svg_is_valid_and_renderer_friendly():
    svg = branding.logo_svg("#1DB954")
    ET.fromstring(svg)                                   # well-formed XML
    assert ' xmlns="http://www.w3.org/2000/svg"' in svg  # Qt/renderers detect SVG by its xmlns attribute
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

# --------------------------------------------------------------------------- Qt GUI
def _wait_for(predicate, timeout=5.0):
    """Poll *predicate*, pumping the Qt event loop so queued signals are delivered."""
    import time

    from PySide6.QtWidgets import QApplication

    end = time.time() + timeout
    while time.time() < end:
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
        try:
            if predicate():
                return True
        except (FileNotFoundError, ValueError, KeyError, RuntimeError):
            pass
        time.sleep(0.02)
    return False


def test_qss_helpers_and_theme_stylesheet():
    p = theme.build_palette(True, "#1DB954")
    assert theme.qss_hex(theme.with_alpha("#FFFFFF", 0.5)) == "#FFFFFF"
    assert theme.qss_color(theme.with_alpha("#FFFFFF", 0.5)) == "rgba(255, 255, 255, 50%)"
    qss = theme.build_qss(p)
    assert "QPushButton#Primary" in qss and "qlineargradient" in qss
    assert theme.qss_hex("#1DB954") in qss


def test_window_starts_dark_apple_blue(window):
    from PySide6.QtWidgets import QApplication

    assert window.palette.dark and window.palette.accent == "#0A84FF"
    assert window.settings.theme_mode == "dark"
    assert window.stack.count() == 4
    assert window.nav_buttons["workspace"].isChecked()
    assert "#0A84FF" in QApplication.instance().styleSheet().upper()


def test_accent_change_repaints_everything_and_is_persisted(window):
    from PySide6.QtWidgets import QApplication

    window._select_accent("spotify_green")
    assert window.palette.accent == "#1DB954"
    assert window.settings.accent_preset == "spotify_green"
    assert "#1DB954" in QApplication.instance().styleSheet().upper()
    assert "Spotify Green" in window.accent_caption.text()
    assert _wait_for(lambda: json.loads(window.store.path.read_text())["accent_preset"]
                     == "spotify_green")


def test_mode_switch_dark_light_system(window):
    window._select_mode("light")
    assert not window.palette.dark
    assert window.palette.glass == "#88FFFFFF"
    window._select_mode("system")
    assert window.settings.theme_mode == "system" and window.palette.dark   # offscreen -> dark
    window._toggle_theme()                                                   # quick flip
    assert window.settings.theme_mode == "light" and not window.palette.dark
    assert _wait_for(lambda: json.loads(window.store.path.read_text())["theme_mode"] == "light")


def test_custom_accent_field(window):
    window.custom_hex_field.setText("ff2d95")
    assert window.settings.accent_preset == CUSTOM_ACCENT_ID
    assert window.palette.accent == "#FF2D95"
    assert "FF2D95" in window.accent_caption.text()
    window.custom_hex_field.setText("zz")                       # incomplete input is ignored
    assert window.palette.accent == "#FF2D95"
    window._select_accent("apple_blue")
    assert window.palette.accent == "#0A84FF"
    window._select_accent(CUSTOM_ACCENT_ID)
    assert window.palette.accent == "#FF2D95"
    assert window.settings.accent_preset == CUSTOM_ACCENT_ID


def test_old_log_lines_are_recoloured_on_theme_change(window):
    window._append_log("OK", "done")
    window._append_log("ERR", "broken", "error")
    dark_ok = window.palette.tag_color("OK")
    window._select_mode("light")
    assert window.palette.tag_color("OK") != dark_ok
    assert window.palette.tag_color("ERR") == window.palette.tags["ERR"]
    assert [tag for _, tag, _, _ in window.log_entries][-2:] == ["OK", "ERR"]


def test_nav_and_segments_follow_selection(window):
    window._show_view("settings")
    assert window.stack.currentWidget() is window.settings_view
    assert window.nav_buttons["settings"].isChecked()
    window._select_accent("sunset_amber")
    assert window.palette.accent == "#FF9F0A"
    window._show_view("workspace")
    assert window.stack.currentWidget() is window.workspace_view
    assert window.nav_buttons["workspace"].isChecked()


def test_progress_card_and_status(window):
    from local_sync_lyrics.models import Event

    assert window.progress_card.isHidden()
    window._set_running(True)
    assert not window.progress_card.isHidden() and window.status_text.text() == "Syncing"
    window._handle_event(Event("progress", message="x.flac", current=1, total=4))
    assert window.progress_bar.value() == 250 and "1 / 4" in window.progress_label.text()
    window._handle_event(Event("done", message="1.0 s", data={"synced_online": 3}))
    window._handle_event(Event("finished"))
    assert window.status_text.text() == "Finished" and window.start_btn.isEnabled()


def test_nothing_in_the_gui_touches_the_disk_on_the_ui_loop(window, monkeypatch):
    """Saving settings / clearing the cache must hop to a thread, not block the handler."""
    import threading

    names = []
    real_save = window.store.save

    def spy(settings):
        names.append(threading.current_thread().name)
        return real_save(settings)

    monkeypatch.setattr(window.store, "save", spy)
    window._select_accent("midnight_purple")
    window._save_settings()
    assert _wait_for(lambda: len(names) >= 1)
    assert all(n != threading.main_thread().name for n in names)
