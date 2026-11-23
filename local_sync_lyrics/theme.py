"""Colour maths and the *Liquid Glass* palette (pure Python - no Qt import).

Everything the GUI paints is derived from two inputs: whether the surface is dark
or light, and one accent colour. :func:`build_palette` turns those into a frozen
:class:`Palette`; the GUI re-applies a new palette to its controls whenever the
user changes the appearance mode or the accent - no restart needed.

Colour strings use the Qt-friendly ``#RRGGBB`` / ``#AARRGGBB`` convention
(alpha FIRST, so ``#18FFFFFF`` is white at ~9 % opacity).  :func:`build_qss`
turns a palette into the complete Qt Style Sheet used by the PySide6 front-end.
"""

from __future__ import annotations

import colorsys
import re
from dataclasses import dataclass, field

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")

DEFAULT_ACCENT = "#0A84FF"


# --------------------------------------------------------------------- colour maths
def parse_hex(value: object) -> tuple[int, int, int] | None:
    """``'#0A84FF'`` / ``'0a84ff'`` / ``'#08f'`` -> (10, 132, 255); ``None`` if invalid."""
    match = _HEX_RE.match(str(value or "").strip())
    if not match:
        return None
    digits = match.group(1)
    if len(digits) == 3:
        digits = "".join(ch * 2 for ch in digits)
    return int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16)


def to_hex(rgb: tuple[float, float, float]) -> str:
    r, g, b = (max(0, min(255, int(round(c)))) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def normalise_hex(value: object, default: str = DEFAULT_ACCENT) -> str:
    """Return a canonical upper-case ``#RRGGBB`` (or *default* when *value* is not a colour)."""
    rgb = parse_hex(value)
    return to_hex(rgb) if rgb else default


def with_alpha(color: str, alpha: float) -> str:
    """``with_alpha('#0A84FF', 0.5)`` -> ``'#800A84FF'`` (``#AARRGGBB``)."""
    rgb = parse_hex(color) or parse_hex(DEFAULT_ACCENT)
    a = max(0, min(255, int(round(alpha * 255))))
    return f"#{a:02X}{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def clear(color: str) -> str:
    """Same colour at 0 % alpha (gradient end-stops: avoids a grey fringe)."""
    return "#00" + str(color).lstrip("#")[-6:]


def mix(a: str, b: str, t: float) -> str:
    """Linear blend: ``t = 0`` -> *a*, ``t = 1`` -> *b*."""
    ra = parse_hex(a) or (0, 0, 0)
    rb = parse_hex(b) or (0, 0, 0)
    t = max(0.0, min(1.0, t))
    return to_hex(tuple(x + (y - x) * t for x, y in zip(ra, rb)))


def lighten(color: str, t: float) -> str:
    return mix(color, "#FFFFFF", t)


def darken(color: str, t: float) -> str:
    return mix(color, "#000000", t)


def shift_hue(color: str, degrees: float, *, saturation: float = 1.0, value: float = 1.0) -> str:
    """Rotate the hue (used for the secondary / tertiary ambient glows)."""
    r, g, b = (c / 255 for c in (parse_hex(color) or (10, 132, 255)))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + degrees / 360.0) % 1.0
    s = max(0.0, min(1.0, s * saturation))
    v = max(0.0, min(1.0, v * value))
    return to_hex(tuple(c * 255 for c in colorsys.hsv_to_rgb(h, s, v)))


def luminance(color: str) -> float:
    """WCAG relative luminance (0 = black, 1 = white)."""
    def lin(c: int) -> float:
        c /= 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = parse_hex(color) or (0, 0, 0)
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast_ratio(a: str, b: str) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def on_color(background: str) -> str:
    """Readable label colour on a solid *background*.

    White on Apple's saturated blue / pink / purple (what Apple itself ships), but a
    near-black label on light accents such as Spotify green or amber, where white
    text would fall below 3:1 contrast.
    """
    return "#FFFFFF" if luminance(background) < 0.30 else "#0B0B10"


# ----------------------------------------------------------------------- the palette
# Log tag colours: the dark set is the familiar terminal palette, the light set is its
# darker twin so the activity log stays readable on Frosted Pearl.
_TAGS_DARK = {
    "API": "#58A6FF", "AI": "#BC8CFF", "OK": "#3FB950", "WARN": "#D29922", "ERR": "#F85149",
    "SKIP": "#8B949E", "SYS": "#79C0FF", "INFO": "#E6EDF3", "Mashup Detected": "#F778BA",
    "Audio": "#56D4DD", "Output": "#A5D6FF",
}
_TAGS_LIGHT = {
    "API": "#0969DA", "AI": "#8250DF", "OK": "#1A7F37", "WARN": "#9A6700", "ERR": "#CF222E",
    "SKIP": "#6E7781", "SYS": "#0550AE", "INFO": "#1F2328", "Mashup Detected": "#BF3989",
    "Audio": "#0E7490", "Output": "#0969DA",
}
_STATS_DARK = {"synced_online": "#3FB950", "synced_aligned": "#56D4DD", "synced_ai": "#BC8CFF",
               "skipped": "#8B949E", "not_found": "#D29922", "failed": "#F85149"}
_STATS_LIGHT = {"synced_online": "#1A7F37", "synced_aligned": "#0E7490", "synced_ai": "#8250DF",
                "skipped": "#6E7781", "not_found": "#9A6700", "failed": "#CF222E"}


@dataclass(frozen=True)
class Palette:
    """Every colour the UI uses, for one (mode, accent) pair."""

    dark: bool
    accent: str
    accent_hi: str            # hover / gradient highlight
    accent_lo: str            # pressed / gradient shade
    on_accent: str            # label colour on a solid accent fill
    accent_fg: str            # accent nudged to stay legible as text / icon colour on glass
    accent_soft: str          # translucent accent tint (active pills, focus washes)
    accent_glow: str          # translucent accent used for outer glows

    bg_top: str               # ambient background gradient
    bg_bottom: str
    orb_a: str                # three ambient light blobs behind the glass
    orb_b: str
    orb_c: str

    glass: str                # card fill
    glass_hi: str             # card fill on hover
    glass_strong: str         # sidebar / floating surfaces
    sheen_top: str            # specular gradient painted inside glass
    sheen_bottom: str
    border: str               # 1px specular edge
    border_hi: str
    field: str                # text-field / dropdown fill
    field_border: str
    shadow: str
    surface: str              # opaque surface for dialogs / dropdown menus / snackbars

    text: str
    text2: str                # secondary captions
    text3: str                # tertiary / disabled

    log_bg: str
    ok: str
    warn: str
    err: str
    tags: dict[str, str] = field(default_factory=dict)
    stats: dict[str, str] = field(default_factory=dict)

    def tag_color(self, tag: str, level: str = "info") -> str:
        """Colour of an activity-log line (``API: Tier 1`` shares the colour of ``API``)."""
        if level == "warning":
            return self.tags["WARN"]
        if level == "error":
            return self.tags["ERR"]
        return (self.tags.get(tag) or self.tags.get(tag.split(":")[0].strip())
                or self.tags["INFO"])


def legible_on(color: str, background: str, minimum: float = 3.0) -> str:
    """Nudge *color* toward white/black until it reaches *minimum* contrast on *background*."""
    target = "#FFFFFF" if luminance(background) < 0.5 else "#000000"
    out = color
    for step in range(1, 21):
        if contrast_ratio(out, background) >= minimum:
            break
        out = mix(color, target, step * 0.05)
    return out


def build_palette(dark: bool, accent: str = DEFAULT_ACCENT) -> Palette:
    """Derive the full Liquid Glass palette from the surface brightness and one accent."""
    accent = normalise_hex(accent)
    accent_hi = lighten(accent, 0.18)
    accent_lo = darken(accent, 0.22)
    bg_probe = "#14141A" if dark else "#EEF0F6"      # typical surface the accent text sits on
    common = dict(
        dark=dark, accent=accent, accent_hi=accent_hi, accent_lo=accent_lo,
        on_accent=on_color(accent), accent_fg=legible_on(accent, bg_probe),
        accent_soft=with_alpha(accent, 0.22 if dark else 0.16),
        accent_glow=with_alpha(accent, 0.55 if dark else 0.38),
    )
    if dark:  # "Liquid Obsidian": near-black, faintly tinted by the accent
        return Palette(
            **common,
            bg_top=mix("#0C0C12", accent, 0.11), bg_bottom=mix("#050508", accent, 0.05),
            orb_a=with_alpha(accent, 0.42),
            orb_b=with_alpha(shift_hue(accent, 42, saturation=0.9), 0.28),
            orb_c=with_alpha(shift_hue(accent, -38, saturation=0.85), 0.22),
            glass="#18FFFFFF", glass_hi="#26FFFFFF", glass_strong="#1FFFFFFF",
            sheen_top="#26FFFFFF", sheen_bottom="#08FFFFFF",
            border="#33FFFFFF", border_hi="#59FFFFFF",
            field="#14FFFFFF", field_border="#2EFFFFFF", shadow="#73000000",
            surface=mix("#202026", accent, 0.06),
            text="#F5F5F7", text2="#A1A1A8", text3="#6C6C72",
            log_bg="#59000000",
            ok=_TAGS_DARK["OK"], warn=_TAGS_DARK["WARN"], err=_TAGS_DARK["ERR"],
            tags=dict(_TAGS_DARK), stats=dict(_STATS_DARK),
        )
    # "Frosted Pearl": soft off-white with an accent-tinted haze
    return Palette(
        **common,
        bg_top=mix("#F8F8FB", accent, 0.07), bg_bottom=mix("#E8EAF1", accent, 0.16),
        orb_a=with_alpha(accent, 0.34),
        orb_b=with_alpha(shift_hue(accent, 42, saturation=0.8), 0.24),
        orb_c=with_alpha(shift_hue(accent, -38, saturation=0.75), 0.20),
        glass="#88FFFFFF", glass_hi="#B0FFFFFF", glass_strong="#A3FFFFFF",
        sheen_top="#B8FFFFFF", sheen_bottom="#4DFFFFFF",
        border="#B3FFFFFF", border_hi="#E6FFFFFF",
        field="#99FFFFFF", field_border="#26000000", shadow="#26283250",
        surface=mix("#F6F6FA", accent, 0.04),
        text="#1C1C1E", text2="#4F4F58", text3="#8A8A92",
        log_bg="#B3FFFFFF",
        ok=_TAGS_LIGHT["OK"], warn=_TAGS_LIGHT["WARN"], err=_TAGS_LIGHT["ERR"],
        tags=dict(_TAGS_LIGHT), stats=dict(_STATS_LIGHT),
    )


def resolve_dark(mode: str, system_is_dark: bool) -> bool:
    """``dark`` / ``light`` are literal; ``system`` follows the OS."""
    if mode == "light":
        return False
    if mode == "dark":
        return True
    return bool(system_is_dark)


# ------------------------------------------------------------------- Qt helpers
def split_rgba(color: str) -> tuple[int, int, int, int]:
    """``#AARRGGBB`` / ``#RRGGBB`` -> ``(r, g, b, a)`` (a = 0..255)."""
    text = str(color or "").strip().lstrip("#")
    if len(text) == 8:
        a, r, g, b = (int(text[i:i + 2], 16) for i in (0, 2, 4, 6))
        return r, g, b, a
    rgb = parse_hex(text) or parse_hex(DEFAULT_ACCENT)
    return rgb[0], rgb[1], rgb[2], 255


def qss_color(color: str) -> str:
    """Qt Style Sheet colour: ``rgba(r, g, b, p%)`` (works for alpha colours too)."""
    r, g, b, a = split_rgba(color)
    return f"rgba({r}, {g}, {b}, {a * 100 // 255}%)"


def qss_hex(color: str) -> str:
    """Opaque ``#RRGGBB`` (drops any alpha)."""
    r, g, b, _ = split_rgba(color)
    return f"#{r:02X}{g:02X}{b:02X}"


def build_qss(p: Palette) -> str:
    """The complete Qt Style Sheet for *p*.

    Qt style sheets have no CSS transitions, so hover / checked / disabled states
    switch colour instantly. Everything visual - including the progress card and
    toasts - is defined here, so one call restyles the whole window.
    """
    accent, accent_hi, accent_lo = qss_hex(p.accent), qss_hex(p.accent_hi), qss_hex(p.accent_lo)
    on_accent = qss_hex(p.on_accent)
    bg_top, bg_bottom = qss_hex(p.bg_top), qss_hex(p.bg_bottom)
    accent_fg = qss_hex(p.accent_fg)
    text, text2, text3 = qss_hex(p.text), qss_hex(p.text2), qss_hex(p.text3)
    ok, warn, err = qss_hex(p.ok), qss_hex(p.warn), qss_hex(p.err)
    surface = qss_hex(p.surface)
    field = qss_color(p.field)
    field_border = qss_color(p.field_border)
    border = qss_color(p.border)
    border_hi = qss_color(p.border_hi)
    glass = qss_color(p.glass)
    glass_hi = qss_color(p.glass_hi)
    glass_strong = qss_color(p.glass_strong)
    accent_soft = qss_color(p.accent_soft)
    log_bg = qss_color(p.log_bg)
    return f"""
* {{
    font-family: "SF Pro Display", "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif;
    outline: none;
}}
QMainWindow {{
    background-color: {bg_bottom};
    color: {text};
}}
QWidget#Root {{
    background: qlineargradient(x1:0, y1:0, x2:0.55, y2:1,
        stop:0 {bg_top}, stop:1 {bg_bottom});
    color: {text};
}}
QWidget#Page, QStackedWidget, QWidget#SettingsInner {{
    background: transparent;
}}
QWidget {{
    color: {text};
    font-size: 13px;
}}
QToolTip {{
    background-color: {surface};
    color: {text};
    border: 1px solid {border_hi};
    border-radius: 8px;
    padding: 6px 9px;
}}
QFrame#Sidebar {{
    background-color: {glass_strong};
    border: 1px solid {border};
    border-radius: 28px;
}}
QFrame#Card {{
    background-color: {glass};
    border: 1px solid {border};
    border-radius: 18px;
}}
QFrame#CardStrong {{
    background-color: {glass_strong};
    border: 1px solid {border_hi};
    border-radius: 20px;
}}
QFrame#ProgressCard {{
    background-color: {accent_soft};
    border: 1px solid {qss_color(p.accent_glow)};
    border-radius: 20px;
}}
QFrame#StatusPill {{
    background-color: {field};
    border: 1px solid {field_border};
    border-radius: 15px;
}}
QFrame#LangChip {{
    background-color: {accent_soft};
    border: 1px solid {qss_color(p.accent_glow)};
    border-radius: 12px;
}}
QFrame#Banner {{
    background-color: {qss_color(with_alpha(p.warn, 0.16))};
    border: 1px solid {qss_color(with_alpha(p.warn, 0.55))};
    border-radius: 14px;
}}
QFrame#Toast {{
    background-color: {surface};
    border: 1px solid {border_hi};
    border-radius: 14px;
}}
QLabel#Title {{
    font-size: 30px;
    font-weight: 700;
    letter-spacing: -0.8px;
    color: {text};
}}
QLabel#Heading {{
    font-size: 17px;
    font-weight: 600;
    color: {text};
}}
QLabel#Caption {{
    font-size: 12.5px;
    color: {text2};
}}
QLabel#Tiny {{
    font-size: 11.5px;
    color: {text3};
}}
QLabel#SectionLabel {{
    font-size: 12.5px;
    font-weight: 600;
    color: {text2};
}}
QLabel#StatValue {{
    font-size: 24px;
    font-weight: 700;
}}
QLabel#StatCaption {{
    font-size: 11.5px;
    color: {text2};
}}
QLabel#Brand {{
    font-size: 15.5px;
    font-weight: 700;
    letter-spacing: -0.3px;
}}
QLabel#AccentCaption {{
    font-size: 12.5px;
    color: {text2};
}}
QLabel#BannerText {{
    color: {text};
    font-size: 12.5px;
}}
QPushButton {{
    background-color: transparent;
    color: {text};
    border: 1px solid transparent;
    border-radius: 12px;
    padding: 9px 14px;
    font-size: 13px;
}}
QPushButton:hover {{
    background-color: {glass_hi};
}}
QPushButton#Primary {{
    background-color: {accent};
    color: {on_accent};
    border: 1px solid {accent};
    border-radius: 13px;
    padding: 11px 22px;
    font-weight: 600;
}}
QPushButton#Primary:hover {{
    background-color: {accent_hi};
    border-color: {accent_hi};
}}
QPushButton#Primary:pressed {{
    background-color: {accent_lo};
    border-color: {accent_lo};
}}
QPushButton#Primary:disabled {{
    background-color: {field};
    border-color: {field_border};
    color: {text3};
}}
QPushButton#Glass {{
    background-color: {glass};
    color: {text};
    border: 1px solid {border};
    border-radius: 13px;
    padding: 11px 20px;
    font-weight: 500;
}}
QPushButton#Glass:hover {{
    background-color: {glass_hi};
    border-color: {border_hi};
}}
QPushButton#Glass:disabled {{
    color: {text3};
    background-color: {field};
    border-color: {field_border};
}}
QPushButton#Danger {{
    background-color: {glass};
    color: {err};
    border: 1px solid {qss_color(with_alpha(p.err, 0.45))};
    border-radius: 13px;
    padding: 11px 20px;
    font-weight: 500;
}}
QPushButton#Danger:hover {{
    background-color: {qss_color(with_alpha(p.err, 0.16))};
    border-color: {err};
}}
QPushButton#Danger:disabled {{
    color: {text3};
    background-color: {field};
    border-color: {field_border};
}}
QPushButton#Quiet {{
    background-color: transparent;
    color: {accent_fg};
    border: 1px solid transparent;
    border-radius: 11px;
    padding: 8px 12px;
}}
QPushButton#Quiet:hover {{
    background-color: {glass_hi};
}}
QPushButton#Nav {{
    background-color: transparent;
    color: {text2};
    border: 1px solid transparent;
    border-radius: 13px;
    padding: 12px 14px;
    text-align: left;
    font-size: 14px;
    font-weight: 500;
}}
QPushButton#Nav:hover {{
    background-color: {glass_hi};
    color: {text};
}}
QPushButton#Nav:checked {{
    background-color: {accent_soft};
    border: 1px solid {qss_color(with_alpha(p.accent, 0.42))};
    color: {text};
    font-weight: 600;
}}
QPushButton#Segment {{
    background-color: {field};
    color: {text2};
    border: 1px solid {field_border};
    border-radius: 13px;
    padding: 10px 12px;
    font-size: 12.5px;
    font-weight: 600;
    text-align: center;
}}
QPushButton#Segment:hover {{
    background-color: {glass_hi};
    color: {text};
}}
QPushButton#Segment:checked {{
    background-color: {accent_soft};
    border: 1.5px solid {qss_color(with_alpha(p.accent, 0.60))};
    color: {text};
}}
QPushButton#Segment:disabled {{
    color: {text3};
}}
QPushButton#ModeSegment {{
    background-color: {field};
    color: {text};
    border: 1px solid {field_border};
    border-radius: 15px;
    padding: 12px 15px;
    text-align: left;
    font-size: 13.5px;
    font-weight: 600;
}}
QPushButton#ModeSegment:hover {{
    background-color: {glass_hi};
}}
QPushButton#ModeSegment:checked {{
    background-color: {accent_soft};
    border: 1.5px solid {qss_color(with_alpha(p.accent, 0.60))};
}}
QPushButton#IconButton {{
    background-color: transparent;
    color: {text2};
    border: 1px solid transparent;
    border-radius: 10px;
    padding: 6px;
    font-size: 15px;
}}
QPushButton#IconButton:hover {{
    background-color: {glass_hi};
    color: {text};
}}
QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox {{
    background-color: {field};
    color: {text};
    border: 1px solid {field_border};
    border-radius: 11px;
    padding: 8px 10px;
    selection-background-color: {accent};
    selection-color: {on_accent};
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus {{
    border: 1px solid {accent};
}}
QLineEdit:disabled, QPlainTextEdit:disabled, QComboBox:disabled {{
    color: {text3};
}}
QLineEdit[error="true"] {{
    border: 1px solid {err};
}}
QPlainTextEdit#Log {{
    background-color: {log_bg};
    border: 1px solid {field_border};
    border-radius: 13px;
    padding: 8px;
    font-family: "SF Mono", "Cascadia Code", "Consolas", "DejaVu Sans Mono", monospace;
    font-size: 12px;
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox QAbstractItemView {{
    background-color: {surface};
    color: {text};
    border: 1px solid {border_hi};
    border-radius: 10px;
    selection-background-color: {accent_soft};
    selection-color: {text};
    padding: 4px;
}}
QCheckBox {{
    color: {text};
    spacing: 9px;
    font-size: 13px;
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 5px;
    border: 1px solid {field_border};
    background-color: {field};
}}
QCheckBox::indicator:hover {{
    border-color: {border_hi};
}}
QCheckBox::indicator:checked {{
    background-color: {accent};
    border: 1px solid {accent};
}}
QCheckBox:disabled {{
    color: {text3};
}}
QProgressBar {{
    background-color: {field};
    border: 1px solid {field_border};
    border-radius: 6px;
    height: 10px;
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {accent_lo}, stop:0.55 {accent}, stop:1 {accent_hi});
    border-radius: 5px;
}}
QScrollArea {{
    background: transparent;
    border: none;
}}
QScrollArea > QWidget > QWidget {{
    background: transparent;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {border_hi};
    border-radius: 4px;
    min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{
    background: {accent};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: {border_hi};
    border-radius: 4px;
    min-width: 28px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}
QDialog {{
    background-color: {surface};
    color: {text};
}}
QMessageBox {{
    background-color: {surface};
}}
QMenu {{
    background-color: {surface};
    color: {text};
    border: 1px solid {border_hi};
    border-radius: 10px;
}}
QMenu::item:selected {{
    background-color: {accent_soft};
}}
QStatusBar {{
    background: transparent;
    color: {text2};
}}
"""
