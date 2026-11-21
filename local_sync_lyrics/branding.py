"""Local Sync Lyrics brand mark: one geometry, two renderers.

The logo is a glass *squircle* holding a waveform whose left half is "already
sung" (bright) and a two-line lyric block - i.e. audio and words, in sync.

* :func:`logo_svg` returns an SVG string recoloured for any accent. The GUI feeds it
  straight to ``ft.Image`` (Flet detects SVG by its ``xmlns`` attribute), so the
  logo follows the selected theme colour live. Only features that ``flutter_svg``
  renders reliably are used: paths, gradients, clip paths - no filters, no CSS.
* :func:`render_png` / :func:`write_assets` draw the very same geometry with Pillow
  (imported lazily - Pillow is a build-time dependency only) to produce
  ``assets/logo.png`` and a multi-size ``assets/logo.ico`` for ``flet pack --icon``.
"""

from __future__ import annotations

import math
from pathlib import Path

from .theme import DEFAULT_ACCENT, lighten, darken, normalise_hex, on_color, parse_hex

CANVAS = 512            # design grid
TILE_HALF = 240         # squircle half-size (16 px breathing room)
SQUIRCLE_N = 5.0        # superellipse exponent: 4 = soft, 5 = Apple-like, inf = square

# waveform: (height) per bar; the first four bars are the "already sung" half
BAR_HEIGHTS = (70, 136, 100, 224, 156, 196, 88)
BAR_WIDTH = 30
BAR_PITCH = 52
BAR_CENTER_Y = 210
PLAYED_BARS = 4
# lyric block: (width, y) - left aligned with the first bar
LYRIC_LINES = ((270, 362), (164, 404))
LYRIC_HEIGHT = 24

ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)


# ---------------------------------------------------------------------- geometry
def squircle_points(cx: float = CANVAS / 2, cy: float = CANVAS / 2, half: float = TILE_HALF,
                    n: float = SQUIRCLE_N, count: int = 720) -> list[tuple[float, float]]:
    """Points on the superellipse ``|x/a|^n + |y/a|^n = 1`` (clockwise from the right)."""
    pts: list[tuple[float, float]] = []
    for i in range(count):
        t = 2 * math.pi * i / count
        c, s = math.cos(t), math.sin(t)
        x = cx + half * math.copysign(abs(c) ** (2 / n), c)
        y = cy + half * math.copysign(abs(s) ** (2 / n), s)
        pts.append((x, y))
    return pts


def squircle_path(cx: float = CANVAS / 2, cy: float = CANVAS / 2, half: float = TILE_HALF,
                  n: float = SQUIRCLE_N, samples: int = 48) -> str:
    """Smooth closed cubic-Bezier path through *samples* superellipse points."""
    pts = squircle_points(cx, cy, half, n, samples)
    m = len(pts)
    parts = [f"M{pts[0][0]:.2f} {pts[0][1]:.2f}"]
    for i in range(m):  # Catmull-Rom -> cubic Bezier
        p0, p1, p2, p3 = pts[(i - 1) % m], pts[i], pts[(i + 1) % m], pts[(i + 2) % m]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        parts.append(f"C{c1[0]:.2f} {c1[1]:.2f} {c2[0]:.2f} {c2[1]:.2f} {p2[0]:.2f} {p2[1]:.2f}")
    parts.append("Z")
    return "".join(parts)


def _bars() -> list[tuple[float, float, float, float, bool]]:
    """(x, y, w, h, played) for every waveform bar."""
    first = CANVAS / 2 - (len(BAR_HEIGHTS) - 1) * BAR_PITCH / 2 - BAR_WIDTH / 2
    return [(first + i * BAR_PITCH, BAR_CENTER_Y - h / 2, BAR_WIDTH, h, i < PLAYED_BARS)
            for i, h in enumerate(BAR_HEIGHTS)]


def _lyric_x() -> float:
    return _bars()[0][0]


def _palette(accent: str) -> dict[str, str]:
    accent = normalise_hex(accent, DEFAULT_ACCENT)
    return {
        "accent": accent,
        "hi": lighten(accent, 0.22),
        "lo": darken(accent, 0.30),
        "fg": on_color(accent),   # glyph colour that stays readable on this accent
    }


# --------------------------------------------------------------------------- SVG
def logo_svg(accent: str = DEFAULT_ACCENT) -> str:
    """The brand mark as a standalone SVG string tinted with *accent*."""
    p = _palette(accent)
    fg = p["fg"]
    path = squircle_path()
    bars = []
    for x, y, w, h, played in _bars():
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w}" height="{h}" rx="{w / 2:.0f}" '
                    f'fill="{fg}" fill-opacity="{0.95 if played else 0.50}"/>')
    lx = _lyric_x()
    lines = []
    for i, (w, y) in enumerate(LYRIC_LINES):
        lines.append(f'<rect x="{lx:.1f}" y="{y}" width="{w}" height="{LYRIC_HEIGHT}" '
                     f'rx="{LYRIC_HEIGHT / 2:.0f}" fill="{fg}" fill-opacity="{0.95 if i == 0 else 0.42}"/>')
    cx = CANVAS / 2
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS}" height="{CANVAS}" '
        f'viewBox="0 0 {CANVAS} {CANVAS}">'
        '<defs>'
        '<linearGradient id="tile" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{p["hi"]}"/>'
        f'<stop offset="0.52" stop-color="{p["accent"]}"/>'
        f'<stop offset="1" stop-color="{p["lo"]}"/></linearGradient>'
        '<linearGradient id="gloss" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#FFFFFF" stop-opacity="0.50"/>'
        '<stop offset="1" stop-color="#FFFFFF" stop-opacity="0"/></linearGradient>'
        '<radialGradient id="glow" cx="0.5" cy="0.5" r="0.5">'
        f'<stop offset="0" stop-color="{p["hi"]}" stop-opacity="0.65"/>'
        f'<stop offset="1" stop-color="{p["hi"]}" stop-opacity="0"/></radialGradient>'
        f'<clipPath id="clip"><path d="{path}"/></clipPath>'
        '</defs>'
        f'<path d="{path}" fill="url(#tile)"/>'
        f'<g clip-path="url(#clip)">'
        f'<circle cx="{cx}" cy="{BAR_CENTER_Y}" r="200" fill="url(#glow)"/>'
        f'<ellipse cx="{cx}" cy="-40" rx="330" ry="250" fill="url(#gloss)"/>'
        '</g>'
        f'<path d="{path}" fill="none" stroke="#FFFFFF" stroke-opacity="0.38" stroke-width="3"/>'
        + "".join(bars) + "".join(lines) +
        '</svg>'
    )


# ----------------------------------------------------------------------- Pillow
def render_png(accent: str = DEFAULT_ACCENT, size: int = 512):
    """Render the logo with Pillow (4x supersampled). Returns a ``PIL.Image`` (RGBA)."""
    from PIL import Image, ImageDraw  # lazy: only needed when building assets
    import numpy as np

    p = _palette(accent)
    ss = 4
    big = size * ss
    k = big / CANVAS

    def pt(points):
        return [(x * k, y * k) for x, y in points]

    rgb = lambda c: parse_hex(c) or (255, 255, 255)  # noqa: E731

    # squircle mask
    mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(mask).polygon(pt(squircle_points()), fill=255)

    # diagonal tile gradient hi -> accent -> lo
    yy, xx = np.mgrid[0:big, 0:big].astype("float32")
    t = (xx + yy) / (2.0 * (big - 1))
    hi, mid, lo = (np.array(rgb(p[key]), dtype="float32") for key in ("hi", "accent", "lo"))
    first = np.clip(t / 0.52, 0, 1)[..., None]
    second = np.clip((t - 0.52) / 0.48, 0, 1)[..., None]
    grad = np.where(t[..., None] < 0.52, hi + (mid - hi) * first, mid + (lo - mid) * second)
    tile = Image.fromarray(grad.astype("uint8"), "RGB").convert("RGBA")

    canvas = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    canvas.paste(tile, (0, 0), mask)

    # inner glow behind the waveform + top gloss, both clipped to the squircle
    def radial(cx, cy, r, color, peak):
        d = np.hypot(xx - cx * k, yy - cy * k) / (r * k)
        a = np.clip(1 - d, 0, 1) * peak * 255
        layer = np.zeros((big, big, 4), dtype="uint8")
        layer[..., :3] = rgb(color)
        layer[..., 3] = a.astype("uint8")
        return Image.fromarray(layer, "RGBA")

    glow = radial(CANVAS / 2, BAR_CENTER_Y, 200, p["hi"], 0.65)
    gloss_a = np.clip(1 - np.hypot((xx - CANVAS / 2 * k) / (330 * k),
                                   (yy - (-40) * k) / (250 * k)), 0, 1) * 0.50 * 255
    gloss_layer = np.zeros((big, big, 4), dtype="uint8")
    gloss_layer[..., :3] = 255
    gloss_layer[..., 3] = gloss_a.astype("uint8")
    gloss = Image.fromarray(gloss_layer, "RGBA")
    for layer in (glow, gloss):
        clipped = Image.new("RGBA", (big, big), (0, 0, 0, 0))
        clipped.paste(layer, (0, 0), mask)
        canvas = Image.alpha_composite(canvas, clipped)

    # specular rim
    rim = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    ImageDraw.Draw(rim).line(pt(squircle_points()) + [pt(squircle_points())[0]],
                             fill=(255, 255, 255, int(0.38 * 255)), width=max(1, int(3 * k)),
                             joint="curve")
    canvas = Image.alpha_composite(canvas, rim)

    # glyphs (waveform + lyric block)
    glyphs = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(glyphs)
    fg = rgb(p["fg"])
    for x, y, w, h, played in _bars():
        draw.rounded_rectangle([x * k, y * k, (x + w) * k, (y + h) * k], radius=w / 2 * k,
                               fill=(*fg, int((0.95 if played else 0.50) * 255)))
    lx = _lyric_x()
    for i, (w, y) in enumerate(LYRIC_LINES):
        draw.rounded_rectangle([lx * k, y * k, (lx + w) * k, (y + LYRIC_HEIGHT) * k],
                               radius=LYRIC_HEIGHT / 2 * k,
                               fill=(*fg, int((0.95 if i == 0 else 0.42) * 255)))
    canvas = Image.alpha_composite(canvas, glyphs)
    return canvas.resize((size, size), Image.Resampling.LANCZOS)


def write_assets(out_dir: str | Path, accent: str = DEFAULT_ACCENT) -> dict[str, Path]:
    """Write ``logo.svg``, ``logo.png`` (512) and a multi-size ``logo.ico`` into *out_dir*."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    svg = out / "logo.svg"
    svg.write_text(logo_svg(accent), encoding="utf-8")
    png = out / "logo.png"
    render_png(accent, 512).save(png, format="PNG", optimize=True)
    ico = out / "logo.ico"
    # Render every size natively (sharper than letting the ICO writer downscale one image).
    frames = [render_png(accent, s) for s in ICO_SIZES]
    frames[-1].save(ico, format="ICO", sizes=[(s, s) for s in ICO_SIZES],
                    append_images=frames[:-1])
    return {"svg": svg, "png": png, "ico": ico}
