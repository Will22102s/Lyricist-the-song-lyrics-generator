"""Generate the Lyricist brand assets (needs Pillow).

    python make_icons.py                    # assets/logo.svg + logo.png (512) + logo.ico (16-256 px)
    python make_icons.py --accent 1DB954    # same logo in another accent colour
    python make_icons.py --out some/folder

The artwork itself lives in ``local_sync_lyrics/branding.py`` (one geometry, drawn by
both the in-app SVG and this Pillow renderer), so the executable icon, the Windows
title-bar / taskbar icon and the logo inside the window always match.
``build.py`` calls :func:`ensure_assets` before packaging, so PyInstaller always
finds a fresh ``assets/logo.ico``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
OUTPUTS = ("logo.svg", "logo.png", "logo.ico", "icon.png", "icon.ico")
SOURCES = (ROOT / "local_sync_lyrics" / "branding.py", ROOT / "local_sync_lyrics" / "theme.py")


def is_stale(out_dir: Path = ASSETS) -> bool:
    """True when an asset is missing or older than the code that draws it."""
    files = [out_dir / name for name in OUTPUTS]
    if not all(f.is_file() for f in files):
        return True
    newest_source = max(src.stat().st_mtime for src in SOURCES if src.is_file())
    return min(f.stat().st_mtime for f in files) < newest_source


def generate(out_dir: Path = ASSETS, accent: str | None = None) -> dict[str, Path]:
    try:
        import PIL  # noqa: F401
    except ImportError:
        raise SystemExit("Pillow is required to generate the icons:  pip install pillow")
    sys.path.insert(0, str(ROOT))
    from local_sync_lyrics.branding import write_assets
    from local_sync_lyrics.theme import DEFAULT_ACCENT, normalise_hex

    return write_assets(out_dir, normalise_hex(accent, DEFAULT_ACCENT) if accent else DEFAULT_ACCENT)


def ensure_assets(out_dir: Path = ASSETS, *, force: bool = False) -> Path:
    """(Re)generate the default-accent assets when missing, stale or *force*d."""
    if force or is_stale(out_dir):
        paths = generate(out_dir)
        print("Generated:", ", ".join(str(p.relative_to(ROOT)) for p in paths.values()))
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ASSETS, help="output folder (default: assets/)")
    parser.add_argument("--accent", help="accent colour, e.g. 1DB954 (default: Apple Blue 0A84FF)")
    args = parser.parse_args(argv)
    paths = generate(args.out, args.accent)
    for kind, path in paths.items():
        print(f"{kind.upper():4} {path}  ({path.stat().st_size / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
