"""Read artist/title/album/duration from tags, falling back to the file name.

Mutagen exposes different key names per container (ID3, MP4, Vorbis, ASF...).
We try the "easy" interface first and then a table of raw keys, and we never
let a damaged file raise out of this module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from .models import TrackInfo
from .text_utils import parse_filename, sanitize_artist, sanitize_title

logger = logging.getLogger(__name__)

_RAW_KEYS = {
    "artist": ("artist", "TPE1", "\xa9ART", "Author", "WM/Author", "ARTIST", "albumartist", "TPE2", "aART"),
    "title": ("title", "TIT2", "\xa9nam", "Title", "WM/Title", "TITLE"),
    "album": ("album", "TALB", "\xa9alb", "WM/AlbumTitle", "ALBUM"),
}


def _first_text(tags: Any, keys: tuple[str, ...]) -> str:
    if not tags:
        return ""
    for key in keys:
        try:
            value = tags.get(key)
        except Exception:
            continue
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ""
        text = getattr(value, "text", None)  # ID3 frames
        if text:
            value = text[0] if isinstance(text, (list, tuple)) else text
        value = str(value).strip()
        if value:
            return value
    return ""


def _open_mutagen(path: Path):
    import mutagen

    for easy in (True, False):
        try:
            audio = mutagen.File(str(path), easy=easy)
        except TypeError:  # format without an "easy" variant
            continue
        except Exception as exc:  # corrupt / unsupported
            logger.debug("mutagen failed on %s (easy=%s): %s", path.name, easy, exc)
            continue
        if audio is not None:
            return audio
    return None


def _duration_via_pyav(path: Path) -> Optional[float]:
    try:
        import av

        with av.open(str(path)) as container:
            if container.duration:
                return float(container.duration) / 1_000_000.0
    except Exception:
        pass
    return None


def read_track_info(path: Path) -> TrackInfo:
    """Return a :class:`TrackInfo`. Never raises for unreadable tags."""
    artist = title = album = ""
    duration: Optional[float] = None

    audio = None
    try:
        audio = _open_mutagen(path)
    except Exception as exc:  # mutagen not importable, etc.
        logger.debug("tag reading unavailable: %s", exc)

    if audio is not None:
        tags = getattr(audio, "tags", None)
        artist = _first_text(tags, _RAW_KEYS["artist"])
        title = _first_text(tags, _RAW_KEYS["title"])
        album = _first_text(tags, _RAW_KEYS["album"])
        info = getattr(audio, "info", None)
        length = getattr(info, "length", None)
        if isinstance(length, (int, float)) and length > 0:
            duration = float(length)

    if duration is None:
        duration = _duration_via_pyav(path)

    parsed = parse_filename(path.stem)
    artist = sanitize_artist(artist) or parsed.artist
    # A tag title like "Queen - Bohemian Rhapsody" is cleaned of the artist prefix.
    title = sanitize_title(title, artist) if title else parsed.title
    if not title:
        title = parsed.title or path.stem
    return TrackInfo(
        path=path,
        artist=artist,
        title=title,
        album=album.strip(),
        duration=duration,
    )
