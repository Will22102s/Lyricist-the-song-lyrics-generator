"""Read artist/title/album/duration from tags - and write lyrics back into them.

Reading
-------
Mutagen exposes different key names per container (ID3, MP4, Vorbis, ASF...).
We try the "easy" interface first and then a table of raw keys, and we never
let a damaged file raise out of this module.

Writing (:func:`embed_lyrics`)
------------------------------
=====================  ===========================================================
MP3                    ``USLT`` (plain text fallback) **and** ``SYLT`` (millisecond
                       timestamps, content type "lyrics"), UTF-16 so ID3v2.3 and
                       v2.4 players both read them
FLAC, OGG, OGA, Opus   ``LYRICS`` Vorbis comment holding the time-stamped LRC text
M4A, MP4, M4B          ``\xa9lyr`` atom holding the time-stamped LRC text
=====================  ===========================================================

Anything else (WAV, AIFF, WMA ...) reports ``supported=False`` so the caller can fall
back to a separate ``.lrc``. Embedding edits the user's audio file in place, so it is
only attempted with a fully validated result, every failure is reported instead of
raised, and the write is verified by reading the tag back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .language import iso639_2
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


# ===================================================================== embedding
MP3_EXTENSIONS = frozenset({".mp3"})
VORBIS_EXTENSIONS = frozenset({".flac", ".ogg", ".oga", ".opus"})
MP4_EXTENSIONS = frozenset({".m4a", ".mp4", ".m4b"})
EMBED_EXTENSIONS = MP3_EXTENSIONS | VORBIS_EXTENSIONS | MP4_EXTENSIONS
_MP4_LYRICS = "\xa9lyr"


@dataclass
class EmbedResult:
    """Outcome of :func:`embed_lyrics` (never raised, always returned)."""

    ok: bool
    supported: bool = True
    fields: list[str] = field(default_factory=list)   # e.g. ["USLT", "SYLT"]
    message: str = ""

    def describe(self) -> str:
        return " + ".join(self.fields) if self.fields else self.message


def supports_embedding(path: Path) -> bool:
    """True when :func:`embed_lyrics` knows this container."""
    return Path(path).suffix.lower() in EMBED_EXTENSIONS


def _plain_from_lrc(lrc_text: str) -> str:
    from .lrc_utils import strip_to_plain

    return strip_to_plain(lrc_text)


def _lrc_body(lrc_text: str) -> str:
    """Time-stamped lines only (no ``[ti:]``/``[by:]`` header, which players show as text)."""
    from .lrc_utils import format_timestamp, parse_lrc

    return "\n".join(f"{format_timestamp(l.time)}{l.text.strip()}" for l in parse_lrc(lrc_text)) + "\n"


def _embed_mp3(path: Path, lrc_text: Optional[str], plain: str, language: Optional[str]) -> list[str]:
    from mutagen.id3 import ID3, SYLT, USLT, Encoding, ID3NoHeaderError

    try:
        tags = ID3(str(path))
    except ID3NoHeaderError:
        tags = ID3()
    lang = iso639_2(language)
    tags.delall("USLT")
    tags.delall("SYLT")
    written = ["USLT"]
    tags.add(USLT(encoding=Encoding.UTF16, lang=lang, desc="", text=plain))
    if lrc_text:
        from .lrc_utils import parse_lrc

        stamped = [(l.text.strip(), int(round(l.time * 1000))) for l in parse_lrc(lrc_text) if l.text.strip()]
        if stamped:
            # format=2 -> absolute milliseconds, type=1 -> lyrics
            tags.add(SYLT(encoding=Encoding.UTF16, lang=lang, format=2, type=1, desc="", text=stamped))
            written.append("SYLT")
    version = 3 if tuple(getattr(tags, "version", (2, 4, 0)))[:2] == (2, 3) else 4
    tags.save(str(path), v2_version=version)
    return written


def _embed_vorbis(path: Path, text: str) -> list[str]:
    import mutagen

    audio = mutagen.File(str(path))
    if audio is None:
        raise ValueError("not a recognised Ogg/FLAC file")
    if audio.tags is None:
        audio.add_tags()
    for key in [k for k in list(audio.tags.keys()) if str(k).lower() in ("lyrics", "unsyncedlyrics")]:
        del audio.tags[key]
    audio.tags["LYRICS"] = [text]
    audio.save()
    return ["LYRICS"]


def _embed_mp4(path: Path, text: str) -> list[str]:
    from mutagen.mp4 import MP4

    audio = MP4(str(path))
    if audio.tags is None:
        audio.add_tags()
    audio.tags[_MP4_LYRICS] = [text]
    audio.save()
    return ["\xa9lyr"]


def embed_lyrics(path: Path, lrc_text: Optional[str] = None, *, plain_text: Optional[str] = None,
                 language: Optional[str] = None) -> EmbedResult:
    """Write lyrics into the audio file's own tags.

    Pass ``lrc_text`` for synced lyrics (a plain-text version is derived for ``USLT``), or
    only ``plain_text`` for unsynced ones. ``language`` (``"ja"``, ``"en"`` ...) fills the
    3-letter language of the ID3 frames.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in EMBED_EXTENSIONS:
        return EmbedResult(False, supported=False,
                           message=f"embedding is not supported for {suffix or 'this'} files")
    if not lrc_text and not (plain_text or "").strip():
        return EmbedResult(False, message="no lyrics to embed")
    plain = (plain_text or "").strip() or _plain_from_lrc(lrc_text or "")
    body = _lrc_body(lrc_text) if lrc_text else plain
    try:
        if suffix in MP3_EXTENSIONS:
            written = _embed_mp3(path, lrc_text, plain, language)
        elif suffix in VORBIS_EXTENSIONS:
            written = _embed_vorbis(path, body)
        else:
            written = _embed_mp4(path, body)
    except PermissionError:
        return EmbedResult(False, message="the file is read-only or locked by another program")
    except Exception as exc:  # mutagen raises many unrelated types for damaged files
        logger.debug("embedding failed for %s", path.name, exc_info=True)
        return EmbedResult(False, message=f"{exc.__class__.__name__}: {exc}")
    if not has_embedded_lyrics(path):                       # trust, but verify
        return EmbedResult(False, message="the tag was written but could not be read back")
    return EmbedResult(True, fields=written)


def read_embedded_lyrics(path: Path) -> Optional[str]:
    """The lyrics stored in the file's tags (synced LRC text when present), else ``None``."""
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix in MP3_EXTENSIONS:
            from mutagen.id3 import ID3, ID3NoHeaderError

            try:
                tags = ID3(str(path))
            except ID3NoHeaderError:
                return None
            sylt = tags.getall("SYLT")
            if sylt and sylt[0].text:
                from .lrc_utils import format_timestamp

                return "\n".join(f"{format_timestamp(ms / 1000.0)}{text}" for text, ms in sylt[0].text) + "\n"
            uslt = tags.getall("USLT")
            return str(uslt[0].text).strip() if uslt and str(uslt[0].text).strip() else None
        if suffix in VORBIS_EXTENSIONS:
            import mutagen

            audio = mutagen.File(str(path))
            for key in ("LYRICS", "UNSYNCEDLYRICS"):
                if audio is not None and audio.tags is not None:
                    for k in audio.tags.keys():
                        if str(k).upper() == key:
                            values = audio.tags[k]
                            text = values[0] if isinstance(values, list) else values
                            return str(text).strip() or None
            return None
        if suffix in MP4_EXTENSIONS:
            from mutagen.mp4 import MP4

            tags = MP4(str(path)).tags
            values = tags.get(_MP4_LYRICS) if tags else None
            return str(values[0]).strip() if values and str(values[0]).strip() else None
    except Exception:
        logger.debug("reading embedded lyrics failed for %s", path.name, exc_info=True)
    return None


def has_embedded_lyrics(path: Path) -> bool:
    return bool(read_embedded_lyrics(path))


# ===================================================================== Phase 3 tags
TAG_EXTENSIONS = MP3_EXTENSIONS | VORBIS_EXTENSIONS | MP4_EXTENSIONS


@dataclass
class TagWriteResult:
    """Outcome of :func:`write_tags` (never raised, always returned)."""

    ok: bool
    supported: bool = True
    fields: list[str] = field(default_factory=list)
    cover_written: bool = False
    message: str = ""

    def describe(self) -> str:
        parts = []
        if self.fields:
            parts.append(", ".join(self.fields))
        if self.cover_written:
            parts.append("cover art")
        return "; ".join(parts) or self.message


@dataclass
class SaveLyricsResult:
    """Outcome of :func:`save_lyrics` (the lyric editor's Save & Apply)."""

    ok: bool
    wrote_lrc: bool = False
    embedded: Optional[EmbedResult] = None
    message: str = ""

    def describe(self) -> str:
        parts = []
        if self.wrote_lrc:
            parts.append(".lrc updated")
        if self.embedded is not None and self.embedded.ok:
            parts.append(f"embedded ({self.embedded.describe()})")
        if self.embedded is not None and not self.embedded.ok:
            parts.append(self.embedded.message or "embedding failed")
        return "; ".join(parts) or self.message


def supports_tagging(path: Path) -> bool:
    return Path(path).suffix.lower() in TAG_EXTENSIONS


def _text_of(value: Any) -> str:
    """String content of an ID3/MP4/Vorbis value (for fill-missing checks)."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return _text_of(value[0]) if value else ""
    text = getattr(value, "text", None)
    if text:
        return _text_of(text)
    return str(value).strip()


def _wants(existing: Any, overwrite: bool) -> bool:
    return overwrite or not _text_of(existing)


def _write_mp3_tags(path: Path, meta: "TrackMetadata", cover: bytes, cover_mime: str,
                    overwrite: bool) -> tuple[list[str], bool]:
    from mutagen.id3 import (
        APIC, ID3, TALB, TCON, TDRC, TIT2, TPE1, TPE2, TRCK, Encoding, ID3NoHeaderError,
    )

    try:
        tags = ID3(str(path))
    except ID3NoHeaderError:
        tags = ID3()
    fields: list[str] = []
    frames = (
        ("TIT2", TIT2, meta.title),
        ("TPE1", TPE1, meta.artist),
        ("TPE2", TPE2, meta.album_artist or meta.artist),
        ("TALB", TALB, meta.album),
        ("TCON", TCON, meta.genre),
        ("TDRC", TDRC, meta.year),
        ("TRCK", TRCK, str(meta.track_number) if meta.track_number else ""),
    )
    for frame_id, frame_cls, value in frames:
        if not str(value).strip():
            continue
        if not _wants(tags.get(frame_id), overwrite):
            continue
        tags.setall(frame_id, [frame_cls(encoding=Encoding.UTF16, text=[str(value)])])
        fields.append(frame_id)
    cover_written = False
    if cover and (overwrite or not tags.getall("APIC")):
        tags.delall("APIC")
        tags.add(APIC(encoding=Encoding.UTF8, mime=cover_mime or "image/jpeg", type=3,
                      desc="Cover", data=cover))
        cover_written = True
    if fields or cover_written:
        version = 3 if tuple(getattr(tags, "version", (2, 4, 0)))[:2] == (2, 3) else 4
        tags.save(str(path), v2_version=version)
    return fields, cover_written


def _write_vorbis_tags(path: Path, meta: "TrackMetadata", cover: bytes, cover_mime: str,
                       overwrite: bool) -> tuple[list[str], bool]:
    import base64

    import mutagen
    from mutagen.flac import FLAC, Picture

    audio = mutagen.File(str(path))
    if audio is None:
        raise ValueError("not a recognised Ogg/FLAC file")
    if audio.tags is None:
        audio.add_tags()
    tags = audio.tags
    mapping = (
        ("TITLE", meta.title), ("ARTIST", meta.artist),
        ("ALBUMARTIST", meta.album_artist or meta.artist), ("ALBUM", meta.album),
        ("GENRE", meta.genre), ("DATE", meta.year),
        ("TRACKNUMBER", str(meta.track_number) if meta.track_number else ""),
    )
    fields: list[str] = []
    for key, value in mapping:
        if not str(value).strip():
            continue
        if not _wants(tags.get(key), overwrite):
            continue
        tags[key] = [str(value)]
        fields.append(key)
    cover_written = False
    if cover:
        if isinstance(audio, FLAC):
            if overwrite or not audio.pictures:
                audio.clear_pictures()
                picture = Picture()
                picture.type = 3
                picture.mime = cover_mime or "image/jpeg"
                picture.desc = "Cover"
                picture.data = cover
                audio.add_picture(picture)
                cover_written = True
        else:
            existing = tags.get("metadata_block_picture")
            if overwrite or not existing:
                picture = Picture()
                picture.type = 3
                picture.mime = cover_mime or "image/jpeg"
                picture.desc = "Cover"
                picture.data = cover
                tags["metadata_block_picture"] = [
                    base64.b64encode(picture.write()).decode("ascii")]
                cover_written = True
    if fields or cover_written:
        audio.save()
    return fields, cover_written


def _write_mp4_tags(path: Path, meta: "TrackMetadata", cover: bytes, cover_mime: str,
                    overwrite: bool) -> tuple[list[str], bool]:
    from mutagen.mp4 import MP4, MP4Cover

    audio = MP4(str(path))
    if audio.tags is None:
        audio.add_tags()
    tags = audio.tags
    mapping = (
        ("\xa9nam", meta.title), ("\xa9ART", meta.artist),
        ("aART", meta.album_artist or meta.artist), ("\xa9alb", meta.album),
        ("\xa9gen", meta.genre), ("\xa9day", meta.year),
    )
    fields: list[str] = []
    for atom, value in mapping:
        if not str(value).strip():
            continue
        if not _wants(tags.get(atom), overwrite):
            continue
        tags[atom] = [str(value)]
        fields.append(atom)
    if meta.track_number:
        if overwrite or not tags.get("trkn"):
            tags["trkn"] = [(int(meta.track_number), 0)]
            fields.append("trkn")
    cover_written = False
    if cover and (overwrite or not tags.get("covr")):
        image_format = (MP4Cover.FORMAT_PNG if "png" in (cover_mime or "").lower()
                        else MP4Cover.FORMAT_JPEG)
        tags["covr"] = [MP4Cover(cover, imageformat=image_format)]
        cover_written = True
    if fields or cover_written:
        audio.save()
    return fields, cover_written


def write_tags(path: Path, meta: "TrackMetadata", *, cover: bytes = b"",
               cover_mime: str = "", mode: str = "fill_missing") -> TagWriteResult:
    """Write canonical tags (and optionally a cover) into an audio file.

    ``mode="fill_missing"`` keeps every value that already exists; ``"overwrite_all"``
    replaces them. Cover art follows the same rule. Never raises.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in TAG_EXTENSIONS:
        return TagWriteResult(False, supported=False,
                              message=f"tag writing is not supported for {suffix or 'this'} files")
    overwrite = mode == "overwrite_all"
    payload = cover or meta.cover
    mime = cover_mime or meta.cover_mime
    try:
        if suffix in MP3_EXTENSIONS:
            fields, cover_written = _write_mp3_tags(path, meta, payload, mime, overwrite)
        elif suffix in VORBIS_EXTENSIONS:
            fields, cover_written = _write_vorbis_tags(path, meta, payload, mime, overwrite)
        else:
            fields, cover_written = _write_mp4_tags(path, meta, payload, mime, overwrite)
    except PermissionError:
        return TagWriteResult(False, message="the file is read-only or locked by another program")
    except Exception as exc:  # mutagen raises many unrelated types for damaged files
        logger.debug("tag writing failed for %s", path.name, exc_info=True)
        return TagWriteResult(False, message=f"{exc.__class__.__name__}: {exc}")
    if not fields and not cover_written:
        return TagWriteResult(True, message="nothing to change (all fields already present)")
    return TagWriteResult(True, fields=fields, cover_written=cover_written)


def save_lyrics(path: Path, lrc_text: str, *, output_mode: str = "lrc",
                language: Optional[str] = None) -> SaveLyricsResult:
    """The lyric editor's Save & Apply: refresh the ``.lrc`` and/or re-embed the tags.

    The ``.lrc`` is also refreshed when one already exists, so an embed-only setup
    never leaves a stale sidecar behind.
    """
    from .lrc_utils import validate_synced, write_text_atomic

    path = Path(path)
    lrc_path = path.with_suffix(".lrc")
    ok, reason, _ = validate_synced(lrc_text, None, min_lines=1)
    if not ok:
        return SaveLyricsResult(False, message=f"the edited lyrics are not valid ({reason})")
    wrote_lrc = False
    embedded: Optional[EmbedResult] = None
    try:
        if output_mode in ("lrc", "both") or lrc_path.exists():
            write_text_atomic(lrc_path, lrc_text)
            wrote_lrc = True
        if output_mode in ("embed", "both"):
            embedded = embed_lyrics(path, lrc_text, language=language)
    except OSError as exc:
        return SaveLyricsResult(False, wrote_lrc=wrote_lrc, embedded=embedded,
                                message=f"could not write the lyrics: {exc.strerror or exc}")
    if not wrote_lrc and (embedded is None or not embedded.ok):
        return SaveLyricsResult(False, embedded=embedded,
                                message=embedded.message if embedded else "nothing was written")
    return SaveLyricsResult(True, wrote_lrc=wrote_lrc, embedded=embedded)
