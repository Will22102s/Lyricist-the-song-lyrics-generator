"""LRC helpers: parse, validate, build, atomically write, and convert TTML.

Everything here is pure Python and I/O-free except :func:`write_text_atomic`.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

GENERATOR = "Local Sync Lyrics"

_TS_RE = re.compile(r"\[(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?\]")
_WORD_TS_RE = re.compile(r"<\d{1,3}:\d{1,2}(?:[.:]\d{1,3})?>")  # enhanced-LRC word tags
_OFFSET_RE = re.compile(r"^\[offset:\s*([+-]?\d+)\s*\]", re.IGNORECASE)
_MAX_TTML_BYTES = 5_000_000


@dataclass
class LrcLine:
    time: float
    text: str


# --------------------------------------------------------------------- parsing
def _to_seconds(minutes: str, seconds: str, fraction: Optional[str]) -> float:
    value = int(minutes) * 60 + int(seconds)
    if fraction:
        value += int(fraction) / (10 ** len(fraction))
    return float(value)


def parse_lrc(text: str) -> list[LrcLine]:
    """Parse LRC text into time-sorted lines (handles multi-stamp lines + offset)."""
    lines: list[LrcLine] = []
    offset = 0.0
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        raw = raw.strip().lstrip("\ufeff")
        if not raw:
            continue
        off = _OFFSET_RE.match(raw)
        if off:
            offset = int(off.group(1)) / 1000.0
            continue
        pos, stamps = 0, []
        while True:
            m = _TS_RE.match(raw, pos)
            if not m:
                break
            stamps.append(_to_seconds(m.group(1), m.group(2), m.group(3)))
            pos = m.end()
        if not stamps:
            continue  # metadata tag or plain text
        body = _WORD_TS_RE.sub("", raw[pos:]).strip()
        for stamp in stamps:
            lines.append(LrcLine(max(0.0, stamp + offset), body))
    lines.sort(key=lambda l: l.time)
    return lines


def strip_to_plain(lrc_text: str) -> str:
    """Timestamps/metadata removed; one lyric line per row."""
    out = [l.text for l in parse_lrc(lrc_text)]
    return "\n".join(out).strip()


# ------------------------------------------------------------------ validation
def validate_synced(
    text: str, duration: Optional[float] = None, min_lines: int = 4
) -> tuple[bool, str, list[LrcLine]]:
    """Decide whether *text* is believable synced lyrics for a track.

    Returns ``(ok, reason, parsed_lines)``.
    """
    lines = parse_lrc(text or "")
    texty = [l for l in lines if len(l.text.strip()) >= 2]
    if len(texty) < min_lines:
        return False, f"only {len(texty)} timed lines", lines
    if texty[-1].time <= 0.5:
        return False, "all timestamps are zero", lines
    distinct = {round(l.time, 1) for l in texty}
    if len(distinct) < min(len(texty), max(3, len(texty) // 3)):
        return False, "timestamps are not distinct", lines
    if duration and duration > 20:
        if texty[0].time > duration:
            return False, "first line starts after the track ends", lines
        if texty[-1].time > duration + 20:
            return False, (
                f"last timestamp {texty[-1].time:.0f}s exceeds track length {duration:.0f}s"
            ), lines
    return True, "ok", lines


# -------------------------------------------------------------------- building
def format_timestamp(seconds: float) -> str:
    total_cs = int(round(max(0.0, seconds) * 100))
    minutes, rest = divmod(total_cs, 6000)
    secs, cs = divmod(rest, 100)
    return f"[{minutes:02d}:{secs:02d}.{cs:02d}]"


def _clean_tag(value: str) -> str:
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def build_lrc(
    lines: list[LrcLine],
    *,
    artist: str = "",
    title: str = "",
    album: str = "",
    duration: Optional[float] = None,
    source: str = "",
) -> str:
    header: list[str] = []
    if title:
        header.append(f"[ti:{_clean_tag(title)}]")
    if artist:
        header.append(f"[ar:{_clean_tag(artist)}]")
    if album:
        header.append(f"[al:{_clean_tag(album)}]")
    if duration and duration > 0:
        m, s = divmod(int(round(duration)), 60)
        header.append(f"[length:{m:02d}:{s:02d}]")
    by = GENERATOR + (f" ({source})" if source else "")
    header.append(f"[by:{by}]")
    body = [f"{format_timestamp(l.time)}{l.text.strip()}" for l in lines]
    return "\n".join(header + body) + "\n"


def normalise_lrc(text: str, **meta) -> str:
    """Re-emit an arbitrary LRC document in clean, consistent form."""
    return build_lrc(parse_lrc(text), **meta)


# ---------------------------------------------------------------------- TTML
def _parse_ttml_time(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    value = value.strip()
    try:
        if value.endswith("ms"):
            return float(value[:-2]) / 1000.0
        if value.endswith("s"):
            return float(value[:-1])
        if value.endswith("m"):
            return float(value[:-1]) * 60
        if value.endswith("h"):
            return float(value[:-1]) * 3600
        parts = value.split(":")
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60 + float(part)
        return seconds
    except ValueError:
        return None


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


_SKIP_ROLES = {"x-translation", "x-roman", "x-transliteration"}


def _collect_text(element: ET.Element) -> str:
    for key, val in element.attrib.items():
        if _local(key) == "role" and val in _SKIP_ROLES:
            return ""
    parts = [element.text or ""]
    for child in element:
        parts.append(_collect_text(child))
        parts.append(child.tail or "")
    return "".join(parts)


def parse_ttml(xml_text: str) -> list[LrcLine]:
    """Convert TTML (``<p begin=...>`` lines) to :class:`LrcLine` objects."""
    if not xml_text or len(xml_text) > _MAX_TTML_BYTES:
        return []
    try:
        root = ET.fromstring(xml_text.strip())
    except ET.ParseError:
        return []
    lines: list[LrcLine] = []
    for element in root.iter():
        if _local(element.tag) != "p":
            continue
        begin = _parse_ttml_time(element.get("begin"))
        if begin is None:  # fall back to the first timed child span
            for child in element.iter():
                begin = _parse_ttml_time(child.get("begin"))
                if begin is not None:
                    break
        if begin is None:
            continue
        text = " ".join(_collect_text(element).split())
        if text:
            lines.append(LrcLine(begin, text))
    lines.sort(key=lambda l: l.time)
    return lines


# ------------------------------------------------------------------------ I/O
def write_text_atomic(path: Path, text: str) -> None:
    """Write UTF-8 text via a temp file + rename so a crash never leaves junk."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=".lsl-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        # mkstemp creates 0600 files; media servers/players often run as another
        # user, so keep the original mode (or a normal 0644) instead.
        try:
            if path.exists():
                shutil.copymode(path, tmp)
            else:
                os.chmod(tmp, 0o644)
        except OSError:
            pass
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
