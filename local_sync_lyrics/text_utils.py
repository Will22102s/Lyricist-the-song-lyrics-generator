"""Text clean-up used before any lyrics query is sent anywhere.

Typical dirty inputs this module understands::

    "01 - Queen - Bohemian Rhapsody (Official Video) [dQw4w9WgXcQ].mp3"
    "Daft Punk - Get Lucky (feat. Pharrell Williams) [Lyrics] | HD"
    "Artist_Name_-_Song_Title_[1080p]"
    tag artist="Queen"  tag title="Queen - Bohemian Rhapsody"
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from thefuzz import fuzz

# Words that, when they are the content of a (...) / [...] group, mark it as junk.
_NOISE = (
    r"official|music\s*video|video|audio|lyrics?|lyric\s*video|visuali[sz]er|"
    r"hd|hq|4k|8k|1080p|720p|480p|remaster(?:ed)?(?:\s*\d{4})?|full\s*(?:song|album)|"
    r"explicit|mv|m/v|topic|vevo|with\s+lyrics|karaoke|free\s*download|"
    r"out\s*now|download"
)
_NOISE_RE = re.compile(rf"\b(?:{_NOISE})\b", re.IGNORECASE)

# (...), [...], {...}, full-width and CJK brackets
_BRACKET_RE = re.compile(r"\s*[\(\[\{（［【「『]([^\)\]\}）］】」』]*)[\)\]\}）］】」』]")
_FEAT_RE = re.compile(
    r"\s*[\(\[]?\s*\b(?:feat(?:uring)?|ft|with)\b\.?\s+[^\)\]]*[\)\]]?\s*$", re.IGNORECASE
)
_TRAILING_TAG_RE = re.compile(
    r"\s*[\|\-–—:~]\s*(?:official\s+)?(?:music\s+)?"
    r"(?:lyrics?|lyric\s*video|audio|video|visuali[sz]er|hd|hq|4k)\s*$",
    re.IGNORECASE,
)
_TRAILING_QUALITY_RE = re.compile(r"(?:\s+(?:4k|8k|1080p|720p|480p|hd|hq))+\s*$", re.IGNORECASE)
_TRACK_NO_RE = re.compile(r"^\s*(?:\d{1,2}[-.]\d{1,2}\s+|\d{1,2}\s*[.)_\-]\s+|\d{1,2}\s+-\s+)")
_SEP_RE = re.compile(r"\s+[-–—~]\s+|\s*[｜|]\s*")
_ARTIST_JUNK_RE = re.compile(r"\s*(?:-\s*topic|vevo|official(?:\s+channel)?)\s*$", re.IGNORECASE)
_SPACES_RE = re.compile(r"\s+")


def clean_whitespace(text: str) -> str:
    return _SPACES_RE.sub(" ", text).strip()


def looks_like_youtube_id(text: str) -> bool:
    """11 chars of ``[A-Za-z0-9_-]`` that are not an ordinary word."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", text):
        return False
    if any(c.isdigit() or c in "-_" for c in text):
        return True
    return any(c.isupper() for c in text[1:]) and any(c.islower() for c in text)


def _strip_brackets(text: str, aggressive: bool) -> str:
    def repl(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if aggressive:
            return ""
        if not inner or looks_like_youtube_id(inner) or _NOISE_RE.search(inner):
            return ""
        return match.group(0)

    return _BRACKET_RE.sub(repl, text)


def strip_noise(text: str, aggressive: bool = False) -> str:
    """Remove YouTube/rip clutter from a title or file stem.

    ``aggressive=True`` also drops *every* bracketed group and ``feat.`` credits.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("_", " ")
    text = _strip_brackets(text, aggressive)
    text = _TRAILING_TAG_RE.sub("", text)
    text = _TRAILING_QUALITY_RE.sub("", text)
    if aggressive:
        text = _FEAT_RE.sub("", text)
    text = text.strip(" -–—~|:.")
    return clean_whitespace(text)


def strip_track_number(text: str) -> str:
    return clean_whitespace(_TRACK_NO_RE.sub("", text, count=1))


def fold(text: str) -> str:
    """Normalise text for *comparison* (case, accents, punctuation)."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.casefold().replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    return clean_whitespace(text)


def sanitize_artist(artist: str) -> str:
    artist = unicodedata.normalize("NFKC", artist or "").replace("_", " ")
    artist = _ARTIST_JUNK_RE.sub("", artist)
    return clean_whitespace(artist.strip(" -–—"))


def primary_artist(artist: str) -> str:
    """First credited artist ("A feat. B; C" -> "A"). Keeps "Simon & Garfunkel"."""
    parts = re.split(r"\s*(?:;|/|\bfeat\.?|\bft\.?|\bfeaturing\b|\bwith\b)\s*", artist, flags=re.I)
    return clean_whitespace(parts[0]) if parts and parts[0].strip() else artist


def sanitize_title(title: str, artist: str = "", aggressive: bool = False) -> str:
    """Clean a title and drop a redundant leading ``"Artist - "``."""
    title = strip_noise(title, aggressive=aggressive)
    title = strip_track_number(title)
    if artist:
        # Exact prefix: "Queen - Bohemian Rhapsody" with artist "Queen"
        pattern = rf"^\s*{re.escape(artist)}\s*[-–—:~]\s*"
        title = re.sub(pattern, "", title, count=1, flags=re.IGNORECASE)
        # Fuzzy prefix: left part of "X - Y" is (almost) the artist name.
        pieces = _SEP_RE.split(title, maxsplit=1)
        if len(pieces) == 2 and fuzz.ratio(fold(pieces[0]), fold(artist)) >= 88:
            title = pieces[1]
    return clean_whitespace(title)


@dataclass(frozen=True)
class ParsedName:
    artist: str
    title: str


def parse_filename(stem: str) -> ParsedName:
    """Best-effort ``Artist - Title`` split of a file name (without extension)."""
    cleaned = strip_noise(stem)
    cleaned = strip_track_number(cleaned)
    pieces = _SEP_RE.split(cleaned, maxsplit=1)
    if len(pieces) == 2 and pieces[0].strip() and pieces[1].strip():
        return ParsedName(sanitize_artist(pieces[0]), clean_whitespace(pieces[1]))
    return ParsedName("", cleaned)


def query_variants(artist: str, title: str, limit: int = 4) -> list[tuple[str, str]]:
    """Ordered, de-duplicated ``(artist, title)`` pairs to try, best guess first."""
    artist_clean = sanitize_artist(artist)
    title_clean = sanitize_title(title, artist_clean)
    title_hard = sanitize_title(title, artist_clean, aggressive=True)
    artist_main = primary_artist(artist_clean)

    candidates = [
        (artist_clean, title_clean),
        (artist_main, title_hard),
        (artist_clean, title_hard),
    ]
    if artist_clean:  # title-only search as a last resort (strict scoring applies)
        candidates.append(("", title_hard))

    seen: set[tuple[str, str]] = set()
    result: list[tuple[str, str]] = []
    for a, t in candidates:
        if not t:
            continue
        key = (fold(a), fold(t))
        if key in seen:
            continue
        seen.add(key)
        result.append((a, t))
        if len(result) >= limit:
            break
    return result


def similarity(a: str, b: str) -> int:
    """0-100 fuzzy similarity of two display strings."""
    fa, fb = fold(a), fold(b)
    if not fa or not fb:
        return 0
    return max(fuzz.token_set_ratio(fa, fb), fuzz.ratio(fa, fb))
