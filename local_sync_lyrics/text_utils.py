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
    """Normalise text for *comparison* (case, accents, punctuation).

    Accents are stripped from Latin letters only: removing combining marks from
    kana/Hangul/Indic text would destroy information (e.g. Japanese dakuten).
    """
    decomposed = unicodedata.normalize("NFKD", text or "")
    kept: list[str] = []
    for ch in decomposed:
        if unicodedata.combining(ch) and kept and ord(kept[-1]) < 0x250:
            continue  # accent on a Latin base letter
        kept.append(ch)
    # NFC re-composes kana/Hangul so voiced marks survive the \w filter below
    text = unicodedata.normalize("NFC", "".join(kept)).casefold().replace("&", " and ")
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


# =============================================================================
# Mashup / merged-track detection
# =============================================================================
#
# A "mashup" file name packs several songs into one title:
#     "Understand x Pluto Projector"        "Song A vs Song B"
#     "Song A / Song B"                     "Song A & Song B (Mashup)"
# We split on the delimiter, but only when the evidence is good enough, because
# many legitimate titles contain "&" or "/" ("Simon & Garfunkel", "AC/DC").
# Evidence is expressed as a confidence value the processor can act on.

_MASHUP_KEYWORD_RE = re.compile(
    r"\b(?:mash[\s-]?ups?|bootleg|mega[\s-]?mix|medley|blend|mixtape)\b", re.IGNORECASE
)
_MASHUP_BRACKET_RE = re.compile(
    r"\s*[\(\[\{]\s*[^)\]}]*\b(?:mash[\s-]?ups?|bootleg|mega[\s-]?mix|medley|blend)\b[^)\]}]*[\)\]\}]",
    re.IGNORECASE,
)
_MASHUP_PREFIX_RE = re.compile(
    r"^\s*(?:a\s+)?(?:mash[\s-]?up|medley|blend)\s+(?:of|between|with)\s+", re.IGNORECASE
)
_MASHUP_TRAILING_KW_RE = re.compile(
    r"\s*[-–—:]?\s*\b(?:mash[\s-]?ups?|bootleg|mega[\s-]?mix|medley)\b\s*$", re.IGNORECASE
)
# (name, regex fragment, confidence without keyword, confidence with keyword)
_MASHUP_DELIMITERS: tuple[tuple[str, str, float, float], ...] = (
    ("x", r"[x×✕]", 0.80, 0.95),
    ("vs", r"(?:vs\.?|versus|v\.s\.)", 0.80, 0.95),
    ("slash", r"/{1,2}", 0.55, 0.90),
    ("plus", r"\+", 0.55, 0.90),
    ("arrow", r"(?:→|->|=>)", 0.60, 0.90),
    ("&", r"(?:&|and)", 0.0, 0.90),        # only with an explicit mashup keyword
)
MASHUP_CONFIDENT = 0.75   # at/above: go straight to the component pipeline
MASHUP_POSSIBLE = 0.50    # at/above: try it if a normal full-title search fails
MAX_COMPONENTS = 6


@dataclass(frozen=True)
class MashupInfo:
    is_mashup: bool = False
    components: tuple[tuple[str, str], ...] = ()   # (artist, title); artist may be ""
    confidence: float = 0.0
    delimiter: str = ""
    keyword: bool = False
    cleaned_title: str = ""

    @property
    def has_named_components(self) -> bool:
        return len(self.components) >= 2

    def describe(self) -> str:
        names = " | ".join(f"{a} - {t}" if a else t for a, t in self.components)
        how = f"delimiter '{self.delimiter}'" if self.delimiter else "keyword"
        return f"{names or 'unnamed components'} ({how}, confidence {self.confidence:.0%})"


def _mask_brackets(text: str) -> tuple[str, list[str]]:
    groups: list[str] = []

    def repl(m: re.Match[str]) -> str:
        groups.append(m.group(0))
        return f"\x00{len(groups) - 1}\x00"

    return _BRACKET_RE.sub(repl, text), groups


def _unmask(text: str, groups: list[str]) -> str:
    return re.sub(r"\x00(\d+)\x00", lambda m: groups[int(m.group(1))], text)


def _valid_component(text: str) -> bool:
    letters = sum(c.isalpha() for c in text)
    return letters >= 2 and len(text) <= 90


def _component_from_piece(piece: str) -> tuple[str, str]:
    """``"Artist - Title"`` or just ``"Title"`` -> (artist, title)."""
    piece = strip_noise(clean_whitespace(piece))
    pieces = _SEP_RE.split(piece, maxsplit=1)
    if len(pieces) == 2 and _valid_component(pieces[0]) and _valid_component(pieces[1]):
        return sanitize_artist(pieces[0]), clean_whitespace(pieces[1])
    return "", piece


def _detect_in(text: str, keyword_hint: bool) -> MashupInfo:
    if not text:
        return MashupInfo()
    text = unicodedata.normalize("NFKC", text).replace("_", " ")
    keyword = keyword_hint or bool(_MASHUP_KEYWORD_RE.search(text))
    work = _MASHUP_BRACKET_RE.sub("", text)
    work = _MASHUP_PREFIX_RE.sub("", work)
    work = _MASHUP_TRAILING_KW_RE.sub("", work)
    work = strip_noise(clean_whitespace(work))
    masked, groups = _mask_brackets(work)

    for name, fragment, conf_plain, conf_kw in _MASHUP_DELIMITERS:
        confidence = conf_kw if keyword else conf_plain
        if confidence <= 0:
            continue
        parts = re.split(rf"\s+{fragment}\s+", masked, flags=re.IGNORECASE)
        if not 2 <= len(parts) <= MAX_COMPONENTS:
            continue
        pieces = [clean_whitespace(_unmask(p, groups)) for p in parts]
        if not all(_valid_component(p) for p in pieces):
            continue
        components = tuple(_component_from_piece(p) for p in pieces)
        return MashupInfo(True, components, confidence, name, keyword, work)

    if keyword:  # "Summer Mashup 2024": we know it is one, but not of what
        return MashupInfo(True, (), 0.60, "", True, work)
    return MashupInfo(cleaned_title=work)


def detect_mashup(title: str, artist: str = "", album: str = "",
                  filename_stem: str = "") -> MashupInfo:
    """Decide whether a track is a mashup / merged song and split its title.

    The artist field is deliberately *not* split (``"Calvin Harris x Dua Lipa"``
    is a collaboration, not a mashup). The title is tried first, then the file
    name, because tags are often truncated while the file name keeps the full
    ``"A x B"`` text.
    """
    keyword_hint = bool(_MASHUP_KEYWORD_RE.search(album or ""))
    candidates = [title]
    if filename_stem:
        stem = parse_filename(filename_stem)
        candidates += [stem.title, strip_track_number(strip_noise(filename_stem))]
    best = MashupInfo()
    for text in candidates:
        info = _detect_in(text or "", keyword_hint)
        if info.is_mashup and info.confidence > best.confidence:
            best = info
        if best.has_named_components:
            break
    return best


# =============================================================================
# Lyric-snippet helpers (song identification from audio)
# =============================================================================
_STOP = frozenset(
    "the a an and or but if i me my you your he she it we they them to of in on at "
    "for with is are was be oh ooh yeah uh la na hey ah woah whoa mm".split()
)


def best_query_phrase(text: str, words: int = 8) -> str:
    """Pick the most search-worthy run of words from a blind transcription.

    Prefers the line with the most *distinct, non-filler* words, then takes the
    first ``words`` words of it - rare words identify a song far better than
    "I love you baby".
    """
    best_line, best_score = "", 0
    for line in re.split(r"[\n.!?;]+", text or ""):
        tokens = [t for t in fold(line).split() if t]
        score = len({t for t in tokens if t not in _STOP and len(t) > 2})
        if score > best_score:
            best_line, best_score = " ".join(tokens), score
    return " ".join(best_line.split()[:words])


def snippet_match_score(snippet: str, lyrics: str) -> int:
    """0-100: does the *sung* snippet appear (fuzzily) inside this lyrics text?

    Used to VERIFY a candidate song before trusting it - a title match alone is
    not enough when identifying the pieces of a mashup.
    """
    a, b = fold(snippet), fold(lyrics)
    if len(a) < 12 or not b:
        return 0
    return int(fuzz.partial_ratio(a, b))
