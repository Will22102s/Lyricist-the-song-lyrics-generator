"""LRC helpers: parse, validate, build, atomically write, and convert TTML.

Everything here is pure Python and I/O-free except :func:`write_text_atomic`.
"""

from __future__ import annotations

import difflib
import html
import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from thefuzz import fuzz, process

from .text_utils import _STOP, fold

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


# =============================================================================
# Plain-text lyrics clean-up (Tier 2 providers)
# =============================================================================
# [Chorus] / [Verse 1: Artist] are headers. "(ooh, ooh)" in parentheses is a SUNG
# background line, so round brackets only count when they name a song section.
_SECTION_RE = re.compile(
    r"^\s*(?:\[[^\]]{1,70}\]|\(\s*(?:verse|chorus|pre-?chorus|post-?chorus|bridge|intro|outro|hook|"
    r"refrain|interlude|instrumental|break|solo|drop|repeat)\b[^)]{0,60}\))\s*$",
    re.IGNORECASE,
)
_BOILERPLATE_RE = re.compile(
    r"^\s*(?:you might also like|see .{0,60} live|get tickets.*|\d*\s*embed|"
    r"\d+\s*contributors?.*|.{0,80}\blyrics$|translations?|read more|"
    r"paroles de la chanson.*|\*{3,}.*commercial use.*\*{3,}|.*lyrics powered by.*|"
    r"submit corrections|the lyrics for this song have yet to be released.*)\s*$",
    re.IGNORECASE,
)
_REPEAT_MARK_RE = re.compile(r"\s*[\(\[]?\s*x\s*\d{1,2}\s*[\)\]]?\s*$", re.IGNORECASE)


def clean_plain_lyrics(text: str) -> str:
    """Strip scraper debris so only the *sung* words remain.

    Removes ``[Chorus]``-style section headers, Genius/Musixmatch boilerplate,
    ``(x2)`` repeat markers and blank-line runs. The result is the ground truth
    the aligner snaps the audio to - headers would otherwise be "words" that are
    never sung.
    """
    if not text:
        return ""
    text = html.unescape(text).replace("\r\n", "\n").replace("\r", "\n").replace("\u200b", "")
    kept: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        if _SECTION_RE.match(line) or _BOILERPLATE_RE.match(line):
            continue
        line = _REPEAT_MARK_RE.sub("", line).strip()
        if line:
            kept.append(line)
    return "\n".join(kept).strip()


def plain_to_lines(text: str) -> list[str]:
    """Non-empty cleaned lyric lines."""
    return [ln for ln in clean_plain_lyrics(text).split("\n") if ln.strip()]


def looks_like_lyrics(text: str, min_lines: int = 4, min_words: int = 20) -> bool:
    """Cheap sanity check against error pages, one-line stubs and HTML soup."""
    lines = plain_to_lines(text)
    if len(lines) < min_lines:
        return False
    words = sum(len(ln.split()) for ln in lines)
    markup = sum(ln.count("<") + ln.count("{") for ln in lines)
    return words >= min_words and markup < max(3, len(lines) // 4)


_CREDIT_RE = re.compile(
    r"^\s*(?:作词|作曲|编曲|制作人?|词|曲|歌手|演唱|lyrics?|lyricist|music|composer|composed|"
    r"written|words|arranged?|arranger|produced?|producer|vocals?)\s*(?:by)?\s*[:：]",
    re.IGNORECASE,
)


def drop_credit_lines(lines: list[LrcLine]) -> list[LrcLine]:
    """Remove leading '作词 : …' / 'Composed by: …' rows some services prepend."""
    i = 0
    while i < len(lines) and i < 8 and _CREDIT_RE.match(lines[i].text):
        i += 1
    return lines[i:]


# =============================================================================
# Text-guided forced alignment
# =============================================================================
#
# Problem: Whisper hears words well enough to know *when* things are sung, but it
# mis-spells and mishears. The official lyrics are spelled right but have no
# timing. The aligner joins the two:
#
#   1. Both texts are reduced to comparable tokens (lower-case, no accents or
#      punctuation, CJK per character).
#   2. Each recognised phrase is matched against the reference with
#      difflib.SequenceMatcher. Before matching, misheard words are snapped to the
#      closest word in that song's vocabulary (thefuzz) - "luv" -> "love".
#   3. Matching is *tracking*: we remember where in each song we are, search a
#      window ahead of that position first, and only jump elsewhere (a repeated
#      chorus, the other song of a mashup) when the local match is poor.
#   4. Matched tokens become time anchors. Anchors form "runs" (a contiguous
#      stretch of one song). Tokens between anchors are interpolated; whole lines
#      are emitted with the OFFICIAL text and the time of their first word.
#
# Because a run ends when the song or position changes, mashups fall out for
# free: A-verse, B-chorus, A-chorus becomes three runs whose boundaries are the
# transition points.

_CJK_CLASS = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af"
_TOKEN_RE = re.compile(rf"[{_CJK_CLASS}]|[^\W_]+")

# tuning constants (seconds / tokens / ratios)
LEAD_IN = 0.10             # LRC stamps a hair before the first sung word
MIN_TOKEN_SEC = 0.07       # fastest plausible word; used to spot "line not sung"
CUT_RATIO = 0.40           # < this share of the expected singing time = section was cut
MAX_SLOT_SEC = 0.60        # interpolated word slots never exceed this
RUN_MAX_TOKEN_SKIP = 60    # anchors further apart than this start a new run
RUN_MAX_GAP_SEC = 15.0     # ... as do pauses longer than this
BACK_TOKENS = 2            # tracked search may look this far behind the pointer
DUP_EPS_SEC = 1.0          # identical emissions closer than this are one
FUZZY_CUTOFF = 78          # thefuzz ratio for snapping misheard words
LOCAL_ACCEPT = 0.55        # tracked match good enough to stop searching
MIN_SCORE, MIN_WEIGHT = 0.50, 1.6
JUMP_SCORE, JUMP_WEIGHT = 0.65, 2.5   # stricter bar for non-local jumps
FOCUS_SCORE, FOCUS_WEIGHT = 0.40, 1.0  # gap refinement knows where it is
STRONG_WEIGHT = 2.4        # evidence needed to accept a phrase on cluster quality alone


@dataclass
class RecWord:
    """One word recognised by Whisper (absolute times, seconds)."""

    text: str
    start: float
    end: float
    prob: float = 1.0


def tokenize_for_alignment(text: str) -> list[str]:
    """Comparable tokens: folded, apostrophes dropped, CJK split per character."""
    text = text.replace("'", "").replace("\u2019", "").replace("`", "")
    return _TOKEN_RE.findall(fold(text))


def _weight(token: str) -> float:
    if token in _STOP:
        return 0.35
    return 0.5 if len(token) <= 2 and not _is_cjk(token) else 1.0


def _is_cjk(token: str) -> bool:
    return bool(token) and "\u3040" <= token[0] <= "\ud7af"


def split_phrases(words: list[RecWord], max_gap: float = 1.2, max_words: int = 16) -> list[list[RecWord]]:
    """Group words into singable phrases (pause or length based)."""
    phrases: list[list[RecWord]] = []
    for word in sorted(words, key=lambda w: w.start):
        if phrases and (word.start - phrases[-1][-1].end) <= max_gap and len(phrases[-1]) < max_words:
            phrases[-1].append(word)
        else:
            phrases.append([word])
    return phrases


@dataclass
class _Anchor:
    comp: int
    idx: int          # token index inside the component
    t0: float
    t1: float


@dataclass
class AlignGap:
    """Reference tokens between two anchors whose timing is only a guess."""

    comp: int
    tok_lo: int
    tok_hi: int       # exclusive
    t_start: float
    t_end: float

    @property
    def size(self) -> int:
        return self.tok_hi - self.tok_lo


@dataclass
class AlignmentResult:
    lines: list[LrcLine]
    match_ratio: float            # share of sung words explained by the reference (0-1)
    coverage: float               # share of reference lines that received a timestamp (0-1)
    emitted_lines: int
    total_lines: int
    phrases_total: int
    phrases_matched: int
    switches: int = 0             # song changes inside the audio (mashup transitions)
    per_component: list[float] = field(default_factory=list)
    gaps: list[AlignGap] = field(default_factory=list)

    def ok(self, min_match: float = 0.40) -> bool:
        needed = min(3, self.total_lines)
        return self.emitted_lines >= needed and self.match_ratio >= min_match

    def summary(self) -> str:
        parts = ", ".join(f"{c:.0%}" for c in self.per_component)
        return (f"{self.emitted_lines}/{self.total_lines} lines timed, "
                f"{self.match_ratio:.0%} of sung words matched, coverage per song: {parts}")


class _Component:
    """Tokenised reference for ONE song."""

    def __init__(self, lines: list[str]) -> None:
        self.lines: list[str] = []
        self.tokens: list[str] = []
        self.line_start: list[int] = []
        self.line_end: list[int] = []
        for raw in lines:
            toks = tokenize_for_alignment(raw)
            if not toks:
                continue
            self.lines.append(raw.strip())
            self.line_start.append(len(self.tokens))
            self.tokens.extend(toks)
            self.line_end.append(len(self.tokens))
        self.line_of = [0] * len(self.tokens)
        for li, (a, b) in enumerate(zip(self.line_start, self.line_end)):
            for k in range(a, b):
                self.line_of[k] = li
        self.vocab = sorted(set(self.tokens))
        self._vocab_set = set(self.vocab)
        self._canon_cache: dict[str, str] = {}
        self.ptr = 0   # next expected token (tracking position)

    def canon(self, token: str) -> str:
        """Snap a (possibly misheard) token onto this song's vocabulary."""
        if token in self._vocab_set or len(token) < 3 or _is_cjk(token):
            return token
        cached = self._canon_cache.get(token)
        if cached is None:
            hit = process.extractOne(token, self.vocab, scorer=fuzz.ratio,
                                     processor=None, score_cutoff=FUZZY_CUTOFF)
            cached = hit[0] if hit else token
            self._canon_cache[token] = cached
        return cached


class ForcedAligner:
    """Maps recognised words onto the official lyrics of one or more songs.

    Typical use (the transcriber does exactly this, window by window)::

        aligner = ForcedAligner([lines_of_song_a, lines_of_song_b])
        for phrases in recognised_windows:      # lists of lists of RecWord
            aligner.feed(phrases)
        result = aligner.finalize()
    """

    def __init__(self, references: list[list[str]]) -> None:
        self.comps = [_Component(lines) for lines in references]
        self.anchors: list[_Anchor] = []
        self.last_comp: Optional[int] = None
        self._w_total = 0.0
        self._w_match = 0.0
        self._phrases = 0
        self._phrases_matched = 0

    # ------------------------------------------------------------ matching
    @staticmethod
    def _flatten(phrase: list[RecWord]) -> list[tuple[str, float, float]]:
        out: list[tuple[str, float, float]] = []
        for word in phrase:
            toks = tokenize_for_alignment(word.text)
            if not toks:
                continue
            span = max(word.end - word.start, 0.01) / len(toks)
            for i, tok in enumerate(toks):
                out.append((tok, word.start + i * span, word.start + (i + 1) * span))
        return out

    def _match(self, comp: _Component, lo: int, hi: int, hyp: list[str]):
        """Match a phrase against ``comp.tokens[lo:hi]``.

        Returns ``(pairs, score, matched_weight, quality)``. ``score`` is the share
        of the WHOLE phrase explained; ``quality`` is how well the matched clusters
        explain the part of the phrase they cover. A glued phrase that is half
        verse, half repeated chorus has a poor score but a perfect quality - the
        caller then accepts the verse and re-places the chorus.
        """
        ref = comp.tokens[lo:hi]
        if not ref or not hyp:
            return [], 0.0, 0.0, 0.0
        canon = [comp.canon(t) for t in hyp]
        matcher = difflib.SequenceMatcher(None, ref, canon, autojunk=False)
        # A lone filler word ("me", "the") matching by coincidence is not evidence:
        # keep a block only if it is a real run or a content word.
        blocks = [b for b in matcher.get_matching_blocks()
                  if b.size >= 2 or (b.size == 1 and _weight(hyp[b.b]) >= 1.0)]
        if not blocks:
            return [], 0.0, 0.0, 0.0

        clusters: list[list] = [[blocks[0]]]
        for prev, blk in zip(blocks, blocks[1:]):
            gap_ref = blk.a - (prev.a + prev.size)
            gap_hyp = blk.b - (prev.b + prev.size)
            if gap_ref > 2 * gap_hyp + 8:      # far apart in the lyrics: another place
                clusters.append([blk])
            else:
                clusters[-1].append(blk)

        def weight_of(cluster) -> float:
            return sum(_weight(hyp[b.b + k]) for b in cluster for k in range(b.size))

        def local_score(cluster) -> float:
            span = sum(_weight(hyp[i]) for i in range(cluster[0].b, cluster[-1].b + cluster[-1].size))
            return weight_of(cluster) / (span or 1e-9)

        # Keep every cluster that is convincing on its own (e.g. the end of one line and
        # the start of a line much later, with a cut in between); otherwise only the
        # heaviest one. Clusters are in phrase order, so the pairs stay monotonic.
        good = [c for c in clusters if local_score(c) >= MIN_SCORE and weight_of(c) >= MIN_WEIGHT]
        chosen = good or [max(clusters, key=weight_of)]
        pairs = [(lo + b.a + k, b.b + k) for c in chosen for b in c for k in range(b.size)]
        total = sum(_weight(t) for t in hyp) or 1e-9
        matched = sum(weight_of(c) for c in chosen)
        covered = sum(
            sum(_weight(hyp[i]) for i in range(c[0].b, c[-1].b + c[-1].size)) for c in chosen
        ) or 1e-9
        return pairs, matched / total, matched, matched / covered

    def feed(self, phrases: list[list[RecWord]],
             focus: Optional[tuple[int, int, int]] = None) -> int:
        """Match recognised phrases. ``focus=(component, tok_lo, tok_hi)`` restricts the
        search to a known region (used when re-listening to a gap). Returns the
        number of phrases that found a home in the reference."""
        accepted = 0
        for phrase in phrases:
            flat = self._flatten(phrase)
            if not flat:
                continue
            self._phrases += 1
            self._w_total += sum(_weight(t) for t, _, _ in flat)
            accepted += self._place(flat, focus, depth=0)
        return accepted

    def _place(self, flat: list[tuple[str, float, float]], focus, depth: int) -> int:
        """Find a home for one phrase; re-place leftovers that belong to another song.

        A seamless mashup transition often lands INSIDE one Whisper segment
        ("...end of song B | start of song A..."). The phrase is matched to the
        song that explains most of it, then the unexplained head/tail/middle is
        offered to the other songs as its own phrase.
        """
        hyp = [t for t, _, _ in flat]
        choice = self._best_choice(hyp, focus)
        if choice is None:
            return 0
        ci, pairs, matched = choice
        self._phrases_matched += 1 if depth == 0 else 0
        self._w_match += matched
        self.last_comp = ci
        self.comps[ci].ptr = pairs[-1][0] + 1
        known = {(a.comp, a.idx): a for a in self.anchors}
        used = set()
        for ref_idx, hyp_idx in pairs:
            used.add(hyp_idx)
            _, t0, t1 = flat[hyp_idx]
            old = known.get((ci, ref_idx))
            if old is not None and abs(old.t0 - t0) < 3.0:
                continue  # same word heard twice in overlapping windows
            self.anchors.append(_Anchor(ci, ref_idx, t0, t1))

        placed = 1
        if depth < 2 and focus is None:
            first, last = min(used), max(used)
            runs: list[list[int]] = []
            for i in range(len(hyp)):
                if i in used:
                    continue
                if runs and runs[-1][-1] == i - 1:
                    runs[-1].append(i)
                else:
                    runs.append([i])
            for run in runs:
                # a short unmatched stretch INSIDE the matched span is just
                # mishearing; only long ones (or head/tail leftovers) are re-placed
                inner = first < run[0] and run[-1] < last
                if len(run) < (6 if inner else 3):
                    continue
                if sum(_weight(hyp[i]) for i in run) < MIN_WEIGHT:
                    continue
                placed += self._place([flat[i] for i in run], None, depth + 1)
        return placed

    def _best_choice(self, hyp: list[str], focus):
        n_hyp = len(hyp)
        if focus is not None:
            ci, lo, hi = focus
            pairs, score, wm, _ = self._match(self.comps[ci], lo, hi, hyp)
            if pairs and score >= FOCUS_SCORE and wm >= FOCUS_WEIGHT:
                return ci, pairs, wm
            return None

        def good_local(score, wm, quality):
            return (score >= LOCAL_ACCEPT and wm >= MIN_WEIGHT) or (
                quality >= 0.70 and wm >= STRONG_WEIGHT)

        best = None   # (rank, ci, pairs, wm, score, quality)
        for ci, comp in enumerate(self.comps):
            lo = max(0, comp.ptr - BACK_TOKENS)
            hi = min(len(comp.tokens), comp.ptr + 40 + 3 * n_hyp)
            pairs, score, wm, quality = self._match(comp, lo, hi, hyp)
            rank = score + (0.06 if ci == self.last_comp else 0.0)
            if pairs and (best is None or rank > best[0]):
                best = (rank, ci, pairs, wm, score, quality)
        if best and good_local(best[4], best[3], best[5]):
            return best[1], best[2], best[3]
        if n_hyp <= 2:
            return None  # too little evidence for a non-local match

        # poor local fit: a repeated chorus, or the OTHER song of a mashup
        jump = None
        for ci, comp in enumerate(self.comps):
            pairs, score, wm, quality = self._match(comp, 0, len(comp.tokens), hyp)
            if pairs and (jump is None or score > jump[3]):
                jump = (ci, pairs, wm, score, quality)
        if jump and ((jump[3] >= JUMP_SCORE and jump[2] >= JUMP_WEIGHT)
                     or (jump[3] >= 0.85 and jump[2] >= 2.0)
                     or (jump[4] >= 0.80 and jump[2] >= 3.0)):
            return jump[0], jump[1], jump[2]
        if best and best[4] >= MIN_SCORE and best[3] >= MIN_WEIGHT:
            return best[1], best[2], best[3]
        return None

    # ----------------------------------------------------------- prompts
    def prompt_hint(self, budget_tokens: int, count_tokens: Callable[[str], int]) -> str:
        """Upcoming official lines to use as Whisper's ``initial_prompt``.

        Whisper only reads ~220 prompt tokens, so we send *the lines expected
        next*: starting at the tracked position of the song we are currently in
        (about two thirds of the budget) and at the tracked position of every other
        song (the rest). For a mashup this primes the vocabulary of both songs.
        """
        if not self.comps:
            return ""
        order = list(range(len(self.comps)))
        if self.last_comp is not None:
            order.remove(self.last_comp)
            order.insert(0, self.last_comp)
        shares = [budget_tokens] if len(order) == 1 else (
            [int(budget_tokens * 0.65)] + [int(budget_tokens * 0.35 / (len(order) - 1))] * (len(order) - 1)
        )
        chunks: list[str] = []
        for ci, share in zip(order, shares):
            comp = self.comps[ci]
            if not comp.lines:
                continue
            start = comp.line_of[min(comp.ptr, len(comp.tokens) - 1)]
            used, picked = 0, []
            for line in comp.lines[start:]:
                cost = max(1, count_tokens(line))
                if picked and used + cost > share:
                    break
                picked.append(line)
                used += cost
            if picked:
                chunks.append("\n".join(picked))
        return "\n".join(chunks)

    # ------------------------------------------------------------- result
    def _runs(self) -> list[list[_Anchor]]:
        runs: list[list[_Anchor]] = []
        for a in sorted(self.anchors, key=lambda x: (x.t0, x.idx)):
            cur = runs[-1] if runs else None
            if cur is not None:
                last = cur[-1]
                if a.comp == last.comp and a.idx == last.idx:
                    continue
                if (a.comp == last.comp and 0 < a.idx - last.idx <= RUN_MAX_TOKEN_SKIP
                        and a.t0 - last.t1 <= RUN_MAX_GAP_SEC):
                    cur.append(a)
                    continue
            runs.append([a])
        return runs

    @staticmethod
    def _run_speed(run: list[_Anchor]) -> float:
        """Typical seconds per word in this stretch (median of neighbouring anchors)."""
        steps = [(n.t0 - p.t0) / (n.idx - p.idx) for p, n in zip(run, run[1:])
                 if n.idx > p.idx and 0 < n.t0 - p.t0 < 1.5 * (n.idx - p.idx)]
        if not steps:
            return 0.40
        steps.sort()
        return min(max(steps[len(steps) // 2], 0.18), 0.70)

    @classmethod
    def _interpolate(cls, run: list[_Anchor]) -> tuple[dict[int, float], list[AlignGap], float]:
        """Token start times for a run + the stretches that are only a guess.

        Words Whisper never heard are placed at the run's own speaking speed and
        *centred* in the silence between their neighbours, because a dropped line
        normally sits after a breath rather than glued to the previous line.
        """
        speed = cls._run_speed(run)
        times = {a.idx: a.t0 for a in run}
        gaps: list[AlignGap] = []
        for prev, nxt in zip(run, run[1:]):
            n = nxt.idx - prev.idx - 1
            if n <= 0:
                continue
            avail = nxt.t0 - prev.t1
            if n >= 3 and avail < n * MIN_TOKEN_SEC:
                continue  # no room to sing that many words: the section was cut
            if n >= 4 and avail < CUT_RATIO * n * speed:
                continue  # far too little time for them at this run's tempo: cut
            slot = min(max(avail, 0.0) / n, speed, MAX_SLOT_SEC)
            offset = max(0.0, (avail - slot * n) / 2)
            for k in range(n):
                times[prev.idx + 1 + k] = prev.t1 + offset + k * slot
            if n >= 3 and avail >= 2.0:
                gaps.append(AlignGap(prev.comp, prev.idx + 1, nxt.idx, prev.t1, nxt.t0))
        return times, gaps, speed

    def finalize(self, duration: Optional[float] = None) -> AlignmentResult:
        runs = self._runs()
        emitted: list[tuple[float, int, int]] = []   # (time, comp, line)
        gaps: list[AlignGap] = []
        prev_run: Optional[list[_Anchor]] = None
        for run in runs:
            comp = self.comps[run[0].comp]
            times, run_gaps, speed = self._interpolate(run)
            gaps.extend(run_gaps)
            first_idx, last_idx = run[0].idx, run[-1].idx
            first_line = comp.line_of[first_idx]
            for li in range(first_line, comp.line_of[last_idx] + 1):
                a, b = comp.line_start[li], comp.line_end[li]
                # A run that starts inside the line the PREVIOUS run ended in is the
                # same occurrence of that line, not a repeat: do not emit it twice.
                if (li == first_line and prev_run is not None
                        and prev_run[0].comp == run[0].comp
                        and a <= prev_run[-1].idx < b):
                    continue
                known = [k for k in range(a, b) if k in times]
                if not known:
                    continue
                stamp = times[known[0]] - (known[0] - a) * speed   # words before the first anchor
                emitted.append((max(0.0, stamp - LEAD_IN), run[0].comp, li))
            prev_run = run

        emitted.sort()
        lines: list[LrcLine] = []
        unique: set[tuple[int, int]] = set()
        last_seen: dict[tuple[int, int], float] = {}
        for t, ci, li in emitted:
            prev_t = last_seen.get((ci, li))
            if prev_t is not None and t - prev_t < DUP_EPS_SEC:
                continue
            last_seen[(ci, li)] = t
            unique.add((ci, li))
            lines.append(LrcLine(round(t, 2), self.comps[ci].lines[li]))
        if duration:
            lines = [l for l in lines if l.time <= duration + 1.0]

        per_comp = []
        for ci, comp in enumerate(self.comps):
            got = sum(1 for (c, _) in unique if c == ci)
            per_comp.append(got / len(comp.lines) if comp.lines else 0.0)
        total_lines = sum(len(c.lines) for c in self.comps)
        switches = sum(1 for r0, r1 in zip(runs, runs[1:]) if r0[0].comp != r1[0].comp)
        gaps.sort(key=lambda g: -g.size)
        return AlignmentResult(
            lines=lines,
            match_ratio=(self._w_match / self._w_total) if self._w_total else 0.0,
            coverage=(len(unique) / total_lines) if total_lines else 0.0,
            emitted_lines=len(unique),
            total_lines=total_lines,
            phrases_total=self._phrases,
            phrases_matched=self._phrases_matched,
            switches=switches,
            per_component=per_comp,
            gaps=gaps,
        )


def align_lines_to_words(reference_lines: list[str], words: list[RecWord],
                         duration: Optional[float] = None) -> AlignmentResult:
    """One-shot alignment of a single song's lyrics to a full word list."""
    aligner = ForcedAligner([reference_lines])
    aligner.feed(split_phrases(words))
    return aligner.finalize(duration)
