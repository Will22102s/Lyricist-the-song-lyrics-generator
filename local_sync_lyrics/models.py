"""Plain data containers shared by every layer (no I/O, no heavy imports)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class Status(str, Enum):
    """Outcome of processing one track (also what the cache stores)."""

    SYNCED_ONLINE = "synced_online"    # Tier 1 provider delivered a ready .lrc
    SYNCED_ALIGNED = "synced_aligned"  # official text + Whisper forced alignment
    SYNCED_AI = "synced_ai"            # blind transcription (last resort)
    PLAIN_ONLY = "plain_only"
    INSTRUMENTAL = "instrumental"
    NOT_FOUND = "not_found"
    FAILED = "failed"
    SKIPPED = "skipped"  # runtime-only: never written to the cache


class MatchKind(str, Enum):
    """What a provider hit gives us."""

    SYNCED = "SYNCED"          # ready-to-save LRC
    PLAIN_TEXT = "PLAIN_TEXT"  # official text only: needs AI timestamp alignment
    INSTRUMENTAL = "INSTRUMENTAL"


@dataclass
class TrackInfo:
    """Everything we know about one audio file."""

    path: Path
    artist: str = ""
    title: str = ""
    album: str = ""
    duration: Optional[float] = None  # seconds

    @property
    def display_name(self) -> str:
        if self.artist and self.title:
            return f"{self.artist} - {self.title}"
        return self.title or self.path.stem


@dataclass
class LyricsResult:
    """A lyrics hit from one provider."""

    provider: str
    synced_lrc: Optional[str] = None
    plain: Optional[str] = None
    instrumental: bool = False
    artist: str = ""
    title: str = ""
    score: float = 0.0
    tier: int = 1
    partial: bool = False  # provider only returns part of the lyrics (e.g. free Musixmatch)

    @property
    def has_synced(self) -> bool:
        return bool(self.synced_lrc)

    @property
    def kind(self) -> MatchKind:
        if self.synced_lrc:
            return MatchKind.SYNCED
        if self.instrumental and not self.plain:
            return MatchKind.INSTRUMENTAL
        return MatchKind.PLAIN_TEXT


@dataclass
class ReferenceComponent:
    """Ground-truth lyrics of ONE song (a normal track has one; a mashup has several)."""

    artist: str
    title: str
    lines: list[str]
    source: str = ""
    partial: bool = False
    first_seen: Optional[float] = None  # seconds into the audio (set by audio identification)

    @property
    def name(self) -> str:
        return f"{self.artist} - {self.title}" if self.artist else self.title

    @property
    def label(self) -> str:
        return self.name


@dataclass
class ReferenceScript:
    """The combined reference lyrics that forced alignment maps the audio onto."""

    components: list[ReferenceComponent] = field(default_factory=list)
    is_mashup: bool = False

    @property
    def line_count(self) -> int:
        return sum(len(c.lines) for c in self.components)

    @property
    def partial(self) -> bool:
        return bool(self.components) and all(c.partial for c in self.components)

    @property
    def any_partial(self) -> bool:
        return any(c.partial for c in self.components)

    def describe(self) -> str:
        return " | ".join(f"{c.name} ({len(c.lines)} lines, {c.source})" for c in self.components)


@dataclass
class Snippet:
    """A few lyric words recognised from a slice of the audio (used to identify songs)."""

    start: float
    text: str


@dataclass
class IdentifiedSong:
    """A song found by searching for a lyric snippet."""

    artist: str
    title: str
    source: str


@dataclass
class TranscriptionResult:
    lrc_text: str
    language: str
    language_probability: float
    line_count: int
    model: str
    device: str
    seconds: float


@dataclass
class Event:
    """Message from the worker thread to the UI / CLI.

    kind: ``log`` | ``progress`` | ``stats`` | ``done`` | ``finished``
    tag : e.g. ``API: Tier 1`` | ``API: Plain Text`` | ``Mashup Detected`` |
          ``AI: Forced Alignment`` | ``AI: Identify`` | ``AI: Blind`` | ``OK`` | ...
    """

    kind: str
    message: str = ""
    tag: str = "INFO"
    level: str = "info"  # info | warning | error
    current: int = 0
    total: int = 0
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchSummary:
    total: int = 0
    synced_online: int = 0
    synced_aligned: int = 0
    synced_ai: int = 0
    plain_only: int = 0
    instrumental: int = 0
    skipped: int = 0
    not_found: int = 0
    failed: int = 0
    cancelled: bool = False

    def bump(self, status: Status) -> None:
        mapping = {
            Status.SYNCED_ONLINE: "synced_online",
            Status.SYNCED_ALIGNED: "synced_aligned",
            Status.SYNCED_AI: "synced_ai",
            Status.PLAIN_ONLY: "plain_only",
            Status.INSTRUMENTAL: "instrumental",
            Status.SKIPPED: "skipped",
            Status.NOT_FOUND: "not_found",
            Status.FAILED: "failed",
        }
        attr = mapping[status]
        setattr(self, attr, getattr(self, attr) + 1)

    @property
    def synced_total(self) -> int:
        return self.synced_online + self.synced_aligned + self.synced_ai

    @property
    def processed(self) -> int:
        return (
            self.synced_online + self.synced_aligned + self.synced_ai + self.plain_only
            + self.instrumental + self.skipped + self.not_found + self.failed
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "synced_online": self.synced_online,
            "synced_aligned": self.synced_aligned,
            "synced_ai": self.synced_ai,
            "plain_only": self.plain_only,
            "instrumental": self.instrumental,
            "skipped": self.skipped,
            "not_found": self.not_found,
            "failed": self.failed,
        }
