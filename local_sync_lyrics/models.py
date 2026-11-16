"""Plain data containers shared by every layer (no I/O, no heavy imports)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class Status(str, Enum):
    """Outcome of processing one track (also what the cache stores)."""

    SYNCED_ONLINE = "synced_online"
    SYNCED_AI = "synced_ai"
    PLAIN_ONLY = "plain_only"
    INSTRUMENTAL = "instrumental"
    NOT_FOUND = "not_found"
    FAILED = "failed"
    SKIPPED = "skipped"  # runtime-only: never written to the cache


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

    @property
    def has_synced(self) -> bool:
        return bool(self.synced_lrc)


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
    """

    kind: str
    message: str = ""
    tag: str = "INFO"  # API | AI | OK | SKIP | WARN | ERR | INFO | SYS
    level: str = "info"  # info | warning | error
    current: int = 0
    total: int = 0
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchSummary:
    total: int = 0
    synced_online: int = 0
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
    def processed(self) -> int:
        return (
            self.synced_online + self.synced_ai + self.plain_only + self.instrumental
            + self.skipped + self.not_found + self.failed
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "synced_online": self.synced_online,
            "synced_ai": self.synced_ai,
            "plain_only": self.plain_only,
            "instrumental": self.instrumental,
            "skipped": self.skipped,
            "not_found": self.not_found,
            "failed": self.failed,
        }
