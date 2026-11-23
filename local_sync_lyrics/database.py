"""SQLite cache of already-processed tracks.

Why a fingerprint instead of just the path?
    Hashing a 40 MB FLAC end-to-end for every run would be slow. We hash the file
    size plus three 64 KiB slices (head / middle / tail) which is effectively
    unique for audio and *survives renames and moves*.

The connection is shared across threads behind a lock (``check_same_thread`` is
off), uses WAL for resilience and is rebuilt automatically if the file is
corrupt.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_SLICE = 64 * 1024


@dataclass
class TrackRecord:
    fingerprint: str
    path: str
    status: str
    artist: str = ""
    title: str = ""
    duration: Optional[float] = None
    source: str = ""
    lrc_path: str = ""
    ai_tried: bool = False
    error: str = ""
    attempts: int = 1
    updated_at: float = 0.0


def fingerprint_file(path: Path) -> str:
    """Fast, rename-proof identity of an audio file."""
    digest = hashlib.sha1()
    try:
        size = path.stat().st_size
        digest.update(str(size).encode())
        with open(path, "rb") as handle:
            for offset in (0, max(0, size // 2 - _SLICE // 2), max(0, size - _SLICE)):
                handle.seek(offset)
                digest.update(handle.read(_SLICE))
    except OSError:
        # Unreadable content: fall back to path + stat so we still get a stable key.
        try:
            st = path.stat()
            digest.update(f"{path}|{st.st_size}|{int(st.st_mtime)}".encode("utf-8", "replace"))
        except OSError:
            digest.update(str(path).encode("utf-8", "replace"))
    return digest.hexdigest()


class LyricsDatabase:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._open()

    # ------------------------------------------------------------- connection
    def _open(self) -> None:
        try:
            self._conn = self._connect()
        except sqlite3.DatabaseError as exc:
            logger.error("Cache database unusable (%s); recreating", exc)
            self._quarantine()
            self._conn = self._connect()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self.path), timeout=30, check_same_thread=False, isolation_level=None
        )
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.DatabaseError:
            pass  # e.g. network drives without WAL support: keep defaults
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tracks (
                fingerprint TEXT PRIMARY KEY,
                path        TEXT NOT NULL,
                artist      TEXT NOT NULL DEFAULT '',
                title       TEXT NOT NULL DEFAULT '',
                duration    REAL,
                status      TEXT NOT NULL,
                source      TEXT NOT NULL DEFAULT '',
                lrc_path    TEXT NOT NULL DEFAULT '',
                ai_tried    INTEGER NOT NULL DEFAULT 0,
                error       TEXT NOT NULL DEFAULT '',
                attempts    INTEGER NOT NULL DEFAULT 1,
                updated_at  REAL NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tracks_path ON tracks(path)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tracks_status ON tracks(status)")
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        conn.execute("SELECT count(*) FROM tracks").fetchone()  # forces corruption errors now
        return conn

    def _quarantine(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except sqlite3.Error:
            pass
        self._conn = None
        stamp = int(time.time())
        for suffix in ("", "-wal", "-shm"):
            src = Path(str(self.path) + suffix)
            if src.exists():
                try:
                    src.replace(Path(f"{self.path}.corrupt-{stamp}{suffix}"))
                except OSError:
                    try:
                        src.unlink()
                    except OSError:
                        pass

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except sqlite3.Error:
                    pass
                self._conn = None

    def __enter__(self) -> "LyricsDatabase":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ queries
    def get(self, fingerprint: str) -> Optional[TrackRecord]:
        with self._lock:
            if self._conn is None:
                return None
            try:
                row = self._conn.execute(
                    "SELECT * FROM tracks WHERE fingerprint=?", (fingerprint,)
                ).fetchone()
            except sqlite3.Error as exc:
                logger.warning("cache read failed: %s", exc)
                return None
        if row is None:
            return None
        return TrackRecord(
            fingerprint=row["fingerprint"], path=row["path"], status=row["status"],
            artist=row["artist"], title=row["title"], duration=row["duration"],
            source=row["source"], lrc_path=row["lrc_path"], ai_tried=bool(row["ai_tried"]),
            error=row["error"], attempts=row["attempts"], updated_at=row["updated_at"],
        )

    def upsert(self, record: TrackRecord) -> None:
        """Insert or update a record. Never raises (cache is best-effort)."""
        with self._lock:
            if self._conn is None:
                return
            try:
                previous = self._conn.execute(
                    "SELECT attempts FROM tracks WHERE fingerprint=?", (record.fingerprint,)
                ).fetchone()
                attempts = (previous["attempts"] + 1) if previous else 1
                self._conn.execute(
                    """
                    INSERT INTO tracks (fingerprint, path, artist, title, duration, status,
                                        source, lrc_path, ai_tried, error, attempts, updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(fingerprint) DO UPDATE SET
                        path=excluded.path, artist=excluded.artist, title=excluded.title,
                        duration=excluded.duration, status=excluded.status,
                        source=excluded.source, lrc_path=excluded.lrc_path,
                        ai_tried=excluded.ai_tried, error=excluded.error,
                        attempts=excluded.attempts, updated_at=excluded.updated_at
                    """,
                    (
                        record.fingerprint, record.path, record.artist, record.title,
                        record.duration, record.status, record.source, record.lrc_path,
                        int(record.ai_tried), record.error[:500], attempts, time.time(),
                    ),
                )
            except sqlite3.Error as exc:
                logger.warning("cache write failed: %s", exc)

    def rekey(self, old: str, new: str) -> None:
        """Keep the cache entry valid after tag writing changed the file's fingerprint."""
        if not old or not new or old == new:
            return
        with self._lock:
            if self._conn is None:
                return
            try:
                self._conn.execute(
                    "UPDATE OR REPLACE tracks SET fingerprint=? WHERE fingerprint=?", (new, old))
            except sqlite3.Error as exc:
                logger.warning("cache rekey failed: %s", exc)

    def stats(self) -> dict[str, int]:
        with self._lock:
            if self._conn is None:
                return {}
            try:
                rows = self._conn.execute(
                    "SELECT status, count(*) AS n FROM tracks GROUP BY status"
                ).fetchall()
            except sqlite3.Error:
                return {}
        return {row["status"]: row["n"] for row in rows}

    def clear(self) -> int:
        with self._lock:
            if self._conn is None:
                return 0
            try:
                count = self._conn.execute("SELECT count(*) FROM tracks").fetchone()[0]
                self._conn.execute("DELETE FROM tracks")
                return int(count)
            except sqlite3.Error as exc:
                logger.warning("cache clear failed: %s", exc)
                return 0


def clear_cache_file(path: Path) -> int:
    """Convenience for the UI: open, wipe, close. Returns rows removed."""
    with LyricsDatabase(path) as db:
        return db.clear()


def cache_file_exists(path: Path) -> bool:
    return os.path.exists(path)
