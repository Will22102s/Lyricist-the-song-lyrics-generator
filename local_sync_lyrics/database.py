"""SQLite cache and sync history of already-processed tracks.

Why a fingerprint instead of just the path?
    Hashing a 40 MB FLAC end-to-end for every run would be slow. We hash the file
    size plus three 64 KiB slices (head / middle / tail) which is effectively
    unique for audio and *survives renames and moves*.

Phase 4 adds a *history / media vault* on top of the cache: album, genre, year,
language, output mode, provider, the moment the track was last synced and a
cached artwork thumbnail. The schema is upgraded in place with ``ALTER TABLE``
so an existing ``cache.db`` is never lost.

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

SCHEMA_VERSION = 2
_SLICE = 64 * 1024
SYNCED_STATUSES = ("synced_online", "synced_aligned", "synced_ai")

# Column name -> SQL declaration.  Missing columns are added to an old database
# in place; every declaration is safe for an ``ALTER TABLE ... ADD COLUMN``.
_HISTORY_COLUMNS: dict[str, str] = {
    "file_path": "TEXT NOT NULL DEFAULT ''",
    "album": "TEXT NOT NULL DEFAULT ''",
    "genre": "TEXT NOT NULL DEFAULT ''",
    "year": "TEXT NOT NULL DEFAULT ''",
    "language": "TEXT NOT NULL DEFAULT ''",
    "output_mode": "TEXT NOT NULL DEFAULT ''",
    "provider_used": "TEXT NOT NULL DEFAULT ''",
    "synced_at": "REAL NOT NULL DEFAULT 0",
    "thumbnail": "BLOB",
    "thumbnail_mime": "TEXT NOT NULL DEFAULT ''",
}


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

    # --- Phase 4: history / media vault -----------------------------------
    album: str = ""
    genre: str = ""
    year: str = ""
    language: str = ""
    output_mode: str = ""
    provider_used: str = ""
    synced_at: float = 0.0
    thumbnail: bytes = b""
    thumbnail_mime: str = ""

    @property
    def file_path(self) -> str:
        """Alias kept for the public history API (``path`` remains the cache key)."""
        return self.path


@dataclass
class HistoryRecord:
    """One row of the History & Library view."""

    fingerprint: str
    path: str
    title: str = ""
    artist: str = ""
    album: str = ""
    genre: str = ""
    year: str = ""
    language: str = ""
    status: str = ""
    output_mode: str = ""
    provider_used: str = ""
    synced_at: float = 0.0
    duration: Optional[float] = None
    lrc_path: str = ""
    error: str = ""
    thumbnail: bytes = b""
    thumbnail_mime: str = ""

    @property
    def file_path(self) -> str:
        return self.path

    @property
    def display_title(self) -> str:
        return self.title or Path(self.path).stem or "Untitled"

    @property
    def display_artist(self) -> str:
        return self.artist or "Unknown artist"

    @property
    def is_synced(self) -> bool:
        return self.status in SYNCED_STATUSES


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
                file_path   TEXT NOT NULL DEFAULT '',
                artist      TEXT NOT NULL DEFAULT '',
                title       TEXT NOT NULL DEFAULT '',
                album       TEXT NOT NULL DEFAULT '',
                genre       TEXT NOT NULL DEFAULT '',
                year        TEXT NOT NULL DEFAULT '',
                language    TEXT NOT NULL DEFAULT '',
                duration    REAL,
                status      TEXT NOT NULL,
                source      TEXT NOT NULL DEFAULT '',
                lrc_path    TEXT NOT NULL DEFAULT '',
                output_mode TEXT NOT NULL DEFAULT '',
                provider_used TEXT NOT NULL DEFAULT '',
                synced_at   REAL NOT NULL DEFAULT 0,
                ai_tried    INTEGER NOT NULL DEFAULT 0,
                error       TEXT NOT NULL DEFAULT '',
                attempts    INTEGER NOT NULL DEFAULT 1,
                updated_at  REAL NOT NULL,
                thumbnail   BLOB,
                thumbnail_mime TEXT NOT NULL DEFAULT ''
            )
            """
        )
        self._migrate(conn)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tracks_path ON tracks(path)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tracks_status ON tracks(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tracks_synced ON tracks(synced_at)")
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        conn.execute("SELECT count(*) FROM tracks").fetchone()  # forces corruption errors now
        return conn

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Add Phase 4 columns to an existing database without touching the rows."""
        try:
            existing = {row["name"] for row in conn.execute("PRAGMA table_info(tracks)")}
        except sqlite3.Error:
            existing = set()
        for name, declaration in _HISTORY_COLUMNS.items():
            if name in existing:
                continue
            try:
                conn.execute(f"ALTER TABLE tracks ADD COLUMN {name} {declaration}")
            except sqlite3.Error as exc:  # pragma: no cover - defensive
                logger.warning("Could not add column %s: %s", name, exc)
        try:
            conn.execute(
                "UPDATE tracks SET file_path=path "
                "WHERE file_path='' OR file_path IS NULL"
            )
            placeholders = ",".join("?" for _ in SYNCED_STATUSES)
            conn.execute(
                f"UPDATE tracks SET synced_at=updated_at "
                f"WHERE synced_at<=0 AND status IN ({placeholders})",
                SYNCED_STATUSES,
            )
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            logger.debug("history backfill skipped: %s", exc)

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
    @staticmethod
    def _row_value(row: sqlite3.Row, name: str, default: object = "") -> object:
        try:
            value = row[name]
        except (IndexError, KeyError):
            return default
        return default if value is None else value

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
        return self._record_from_row(row)

    def _record_from_row(self, row: sqlite3.Row) -> TrackRecord:
        return TrackRecord(
            fingerprint=row["fingerprint"], path=row["path"], status=row["status"],
            artist=self._row_value(row, "artist"), title=self._row_value(row, "title"),
            duration=row["duration"], source=self._row_value(row, "source"),
            lrc_path=self._row_value(row, "lrc_path"), ai_tried=bool(row["ai_tried"]),
            error=self._row_value(row, "error"), attempts=row["attempts"],
            updated_at=row["updated_at"],
            album=self._row_value(row, "album"), genre=self._row_value(row, "genre"),
            year=self._row_value(row, "year"), language=self._row_value(row, "language"),
            output_mode=self._row_value(row, "output_mode"),
            provider_used=self._row_value(row, "provider_used"),
            synced_at=float(self._row_value(row, "synced_at", 0.0) or 0.0),
            thumbnail=self._row_value(row, "thumbnail", b"") or b"",
            thumbnail_mime=self._row_value(row, "thumbnail_mime"),
        )

    def upsert(self, record: TrackRecord) -> None:
        """Insert or update a record. Never raises (cache is best-effort).

        Empty values never erase data that is already there (a later failure
        record must not wipe the album art of an earlier successful sync), and
        an empty ``synced_at`` keeps the previous timestamp.
        """
        with self._lock:
            if self._conn is None:
                return
            try:
                previous = self._conn.execute(
                    "SELECT attempts FROM tracks WHERE fingerprint=?", (record.fingerprint,)
                ).fetchone()
                attempts = (previous["attempts"] + 1) if previous else 1
                synced_at = record.synced_at
                if not synced_at and record.status in SYNCED_STATUSES:
                    synced_at = time.time()
                self._conn.execute(
                    """
                    INSERT INTO tracks (fingerprint, path, file_path, artist, title, album,
                                        genre, year, language, duration, status, source,
                                        lrc_path, output_mode, provider_used, synced_at,
                                        ai_tried, error, attempts, updated_at,
                                        thumbnail, thumbnail_mime)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(fingerprint) DO UPDATE SET
                        path=excluded.path,
                        file_path=CASE WHEN excluded.file_path<>'' THEN excluded.file_path
                                       ELSE tracks.file_path END,
                        artist=CASE WHEN excluded.artist<>'' THEN excluded.artist
                                    ELSE tracks.artist END,
                        title=CASE WHEN excluded.title<>'' THEN excluded.title
                                   ELSE tracks.title END,
                        album=CASE WHEN excluded.album<>'' THEN excluded.album
                                   ELSE tracks.album END,
                        genre=CASE WHEN excluded.genre<>'' THEN excluded.genre
                                   ELSE tracks.genre END,
                        year=CASE WHEN excluded.year<>'' THEN excluded.year
                                  ELSE tracks.year END,
                        language=CASE WHEN excluded.language<>'' THEN excluded.language
                                      ELSE tracks.language END,
                        duration=COALESCE(excluded.duration, tracks.duration),
                        status=excluded.status,
                        source=CASE WHEN excluded.source<>'' THEN excluded.source
                                    ELSE tracks.source END,
                        lrc_path=CASE WHEN excluded.lrc_path<>'' THEN excluded.lrc_path
                                      ELSE tracks.lrc_path END,
                        output_mode=CASE WHEN excluded.output_mode<>'' THEN excluded.output_mode
                                         ELSE tracks.output_mode END,
                        provider_used=CASE WHEN excluded.provider_used<>''
                                           THEN excluded.provider_used
                                           ELSE tracks.provider_used END,
                        synced_at=CASE WHEN excluded.synced_at>0 THEN excluded.synced_at
                                       ELSE tracks.synced_at END,
                        ai_tried=excluded.ai_tried, error=excluded.error,
                        attempts=excluded.attempts, updated_at=excluded.updated_at,
                        thumbnail=CASE WHEN length(excluded.thumbnail)>0
                                       THEN excluded.thumbnail ELSE tracks.thumbnail END,
                        thumbnail_mime=CASE WHEN excluded.thumbnail_mime<>''
                                            THEN excluded.thumbnail_mime
                                            ELSE tracks.thumbnail_mime END
                    """,
                    (
                        record.fingerprint, record.path, record.path, record.artist,
                        record.title, record.album, record.genre, record.year,
                        record.language, record.duration, record.status, record.source,
                        record.lrc_path, record.output_mode, record.provider_used,
                        synced_at, int(record.ai_tried), record.error[:500], attempts,
                        time.time(), record.thumbnail or b"", record.thumbnail_mime,
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

    # ----------------------------------------------------------------- history
    def history(self, *, search: str = "", status: str = "", limit: int = 400,
                offset: int = 0) -> list[HistoryRecord]:
        """Rows for the History & Library view, newest sync first.

        Runs on the caller's thread; the GUI calls it from a ``QThreadPool`` job.
        """
        clauses: list[str] = []
        params: list[object] = []
        needle = (search or "").strip()
        if needle:
            clauses.append("(title LIKE ? OR artist LIKE ? OR album LIKE ? OR path LIKE ?)")
            like = f"%{needle}%"
            params.extend([like, like, like, like])
        wanted = (status or "").strip()
        if wanted and wanted != "all":
            clauses.append("status=?")
            params.append(wanted)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            "SELECT * FROM tracks"
            + where
            + " ORDER BY CASE WHEN synced_at>0 THEN synced_at ELSE updated_at END DESC"
            " LIMIT ? OFFSET ?"
        )
        params.extend([max(1, int(limit)), max(0, int(offset))])
        with self._lock:
            if self._conn is None:
                return []
            try:
                rows = self._conn.execute(sql, params).fetchall()
            except sqlite3.Error as exc:
                logger.warning("history read failed: %s", exc)
                return []
        return [self._history_from_row(row) for row in rows]

    def history_count(self, *, search: str = "", status: str = "") -> int:
        clauses: list[str] = []
        params: list[object] = []
        needle = (search or "").strip()
        if needle:
            clauses.append("(title LIKE ? OR artist LIKE ? OR album LIKE ? OR path LIKE ?)")
            like = f"%{needle}%"
            params.extend([like, like, like, like])
        wanted = (status or "").strip()
        if wanted and wanted != "all":
            clauses.append("status=?")
            params.append(wanted)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            if self._conn is None:
                return 0
            try:
                return int(self._conn.execute(
                    f"SELECT count(*) FROM tracks{where}", params).fetchone()[0])
            except sqlite3.Error:
                return 0

    @staticmethod
    def _history_from_row(row: sqlite3.Row) -> HistoryRecord:
        def value(name: str, default: object = "") -> object:
            try:
                raw = row[name]
            except (IndexError, KeyError):
                return default
            return default if raw is None else raw

        return HistoryRecord(
            fingerprint=row["fingerprint"], path=row["path"],
            title=value("title"), artist=value("artist"), album=value("album"),
            genre=value("genre"), year=value("year"), language=value("language"),
            status=row["status"], output_mode=value("output_mode"),
            provider_used=value("provider_used"),
            synced_at=float(value("synced_at", 0.0) or 0.0),
            duration=row["duration"], lrc_path=value("lrc_path"), error=value("error"),
            thumbnail=value("thumbnail", b"") or b"",
            thumbnail_mime=value("thumbnail_mime"),
        )

    def get_by_path(self, path: str | Path) -> Optional[HistoryRecord]:
        target = str(path)
        with self._lock:
            if self._conn is None:
                return None
            try:
                row = self._conn.execute(
                    "SELECT * FROM tracks WHERE path=? OR file_path=? "
                    "ORDER BY updated_at DESC LIMIT 1", (target, target)).fetchone()
            except sqlite3.Error:
                return None
        return self._history_from_row(row) if row is not None else None

    def set_thumbnail(self, fingerprint: str, data: bytes, mime: str = "") -> None:
        if not data:
            return
        with self._lock:
            if self._conn is None:
                return
            try:
                self._conn.execute(
                    "UPDATE tracks SET thumbnail=?, thumbnail_mime=? WHERE fingerprint=?",
                    (data, mime, fingerprint))
            except sqlite3.Error as exc:
                logger.debug("thumbnail write failed: %s", exc)

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
