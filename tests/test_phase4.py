"""Phase 4: history vault, artwork, updater, frameless GUI and packaging."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------- database
def _v1_database(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE tracks (
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
    conn.execute(
        "INSERT INTO tracks (fingerprint, path, artist, title, status, updated_at) "
        "VALUES ('old', '/music/old.mp3', 'Artist', 'Old Song', 'synced_online', 1234.0)"
    )
    conn.execute("PRAGMA user_version=1")
    conn.commit()
    conn.close()


def test_schema_v1_is_migrated_without_losing_rows(tmp_path):
    from local_sync_lyrics.database import SCHEMA_VERSION, LyricsDatabase

    path = tmp_path / "cache.db"
    _v1_database(path)
    with LyricsDatabase(path) as db:
        records = db.history()
        assert len(records) == 1
        record = records[0]
        assert record.title == "Old Song"
        assert record.path == "/music/old.mp3"
        assert record.file_path == "/music/old.mp3"      # backfilled from path
        assert record.synced_at == 1234.0                 # backfilled from updated_at
        assert record.album == "" and record.language == ""
        assert SCHEMA_VERSION >= 2


def test_history_search_filter_and_upsert_preserves_metadata(tmp_path):
    from local_sync_lyrics.database import LyricsDatabase, TrackRecord

    with LyricsDatabase(tmp_path / "cache.db") as db:
        db.upsert(TrackRecord("a", "/m/one.mp3", "synced_online", artist="A", title="Alpha",
                              album="First", language="en", synced_at=10.0))
        db.upsert(TrackRecord("b", "/m/two.mp3", "failed", artist="B", title="Beta"))
        db.upsert(TrackRecord("c", "/m/three.mp3", "synced_ai", artist="C", title="Gamma",
                              album="Third", synced_at=20.0))

        assert [r.title for r in db.history(search="alp")] == ["Alpha"]
        assert [r.title for r in db.history(status="synced_online")] == ["Alpha"]
        assert db.history_count() == 3
        assert db.history_count(search="Beta") == 1

        # A later failure must not erase the album/language of the earlier success.
        db.upsert(TrackRecord("a", "/m/one.mp3", "failed", artist="A", title="Alpha",
                              error="network"))
        record = db.get("a")
        assert record is not None and record.status == "failed"
        assert record.album == "First" and record.language == "en"
        assert record.synced_at == 10.0


# ---------------------------------------------------------------------- artwork
def _png_bytes() -> bytes:
    from PIL import Image

    import io

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (220, 30, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_embedded_artwork_is_read_and_thumbnailed(tmp_path):
    from audio_fixtures import make_audio
    from local_sync_lyrics.metadata import (
        image_mime, make_thumbnail, read_artwork, read_extra_tags, write_tags,
    )
    from local_sync_lyrics.models import TrackMetadata

    path = tmp_path / "cover.mp3"
    if not make_audio(path):
        pytest.skip("this PyAV build has no MP3 encoder")
    png = _png_bytes()
    result = write_tags(path, TrackMetadata(title="Cover", genre="Pop", year="2024-05-01"),
                        cover=png, cover_mime="image/png")
    assert result.ok and result.cover_written

    data, mime = read_artwork(path)
    assert data == png
    assert mime == "image/png"
    assert image_mime(data) == "image/png"

    extra = read_extra_tags(path)
    assert extra["genre"] == "Pop"
    assert extra["year"] == "2024"

    thumb, thumb_mime = make_thumbnail(data, mime, max_size=6)
    assert thumb and thumb_mime == "image/jpeg"


def test_artwork_missing_returns_empty(tmp_path):
    from audio_fixtures import make_audio
    from local_sync_lyrics.metadata import read_artwork

    path = tmp_path / "plain.flac"
    if not make_audio(path):
        pytest.skip("this PyAV build has no FLAC encoder")
    assert read_artwork(path) == (b"", "")


# ---------------------------------------------------------------------- updater
def test_semantic_version_comparison():
    from local_sync_lyrics.updater import is_newer

    assert is_newer("2.2.0", "v2.3.0")
    assert is_newer("2.3.0", "2.10.0")
    assert not is_newer("2.3.0", "2.3.0")
    assert not is_newer("2.3.0", "2.2.9")
    assert not is_newer("2.3.0", "")


def test_release_payload_selects_platform_asset_and_changelog():
    from local_sync_lyrics.updater import check_for_update, parse_release

    payload = {
        "tag_name": "v9.9.9",
        "name": "Lyricist 9.9.9",
        "body": "## Highlights\n- Faster sync",
        "html_url": "https://github.com/o/r/releases/tag/v9.9.9",
        "assets": [
            {"name": "Lyricist-9.9.9-win.exe",
             "browser_download_url": "https://example.test/win.exe", "size": 42},
            {"name": "Lyricist-9.9.9-linux.zip",
             "browser_download_url": "https://example.test/linux.zip", "size": 43},
        ],
    }
    info = parse_release(payload, current="2.3.0", platform="win32")
    assert info.available
    assert info.latest == "9.9.9"
    assert info.asset is not None and info.asset.name.endswith(".exe")
    assert info.download_url.endswith("win.exe")
    assert "Highlights" in info.notes
    assert "9.9.9" in info.summary()

    def opener(url: str, timeout: float) -> dict:
        assert "api.github.com/repos/o/r/releases/latest" in url
        return payload

    result = check_for_update("2.3.0", "o/r", opener=opener, platform="linux")
    assert result.available and result.asset is not None
    assert result.asset.name.endswith(".zip")


def test_update_check_handles_network_errors_without_raising():
    from local_sync_lyrics.updater import check_for_update

    def missing(url: str, timeout: float) -> dict:
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    info = check_for_update("2.3.0", "o/r", opener=missing)
    assert not info.available
    assert info.error and "releases" in info.error.lower()


# -------------------------------------------------------------------------- GUI
def test_window_has_four_icon_views_and_switch_animation(window, qapp):
    assert window.stack.count() == 4
    assert set(window.nav_buttons) == {"workspace", "history", "providers", "settings"}
    for name in ("history", "providers", "settings", "workspace"):
        window._show_view(name)
        qapp.processEvents()
        assert window.view_name == name
        assert window.stack.currentWidget() is getattr(window, f"{name}_view")
        assert window.nav_buttons[name].isChecked()
        assert not window.nav_buttons[name].icon().isNull()


def test_queue_badges_language_and_clear(window, tmp_path, qapp):
    first = tmp_path / "first.mp3"
    second = tmp_path / "second.flac"
    first.write_bytes(b"x")
    second.write_bytes(b"x")

    window._queue_add([first, second])
    window._queue_add([first])          # duplicate ignored
    qapp.processEvents()
    assert window.queue_table.rowCount() == 2
    assert len(window.queue_paths) == 2
    assert "2 track" in window.queue_count.text()

    window._queue_set_status(str(first), "syncing", "English")
    window._queue_set_status(str(second), "done")
    assert window.queue_table.item(0, 1).text() == "Syncing"
    assert window.queue_table.item(0, 2).text() == "English"
    assert window.queue_table.item(1, 1).text() == "Done"
    assert "1 done" in window.queue_count.text()

    window._queue_clear()
    assert window.queue_table.rowCount() == 0
    assert window.queue_count.text() == "No tracks queued"


def test_drop_zone_emits_paths_and_queue_scans_them(window, tmp_path, qapp):
    from PySide6.QtCore import QMimeData, QPointF, QUrl, Qt
    from PySide6.QtGui import QDropEvent

    from local_sync_lyrics.gui import DropZone

    song = tmp_path / "dropped.mp3"
    song.write_bytes(b"x")
    zone = DropZone()
    seen: list[list[str]] = []
    zone.pathsDropped.connect(seen.append)
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(song))])
    event = QDropEvent(QPointF(4, 4), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    zone.dropEvent(event)
    assert seen == [[str(song)]]

    window._on_paths_dropped(seen[0])
    deadline = time.time() + 5
    while time.time() < deadline and window.queue_table.rowCount() == 0:
        qapp.processEvents()
        time.sleep(0.02)
    assert window.queue_table.rowCount() == 1
    assert window.queue_table.item(0, 0).text() == "dropped.mp3"


def test_history_view_renders_embedded_thumbnail(window, tmp_path, qapp):
    from local_sync_lyrics.app_paths import database_path
    from local_sync_lyrics.database import LyricsDatabase, TrackRecord

    song = tmp_path / "history-song.mp3"
    song.write_bytes(b"x")
    png = _png_bytes()
    with LyricsDatabase(database_path()) as db:
        db.upsert(TrackRecord(
            "history-one", str(song), "synced_online", artist="The Artist",
            title="The Song", album="The Album", year="2025", language="English",
            synced_at=time.time(), thumbnail=png, thumbnail_mime="image/png"))

    window._reload_history()
    deadline = time.time() + 5
    while time.time() < deadline and window.history_list.count() == 0:
        qapp.processEvents()
        time.sleep(0.02)
    assert window.history_list.count() == 1
    item = window.history_list.item(0)
    assert item.text().startswith("The Song")
    assert "The Artist" in item.text()
    assert not item.icon().isNull()
    assert window.history_count_label.text().startswith("1")


def test_settings_round_trip_for_update_preferences(tmp_path):
    from local_sync_lyrics.config import DEFAULT_UPDATE_REPO, Settings

    settings = Settings(auto_check_updates=False, update_repo=" owner/repo ",
                        last_update_check=123.5)
    settings._normalise()
    assert settings.auto_check_updates is False
    assert settings.update_repo == "owner/repo"
    assert settings.last_update_check == 123.5

    restored = Settings.from_dict(settings.to_dict())
    assert restored.auto_check_updates is False
    assert restored.update_repo == "owner/repo"

    broken = Settings(update_repo="not a repo", last_update_check="nope")
    broken._normalise()
    assert broken.update_repo == DEFAULT_UPDATE_REPO
    assert broken.last_update_check == 0.0


def test_version_has_one_source_of_truth():
    import local_sync_lyrics
    from local_sync_lyrics.config import __version__

    assert local_sync_lyrics.__version__ == __version__


# -------------------------------------------------------------------- discovery
def test_discover_audio_accepts_a_single_file_and_a_folder(tmp_path):
    from local_sync_lyrics.processor import discover_audio

    song = tmp_path / "song.mp3"
    song.write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("ignore me")
    assert discover_audio(song) == [song]
    assert discover_audio(tmp_path) == [song]


# ----------------------------------------------------------------------- theme
def test_qss_covers_the_phase4_shell():
    from local_sync_lyrics import theme

    qss = theme.build_qss(theme.build_palette(True))
    for selector in ("QFrame#WindowFrame", "QFrame#TitleBar", "QPushButton#TitleButton",
                     "QFrame#DropZone", "QListWidget#HistoryGrid",
                     "QFrame#UpdateBanner", "QLabel#DropGlyph"):
        assert selector in qss


# ------------------------------------------------------------------- packaging
def test_build_dry_run_prints_a_onedir_command():
    result = subprocess.run(
        [sys.executable, str(ROOT / "build.py"), "--dry-run"],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    command = result.stdout
    assert "--onedir" in command
    assert "--add-data" in command
    assert "--hidden-import mutagen" in command
    assert "--exclude-module PySide6.QtWebEngineCore" in command
    assert "--collect-all faster_whisper" in command


def test_build_dry_run_supports_onefile_and_clean():
    result = subprocess.run(
        [sys.executable, str(ROOT / "build.py"), "--onefile", "--clean", "--dry-run"],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "--onefile" in result.stdout
    assert "--clean" in result.stdout
