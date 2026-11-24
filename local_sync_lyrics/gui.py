"""Native PySide6 (Qt 6) desktop front-end for **Lyricist** - *Liquid Glass* edition.

Design
------
Layered translucent glass over an ambient gradient: a dark (or frosted) gradient plane,
``rgba(...)`` cards with a 1 px specular rim, soft 14-24 px corners and one accent colour.
Everything is painted by the generated Qt Style Sheet in :func:`local_sync_lyrics.theme.build_qss`,
so switching dark/light or the accent repaints the whole window instantly - no restart.

Threading model
---------------
* All heavy work (network, Whisper, vocal separation, metadata) runs in a
  :class:`BatchWorker` (a ``QThread``).  It never touches a widget: it emits
  :class:`~local_sync_lyrics.models.Event` objects through a Qt signal which the GUI
  receives on the UI thread (queued connection).
* Disk work started from a handler (saving settings, clearing the cache, the startup
  hardware probe) runs on :class:`Job` runnables in a ``QThreadPool``.
* :class:`QtLogHandler` is a thread-safe ``logging.Handler`` that forwards records to the
  activity console through a Qt signal, so log output from any thread is safe.
"""

from __future__ import annotations

import logging
import math
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from PySide6.QtCore import (
    QEasingCurve, QObject, QPointF, QProcess, QPropertyAnimation, QRunnable, QRectF, QSize,
    Qt, QThread, QThreadPool, QTimer, QUrl, Signal, Slot,
)
from PySide6.QtGui import (
    QColor, QDesktopServices, QFont, QGuiApplication, QIcon, QImage, QKeySequence,
    QPainter, QPen, QPixmap, QShortcut, QTextCharFormat, QTextCursor,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog,
    QFileDialog, QFormLayout, QFrame, QGraphicsDropShadowEffect, QGraphicsOpacityEffect,
    QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMenu, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QScrollArea, QSizeGrip, QSizePolicy, QSpinBox, QStackedWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from . import __version__
from .api_providers import test_custom_provider
from .app_paths import asset_path, data_dir, database_path, logs_dir
from .config import (
    ACCENT_PRESETS, CREDITS, CUSTOM_ACCENT_ID, GITHUB_RELEASES_URL, OUTPUT_MODE_LABELS,
    OUTPUT_MODES, PROVIDER_IDS, SUPPORT_LINKS, TAG_MODE_LABELS, TAG_MODES, TIER1_IDS,
    TIER2_IDS, CustomProvider, Settings, SettingsStore, find_accent_preset,
)
from .database import HistoryRecord, LyricsDatabase, clear_cache_file
from .hardware_profiler import HardwareProfile, get_profile, run_preflight
from .language import detect_text_language
from .lrc_utils import (
    LrcLine, compose_lrc, format_timestamp, parse_timestamp, repair_lrc, split_lrc,
    write_text_atomic,
)
from .metadata import read_embedded_lyrics, save_lyrics
from .models import Event
from .processor import AUDIO_EXTENSIONS, BatchProcessor, discover_audio
from .theme import (
    Palette, build_palette, build_qss, darken, lighten, normalise_hex, on_color, parse_hex,
    qss_color, qss_hex, resolve_dark, with_alpha,
)
from .updater import UpdateInfo, check_for_update, release_notes

logger = logging.getLogger(__name__)

APP_TITLE = "Lyricist"
MAX_LOG_LINES = 1500
SIDEBAR_WIDTH = 236
SHADOW_MARGIN = 18
VIEWS = (
    ("workspace", "Sync Workspace", "workspace"),
    ("history", "History & Library", "history"),
    ("providers", "Providers & Rules", "providers"),
    ("settings", "Settings & About", "settings"),
)
QUEUE_STATUS = {
    "queued": ("Queued", "text3"),
    "isolating": ("Isolating Vocals", "accent"),
    "syncing": ("Syncing", "accent"),
    "tagging": ("Auto-Tagging", "warn"),
    "done": ("Done", "ok"),
    "error": ("Error", "err"),
    "not_found": ("Not Found", "warn"),
    "skipped": ("Skipped", "text3"),
    "instrumental": ("Instrumental", "text3"),
}
HISTORY_STATUS = {
    "all": "All tracks",
    "synced_online": "Synced online",
    "synced_aligned": "Synced (aligned)",
    "synced_ai": "Synced (AI)",
    "plain_only": "Plain text only",
    "instrumental": "Instrumental",
    "not_found": "Not found",
    "failed": "Failed",
}
QUEUE_DONE_STATUS = {
    "synced_online": "done",
    "synced_aligned": "done",
    "synced_ai": "done",
    "plain_only": "done",
    "not_found": "not_found",
    "failed": "error",
    "skipped": "skipped",
    "instrumental": "instrumental",
}

LANGUAGES = [
    ("auto", "Auto-detect"), ("en", "English"), ("es", "Spanish"), ("fr", "French"),
    ("de", "German"), ("it", "Italian"), ("pt", "Portuguese"), ("nl", "Dutch"),
    ("sv", "Swedish"), ("pl", "Polish"), ("tr", "Turkish"), ("ru", "Russian"),
    ("uk", "Ukrainian"), ("ar", "Arabic"), ("hi", "Hindi"), ("ja", "Japanese"),
    ("ko", "Korean"), ("zh", "Chinese"), ("id", "Indonesian"), ("vi", "Vietnamese"),
    ("th", "Thai"), ("he", "Hebrew"), ("fa", "Persian"),
]
MODEL_LABELS = {
    "auto": "Auto (recommended for this PC)",
    "tiny": "tiny - fastest, lowest accuracy",
    "base": "base - fast",
    "small": "small - balanced",
    "medium": "medium - accurate, heavy",
    "large-v3-turbo": "large-v3-turbo - fast & accurate (GPU)",
    "large-v3": "large-v3 - best accuracy (GPU)",
}
PROVIDER_LABELS = {
    "lrclib": "LRCLIB", "betterlyrics": "BetterLyrics", "rclyricsband": "RCLyricsBand",
    "megalobiz": "Megalobiz", "netease": "NetEase", "qqmusic": "QQ Music",
    "genius": "Genius", "lyricsovh": "Lyrics.ovh", "musixmatch": "Musixmatch (needs API key)",
}
MODE_SEGMENTS = (
    ("dark", "Dark", "Liquid Obsidian"),
    ("light", "Light", "Frosted Pearl"),
    ("system", "System", "Follows your OS"),
)
STAT_LABELS = (
    ("synced_online", "Online"), ("synced_aligned", "Aligned"), ("synced_ai", "Blind AI"),
    ("skipped", "Skipped"), ("not_found", "Not found"), ("failed", "Failed"),
)
RAINBOW = ("#FF453A", "#FF9F0A", "#FFD60A", "#32D74B", "#0A84FF", "#BF5AF2", "#FF453A")


# --------------------------------------------------------------------------- helpers
def _system_is_dark() -> bool:
    """Best-effort OS colour-scheme probe (unknown -> dark)."""
    try:
        hints = QGuiApplication.styleHints()
        return hints.colorScheme() != Qt.ColorScheme.Light
    except Exception:  # pragma: no cover - very old Qt / headless
        return True


def _icon_pixmap(size: int) -> Optional[QPixmap]:
    path = asset_path("logo.png")
    if path is None:
        return None
    pix = QPixmap(str(path))
    if pix.isNull():
        return None
    return pix.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def _app_icon() -> QIcon:
    """Best available high-resolution icon for the window/taskbar."""
    if sys.platform.startswith("win"):
        names = ("icon.ico", "logo.png")
    elif sys.platform == "darwin":
        names = ("icon.icns", "icon.png", "logo.png")
    else:
        names = ("icon.png", "logo.png")
    for name in names:
        path = asset_path(name)
        if path is None:
            continue
        icon = QIcon(str(path))
        if not icon.isNull():
            return icon
    return QIcon()


def _vector_icon(kind: str, color: str, size: int = 22) -> QIcon:
    """Crisp, DPI-aware line icons painted with QPainter (no SVG plugin needed)."""
    ratio = 2
    pix = QPixmap(size * ratio, size * ratio)
    pix.setDevicePixelRatio(ratio)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing, True)
    pen = QPen(QColor(color))
    pen.setWidthF(1.75)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    centre = size / 2.0

    if kind == "workspace":                      # two beamed eighth notes
        painter.drawLine(QPointF(8.2, 16.0), QPointF(8.2, 5.4))
        painter.drawLine(QPointF(16.0, 13.9), QPointF(16.0, 3.3))
        painter.drawLine(QPointF(8.2, 5.4), QPointF(16.0, 3.3))
        painter.drawLine(QPointF(8.2, 8.3), QPointF(16.0, 6.2))
        painter.setBrush(QColor(color))
        painter.drawEllipse(QPointF(6.0, 16.2), 2.3, 1.9)
        painter.drawEllipse(QPointF(13.8, 14.1), 2.3, 1.9)
    elif kind == "history":                      # clock with a back arrow
        painter.drawArc(QRectF(4.2, 4.2, 14.2, 14.2), 35 * 16, 285 * 16)
        painter.drawLine(QPointF(4.4, 7.2), QPointF(4.4, 3.6))
        painter.drawLine(QPointF(4.4, 3.6), QPointF(8.0, 3.6))
        painter.drawLine(QPointF(centre + 0.2, centre - 1.6), QPointF(centre + 0.2, centre + 1.8))
        painter.drawLine(QPointF(centre + 0.2, centre + 1.8), QPointF(centre + 3.2, centre + 1.8))
    elif kind == "providers":                    # sliders
        for y, knob in ((6.4, 8.0), (11.0, 14.0), (15.6, 6.2)):
            painter.drawLine(QPointF(3.8, y), QPointF(size - 3.8, y))
            painter.setBrush(QColor(color))
            painter.drawEllipse(QPointF(knob, y), 1.9, 1.9)
            painter.setBrush(Qt.NoBrush)
    else:                                        # settings: cog
        painter.drawEllipse(QPointF(centre, centre), 4.4, 4.4)
        for index in range(8):
            angle = math.radians(index * 45)
            inner, outer = 5.6, 8.0
            painter.drawLine(
                QPointF(centre + inner * math.cos(angle), centre + inner * math.sin(angle)),
                QPointF(centre + outer * math.cos(angle), centre + outer * math.sin(angle)))
        painter.setBrush(QColor(color))
        painter.drawEllipse(QPointF(centre, centre), 1.5, 1.5)
    painter.end()
    return QIcon(pix)


def _fallback_artwork(palette: Palette, size: int = 168) -> QPixmap:
    """Placeholder album tile used when a track has no embedded cover art."""
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing, True)
    fill = QColor(palette.accent)
    fill.setAlpha(52)
    edge = QColor(palette.accent)
    edge.setAlpha(110)
    painter.setBrush(fill)
    painter.setPen(QPen(edge, 1))
    painter.drawRoundedRect(QRectF(0.5, 0.5, size - 1, size - 1), 22, 22)
    pen = QPen(QColor(palette.accent_fg))
    pen.setWidthF(3.0)
    pen.setCapStyle(Qt.RoundCap)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    x = size * 0.38
    painter.drawLine(QPointF(x, size * 0.68), QPointF(x, size * 0.32))
    painter.drawLine(QPointF(x, size * 0.32), QPointF(size * 0.62, size * 0.27))
    painter.setBrush(QColor(palette.accent_fg))
    painter.drawEllipse(QPointF(x - size * 0.045, size * 0.70), size * 0.055, size * 0.045)
    painter.drawEllipse(QPointF(size * 0.575, size * 0.66), size * 0.055, size * 0.045)
    painter.end()
    return pix


def _label(text: str = "", name: Optional[str] = None, *, wrap: bool = False,
           selectable: bool = False) -> QLabel:
    label = QLabel(text)
    if name:
        label.setObjectName(name)
    label.setWordWrap(wrap)
    if selectable:
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return label


def _button(text: str, name: str = "Glass", *, checkable: bool = False) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName(name)
    button.setCursor(Qt.PointingHandCursor)
    button.setCheckable(checkable)
    return button


def _hline(*items: Any, spacing: int = 10) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(spacing)
    row.setContentsMargins(0, 0, 0, 0)
    for item in items:
        if isinstance(item, int):
            row.addSpacing(item)
        elif isinstance(item, QWidget):
            row.addWidget(item)
        else:
            row.addLayout(item)
    return row


def _vbox(*items: Any, spacing: int = 10) -> QVBoxLayout:
    box = QVBoxLayout()
    box.setSpacing(spacing)
    box.setContentsMargins(0, 0, 0, 0)
    for item in items:
        if isinstance(item, int):
            box.addSpacing(item)
        elif isinstance(item, QWidget):
            box.addWidget(item)
        else:
            box.addLayout(item)
    return box


def _repolish(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)


# --------------------------------------------------------------------------- logging
class QtLogBridge(QObject):
    """Carries log records from any thread to the UI thread."""

    message = Signal(str, str, str)  # text, level name, logger name


class QtLogHandler(logging.Handler):
    """Thread-safe ``logging.Handler`` that emits Qt signals into the console."""

    def __init__(self, bridge: QtLogBridge, level: int = logging.WARNING) -> None:
        super().__init__(level)
        self._bridge = bridge
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102 - logging API
        try:
            self._bridge.message.emit(self.format(record), record.levelname, record.name)
        except Exception:  # a closed window must never break logging
            pass


# --------------------------------------------------------------------------- jobs
class Job(QRunnable):
    """Run a callable on the global thread pool and report back with signals."""

    class Signals(QObject):
        done = Signal(object)
        error = Signal(str)

    def __init__(self, fn: Callable[[], Any]) -> None:
        super().__init__()
        self.fn = fn
        self.signals = Job.Signals()

    def run(self) -> None:  # noqa: D102 - QRunnable API
        try:
            result = self.fn()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("background job failed", exc_info=True)
            try:
                self.signals.error.emit(f"{exc.__class__.__name__}: {exc}")
            except RuntimeError:  # the window was closed while the job ran
                pass
            return
        try:
            self.signals.done.emit(result)
        except RuntimeError:  # the window was closed while the job ran
            pass


# --------------------------------------------------------------------------- worker
class BatchWorker(QThread):
    """Runs the whole batch (or one custom-lyrics file) off the UI thread."""

    event = Signal(object)
    confirm = Signal(object)

    def __init__(self, target: Path | list[Path], settings: Settings,
                 cancel: threading.Event, lyrics: Optional[str] = None) -> None:
        super().__init__()
        self.target = target
        self.settings = settings
        self.cancel = cancel
        self.lyrics = lyrics

    # --------------------------------------------------------------- processor
    def run(self) -> None:  # noqa: D102 - QThread API
        db: Optional[LyricsDatabase] = None
        try:
            db = LyricsDatabase(database_path())
            processor = BatchProcessor(self.settings, db, self.event.emit, self.cancel,
                                       ask=self._ask_user)
            if self.lyrics is not None:
                processor.run_custom(self.target, self.lyrics)
            else:
                processor.run(self.target)
        except Exception as exc:
            logger.exception("worker crashed")
            self.event.emit(Event("log", tag="ERR", level="error",
                                  message=f"Fatal error: {exc.__class__.__name__}: {exc}"))
        finally:
            if db is not None:
                try:
                    db.close()
                except Exception:
                    pass
            self.event.emit(Event("finished"))

    def _ask_user(self, payload: dict) -> bool:
        """Called on the worker thread: ask the window, wait for the answer."""
        done = threading.Event()
        box: dict[str, bool] = {"force": False}
        self.confirm.emit({**payload, "_box": box, "_done": done})
        while not done.wait(0.25):
            if self.cancel.is_set():
                return False
        return bool(box["force"])


# --------------------------------------------------------------------------- dialogs
class ConfirmDialog(QDialog):
    """Glass warning shown when pasted lyrics do not match the audio."""

    def __init__(self, data: dict, palette: Palette, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Check the lyrics")
        self.setModal(True)
        self.setMinimumWidth(470)
        self._palette = palette

        percent = int(data.get("percent", 0))
        need = int(round(float(data.get("threshold", 0.35)) * 100))
        weak = data.get("kind") == "weak"
        title = data.get("title") or "this track"
        headline = ("The alignment does not fit these lyrics" if weak
                    else "These lyrics don't seem to match")
        if weak:
            body = (f"After listening to the whole track, only {percent}% of the sung words "
                    f"match the lyrics you pasted (at least {need}% expected).")
        else:
            body = (f"I transcribed a few slices of “{title}” and compared them with your lyrics: "
                    f"only {percent}% similar (at least {need}% expected).")
        audio_lang = str(data.get("audio_language", ""))
        lyrics_lang = str(data.get("lyrics_language", ""))
        note = (f"The audio sounds {audio_lang}, but the lyrics look {lyrics_lang}. "
                if audio_lang and lyrics_lang and audio_lang != lyrics_lang else "")
        advice = ("This usually means the lyrics belong to a different song or version, or are in "
                  "another language. Forcing the sync can produce wrong timestamps.")

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 20)
        root.setSpacing(14)

        head = QHBoxLayout()
        chip = QLabel("⚠")
        chip.setFixedSize(46, 46)
        chip.setAlignment(Qt.AlignCenter)
        chip.setStyleSheet(
            f"background-color: {qss_color(with_alpha(palette.warn, 0.18))};"
            f"border-radius: 15px; font-size: 22px; color: {qss_hex(palette.warn)};")
        titles = QVBoxLayout()
        titles.setSpacing(2)
        titles.addWidget(_label(headline, "Heading"))
        titles.addWidget(_label("Nothing has been written yet.", "Tiny"))
        head.addWidget(chip)
        head.addLayout(titles, 1)
        root.addLayout(head)

        root.addWidget(_label(body, "Caption", wrap=True))
        root.addWidget(_label(note + advice, "Tiny", wrap=True))

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = _button("Cancel", "Glass")
        force = _button("Force Sync Anyway", "Primary")
        force.setStyleSheet(f"background-color: {qss_hex(palette.warn)};"
                            f"color: {qss_hex(on_color(palette.warn))};")
        cancel.clicked.connect(self.reject)
        force.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(force)
        root.addLayout(buttons)


class CustomLyricsDialog(QDialog):
    """“Your Own Lyrics”: pick one file, paste the words, match & sync."""

    submitted = Signal(str, str)  # audio path, lyrics

    def __init__(self, parent: QWidget, *, output_mode: str, isolate_on: bool) -> None:
        super().__init__(parent)
        self.setWindowTitle("Your Own Lyrics")
        self.setMinimumSize(560, 600)
        self.audio_path = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 20)
        root.setSpacing(14)

        root.addWidget(_label("Your Own Lyrics", "Heading"))
        root.addWidget(_label("Checked against the audio before syncing.", "Caption"))

        self.audio_name = _label("No audio file selected", "Caption")
        pick = _button("Choose audio…", "Glass")
        pick.clicked.connect(self._pick_audio)
        audio_row = QHBoxLayout()
        audio_row.setSpacing(12)
        audio_row.addWidget(pick)
        audio_row.addWidget(self.audio_name, 1)
        root.addLayout(audio_row)

        self.lyrics = QPlainTextEdit()
        self.lyrics.setPlaceholderText(
            "Paste the lyrics here - plain text, one line per sung line.\n"
            "Section labels such as [Chorus] are ignored.")
        self.lyrics.setObjectName("Lyrics")
        self.lyrics.textChanged.connect(self._update_counter)
        root.addWidget(self.lyrics, 1)

        self.counter = _label("0 lines", "Tiny")
        paste = _button("Paste", "Quiet")
        clear = _button("Clear", "Quiet")
        paste.clicked.connect(self._paste)
        clear.clicked.connect(self.lyrics.clear)
        counter_row = QHBoxLayout()
        counter_row.setSpacing(6)
        counter_row.addWidget(self.counter)
        counter_row.addStretch(1)
        counter_row.addWidget(paste)
        counter_row.addWidget(clear)
        root.addLayout(counter_row)

        options = (f"Output: {OUTPUT_MODE_LABELS.get(output_mode, '')}"
                   f"   ·   Vocal isolation: {'on' if isolate_on else 'off'}")
        self.options_label = _label(options, "Tiny", wrap=True)
        root.addWidget(self.options_label)

        bottom = QHBoxLayout()
        bottom.addStretch(1)
        close = _button("Cancel", "Glass")
        sync = _button("Match && Sync", "Primary")
        close.clicked.connect(self.reject)
        sync.clicked.connect(self._submit)
        bottom.addWidget(close)
        bottom.addWidget(sync)
        root.addLayout(bottom)

    # ------------------------------------------------------------------ actions
    def _pick_audio(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in sorted(AUDIO_EXTENSIONS))
        path, _ = QFileDialog.getOpenFileName(
            self, "Select the song", "", f"Audio files ({patterns});;All files (*)")
        if path:
            self.audio_path = path
            self.audio_name.setText(Path(path).name)

    def _paste(self) -> None:
        text = QGuiApplication.clipboard().text()
        if text:
            self.lyrics.setPlainText(text)

    def _update_counter(self) -> None:
        lines = [ln for ln in self.lyrics.toPlainText().splitlines() if ln.strip()]
        self.counter.setText(f"{len(lines)} line{'s' if len(lines) != 1 else ''}")

    def refresh_options(self, *, output_mode: str, isolate_on: bool) -> None:
        self.options_label.setText(
            f"Output: {OUTPUT_MODE_LABELS.get(output_mode, '')}"
            f"   ·   Vocal isolation: {'on' if isolate_on else 'off'}")

    def _submit(self) -> None:
        audio = Path(os.path.expanduser(self.audio_path)) if self.audio_path else None
        if audio is None or not audio.is_file():
            QMessageBox.information(self, APP_TITLE, "Choose the audio file first.")
            return
        lyrics = self.lyrics.toPlainText().strip()
        if len([ln for ln in lyrics.splitlines() if ln.strip()]) < 2:
            QMessageBox.information(self, APP_TITLE, "Paste at least two lines of lyrics first.")
            return
        self.submitted.emit(str(audio), lyrics)
        self.accept()


# --------------------------------------------------------------------------- window
def _stamp_text(seconds: float) -> str:
    """Editable timestamp cell text (``mm:ss.cc`` - no brackets)."""
    return format_timestamp(seconds).strip("[]")


class LyricEditorDialog(QDialog):
    """Review / edit / re-time lyrics, then write them back (Phase 3).

    The table and the raw ``.lrc`` view are two faces of the same document. Saving runs
    on a thread-pool job, so a slow disk or tag rewrite never freezes the UI.
    """

    saved = Signal(object)   # SaveLyricsResult

    def __init__(self, path: Path, text: str, palette: Palette, *,
                 output_mode: str = "lrc",
                 saver: Optional[Callable[[str], Any]] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("EditorDialog")
        self.setWindowTitle(f"Lyric editor - {Path(path).name}")
        self.resize(920, 660)
        self.setMinimumSize(760, 520)
        self.path = Path(path)
        self.palette = palette
        self.output_mode = output_mode
        self._saver = saver or (lambda value: save_lyrics(self.path, value,
                                                          output_mode=self.output_mode))
        self._header, lines = split_lrc(text or "")
        self._jobs: set[Job] = set()
        self.pool = QThreadPool.globalInstance()
        self.edited_text = ""
        self._build()
        self._load_lines(lines)
        if not lines:
            self._set_status("No timed lyrics yet - add a line and a timestamp.", ok=False)

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        box = QVBoxLayout(self)
        box.setContentsMargins(22, 20, 22, 18)
        box.setSpacing(12)

        header = QVBoxLayout()
        header.setSpacing(2)
        header.addWidget(_label("Lyric Editor", "Title"))
        header.addWidget(_label(str(self.path), "Tiny", wrap=True, selectable=True))
        box.addLayout(header)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        self.insert_btn = _button("Insert Line", "Glass")
        self.insert_btn.clicked.connect(self._insert_row)
        self.delete_btn = _button("Delete Line", "Danger")
        self.delete_btn.clicked.connect(self._delete_rows)
        self.shift_spin = QSpinBox()
        self.shift_spin.setRange(10, 10_000)
        self.shift_spin.setSingleStep(50)
        self.shift_spin.setValue(100)
        self.shift_spin.setSuffix(" ms")
        self.shift_spin.setFixedWidth(110)
        self.shift_back = _button("Shift -", "Glass")
        self.shift_back.clicked.connect(lambda: self._shift(-self.shift_spin.value()))
        self.shift_fwd = _button("Shift +", "Glass")
        self.shift_fwd.clicked.connect(lambda: self._shift(self.shift_spin.value()))
        self.validate_btn = _button("Validate Timestamps", "Quiet")
        self.validate_btn.clicked.connect(self._validate)
        toolbar.addWidget(self.insert_btn)
        toolbar.addWidget(self.delete_btn)
        toolbar.addSpacing(8)
        toolbar.addWidget(self.shift_back)
        toolbar.addWidget(self.shift_spin)
        toolbar.addWidget(self.shift_fwd)
        toolbar.addWidget(self.validate_btn)
        toolbar.addStretch(1)
        box.addLayout(toolbar)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Timestamp [mm:ss.xx]", "Lyric line"])
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
                                   | QAbstractItemView.AnyKeyPressed)
        header_view = self.table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.Fixed)
        header_view.setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setColumnWidth(0, 160)
        self.table.itemChanged.connect(lambda *_: None)

        self.raw_view = QPlainTextEdit()
        self.raw_view.setObjectName("RawLrc")
        self.raw_view.setPlaceholderText("[00:12.34]A lyric line")

        self.stack = QStackedWidget()
        self.stack.addWidget(self.table)
        self.stack.addWidget(self.raw_view)
        box.addWidget(self.stack, 1)

        self.status = _label("", "Caption", wrap=True)
        box.addWidget(self.status)

        footer = QHBoxLayout()
        footer.setSpacing(10)
        self.raw_toggle = QCheckBox("Raw .lrc view")
        self.raw_toggle.toggled.connect(self._toggle_raw)
        self.cancel_btn = _button("Cancel", "Glass")
        self.cancel_btn.clicked.connect(self.reject)
        self.save_btn = _button("Save && Apply", "Primary")
        self.save_btn.clicked.connect(self._save)
        footer.addWidget(self.raw_toggle)
        footer.addStretch(1)
        footer.addWidget(self.cancel_btn)
        footer.addWidget(self.save_btn)
        box.addLayout(footer)

    # ------------------------------------------------------------------- data
    def _append_row(self, time: float, text: str, row: Optional[int] = None) -> None:
        row = self.table.rowCount() if row is None else row
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(_stamp_text(time)))
        self.table.setItem(row, 1, QTableWidgetItem(text))

    def _load_lines(self, lines: list[LrcLine]) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for line in lines:
            self._append_row(line.time, line.text)
        self.table.blockSignals(False)

    def _row_lines(self) -> tuple[list[LrcLine], list[str]]:
        lines: list[LrcLine] = []
        problems: list[str] = []
        for row in range(self.table.rowCount()):
            stamp_item = self.table.item(row, 0)
            text_item = self.table.item(row, 1)
            stamp = stamp_item.text().strip() if stamp_item else ""
            text = text_item.text().strip() if text_item else ""
            if not stamp and not text:
                continue
            seconds = parse_timestamp(stamp)
            if seconds is None:
                problems.append(f"Row {row + 1}: “{stamp}” is not a mm:ss.xx timestamp")
                continue
            lines.append(LrcLine(seconds, text))
        lines.sort(key=lambda line: line.time)
        return lines, problems

    def _current_text(self) -> tuple[str, list[str]]:
        if self.raw_toggle.isChecked():
            fixed, problems = repair_lrc(self.raw_view.toPlainText())
            header, lines = split_lrc(fixed)
            return compose_lrc(header or self._header, lines), problems
        lines, problems = self._row_lines()
        return compose_lrc(self._header, lines), problems

    # ----------------------------------------------------------------- actions
    def _insert_row(self) -> None:
        current = self.table.currentRow()
        row = self.table.rowCount() if current < 0 else current + 1
        lines, _ = self._row_lines()
        last = max((line.time for line in lines), default=0.0)
        if current >= 0:
            base = parse_timestamp(self.table.item(current, 0).text()) or last
            new_time = base + 2.0
        else:
            new_time = last + 2.0 if lines else 0.0
        self._append_row(new_time, "", row)
        self.table.setCurrentCell(row, 1)
        self.table.editItem(self.table.item(row, 1))
        self._set_status("Line inserted - type the words and a timestamp.", ok=True)

    def _delete_rows(self) -> None:
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        if not rows:
            rows = [self.table.currentRow()] if self.table.currentRow() >= 0 else []
        for row in rows:
            if 0 <= row < self.table.rowCount():
                self.table.removeRow(row)
        self._set_status(f"Removed {len(rows)} line(s).", ok=True)

    def _shift(self, delta_ms: int) -> None:
        if self.raw_toggle.isChecked():
            fixed, problems = repair_lrc(self.raw_view.toPlainText())
            header, lines = split_lrc(fixed)
            shifted = sorted(
                (LrcLine(max(0.0, line.time + delta_ms / 1000.0), line.text) for line in lines),
                key=lambda line: line.time)
            self.raw_view.setPlainText(compose_lrc(header or self._header, shifted))
            message = f"Shifted every timestamp by {delta_ms:+d} ms."
            if problems:
                message += f" {len(problems)} untimed line(s) were left alone."
            self._set_status(message, ok=not problems)
            return
        problems = 0
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is None:
                continue
            seconds = parse_timestamp(item.text())
            if seconds is None:
                problems += 1
                continue
            item.setText(_stamp_text(max(0.0, seconds + delta_ms / 1000.0)))
        self.table.blockSignals(False)
        message = f"Shifted every timestamp by {delta_ms:+d} ms."
        if problems:
            message += f" {problems} invalid timestamp(s) were left alone."
        self._set_status(message, ok=not problems)

    def _validate(self) -> None:
        text, problems = self._current_text()
        fixed, repair_problems = repair_lrc(text)
        header, lines = split_lrc(fixed)
        self._load_lines(lines)
        if self.raw_toggle.isChecked():
            self.raw_view.setPlainText(compose_lrc(header, lines))
        if problems or repair_problems:
            shown = "; ".join((problems + repair_problems)[:4])
            self._set_status(f"Could not time {len(problems + repair_problems)} line(s): {shown}",
                             ok=False)
        else:
            self._set_status(f"All {len(lines)} timestamps are valid and sorted.", ok=True)

    def _toggle_raw(self, raw: bool) -> None:
        self.insert_btn.setEnabled(not raw)
        self.delete_btn.setEnabled(not raw)
        if raw:
            lines, problems = self._row_lines()          # read the table, not the empty view
            self.raw_view.setPlainText(compose_lrc(self._header, lines))
            self.stack.setCurrentWidget(self.raw_view)
            if problems:
                self._set_status("Fix the raw text before switching back.", ok=False)
        else:
            fixed, problems = repair_lrc(self.raw_view.toPlainText())
            header, lines = split_lrc(fixed)
            if header:
                self._header = header
            self._load_lines(lines)
            self.stack.setCurrentWidget(self.table)
            if problems:
                self._set_status(f"{len(problems)} line(s) had no timestamp and were ignored.",
                                 ok=False)
            else:
                self._set_status(f"Loaded {len(lines)} timed line(s).", ok=True)

    # ------------------------------------------------------------------- save
    def _set_busy(self, busy: bool) -> None:
        for widget in (self.insert_btn, self.delete_btn, self.shift_back, self.shift_fwd,
                       self.shift_spin, self.validate_btn, self.raw_toggle, self.cancel_btn,
                       self.table, self.raw_view):
            widget.setEnabled(not busy)
        self.save_btn.setEnabled(not busy)
        self.save_btn.setText("Saving…" if busy else "Save && Apply")

    def _set_status(self, message: str, *, ok: bool = True) -> None:
        self.status.setText(message)
        self.status.setObjectName("SuccessText" if ok else "ErrorText")
        _repolish(self.status)

    def _save(self) -> None:
        text, problems = self._current_text()
        if problems:
            self._set_status("Fix these first: " + "; ".join(problems[:4]), ok=False)
            return
        header, lines = split_lrc(text)
        if not lines:
            self._set_status("There are no timed lines to save.", ok=False)
            return
        if self.raw_toggle.isChecked() and header:
            self._header = header
        text = compose_lrc(self._header, lines)
        self.edited_text = text
        self._set_busy(True)
        self._set_status("Writing the lyrics…", ok=True)

        job = Job(lambda: self._saver(text))
        job.signals.done.connect(self._on_saved)
        job.signals.error.connect(self._on_save_error)
        self._jobs.add(job)
        self.pool.start(job)

    @Slot(object)
    def _on_saved(self, result: Any) -> None:
        self._jobs.clear()
        self._set_busy(False)
        ok = bool(getattr(result, "ok", False))
        if ok:
            self.saved.emit(result)
            self.accept()
            return
        message = getattr(result, "describe", lambda: str(result))()
        self._set_status(f"Could not save: {message}", ok=False)

    @Slot(str)
    def _on_save_error(self, message: str) -> None:
        self._jobs.clear()
        self._set_busy(False)
        self._set_status(f"Could not save: {message}", ok=False)


class CustomProviderForm(QDialog):
    """Add / edit one custom lyrics endpoint."""

    def __init__(self, provider: Optional[CustomProvider], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ProviderDialog")
        self.setWindowTitle("Custom lyrics provider")
        self.setMinimumWidth(560)
        self._original = provider or CustomProvider()
        self._build()

    def _build(self) -> None:
        box = QVBoxLayout(self)
        box.setContentsMargins(22, 20, 22, 18)
        box.setSpacing(12)
        box.addWidget(_label("Custom provider", "Title"))
        box.addWidget(_label(
            "Use {artist}, {title} and {album} in the URL. Values are URL-encoded automatically.",
            "Caption", wrap=True))

        form = QFormLayout()
        form.setSpacing(10)
        self.name_field = QLineEdit(self._original.name)
        self.name_field.setPlaceholderText("My lyrics site")
        self.url_field = QLineEdit(self._original.url_template)
        self.url_field.setPlaceholderText("https://example.com/api/lyrics?artist={artist}&title={title}")
        self.type_dd = QComboBox()
        self.type_dd.addItem("LRC document (the response body is .lrc)", "lrc")
        self.type_dd.addItem("JSON (extract the lyrics with a dot path)", "json")
        index = self.type_dd.findData(self._original.response_type)
        self.type_dd.setCurrentIndex(max(0, index))
        self.path_field = QLineEdit(self._original.json_lyrics_path)
        self.path_field.setPlaceholderText("data.lyrics")
        self.enabled_check = QCheckBox("Enabled")
        self.enabled_check.setChecked(self._original.enabled)
        form.addRow("Name", self.name_field)
        form.addRow("URL template", self.url_field)
        form.addRow("Response", self.type_dd)
        form.addRow("JSON lyrics path", self.path_field)
        form.addRow("", self.enabled_check)
        box.addLayout(form)

        self.error = _label("", "ErrorText", wrap=True)
        box.addWidget(self.error)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = _button("Cancel", "Glass")
        cancel.clicked.connect(self.reject)
        save = _button("Save", "Primary")
        save.clicked.connect(self._accept)
        buttons.addWidget(cancel)
        buttons.addWidget(save)
        box.addLayout(buttons)

        self.type_dd.currentIndexChanged.connect(self._sync_type)
        self._sync_type()

    def _sync_type(self) -> None:
        is_json = self.type_dd.currentData() == "json"
        self.path_field.setEnabled(is_json)
        if is_json and not self.path_field.text().strip():
            self.path_field.setText("data.lyrics")

    def _accept(self) -> None:
        provider = CustomProvider(
            name=self.name_field.text(),
            enabled=self.enabled_check.isChecked(),
            url_template=self.url_field.text(),
            response_type=self.type_dd.currentData() or "lrc",
            json_lyrics_path=self.path_field.text(),
        ).normalised()
        if not provider.name:
            self.error.setText("Give the provider a name.")
            return
        if not provider.url_template.startswith(("http://", "https://")):
            self.error.setText("The URL must start with http:// or https://.")
            return
        if "{title}" not in provider.url_template and "{artist}" not in provider.url_template:
            self.error.setText("The URL needs a {title} or {artist} placeholder.")
            return
        if provider.response_type == "json" and not provider.json_lyrics_path:
            self.error.setText("A JSON provider needs a dot path to the lyrics, e.g. data.lyrics.")
            return
        self.provider = provider
        self.accept()


class CustomProvidersDialog(QDialog):
    """Manage the user's own lyrics endpoints (add / edit / toggle / test / delete)."""

    def __init__(self, providers: list[CustomProvider], palette: Palette,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ProviderDialog")
        self.setWindowTitle("Custom lyrics providers")
        self.resize(860, 520)
        self.setMinimumSize(700, 420)
        self.palette = palette
        self.providers = [p.normalised() for p in providers]
        self._jobs: set[Job] = set()
        self.pool = QThreadPool.globalInstance()
        self._build()
        self._refresh()

    def _build(self) -> None:
        box = QVBoxLayout(self)
        box.setContentsMargins(22, 20, 22, 18)
        box.setSpacing(12)
        box.addWidget(_label("Custom lyrics providers", "Title"))
        box.addWidget(_label(
            "Queried after the built-in providers, in this order. A broken endpoint is "
            "logged and skipped; it can never abort a batch.", "Caption", wrap=True))

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["On", "Name", "Type", "URL template"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setColumnWidth(0, 46)
        self.table.setColumnWidth(1, 180)
        self.table.setColumnWidth(2, 90)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemDoubleClicked.connect(lambda *_: self._edit())
        box.addWidget(self.table, 1)

        self.status = _label("", "Caption", wrap=True)
        box.addWidget(self.status)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        add = _button("Add…", "Primary")
        add.clicked.connect(self._add)
        edit = _button("Edit…", "Glass")
        edit.clicked.connect(self._edit)
        self.test_btn = _button("Test", "Glass")
        self.test_btn.clicked.connect(self._test)
        delete = _button("Delete", "Danger")
        delete.clicked.connect(self._delete)
        close = _button("Close", "Glass")
        close.clicked.connect(self.accept)
        buttons.addWidget(add)
        buttons.addWidget(edit)
        buttons.addWidget(self.test_btn)
        buttons.addWidget(delete)
        buttons.addStretch(1)
        buttons.addWidget(close)
        box.addLayout(buttons)

    # ------------------------------------------------------------------ table
    def _refresh(self) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for provider in self.providers:
            row = self.table.rowCount()
            self.table.insertRow(row)
            on = QTableWidgetItem("")
            on.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
            on.setCheckState(Qt.Checked if provider.enabled else Qt.Unchecked)
            self.table.setItem(row, 0, on)
            self.table.setItem(row, 1, QTableWidgetItem(provider.name))
            self.table.setItem(row, 2, QTableWidgetItem(
                "LRC" if provider.response_type == "lrc" else "JSON"))
            self.table.setItem(row, 3, QTableWidgetItem(provider.url_template))
        self.table.blockSignals(False)
        enabled = sum(1 for provider in self.providers if provider.enabled)
        self._set_status(f"{len(self.providers)} endpoint(s), {enabled} enabled.", ok=True)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 0:
            return
        index = item.row()
        if 0 <= index < len(self.providers):
            self.providers[index].enabled = item.checkState() == Qt.Checked

    def _selected(self) -> Optional[int]:
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        row = rows[0] if rows else self.table.currentRow()
        return row if 0 <= row < len(self.providers) else None

    def _set_status(self, message: str, *, ok: bool = True) -> None:
        self.status.setText(message)
        self.status.setObjectName("SuccessText" if ok else "ErrorText")
        _repolish(self.status)

    def _add(self) -> None:
        form = CustomProviderForm(None, self)
        if form.exec() == QDialog.Accepted:
            self.providers.append(form.provider)
            self._refresh()
            self.table.selectRow(self.table.rowCount() - 1)
            self._set_status(f"Added “{form.provider.name}”.", ok=True)

    def _edit(self) -> None:
        index = self._selected()
        if index is None:
            self._set_status("Select a provider first.", ok=False)
            return
        form = CustomProviderForm(self.providers[index], self)
        if form.exec() == QDialog.Accepted:
            self.providers[index] = form.provider
            self._refresh()
            self.table.selectRow(index)
            self._set_status(f"Updated “{form.provider.name}”.", ok=True)

    def _delete(self) -> None:
        index = self._selected()
        if index is None:
            self._set_status("Select a provider first.", ok=False)
            return
        provider = self.providers.pop(index)
        self._refresh()
        self._set_status(f"Deleted “{provider.name}”.", ok=True)

    def _test(self) -> None:
        index = self._selected()
        if index is None:
            self._set_status("Select a provider first.", ok=False)
            return
        provider = self.providers[index]
        self.test_btn.setEnabled(False)
        self._set_status(f"Testing “{provider.name}” with Coldplay - Yellow…", ok=True)

        job = Job(lambda: test_custom_provider(provider))
        job.signals.done.connect(self._on_test_done)
        job.signals.error.connect(self._on_test_error)
        self._jobs.add(job)
        self.pool.start(job)

    @Slot(object)
    def _on_test_done(self, result: Any) -> None:
        self._jobs.clear()
        self.test_btn.setEnabled(True)
        self._set_status(str(result), ok=True)

    @Slot(str)
    def _on_test_error(self, message: str) -> None:
        self._jobs.clear()
        self.test_btn.setEnabled(True)
        self._set_status(f"Test failed: {message}", ok=False)


class TitleBar(QFrame):
    """Frameless window chrome: drag, double-click maximise, minimise / close."""

    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self.setObjectName("TitleBar")
        self.setFixedHeight(48)
        self._window = window
        self._drag_offset = None

        row = QHBoxLayout(self)
        row.setContentsMargins(18, 6, 10, 6)
        row.setSpacing(10)

        logo = QLabel()
        pix = _icon_pixmap(26)
        if pix is not None:
            logo.setPixmap(pix)
        else:
            logo.setText("♪")
        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(_label(APP_TITLE, "Brand"))
        titles.addWidget(_label("Synced lyrics, beautifully.", "Tiny"))
        row.addWidget(logo)
        row.addLayout(titles)
        row.addStretch(1)

        self.min_btn = self._title_button("—", "Minimise")
        self.max_btn = self._title_button("▢", "Maximise")
        self.close_btn = self._title_button("✕", "Close")
        self.close_btn.setObjectName("CloseButton")
        self.min_btn.clicked.connect(self._window.showMinimized)
        self.max_btn.clicked.connect(self._window._toggle_maximised)
        self.close_btn.clicked.connect(self._window.close)
        row.addWidget(self.min_btn)
        row.addWidget(self.max_btn)
        row.addWidget(self.close_btn)

    @staticmethod
    def _title_button(glyph: str, tooltip: str) -> QPushButton:
        button = QPushButton(glyph)
        button.setObjectName("TitleButton")
        button.setFixedSize(34, 30)
        button.setToolTip(tooltip)
        button.setCursor(Qt.PointingHandCursor)
        return button

    # ------------------------------------------------------------------ drag
    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.LeftButton:
            handle = self.window().windowHandle()
            if handle is not None:
                try:
                    if handle.startSystemMove():
                        event.accept()
                        return
                except Exception:
                    pass
            self._drag_offset = (event.globalPosition().toPoint()
                                 - self.window().frameGeometry().topLeft())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._drag_offset is not None and event.buttons() & Qt.LeftButton:
            self.window().move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.LeftButton:
            self._window._toggle_maximised()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class DropZone(QFrame):
    """Accept folders and audio files dragged from the file manager."""

    pathsDropped = Signal(list)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("DropZone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(78)
        row = QHBoxLayout(self)
        row.setContentsMargins(18, 12, 18, 12)
        row.setSpacing(14)
        glyph = QLabel("⇩")
        glyph.setObjectName("DropGlyph")
        text = QVBoxLayout()
        text.setSpacing(1)
        text.addWidget(_label("Drop audio files or folders here", "Heading"))
        text.addWidget(_label("MP3 · FLAC · M4A · WAV · OGG · Opus  -  folders are scanned "
                              "recursively", "Tiny"))
        row.addWidget(glyph)
        row.addLayout(text, 1)

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt API
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.pathsDropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()


class UpdateCheckThread(QThread):
    """Runs the GitHub API call off the UI thread."""

    checked = Signal(object)   # UpdateInfo

    def __init__(self, current: str, repo: str, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.current = current
        self.repo = repo

    def run(self) -> None:  # noqa: D102 - QThread API
        try:
            info = check_for_update(self.current, self.repo)
        except Exception as exc:  # check_for_update is defensive, belt and braces
            info = UpdateInfo(current=self.current, error=str(exc))
        self.checked.emit(info)


class UpdateDialog(QDialog):
    """Non-blocking-style update details with the changelog and a download button."""

    def __init__(self, info: UpdateInfo, palette: Palette,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("EditorDialog")
        self.setWindowTitle("Update available")
        self.resize(680, 560)
        self.setMinimumSize(520, 420)
        self.info = info

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 20)
        root.setSpacing(12)
        root.addWidget(_label(f"Lyricist v{info.latest} is available", "Title"))
        root.addWidget(_label(f"You are running v{info.current}.", "Caption"))
        if info.asset_name:
            root.addWidget(_label(f"Download: {info.asset_name}", "Tiny", selectable=True))

        notes = QPlainTextEdit()
        notes.setObjectName("RawLrc")
        notes.setReadOnly(True)
        notes.setPlainText(release_notes(info))
        root.addWidget(notes, 1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        later = _button("Later", "Glass")
        later.clicked.connect(self.reject)
        download = _button("Open Download Page", "Primary")
        download.clicked.connect(self._download)
        buttons.addWidget(later)
        buttons.addWidget(download)
        root.addLayout(buttons)

    def _download(self) -> None:
        QDesktopServices.openUrl(QUrl(self.info.download_url))
        self.accept()


class MainWindow(QMainWindow):
    """The whole application window (titlebar + sidebar + four views)."""

    def __init__(self, store: Optional[SettingsStore] = None,
                 settings: Optional[Settings] = None) -> None:
        super().__init__()
        self.store = store if store is not None else SettingsStore()
        self.settings: Settings = settings if settings is not None else self.store.load()

        self.cancel = threading.Event()
        self.worker: Optional[BatchWorker] = None
        self.profile: Optional[HardwareProfile] = None
        self.pool = QThreadPool.globalInstance()
        self._jobs: set[Job] = set()
        self._save_seq = 0
        self._saved_seq = 0
        self._save_lock = threading.Lock()

        self.view_name = "workspace"
        self._status = "idle"
        self._status_after = "idle"
        self._running = False
        self._run_id = 0
        self.isolation_ready: Optional[bool] = None
        self.log_entries: deque[tuple[str, str, str, str]] = deque(maxlen=MAX_LOG_LINES)
        self.custom_dialog: Optional[CustomLyricsDialog] = None
        self.providers_dialog: Optional[CustomProvidersDialog] = None
        self.editor_dialog: Optional[LyricEditorDialog] = None
        self._last_saved: Optional[dict[str, str]] = None

        # --- Phase 4 state ------------------------------------------------
        self.update_checker: Optional[UpdateCheckThread] = None
        self.update_info: Optional[UpdateInfo] = None
        self._update_manual = False
        self._view_anim: Optional[QPropertyAnimation] = None
        self.history_entries: list[HistoryRecord] = []
        self.history_job_running = False
        self._history_pending = False
        self._history_images: dict[str, QImage] = {}
        self.queue_paths: list[Path] = []
        self.queue_rows: dict[str, int] = {}
        self._current_track = ""
        self._queue_from_folder = False
        self._fallback_art: Optional[QPixmap] = None
        self._shadow: Optional[QGraphicsDropShadowEffect] = None
        self._shortcuts: list[QShortcut] = []

        self.palette = build_palette(resolve_dark(self.settings.theme_mode, _system_is_dark()),
                                     self.settings.accent_color())

        self.setWindowTitle(APP_TITLE)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.resize(1280, 860)
        self.setMinimumSize(1060, 720)
        self.setWindowIcon(_app_icon())

        self._build_ui()
        self._apply_settings_to_controls()
        self._apply_theme()
        self._install_shortcuts()

        self._log_bridge = QtLogBridge()
        self._log_bridge.message.connect(self._on_log_record)
        self._log_handler = QtLogHandler(self._log_bridge)
        logging.getLogger("local_sync_lyrics").addHandler(self._log_handler)

        self._toast_timer = QTimer(self)
        self._toast_timer.setSingleShot(True)
        self._toast_timer.timeout.connect(self._hide_toast)

        QTimer.singleShot(0, self._startup)

    # ================================================================== building
    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        self._root_layout = QVBoxLayout(root)
        self._root_layout.setContentsMargins(SHADOW_MARGIN, SHADOW_MARGIN,
                                             SHADOW_MARGIN, SHADOW_MARGIN)
        self._root_layout.setSpacing(0)

        self.window_frame = QFrame()
        self.window_frame.setObjectName("WindowFrame")
        self._shadow = QGraphicsDropShadowEffect(self.window_frame)
        self._shadow.setBlurRadius(42)
        self._shadow.setOffset(0, 12)
        self._shadow.setColor(QColor(0, 0, 0, 120))
        self.window_frame.setGraphicsEffect(self._shadow)
        self._root_layout.addWidget(self.window_frame)

        frame_layout = QVBoxLayout(self.window_frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)
        self.title_bar = TitleBar(self)
        frame_layout.addWidget(self.title_bar)

        body = QWidget()
        body.setObjectName("Body")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(16, 6, 16, 16)
        body_layout.setSpacing(16)
        body_layout.addWidget(self._build_sidebar())

        self.stack = QStackedWidget()
        self.stack.setObjectName("Stack")
        self.workspace_view = self._build_workspace()
        self.history_view = self._build_history()
        self.providers_view = self._build_providers_view()
        self.settings_view = self._build_settings()
        for view in (self.workspace_view, self.history_view,
                     self.providers_view, self.settings_view):
            self.stack.addWidget(view)
        body_layout.addWidget(self.stack, 1)
        frame_layout.addWidget(body, 1)

        self.size_grip = QSizeGrip(self.window_frame)
        self.size_grip.setFixedSize(18, 18)
        self.size_grip.raise_()

        self.toast = QFrame(root)
        self.toast.setObjectName("Toast")
        self.toast.setVisible(False)
        toast_layout = QHBoxLayout(self.toast)
        toast_layout.setContentsMargins(16, 10, 16, 10)
        self.toast_label = _label("", "Caption", wrap=True)
        toast_layout.addWidget(self.toast_label)

    # ------------------------------------------------------------------ sidebar
    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(SIDEBAR_WIDTH)
        box = QVBoxLayout(sidebar)
        box.setContentsMargins(18, 22, 18, 18)
        box.setSpacing(12)

        brand_row = QHBoxLayout()
        brand_row.setSpacing(12)
        logo = QLabel()
        pix = _icon_pixmap(42)
        if pix is not None:
            logo.setPixmap(pix)
        else:
            logo.setText("♪")
            logo.setStyleSheet("font-size: 28px;")
        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(_label(APP_TITLE, "Brand"))
        titles.addWidget(_label(f"v{__version__}", "Tiny"))
        brand_row.addWidget(logo)
        brand_row.addLayout(titles, 1)
        box.addLayout(brand_row)
        box.addSpacing(6)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_buttons: dict[str, QPushButton] = {}
        for key, text, icon_kind in VIEWS:
            button = _button(f"  {text}", "Nav", checkable=True)
            button.setIconSize(QSize(20, 20))
            button.setProperty("iconKind", icon_kind)
            button.clicked.connect(lambda _=False, k=key: self._show_view(k))
            self.nav_group.addButton(button)
            self.nav_buttons[key] = button
            box.addWidget(button)
        self.nav_buttons["workspace"].setChecked(True)

        box.addStretch(1)

        self.update_badge = _label("", "Tiny", wrap=True)
        self.update_badge.setVisible(False)
        box.addWidget(self.update_badge)

        self.hw_badge = _label("Analysing hardware…", "Tiny", wrap=True)
        box.addWidget(self.hw_badge)

        theme_btn = _button("Toggle light / dark", "Glass")
        theme_btn.clicked.connect(self._toggle_theme)
        box.addWidget(theme_btn)

        box.addWidget(_label("Search first · listen second · guess last", "Tiny", wrap=True))
        return sidebar

    # ---------------------------------------------------------------- workspace
    def _build_workspace(self) -> QWidget:
        page = QWidget()
        page.setObjectName("Page")
        box = QVBoxLayout(page)
        box.setContentsMargins(6, 4, 6, 4)
        box.setSpacing(14)

        # header ---------------------------------------------------------------
        title = _label("Sync Workspace", "Title")
        caption = _label("Synced lyrics for your local music library", "Caption")
        titles = _vbox(title, caption, spacing=2)
        self.custom_btn = _button("Your Own Lyrics", "Glass")
        self.custom_btn.clicked.connect(self._open_custom)
        self.status_dot = QLabel()
        self.status_dot.setFixedSize(8, 8)
        self.status_text = _label("Ready", "Caption")
        status_pill = QFrame()
        status_pill.setObjectName("StatusPill")
        pill_layout = QHBoxLayout(status_pill)
        pill_layout.setContentsMargins(14, 7, 14, 7)
        pill_layout.setSpacing(8)
        pill_layout.addWidget(self.status_dot)
        pill_layout.addWidget(self.status_text)
        header = QHBoxLayout()
        header.setSpacing(12)
        header.addLayout(titles, 1)
        header.addWidget(self.custom_btn)
        header.addWidget(status_pill)
        box.addLayout(header)

        # update banner --------------------------------------------------------
        self.update_banner = QFrame()
        self.update_banner.setObjectName("UpdateBanner")
        update_layout = QHBoxLayout(self.update_banner)
        update_layout.setContentsMargins(14, 10, 14, 10)
        update_layout.setSpacing(10)
        self.update_text = _label("", "BannerText", wrap=True)
        self.update_view_btn = _button("What's new", "Quiet")
        self.update_view_btn.clicked.connect(self._open_update_dialog)
        self.update_dismiss_btn = _button("Dismiss", "Quiet")
        self.update_dismiss_btn.clicked.connect(self._dismiss_update_banner)
        update_layout.addWidget(self.update_text, 1)
        update_layout.addWidget(self.update_view_btn)
        update_layout.addWidget(self.update_dismiss_btn)
        self.update_banner.setVisible(False)
        box.addWidget(self.update_banner)

        # pre-flight banner ----------------------------------------------------
        self.banner = QFrame()
        self.banner.setObjectName("Banner")
        banner_layout = QHBoxLayout(self.banner)
        banner_layout.setContentsMargins(14, 10, 14, 10)
        self.banner_text = _label("", "BannerText", wrap=True)
        banner_layout.addWidget(self.banner_text, 1)
        self.banner.setVisible(False)
        box.addWidget(self.banner)

        # drop zone ------------------------------------------------------------
        self.drop_zone = DropZone()
        self.drop_zone.pathsDropped.connect(self._on_paths_dropped)
        box.addWidget(self.drop_zone)

        # library card ---------------------------------------------------------
        self.folder_field = QLineEdit()
        self.folder_field.setPlaceholderText("Music folder - browse… or paste a path")
        self.folder_field.returnPressed.connect(self._start_batch)
        self.browse_btn = _button("Browse…", "Glass")
        self.browse_btn.clicked.connect(self._browse)
        self.add_files_btn = _button("Add files…", "Glass")
        self.add_files_btn.clicked.connect(self._add_files)
        self.start_btn = _button("Start sync", "Primary")
        self.start_btn.clicked.connect(self._start_batch)
        self.stop_btn = _button("Stop", "Danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)

        folder_row = QHBoxLayout()
        folder_row.setSpacing(10)
        folder_row.addWidget(self.folder_field, 1)
        folder_row.addWidget(self.browse_btn)
        folder_row.addWidget(self.add_files_btn)
        folder_row.addWidget(self.start_btn)
        folder_row.addWidget(self.stop_btn)

        self.output_group = QButtonGroup(self)
        self.output_group.setExclusive(True)
        self.output_buttons: dict[str, QPushButton] = {}
        output_row = QHBoxLayout()
        output_row.setSpacing(8)
        for mode in OUTPUT_MODES:
            button = _button(OUTPUT_MODE_LABELS[mode], "Segment", checkable=True)
            button.clicked.connect(lambda _=False, m=mode: self._select_output(m))
            self.output_group.addButton(button)
            self.output_buttons[mode] = button
            output_row.addWidget(button, 1)

        self.sw_isolate = QCheckBox("Isolate Vocals (Strip Background Music)")
        self.sw_isolate.toggled.connect(self._toggle_isolate)
        self.isolate_hint = _label("Slower - best for dense or noisy mixes", "Tiny")
        isolate_row = QHBoxLayout()
        isolate_row.setSpacing(10)
        isolate_row.addWidget(self.sw_isolate)
        isolate_row.addWidget(self.isolate_hint)
        isolate_row.addStretch(1)

        self.sw_autotag = QCheckBox("Auto-tag & cover art")
        self.sw_autotag.toggled.connect(self._toggle_autotag)
        self.tag_mode_dd = QComboBox()
        for mode in TAG_MODES:
            self.tag_mode_dd.addItem(TAG_MODE_LABELS[mode], mode)
        self.tag_mode_dd.setFixedWidth(150)
        self.tag_mode_dd.currentIndexChanged.connect(self._select_tag_mode)
        self.tag_hint = _label("MusicBrainz + Cover Art Archive, iTunes fallback", "Tiny")
        tag_row = QHBoxLayout()
        tag_row.setSpacing(10)
        tag_row.addWidget(self.sw_autotag)
        tag_row.addWidget(self.tag_mode_dd)
        tag_row.addWidget(self.tag_hint)
        tag_row.addStretch(1)

        library = self._card(_vbox(folder_row, output_row, isolate_row, tag_row, spacing=12))
        box.addWidget(library)

        # queue card -----------------------------------------------------------
        queue_header = QHBoxLayout()
        queue_header.setSpacing(8)
        queue_header.addWidget(_label("Sync queue", "SectionLabel"))
        self.queue_count = _label("No tracks queued", "Tiny")
        queue_header.addWidget(self.queue_count)
        queue_header.addStretch(1)
        self.queue_clear_btn = _button("Clear", "Quiet")
        self.queue_clear_btn.clicked.connect(self._queue_clear)
        queue_header.addWidget(self.queue_clear_btn)

        self.queue_table = QTableWidget(0, 4)
        self.queue_table.setObjectName("QueueTable")
        self.queue_table.setHorizontalHeaderLabels(["Track", "Status", "Language", "Folder"])
        self.queue_table.verticalHeader().setVisible(False)
        self.queue_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.queue_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.queue_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.queue_table.setAlternatingRowColors(True)
        self.queue_table.setMinimumHeight(138)
        self.queue_table.setMaximumHeight(210)
        queue_head = self.queue_table.horizontalHeader()
        queue_head.setSectionResizeMode(0, QHeaderView.Stretch)
        queue_head.setSectionResizeMode(1, QHeaderView.Fixed)
        queue_head.setSectionResizeMode(2, QHeaderView.Fixed)
        queue_head.setSectionResizeMode(3, QHeaderView.Fixed)
        self.queue_table.setColumnWidth(1, 142)
        self.queue_table.setColumnWidth(2, 96)
        self.queue_table.setColumnWidth(3, 190)
        self.queue_table.itemSelectionChanged.connect(self._queue_selection_changed)
        self.queue_table.itemDoubleClicked.connect(lambda *_: self._queue_open_editor())

        queue_actions = QHBoxLayout()
        queue_actions.setSpacing(8)
        self.queue_edit_btn = _button("Edit Lyrics", "Glass")
        self.queue_edit_btn.clicked.connect(self._queue_open_editor)
        self.queue_location_btn = _button("Open Location", "Quiet")
        self.queue_location_btn.clicked.connect(self._queue_open_location)
        self.queue_remove_btn = _button("Remove", "Danger")
        self.queue_remove_btn.clicked.connect(self._queue_remove_selected)
        for button in (self.queue_edit_btn, self.queue_location_btn, self.queue_remove_btn):
            button.setEnabled(False)
            queue_actions.addWidget(button)
        queue_actions.addStretch(1)
        queue_card = self._card(_vbox(queue_header, self.queue_table, queue_actions, spacing=8))
        box.addWidget(queue_card)

        # progress card --------------------------------------------------------
        self.progress_card = QFrame()
        self.progress_card.setObjectName("ProgressCard")
        progress_layout = QVBoxLayout(self.progress_card)
        progress_layout.setContentsMargins(20, 16, 20, 16)
        progress_layout.setSpacing(12)
        self.progress_label = _label("Idle", "Heading")
        self.lang_chip = QFrame()
        self.lang_chip.setObjectName("LangChip")
        lang_layout = QHBoxLayout(self.lang_chip)
        lang_layout.setContentsMargins(10, 5, 10, 5)
        self.lang_text = _label("", "Tiny")
        lang_layout.addWidget(self.lang_text)
        self.lang_chip.setVisible(False)
        self.current_file = _label("", "Caption")
        self.current_file.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.progress_stop_btn = _button("Stop", "Danger")
        self.progress_stop_btn.setEnabled(False)
        self.progress_stop_btn.clicked.connect(self._stop)
        top = QHBoxLayout()
        top.setSpacing(12)
        top.addWidget(self.progress_label)
        top.addWidget(self.lang_chip)
        top.addWidget(self.current_file, 1)
        top.addWidget(self.progress_stop_btn)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(10)
        progress_layout.addLayout(top)
        progress_layout.addWidget(self.progress_bar)
        self.progress_card.setVisible(False)
        box.addWidget(self.progress_card)

        # completion card (Phase 3) -------------------------------------------
        self.completion_card = QFrame()
        self.completion_card.setObjectName("CompletionCard")
        completion_layout = QHBoxLayout(self.completion_card)
        completion_layout.setContentsMargins(20, 14, 20, 14)
        completion_layout.setSpacing(12)
        completion_text = QVBoxLayout()
        completion_text.setSpacing(2)
        self.completion_title = _label("Lyrics ready", "Heading")
        self.completion_detail = _label("", "Caption", wrap=True)
        completion_text.addWidget(self.completion_title)
        completion_text.addWidget(self.completion_detail)
        self.edit_btn = _button("View / Edit Lyrics", "Primary")
        self.edit_btn.clicked.connect(self._open_editor)
        self.done_btn = _button("Done", "Glass")
        self.done_btn.clicked.connect(self._dismiss_completion)
        completion_layout.addLayout(completion_text, 1)
        completion_layout.addWidget(self.edit_btn)
        completion_layout.addWidget(self.done_btn)
        self.completion_card.setVisible(False)
        box.addWidget(self.completion_card)

        # stats ----------------------------------------------------------------
        stats = QHBoxLayout()
        stats.setSpacing(10)
        self.stat_values: dict[str, QLabel] = {}
        for key, caption in STAT_LABELS:
            tile = QFrame()
            tile.setObjectName("Card")
            tile_layout = QVBoxLayout(tile)
            tile_layout.setContentsMargins(16, 12, 16, 12)
            tile_layout.setSpacing(2)
            value = _label("0", "StatValue")
            value.setStyleSheet(f"color: {qss_hex(self.palette.text)};")
            tile_layout.addWidget(value)
            tile_layout.addWidget(_label(caption, "StatCaption"))
            self.stat_values[key] = value
            stats.addWidget(tile, 1)
        box.addLayout(stats)

        # log ------------------------------------------------------------------
        log_header = QHBoxLayout()
        log_header.setSpacing(8)
        log_header.addWidget(_label("Activity log", "SectionLabel"))
        log_header.addStretch(1)
        copy_btn = _button("Copy", "Quiet")
        clear_btn = _button("Clear", "Quiet")
        copy_btn.clicked.connect(self._copy_log)
        clear_btn.clicked.connect(self._clear_log)
        log_header.addWidget(copy_btn)
        log_header.addWidget(clear_btn)
        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("Log")
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(MAX_LOG_LINES)
        self.log_view.setFont(QFont("DejaVu Sans Mono", 10))
        log_card = self._card(_vbox(log_header, self.log_view, spacing=8), strong=True)
        box.addWidget(log_card, 1)
        return page

    # ----------------------------------------------------------------- settings
    def _build_settings(self) -> QWidget:
        page = QWidget()
        page.setObjectName("Page")
        box = QVBoxLayout(page)
        box.setContentsMargins(6, 4, 6, 4)
        box.setSpacing(14)

        titles = _vbox(_label("Settings & About", "Title"),
                       _label("Appearance, local AI, updates and support", "Caption"), spacing=2)
        self.clear_cache_btn = _button("Clear cache", "Glass")
        self.clear_cache_btn.clicked.connect(self._clear_cache)
        self.save_btn = _button("Save settings", "Primary")
        self.save_btn.clicked.connect(self._save_settings)
        header = QHBoxLayout()
        header.setSpacing(10)
        header.addLayout(titles, 1)
        header.addWidget(self.clear_cache_btn)
        header.addWidget(self.save_btn)
        box.addLayout(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        inner.setObjectName("SettingsInner")
        column = QVBoxLayout(inner)
        column.setContentsMargins(2, 2, 10, 8)
        column.setSpacing(14)

        column.addWidget(self._build_appearance_card())
        column.addWidget(self._build_ai_card())
        column.addWidget(self._build_about_card())
        column.addStretch(1)
        scroll.setWidget(inner)
        box.addWidget(scroll, 1)
        return page

    # ------------------------------------------------------- providers & rules
    def _build_providers_view(self) -> QWidget:
        page = QWidget()
        page.setObjectName("Page")
        box = QVBoxLayout(page)
        box.setContentsMargins(6, 4, 6, 4)
        box.setSpacing(14)

        titles = _vbox(_label("Providers & Rules", "Title"),
                       _label("Lyrics sources, smart-pipeline rules, tagging and batch behaviour",
                              "Caption"), spacing=2)
        save = _button("Save changes", "Primary")
        save.clicked.connect(self._save_settings)
        header = QHBoxLayout()
        header.setSpacing(10)
        header.addLayout(titles, 1)
        header.addWidget(save)
        box.addLayout(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        inner.setObjectName("SettingsInner")
        column = QVBoxLayout(inner)
        column.setContentsMargins(2, 2, 10, 8)
        column.setSpacing(14)
        column.addWidget(self._build_pipeline_card())
        column.addWidget(self._build_providers_card())
        column.addWidget(self._build_tags_card())
        column.addWidget(self._build_batch_card())
        column.addStretch(1)
        scroll.setWidget(inner)
        box.addWidget(scroll, 1)
        return page

    # ------------------------------------------------------- history & library
    def _build_history(self) -> QWidget:
        page = QWidget()
        page.setObjectName("Page")
        box = QVBoxLayout(page)
        box.setContentsMargins(6, 4, 6, 4)
        box.setSpacing(14)

        titles = _vbox(_label("History & Library", "Title"),
                       _label("Every track Lyricist has synced - with its cover art and "
                              "one-click actions", "Caption"), spacing=2)
        self.history_refresh_btn = _button("Refresh", "Glass")
        self.history_refresh_btn.clicked.connect(self._reload_history)
        header = QHBoxLayout()
        header.setSpacing(10)
        header.addLayout(titles, 1)
        header.addWidget(self.history_refresh_btn)
        box.addLayout(header)

        self.history_search = QLineEdit()
        self.history_search.setPlaceholderText("Search title, artist, album or path…")
        self.history_search.setClearButtonEnabled(True)
        self.history_filter = QComboBox()
        for key, text in HISTORY_STATUS.items():
            self.history_filter.addItem(text, key)
        self.history_filter.setFixedWidth(190)
        self.history_count_label = _label("", "Tiny")
        filters = QHBoxLayout()
        filters.setSpacing(10)
        filters.addWidget(self.history_search, 1)
        filters.addWidget(self.history_filter)
        filters.addWidget(self.history_count_label)
        box.addLayout(filters)

        self._history_timer = QTimer(self)
        self._history_timer.setSingleShot(True)
        self._history_timer.setInterval(260)
        self._history_timer.timeout.connect(self._reload_history)
        self.history_search.textChanged.connect(lambda *_: self._history_timer.start())
        self.history_filter.currentIndexChanged.connect(lambda *_: self._reload_history())

        self.history_list = QListWidget()
        self.history_list.setObjectName("HistoryGrid")
        self.history_list.setViewMode(QListWidget.IconMode)
        self.history_list.setResizeMode(QListWidget.Adjust)
        self.history_list.setMovement(QListWidget.Static)
        self.history_list.setIconSize(QSize(156, 156))
        self.history_list.setGridSize(QSize(214, 262))
        self.history_list.setSpacing(8)
        self.history_list.setWordWrap(True)
        self.history_list.setUniformItemSizes(True)
        self.history_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.history_list.itemSelectionChanged.connect(self._history_selection_changed)
        self.history_list.itemDoubleClicked.connect(lambda *_: self._history_open_editor())
        box.addWidget(self.history_list, 1)

        self.history_empty = _label(
            "No tracks yet - sync a folder or drop a few songs to build your library.",
            "Caption", wrap=True)
        self.history_empty.setAlignment(Qt.AlignCenter)
        self.history_empty.setVisible(False)
        box.addWidget(self.history_empty, 1)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.history_edit_btn = _button("Open in Lyric Editor", "Primary")
        self.history_edit_btn.clicked.connect(self._history_open_editor)
        self.history_location_btn = _button("Open File Location", "Glass")
        self.history_location_btn.clicked.connect(self._history_open_location)
        self.history_resync_btn = _button("Re-Sync Track", "Glass")
        self.history_resync_btn.clicked.connect(self._history_resync)
        self.history_export_btn = _button("Export .lrc", "Glass")
        self.history_export_btn.clicked.connect(self._history_export)
        for button in (self.history_edit_btn, self.history_location_btn,
                       self.history_resync_btn, self.history_export_btn):
            button.setEnabled(False)
            actions.addWidget(button)
        actions.addStretch(1)
        box.addLayout(actions)
        return page

    def _build_appearance_card(self) -> QWidget:
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_buttons: dict[str, QPushButton] = {}
        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        for key, title, caption in MODE_SEGMENTS:
            button = _button(f"{title}\n{caption}", "ModeSegment", checkable=True)
            button.clicked.connect(lambda _=False, k=key: self._select_mode(k))
            self.mode_group.addButton(button)
            self.mode_buttons[key] = button
            mode_row.addWidget(button, 1)

        self.swatches: dict[str, QPushButton] = {}
        swatch_row = QHBoxLayout()
        swatch_row.setSpacing(12)
        for preset in ACCENT_PRESETS:
            swatch_row.addWidget(self._swatch(preset.id, preset.label))
        swatch_row.addWidget(self._swatch(CUSTOM_ACCENT_ID, "Custom colour"))
        swatch_row.addStretch(1)

        self.accent_caption = _label("", "AccentCaption")
        self.custom_hex_field = QLineEdit()
        self.custom_hex_field.setPlaceholderText("0A84FF")
        self.custom_hex_field.setMaxLength(7)
        self.custom_hex_field.setFixedWidth(190)
        self.custom_hex_field.textChanged.connect(self._custom_accent_changed)
        self.custom_hex_field.editingFinished.connect(self._custom_accent_submitted)

        body = _vbox(
            _label("Appearance", "SectionLabel"), mode_row,
            _label("Accent colour", "SectionLabel"), swatch_row,
            self.accent_caption, self.custom_hex_field, spacing=10)
        return self._card(body, title="Appearance & Theme",
                          caption="Pick a surface and an accent - changes apply instantly.")

    def _build_ai_card(self) -> QWidget:
        self.model_dd = QComboBox()
        for key, label in MODEL_LABELS.items():
            self.model_dd.addItem(label, key)
        self.lang_dd = QComboBox()
        for key, name in LANGUAGES:
            self.lang_dd.addItem(name if key == "auto" else f"{name} ({key})", key)
        combo_row = QHBoxLayout()
        combo_row.setSpacing(10)
        combo_row.addWidget(self.model_dd, 1)
        combo_row.addWidget(self.lang_dd, 1)

        self.model_hint = _label("", "Caption", wrap=True)
        self.apply_rec_btn = _button("Use recommended model", "Quiet")
        self.apply_rec_btn.setEnabled(False)
        self.apply_rec_btn.clicked.connect(self._apply_recommendation)
        hint_row = QHBoxLayout()
        hint_row.setSpacing(8)
        hint_row.addWidget(self.model_hint, 1)
        hint_row.addWidget(self.apply_rec_btn)

        helper = ("Using HF_TOKEN from the environment" if self.settings.token_from_environment()
                  else "Optional. Speeds up / unlocks model downloads.")
        self.token_field = QLineEdit()
        self.token_field.setEchoMode(QLineEdit.Password)
        self.token_field.setPlaceholderText(f"Hugging Face token - {helper}")

        self.sw_ai = QCheckBox("Use local AI when no online match")
        self.sw_gpu = QCheckBox("Prefer GPU when available")
        self.sw_vad = QCheckBox("Skip instrumental parts (VAD)")
        body = _vbox(combo_row, hint_row, self.token_field,
                     self._switch_grid(self.sw_ai, self.sw_gpu, self.sw_vad), spacing=10)
        return self._card(body, title="Local AI (Whisper)",
                          caption="Used when no online lyrics match the track.")

    def _build_pipeline_card(self) -> QWidget:
        self.sw_align = QCheckBox("Align official lyrics to the audio")
        self.sw_mashup = QCheckBox("Detect mashups / merged tracks")
        self.sw_ident = QCheckBox("Name unknown songs by listening")
        self.sw_detect_lang = QCheckBox("Detect language before transcribing")
        self.isolation_status = _label("Checking…", "Caption", wrap=True, selectable=True)
        body = _vbox(
            self._switch_grid(self.sw_align, self.sw_mashup, self.sw_ident, self.sw_detect_lang),
            _label("Vocal isolation (switch it on in the Sync Workspace)", "SectionLabel"),
            self.isolation_status, spacing=10)
        return self._card(body, title="Smart pipeline",
                          caption="Search first, listen second, guess last.")

    def _build_providers_card(self) -> QWidget:
        self.provider_checks: dict[str, QCheckBox] = {}
        for pid in PROVIDER_IDS:
            self.provider_checks[pid] = QCheckBox(PROVIDER_LABELS[pid])
        tier1 = self._switch_grid(*[self.provider_checks[pid] for pid in TIER1_IDS])
        tier2 = self._switch_grid(*[self.provider_checks[pid] for pid in TIER2_IDS])

        self.genius_field = QLineEdit()
        self.genius_field.setEchoMode(QLineEdit.Password)
        self.genius_field.setPlaceholderText(
            "Genius API token (optional) - without one, public search is used")
        self.musixmatch_field = QLineEdit()
        self.musixmatch_field.setEchoMode(QLineEdit.Password)
        self.musixmatch_field.setPlaceholderText(
            "Musixmatch API key (optional) - free plan returns excerpts only")
        fields = QHBoxLayout()
        fields.setSpacing(10)
        fields.addWidget(self.genius_field, 1)
        fields.addWidget(self.musixmatch_field, 1)

        self.custom_providers_btn = _button("Custom providers…", "Glass")
        self.custom_providers_btn.clicked.connect(self._open_providers)
        self.custom_providers_summary = _label("No custom endpoints yet.", "Caption", wrap=True)
        custom_row = QHBoxLayout()
        custom_row.setSpacing(10)
        custom_row.addWidget(self.custom_providers_summary, 1)
        custom_row.addWidget(self.custom_providers_btn)

        body = _vbox(_label("Tier 1 - synced lyrics", "SectionLabel"), tier1,
                     _label("Tier 2 - plain text (aligned to the audio)", "SectionLabel"), tier2,
                     fields, _label("Your own endpoints", "SectionLabel"), custom_row, spacing=10)
        return self._card(body, title="Lyrics providers",
                          caption="Tier 1 is searched first, in this order.")

    def _build_tags_card(self) -> QWidget:
        self.sw_autotag_settings = QCheckBox("Complete tags and embed cover art")
        self.sw_autotag_settings.toggled.connect(self._sync_autotag_controls)
        self.tag_mode_settings = QComboBox()
        for mode in TAG_MODES:
            self.tag_mode_settings.addItem(TAG_MODE_LABELS[mode], mode)
        self.tag_mode_settings.setFixedWidth(170)
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(self.sw_autotag_settings)
        row.addWidget(self.tag_mode_settings)
        row.addStretch(1)
        body = _vbox(
            row,
            _label("Zero API keys: MusicBrainz + Cover Art Archive first, iTunes as a "
                   "fallback. Writes title, artist, album, album artist, genre, year, "
                   "track number and a high-resolution cover.", "Caption", wrap=True),
            spacing=10)
        return self._card(body, title="Tags & artwork",
                          caption="Runs after lyrics are resolved, inside the same background job.")

    def _build_batch_card(self) -> QWidget:
        self.sw_recursive = QCheckBox("Include sub-folders")
        self.sw_overwrite = QCheckBox("Overwrite existing .lrc files")
        self.sw_retry = QCheckBox("Retry tracks that failed before")
        self.sw_plain = QCheckBox("Save plain .txt as a last resort")
        body = self._switch_grid(self.sw_recursive, self.sw_overwrite, self.sw_retry, self.sw_plain)
        return self._card(body, title="Batch behaviour",
                          caption="How a sync run treats your files.")

    def _build_about_card(self) -> QWidget:
        logo = QLabel()
        pix = _icon_pixmap(64)
        if pix is not None:
            logo.setPixmap(pix)
        else:
            logo.setText("♪")
            logo.setStyleSheet("font-size: 40px;")
        self.data_path_text = _label("Data folder: …", "Tiny", wrap=True, selectable=True)
        self.logs_path_text = _label("Log files: …", "Tiny", wrap=True, selectable=True)
        self.hw_summary = _label("Hardware: analysing…", "Caption", wrap=True, selectable=True)
        text = _vbox(_label(APP_TITLE, "Heading"),
                     _label(f"Version {__version__}  ·  Qt 6 / Liquid Glass interface",
                            "Caption"),
                     self.hw_summary, self.data_path_text, self.logs_path_text, spacing=3)
        brand = QHBoxLayout()
        brand.setSpacing(18)
        brand.addWidget(logo)
        brand.addLayout(text, 1)

        credits = _label(
            "Built with " + ", ".join(name for name, _ in CREDITS) + ". Thank you to every "
            "open-source maintainer who made this possible.", "Tiny", wrap=True)

        support_row = QHBoxLayout()
        support_row.setSpacing(8)
        for label, url in SUPPORT_LINKS:
            button = _button(label, "Quiet")
            button.clicked.connect(lambda _=False, target=url: QDesktopServices.openUrl(QUrl(target)))
            support_row.addWidget(button)
        support_row.addStretch(1)

        self.sw_updates = QCheckBox("Automatically check for updates at startup")
        self.sw_updates.toggled.connect(self._toggle_auto_updates)
        self.check_updates_btn = _button("Check Now", "Glass")
        self.check_updates_btn.clicked.connect(lambda: self._check_updates(manual=True))
        self.update_status = _label("Updates are checked against GitHub Releases.", "Tiny", wrap=True)
        update_row = QHBoxLayout()
        update_row.setSpacing(10)
        update_row.addWidget(self.sw_updates, 1)
        update_row.addWidget(self.check_updates_btn)

        body = _vbox(
            brand,
            _label("About & Support", "SectionLabel"),
            credits,
            support_row,
            _label("Updates", "SectionLabel"),
            update_row,
            self.update_status,
            spacing=10)
        return self._card(body, title="About Lyricist",
                          caption="Version, hardware profile, credits and ways to support the project.")

    # ---------------------------------------------------------------- factories
    def _card(self, body: Any, *, title: Optional[str] = None,
              caption: Optional[str] = None, strong: bool = False) -> QFrame:
        frame = QFrame()
        frame.setObjectName("CardStrong" if strong else "Card")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)
        if title:
            layout.addWidget(_label(title, "Heading"))
        if caption:
            layout.addWidget(_label(caption, "Caption", wrap=True))
        if isinstance(body, QWidget):
            layout.addWidget(body)
        elif isinstance(body, int):
            pass
        else:
            layout.addLayout(body)
        return frame

    @staticmethod
    def _switch_grid(*checks: QCheckBox) -> QGridLayout:
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(8)
        grid.setContentsMargins(0, 0, 0, 0)
        for index, check in enumerate(checks):
            grid.addWidget(check, index // 2, index % 2)
        return grid

    def _swatch(self, key: str, tooltip: str) -> QPushButton:
        button = QPushButton()
        button.setFixedSize(40, 40)
        button.setToolTip(tooltip)
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(lambda _=False, k=key: self._select_accent(k))
        self.swatches[key] = button
        return button

    # ================================================================== theming
    def _apply_theme(self) -> None:
        self.palette = build_palette(resolve_dark(self.settings.theme_mode, _system_is_dark()),
                                     self.settings.accent_color())
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(build_qss(self.palette))
        self._fallback_art = None
        if self._shadow is not None:
            self._shadow.setColor(QColor(0, 0, 0, 90 if not self.palette.dark else 135))
        self._refresh_status()
        self._refresh_appearance()
        self._refresh_output()
        self._refresh_stat_colors()
        self._refresh_nav_icons()
        self._refresh_queue_colors()
        self._rebuild_log()
        self._update_toast_style()
        if self.history_entries:
            self._reload_history()

    def _refresh_nav_icons(self) -> None:
        """Tint the active sidebar icon with the accent, the others with body text."""
        if not hasattr(self, "nav_buttons"):
            return
        for key, button in self.nav_buttons.items():
            colour = self.palette.accent if key == self.view_name else self.palette.text2
            button.setIcon(_vector_icon(button.property("iconKind") or key, colour, 20))

    def _refresh_stat_colors(self) -> None:
        for value in self.stat_values.values():
            value.setStyleSheet(f"color: {qss_hex(self.palette.text)};")

    def _refresh_status(self) -> None:
        p = self.palette
        colour, text = {
            "idle": (p.text3, "Ready"),
            "running": (p.accent, "Syncing"),
            "done": (p.ok, "Finished"),
        }.get(self._status, (p.text3, "Ready"))
        self.status_dot.setStyleSheet(
            f"background-color: {qss_hex(colour)}; border-radius: 4px;")
        self.status_text.setText(text)

    def _refresh_appearance(self) -> None:
        p = self.palette
        for key, button in self.mode_buttons.items():
            button.setChecked(key == self.settings.theme_mode)
        chosen = self.settings.accent_preset
        for key, button in self.swatches.items():
            if key == CUSTOM_ACCENT_ID and chosen != CUSTOM_ACCENT_ID:
                stops = ", ".join(f"stop:{i / (len(RAINBOW) - 1):.2f} {c}"
                                  for i, c in enumerate(RAINBOW))
                fill = f"qlineargradient(x1:0, y1:0, x2:1, y2:1, {stops})"
            else:
                base = (normalise_hex(self.settings.custom_accent) if key == CUSTOM_ACCENT_ID
                        else (find_accent_preset(key).color if find_accent_preset(key) else p.accent))
                fill = (f"qlineargradient(x1:0, y1:0, x2:1, y2:1, "
                        f"stop:0 {qss_hex(lighten(base, 0.18))}, "
                        f"stop:0.55 {qss_hex(base)}, stop:1 {qss_hex(darken(base, 0.22))})")
            ring = qss_hex(p.text) if key == chosen else "transparent"
            button.setStyleSheet(
                f"QPushButton {{ background: {fill}; border: 3px solid {ring};"
                f" border-radius: 20px; }}")
        preset = find_accent_preset(chosen)
        self.accent_caption.setText(
            f"{preset.label}  -  {preset.caption}" if preset
            else f"Custom colour  -  #{self.settings.custom_accent.lstrip('#')}")
        if self.custom_hex_field.text().lstrip("#").upper() != self.settings.custom_accent.lstrip("#").upper():
            self.custom_hex_field.blockSignals(True)
            self.custom_hex_field.setText(self.settings.custom_accent.lstrip("#"))
            self.custom_hex_field.blockSignals(False)

    def _refresh_output(self) -> None:
        for mode, button in self.output_buttons.items():
            button.setChecked(mode == self.settings.output_mode)
            button.setEnabled(not self._running)

    def _update_toast_style(self) -> None:
        self.toast.setStyleSheet(
            f"QFrame#Toast {{ background-color: {qss_hex(self.palette.surface)};"
            f" border: 1px solid {qss_color(self.palette.border_hi)}; border-radius: 14px; }}")
        self.toast_label.setStyleSheet(f"color: {qss_hex(self.palette.text)};")

    def _queue_status_color(self, key: str) -> str:
        attr = QUEUE_STATUS.get(key, ("", "text3"))[1]
        return getattr(self.palette, attr, self.palette.text)

    def _refresh_queue_colors(self) -> None:
        if not hasattr(self, "queue_table"):
            return
        for row in range(self.queue_table.rowCount()):
            item = self.queue_table.item(row, 1)
            if item is None:
                continue
            key = item.data(Qt.UserRole + 1) or "queued"
            item.setForeground(QColor(self._queue_status_color(key)))

    # ================================================================== jobs
    def _start_job(self, job: Job) -> None:
        """Keep a Python reference to *job* until its signals have been delivered.

        ``QThreadPool`` owns the C++ runnable, but not the Python wrapper - without
        this set the wrapper (and its ``Signals`` QObject) could be garbage collected
        while the job is still running.
        """
        self._jobs.add(job)

        def release(*_args: Any) -> None:
            QTimer.singleShot(500, lambda: self._jobs.discard(job))

        job.signals.done.connect(release, Qt.QueuedConnection)
        job.signals.error.connect(release, Qt.QueuedConnection)
        self.pool.start(job)

    # ================================================================== startup
    def _startup(self) -> None:
        self._append_log("SYS", "Analysing hardware…")

        def probe() -> tuple[Any, str, str]:
            return run_preflight(), str(data_dir()), str(logs_dir())

        job = Job(probe)

        def done(result: Any) -> None:
            report, data, logs = result
            self.data_path_text.setText(f"Data folder: {data}")
            self.logs_path_text.setText(f"Log files: {logs}")
            self.isolation_ready = report.isolation_ready
            if report.isolation_ready:
                self.isolation_status.setText(f"Vocal isolation: ready ({report.backend}).")
                self.banner.setVisible(False)
            else:
                reason = report.isolation_reason()
                self.isolation_status.setText(f"Vocal isolation: unavailable - {reason}.")
                self.isolate_hint.setText(f"Unavailable - {reason}")
                self.banner_text.setText(report.warning())
                self.banner.setVisible(True)
            for line in report.summary_lines():
                self._append_log("SYS", line)
            self._refresh_custom_options()

        job.signals.done.connect(done, Qt.QueuedConnection)
        job.signals.error.connect(
            lambda msg: self._append_log("ERR", f"Pre-flight failed: {msg}", "error"),
            Qt.QueuedConnection)
        self._start_job(job)

        profile_job = Job(get_profile)

        def profile_done(profile: Any) -> None:
            self.profile = profile
            self.hw_badge.setText(profile.badge())
            gpu = (f" · GPU: {profile.gpu_name}"
                   + (f" ({profile.gpu_vram_gb:.0f} GB VRAM)" if profile.gpu_vram_gb else "")
                   if profile.gpu_name else "")
            self.hw_summary.setText(
                f"CPU: {profile.cpu_name or 'unknown'} · {profile.physical_cores} cores · "
                f"{profile.ram_total_gb:.0f} GB RAM{gpu}")
            for line in profile.summary_lines():
                self._append_log("WARN" if line.startswith("Note:") else "SYS", line,
                                 "warning" if line.startswith("Note:") else "info")
            rec = profile.recommendation
            self.model_hint.setText(
                f"Recommended for this PC: {rec.model} on {rec.device.upper()} ({rec.compute_type})")
            self.apply_rec_btn.setEnabled(True)

        profile_job.signals.done.connect(profile_done, Qt.QueuedConnection)
        profile_job.signals.error.connect(
            lambda msg: self._append_log("WARN", f"Hardware analysis failed: {msg}", "warning"),
            Qt.QueuedConnection)
        self._start_job(profile_job)

        QTimer.singleShot(250, self._reload_history)
        if (self.settings.auto_check_updates
                and not os.environ.get("LYRICIST_NO_UPDATE_CHECK")
                and os.environ.get("QT_QPA_PLATFORM") != "offscreen"):
            QTimer.singleShot(1600, lambda: self._check_updates(manual=False))

    # ================================================================== log
    def _append_log(self, tag: str, message: str, level: str = "info") -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_entries.append((stamp, tag, message, level))
        self._write_log_line(stamp, tag, message, level)

    def _write_log_line(self, stamp: str, tag: str, message: str, level: str) -> None:
        cursor = self.log_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(self.palette.tag_color(tag, level)))
        cursor.setCharFormat(fmt)
        cursor.insertText(f"{stamp}  [{tag}] {message}\n")
        self.log_view.setTextCursor(cursor)
        self.log_view.ensureCursorVisible()

    def _rebuild_log(self) -> None:
        self.log_view.clear()
        for stamp, tag, message, level in self.log_entries:
            self._write_log_line(stamp, tag, message, level)

    def _on_log_record(self, message: str, levelname: str, name: str) -> None:
        if name == "local_sync_lyrics.processor":
            return  # already delivered as a processor Event - avoid a duplicate line
        level = "error" if levelname in ("ERROR", "CRITICAL") else (
            "warning" if levelname == "WARNING" else "info")
        self._append_log("ERR" if level == "error" else "WARN", message, level)

    def _copy_log(self) -> None:
        text = "\n".join(f"{stamp}  [{tag}] {msg}" for stamp, tag, msg, _ in self.log_entries)
        QGuiApplication.clipboard().setText(text)
        self._snack("Log copied to clipboard.")

    def _clear_log(self) -> None:
        self.log_entries.clear()
        self.log_view.clear()

    # ================================================================== events
    def _handle_event(self, event: Event) -> None:
        if event.kind == "log":
            self._append_log(event.tag, event.message, event.level)
        elif event.kind == "progress":
            total = max(event.total, 1)
            self.progress_bar.setValue(int(min(1.0, event.current / total) * 1000))
            pct = 100.0 * event.current / total
            self.progress_label.setText(f"{event.current} / {event.total}  ·  {pct:.0f}%")
            self.current_file.setText(event.message)
        elif event.kind == "stats":
            for key, label in self.stat_values.items():
                label.setText(str(event.data.get(key, 0)))
        elif event.kind == "done":
            data = event.data
            synced = (data.get("synced_online", 0) + data.get("synced_aligned", 0)
                      + data.get("synced_ai", 0))
            stopped = self.cancel.is_set()
            verb = "Stopped" if stopped else "Finished"
            self._append_log(
                "OK", f"{verb} in {event.message}: {synced} synced, {data.get('skipped', 0)} skipped, "
                      f"{data.get('not_found', 0)} not found, {data.get('failed', 0)} failed.")
            self.progress_label.setText(f"{verb}  ·  {synced} synced")
            if not stopped:
                self.progress_bar.setValue(1000)
            self.current_file.setText("")
            self._status_after = "idle" if stopped else "done"
            self._snack(f"{verb}: {synced} track(s) synced.")
            QTimer.singleShot(400, self._reload_history)
        elif event.kind == "finished":
            self._set_running(False)
        elif event.kind == "track_start":
            data = event.data or {}
            self._current_track = str(data.get("path", ""))
            self._queue_set_status(self._current_track, data.get("status") or "syncing")
        elif event.kind == "track_done":
            data = event.data or {}
            path = str(data.get("path", ""))
            status = str(data.get("status", ""))
            self._queue_set_status(
                path, QUEUE_DONE_STATUS.get(status, "queued"),
                data.get("language_name") or data.get("language") or "")
            if status in ("synced_online", "synced_aligned", "synced_ai", "plain_only"):
                QTimer.singleShot(600, self._reload_history)
        elif event.kind == "toast":
            self._snack(event.message)
        elif event.kind == "language":
            data = event.data or {}
            if data.get("code"):
                name = data.get("name", data["code"])
                self.lang_text.setText(f"{name} · {round(data.get('probability', 0) * 100)}%")
                self.lang_chip.setToolTip(
                    f"Whisper is locked to {name} for this track "
                    f"(detected from: {data.get('source', 'audio')})")
                self.lang_chip.setVisible(True)
            else:
                self.lang_chip.setVisible(False)
        elif event.kind == "confirm":
            self._show_confirm(event.data or {})
        elif event.kind == "saved":
            self._show_completion(event.data or {})
        elif event.kind == "tagged":
            data = event.data or {}
            fields = ", ".join(data.get("fields") or [])
            extra = " + cover art" if data.get("cover") else ""
            self._append_log("OK", f"Tags written for {event.message} via "
                                   f"{data.get('source', 'lookup')}: {fields}{extra}.")
            self._queue_set_status(data.get("path", ""), "tagging")

    @Slot(object)
    def _on_event(self, event: Event) -> None:
        self._handle_event(event)

    @Slot(object)
    def _on_confirm(self, payload: dict) -> None:
        self._show_confirm(payload)

    def _show_confirm(self, data: dict) -> None:
        box = data.get("_box", {})
        done = data.get("_done")
        dialog = ConfirmDialog(data, self.palette, self)
        force = dialog.exec() == QDialog.Accepted
        box["force"] = force
        if done is not None:
            done.set()

    # ================================================================== run state
    def _set_running(self, running: bool) -> None:
        self._running = running
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.progress_stop_btn.setEnabled(running)
        self.browse_btn.setEnabled(not running)
        self.folder_field.setEnabled(not running)
        self.clear_cache_btn.setEnabled(not running)
        self.custom_btn.setEnabled(not running)
        self.sw_isolate.setEnabled(not running)
        self.sw_autotag.setEnabled(not running)
        self.tag_mode_dd.setEnabled(not running and self.sw_autotag.isChecked())
        self.custom_providers_btn.setEnabled(not running)
        self.queue_clear_btn.setEnabled(not running)
        self.drop_zone.setEnabled(not running)
        self._queue_selection_changed()
        self._refresh_output()
        if running:
            self._status = "running"
            self._refresh_status()
            self.progress_card.setVisible(True)
        else:
            self.lang_chip.setVisible(False)
            self._status = self._status_after
            self._status_after = "idle"
            self._refresh_status()
            run_id = self._run_id
            QTimer.singleShot(7000, lambda: self._collapse_progress(run_id))

    def _collapse_progress(self, run_id: int) -> None:
        if run_id == self._run_id and not self._running:
            self.progress_card.setVisible(False)

    def _show_view(self, name: str) -> None:
        if name not in self.nav_buttons:
            name = "workspace"
        self.view_name = name
        target = {
            "workspace": self.workspace_view,
            "history": self.history_view,
            "providers": self.providers_view,
            "settings": self.settings_view,
        }[name]
        if self.stack.currentWidget() is not target:
            self.stack.setCurrentWidget(target)
            self._animate_view(target)
        button = self.nav_buttons.get(name)
        if button is not None and not button.isChecked():
            button.setChecked(True)
        self._refresh_nav_icons()
        if name == "history":
            self._reload_history()

    def _animate_view(self, view: QWidget) -> None:
        """Short cross-fade when switching pages (native effects only, no blocking)."""
        previous = self._view_anim
        if previous is not None:
            try:
                previous.stop()
            except RuntimeError:      # already deleted by Qt
                pass
        effect = QGraphicsOpacityEffect(view)
        effect.setOpacity(0.0)
        view.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(170)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)

        def finish() -> None:
            try:
                view.setGraphicsEffect(None)
            except RuntimeError:
                pass
            if self._view_anim is anim:
                self._view_anim = None
            anim.deleteLater()

        anim.finished.connect(finish)
        # The window owns the animation; deleteLater() in ``finish`` avoids the
        # DeleteWhenStopped wrapper/QObject ownership mismatch that Qt reports as
        # "shared QObject was deleted directly" at interpreter shutdown.
        anim.start()
        self._view_anim = anim

    # ------------------------------------------------------------- window chrome
    def _toggle_maximised(self) -> None:
        maximised = not self.isMaximized()
        if maximised:
            self.showMaximized()
        else:
            self.showNormal()
        margin = 0 if maximised else SHADOW_MARGIN
        self._root_layout.setContentsMargins(margin, margin, margin, margin)
        if self._shadow is not None:
            self._shadow.setEnabled(not maximised)
        if hasattr(self, "size_grip"):
            self.size_grip.setVisible(not maximised)

    def _install_shortcuts(self) -> None:
        bindings: tuple[tuple[str, Callable[[], Any]], ...] = (
            ("Ctrl+O", self._browse),
            ("Ctrl+Shift+O", self._add_files),
            ("Ctrl+R", self._start_batch),
            ("Ctrl+,", lambda: self._show_view("settings")),
            ("Ctrl+1", lambda: self._show_view("workspace")),
            ("Ctrl+2", lambda: self._show_view("history")),
            ("Ctrl+3", lambda: self._show_view("providers")),
            ("Ctrl+4", lambda: self._show_view("settings")),
            ("Ctrl+E", self._queue_open_editor),
            ("Ctrl+F", self._focus_search),
            ("Ctrl+U", lambda: self._check_updates(manual=True)),
            ("Escape", self._stop),
            ("Ctrl+Q", self.close),
        )
        for sequence, slot in bindings:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(slot)
            self._shortcuts.append(shortcut)

    def _focus_search(self) -> None:
        if self.view_name == "history":
            self.history_search.setFocus()
            self.history_search.selectAll()
        else:
            self.folder_field.setFocus()
            self.folder_field.selectAll()

    def _reveal_path(self, path: Path) -> None:
        """Show *path* in the platform file manager (selecting the file when possible)."""
        path = Path(path)
        target = path if path.exists() else path.parent
        try:
            if sys.platform.startswith("win"):
                if path.exists():
                    subprocess.Popen(["explorer", "/select,", str(path)])
                else:
                    os.startfile(str(target))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", str(path)] if path.exists()
                                 else ["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except Exception as exc:
            logger.debug("could not reveal %s: %s", path, exc)
            self._snack(f"Could not open the file manager: {exc}")

    # ================================================================== toast
    def _snack(self, message: str) -> None:
        self.toast_label.setText(message)
        self.toast.adjustSize()
        root = self.centralWidget()
        if root is not None:
            width = min(460, max(240, self.toast.sizeHint().width() + 32))
            self.toast.resize(width, self.toast.sizeHint().height())
            self.toast.move(max(16, root.width() - width - 26), max(16, root.height() - self.toast.height() - 26))
        self.toast.setVisible(True)
        self.toast.raise_()
        self._toast_timer.start(4500)

    def _hide_toast(self) -> None:
        self.toast.setVisible(False)

    # ================================================================== handlers
    def _browse(self) -> None:
        start = self.folder_field.text().strip()
        path = QFileDialog.getExistingDirectory(
            self, "Select your music folder", start if os.path.isdir(start) else "")
        if path:
            self.folder_field.setText(path)

    def _add_files(self) -> None:
        if self._running:
            return
        start = self.folder_field.text().strip()
        patterns = " ".join(f"*{ext}" for ext in sorted(AUDIO_EXTENSIONS))
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add audio files", start if os.path.isdir(start) else "",
            f"Audio files ({patterns});;All files (*)")
        if paths:
            self._on_paths_dropped(paths)

    # -------------------------------------------------------------------- queue
    def _on_paths_dropped(self, paths: list[str]) -> None:
        """Expand dropped folders on a worker thread, then fill the queue."""
        if self._running:
            self._snack("Stop the current sync before adding more tracks.")
            return
        raw_paths = [Path(os.path.expanduser(str(p))) for p in paths if str(p).strip()]
        if not raw_paths:
            return

        def scan() -> list[Path]:
            found: list[Path] = []
            for entry in raw_paths:
                if entry.is_dir():
                    found.extend(discover_audio(entry, self.settings.recursive))
                elif entry.is_file() and entry.suffix.lower() in AUDIO_EXTENSIONS:
                    found.append(entry)
            return found

        job = Job(scan)
        job.signals.done.connect(self._queue_add, Qt.QueuedConnection)
        job.signals.error.connect(
            lambda msg: self._snack(f"Could not read that folder: {msg}"), Qt.QueuedConnection)
        self._start_job(job)

    def _queue_add(self, files: Any) -> None:
        if self._running:
            self._snack("Stop the current sync before adding more tracks.")
            return
        existing = {str(path) for path in self.queue_paths}
        added = 0
        for entry in files or []:
            path = Path(entry)
            key = str(path)
            if key in existing:
                continue
            existing.add(key)
            self.queue_paths.append(path)
            row = self.queue_table.rowCount()
            self.queue_table.insertRow(row)
            name = QTableWidgetItem(path.name)
            name.setToolTip(key)
            name.setData(Qt.UserRole, key)
            status = QTableWidgetItem("Queued")
            status.setData(Qt.UserRole, key)
            status.setData(Qt.UserRole + 1, "queued")
            status.setForeground(QColor(self._queue_status_color("queued")))
            language = QTableWidgetItem("-")
            language.setData(Qt.UserRole, key)
            folder = QTableWidgetItem(path.parent.name or str(path.parent))
            folder.setToolTip(str(path.parent))
            folder.setData(Qt.UserRole, key)
            for column, item in enumerate((name, status, language, folder)):
                self.queue_table.setItem(row, column, item)
            self.queue_rows[key] = row
            added += 1
        if added:
            self._update_queue_count()
            self._append_log("SYS", f"Added {added} track(s) to the queue.")
        elif files:
            self._snack("Those tracks are already queued.")

    def _queue_find_row(self, key: str) -> int:
        for row in range(self.queue_table.rowCount()):
            item = self.queue_table.item(row, 0)
            if item is not None and item.data(Qt.UserRole) == key:
                return row
        return -1

    def _queue_set_status(self, path: Any, key: str, language: str = "") -> None:
        key_path = str(path or "")
        if not key_path or not hasattr(self, "queue_table"):
            return
        row = self._queue_find_row(key_path)
        if row < 0:
            return
        label = QUEUE_STATUS.get(key, QUEUE_STATUS["queued"])[0]
        item = self.queue_table.item(row, 1)
        if item is not None:
            item.setText(label)
            item.setData(Qt.UserRole + 1, key)
            item.setForeground(QColor(self._queue_status_color(key)))
        if language:
            lang_item = self.queue_table.item(row, 2)
            if lang_item is not None:
                lang_item.setText(language)
        self._update_queue_count()

    def _update_queue_count(self) -> None:
        total = len(self.queue_paths)
        if not total:
            self.queue_count.setText("No tracks queued")
            return
        done = 0
        for row in range(self.queue_table.rowCount()):
            item = self.queue_table.item(row, 1)
            if item is not None and item.data(Qt.UserRole + 1) == "done":
                done += 1
        suffix = f" · {done} done" if done else ""
        self.queue_count.setText(f"{total} track(s){suffix}")

    def _queue_selected_path(self) -> Optional[Path]:
        items = self.queue_table.selectedItems()
        if not items:
            return None
        row = items[0].row()
        item = self.queue_table.item(row, 0)
        raw = item.data(Qt.UserRole) if item is not None else None
        return Path(raw) if raw else None

    def _queue_selection_changed(self) -> None:
        enabled = self._queue_selected_path() is not None and not self._running
        for button in (self.queue_edit_btn, self.queue_location_btn, self.queue_remove_btn):
            button.setEnabled(enabled)

    def _queue_open_editor(self) -> None:
        path = self._queue_selected_path()
        if path is None:
            self._snack("Select a queued track first.")
            return
        self._open_editor_for(path)

    def _queue_open_location(self) -> None:
        path = self._queue_selected_path()
        if path is None:
            self._snack("Select a queued track first.")
            return
        self._reveal_path(path)

    def _queue_remove_selected(self) -> None:
        if self._running:
            return
        path = self._queue_selected_path()
        if path is None:
            return
        row = self._queue_find_row(str(path))
        if row >= 0:
            self.queue_table.removeRow(row)
        self.queue_paths = [p for p in self.queue_paths if str(p) != str(path)]
        self.queue_rows.pop(str(path), None)
        self.queue_rows = {str(p): index for index, p in enumerate(self.queue_paths)}
        self._update_queue_count()
        self._queue_selection_changed()

    def _queue_clear(self) -> None:
        if self._running:
            return
        self.queue_table.setRowCount(0)
        self.queue_paths.clear()
        self.queue_rows.clear()
        self._update_queue_count()
        self._queue_selection_changed()

    def _open_editor_for(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            self._snack("That file has moved or been deleted.")
            return
        text = ""
        lrc_path = path.with_suffix(".lrc")
        if lrc_path.exists():
            try:
                text = lrc_path.read_text(encoding="utf-8")
            except OSError:
                text = ""
        if not text:
            try:
                text = read_embedded_lyrics(path) or ""
            except Exception:
                text = ""
        dialog = LyricEditorDialog(path, text, self.palette,
                                   output_mode=self.settings.output_mode, parent=self)
        dialog.saved.connect(self._on_lyrics_saved)
        self.editor_dialog = dialog
        dialog.exec()
        self._reload_history()

    # ----------------------------------------------------------- history & vault
    def _reload_history(self) -> None:
        if self.history_job_running:
            self._history_pending = True
            return
        self.history_job_running = True
        needle = self.history_search.text().strip()
        status = self.history_filter.currentData() or "all"

        def load() -> tuple[list[HistoryRecord], int, dict[str, QImage]]:
            with LyricsDatabase(database_path()) as db:
                records = db.history(search=needle, status=status, limit=400)
                total = db.history_count(search=needle, status=status)
            images: dict[str, QImage] = {}
            for record in records:
                if not record.thumbnail:
                    continue
                image = QImage.fromData(record.thumbnail, record.thumbnail_mime or None)
                if not image.isNull():
                    images[record.fingerprint] = image
            return records, total, images

        job = Job(load)

        def done(result: Any) -> None:
            self.history_job_running = False
            records, total, images = result
            self._render_history(records, total, images)
            if self._history_pending:
                self._history_pending = False
                self._reload_history()

        job.signals.done.connect(done, Qt.QueuedConnection)
        job.signals.error.connect(self._history_failed, Qt.QueuedConnection)
        self._start_job(job)

    def _history_failed(self, message: str) -> None:
        self.history_job_running = False
        self.history_empty.setText(f"Could not read the library: {message}")
        self.history_empty.setVisible(True)
        self.history_list.setVisible(False)

    def _artwork_placeholder(self) -> QPixmap:
        if self._fallback_art is None or self._fallback_art.isNull():
            self._fallback_art = _fallback_artwork(self.palette, 156)
        return self._fallback_art

    def _render_history(self, records: list[HistoryRecord], total: int,
                        images: dict[str, QImage]) -> None:
        self.history_entries = list(records)
        self._history_images = images
        self.history_list.clear()
        placeholder = self._artwork_placeholder()
        for record in records:
            lines = [record.display_title, record.display_artist]
            extra = " · ".join(part for part in (record.album, record.year) if part)
            if extra:
                lines.append(extra)
            item = QListWidgetItem("\n".join(lines))
            item.setData(Qt.UserRole, record.fingerprint)
            item.setTextAlignment(Qt.AlignHCenter | Qt.AlignTop)
            item.setToolTip(f"{record.path}\nStatus: {record.status}"
                            + (f"\nSynced: {datetime.fromtimestamp(record.synced_at):%d %b %Y %H:%M}"
                               if record.synced_at else ""))
            image = images.get(record.fingerprint)
            if image is not None and not image.isNull():
                item.setIcon(QIcon(QPixmap.fromImage(image)))
            else:
                item.setIcon(QIcon(placeholder))
            self.history_list.addItem(item)
        empty = not records
        self.history_empty.setText(
            "No tracks match that search." if (empty and (self.history_search.text().strip()
                                                          or (self.history_filter.currentData() or "all") != "all"))
            else "No tracks yet - sync a folder or drop a few songs to build your library.")
        self.history_empty.setVisible(empty)
        self.history_list.setVisible(not empty)
        self.history_count_label.setText(f"{total} track(s)" if total else "")
        self._history_selection_changed()

    def _history_selected(self) -> Optional[HistoryRecord]:
        item = self.history_list.currentItem()
        if item is None:
            return None
        fingerprint = item.data(Qt.UserRole)
        for record in self.history_entries:
            if record.fingerprint == fingerprint:
                return record
        return None

    def _history_selection_changed(self) -> None:
        enabled = self._history_selected() is not None
        for button in (self.history_edit_btn, self.history_location_btn,
                       self.history_resync_btn, self.history_export_btn):
            button.setEnabled(enabled)

    def _history_open_editor(self) -> None:
        record = self._history_selected()
        if record is None:
            self._snack("Select a track from the library first.")
            return
        self._open_editor_for(Path(record.path))

    def _history_open_location(self) -> None:
        record = self._history_selected()
        if record is None:
            return
        self._reveal_path(Path(record.path))

    def _history_resync(self) -> None:
        record = self._history_selected()
        if record is None:
            return
        if self._running:
            self._snack("Stop the current sync first.")
            return
        path = Path(record.path)
        if not path.exists():
            self._snack("That file has moved or been deleted - re-add it from the workspace.")
            return
        self._queue_clear()
        self._queue_add([path])
        self._show_view("workspace")
        self._start_batch()

    def _history_export(self) -> None:
        record = self._history_selected()
        if record is None:
            return
        path = Path(record.path)
        text = ""
        lrc_path = path.with_suffix(".lrc")
        if lrc_path.exists():
            try:
                text = lrc_path.read_text(encoding="utf-8")
            except OSError:
                text = ""
        if not text and path.exists():
            try:
                text = read_embedded_lyrics(path) or ""
            except Exception:
                text = ""
        if not text:
            self._snack("No lyrics are stored for this track yet.")
            return
        target, _ = QFileDialog.getSaveFileName(
            self, "Export .lrc", str(lrc_path), "LRC lyrics (*.lrc);;All files (*)")
        if not target:
            return
        try:
            write_text_atomic(Path(target), text)
        except OSError as exc:
            self._snack(f"Could not write the file: {exc}")
            return
        self._snack(f"Exported {Path(target).name}.")

    # ------------------------------------------------------------------ updates
    def _check_updates(self, manual: bool = False) -> None:
        if self.update_checker is not None and self.update_checker.isRunning():
            if manual:
                self._snack("An update check is already running.")
            return
        self._update_manual = manual
        repo = self.settings.update_repo
        self.update_status.setText(f"Checking {repo} for updates…")
        if manual:
            self.check_updates_btn.setEnabled(False)
        thread = UpdateCheckThread(__version__, repo, self)
        thread.checked.connect(self._on_update_checked, Qt.QueuedConnection)
        thread.finished.connect(self._on_update_thread_finished, Qt.QueuedConnection)
        self.update_checker = thread
        thread.start()

    def _on_update_thread_finished(self) -> None:
        if hasattr(self, "check_updates_btn"):
            self.check_updates_btn.setEnabled(True)
        checker = self.update_checker
        if checker is not None and not checker.isRunning():
            checker.deleteLater()
            self.update_checker = None

    def _on_update_checked(self, info: UpdateInfo) -> None:
        self.update_info = info
        manual = self._update_manual
        self._update_manual = False
        if not info.error:
            self.settings.last_update_check = time.time()
        self.update_status.setText(info.summary())
        if info.error:
            if manual:
                self._snack(info.summary())
            return
        if info.available:
            self.update_text.setText(
                f"Lyricist v{info.latest} is available - you have v{info.current}.")
            self.update_banner.setVisible(True)
            self.update_badge.setText(f"Update available: v{info.latest}")
            self.update_badge.setVisible(True)
            self._append_log("SYS", info.summary())
            if manual:
                self._open_update_dialog()
            else:
                self._snack(f"Lyricist v{info.latest} is available.")
        else:
            self.update_banner.setVisible(False)
            self.update_badge.setVisible(False)
            if manual:
                self._snack(info.summary())

    def _open_update_dialog(self) -> None:
        info = self.update_info
        if info is None:
            self._check_updates(manual=True)
            return
        if info.error or not info.available:
            self._snack(info.summary())
            return
        UpdateDialog(info, self.palette, self).exec()

    def _dismiss_update_banner(self) -> None:
        self.update_banner.setVisible(False)
        self.update_badge.setVisible(False)

    def _toggle_auto_updates(self, checked: bool) -> None:
        self.settings.auto_check_updates = bool(checked)
        self._save_bg(self.settings.copy())

    def _select_output(self, mode: str) -> None:
        if self._running or mode == self.settings.output_mode:
            self._refresh_output()
            return
        self.settings.output_mode = mode
        self._refresh_output()
        self._refresh_custom_options()
        self._save_bg(self.settings.copy())

    def _toggle_isolate(self, checked: bool) -> None:
        self.settings.isolate_vocals = bool(checked)
        self._refresh_custom_options()
        self._save_bg(self.settings.copy())
        if checked and self.isolation_ready is False:
            self._snack('Vocal isolation needs "audio-separator" '
                        '(pip install "audio-separator[cpu]"). Until it is installed the '
                        "original audio is used.")

    def _toggle_autotag(self, checked: bool) -> None:
        self.settings.auto_tag = bool(checked)
        self.sw_autotag_settings.blockSignals(True)
        self.sw_autotag_settings.setChecked(bool(checked))
        self.sw_autotag_settings.blockSignals(False)
        self._sync_autotag_controls()
        self._save_bg(self.settings.copy())

    def _select_tag_mode(self, _index: int = 0) -> None:
        mode = self.tag_mode_dd.currentData() or "fill_missing"
        self.settings.tag_mode = mode
        index = self.tag_mode_settings.findData(mode)
        if index >= 0 and index != self.tag_mode_settings.currentIndex():
            self.tag_mode_settings.blockSignals(True)
            self.tag_mode_settings.setCurrentIndex(index)
            self.tag_mode_settings.blockSignals(False)
        self._save_bg(self.settings.copy())

    def _sync_autotag_controls(self) -> None:
        """Keep the workspace switch and the Settings switch in step."""
        enabled = self.sw_autotag_settings.isChecked()
        if self.sw_autotag.isChecked() != enabled:
            self.sw_autotag.blockSignals(True)
            self.sw_autotag.setChecked(enabled)
            self.sw_autotag.blockSignals(False)
        self.tag_mode_dd.setEnabled(enabled)
        self.tag_mode_settings.setEnabled(enabled)

    def _refresh_provider_summary(self) -> None:
        providers = self.settings.custom_providers
        enabled = sum(1 for provider in providers if provider.enabled)
        if not providers:
            self.custom_providers_summary.setText("No custom endpoints yet.")
        else:
            self.custom_providers_summary.setText(
                f"{len(providers)} custom endpoint(s), {enabled} enabled.")

    def _open_providers(self) -> None:
        dialog = CustomProvidersDialog(self.settings.custom_providers, self.palette, self)
        self.providers_dialog = dialog
        if dialog.exec() == QDialog.Accepted:
            self.settings.custom_providers = [p.normalised() for p in dialog.providers
                                              if p.valid()]
            self.settings._normalise()
            self._refresh_provider_summary()
            self._save_bg(self.settings.copy(), toast=True)

    # ------------------------------------------------------------------- editor
    def _show_completion(self, data: dict) -> None:
        self._last_saved = {
            "path": str(data.get("path", "")),
            "lrc": str(data.get("lrc", "")),
            "source": str(data.get("source", "")),
        }
        name = Path(self._last_saved["path"]).name or "your track"
        self.completion_title.setText("Lyrics ready to review")
        self.completion_detail.setText(
            f"{name} - {self._last_saved['source'] or 'synced'}. "
            "Open the editor to fine-tune timestamps or wording.")
        self.completion_card.setVisible(True)

    def _dismiss_completion(self) -> None:
        self.completion_card.setVisible(False)

    def _open_editor(self) -> None:
        if not self._last_saved or not self._last_saved.get("path"):
            self._snack("Sync a track first, then edit its lyrics here.")
            return
        path = Path(self._last_saved["path"])
        text = self._last_saved.get("lrc") or ""
        if path.exists():
            lrc_path = path.with_suffix(".lrc")
            if lrc_path.exists():
                try:
                    text = lrc_path.read_text(encoding="utf-8")
                except OSError:
                    pass
        dialog = LyricEditorDialog(path, text, self.palette,
                                   output_mode=self.settings.output_mode, parent=self)
        dialog.saved.connect(self._on_lyrics_saved)
        self.editor_dialog = dialog
        dialog.exec()

    def _on_lyrics_saved(self, result: Any) -> None:
        self._snack(f"Lyrics saved: {result.describe()}")
        if self._last_saved is not None:
            self._last_saved["lrc"] = self.editor_dialog.edited_text if self.editor_dialog else ""


    def _refresh_custom_options(self) -> None:
        if self.custom_dialog is not None and self.custom_dialog.isVisible():
            self.custom_dialog.refresh_options(
                output_mode=self.settings.output_mode,
                isolate_on=self.sw_isolate.isChecked())

    def _select_mode(self, mode: str) -> None:
        if mode == self.settings.theme_mode:
            return
        self.settings.theme_mode = mode
        self._after_appearance_change()

    def _toggle_theme(self) -> None:
        self.settings.theme_mode = "light" if self.palette.dark else "dark"
        self._after_appearance_change()

    def _select_accent(self, key: str) -> None:
        if key == CUSTOM_ACCENT_ID:
            raw = (self.custom_hex_field.text() or "").strip().lstrip("#")
            self.settings.custom_accent = normalise_hex(raw, self.settings.custom_accent)
        if key == self.settings.accent_preset and key != CUSTOM_ACCENT_ID:
            return
        self.settings.accent_preset = key
        self._after_appearance_change()

    def _custom_accent_changed(self, text: str) -> None:
        raw = text.strip().lstrip("#")
        if len(raw) in (3, 6) and parse_hex(raw) is not None:
            self.custom_hex_field.setProperty("error", False)
            _repolish(self.custom_hex_field)
            self.settings.custom_accent = normalise_hex(raw)
            self.settings.accent_preset = CUSTOM_ACCENT_ID
            self._after_appearance_change()

    def _custom_accent_submitted(self) -> None:
        raw = (self.custom_hex_field.text() or "").strip().lstrip("#")
        if not (len(raw) in (3, 6) and parse_hex(raw) is not None):
            self.custom_hex_field.setProperty("error", True)
            _repolish(self.custom_hex_field)
            self._snack("Enter 3 or 6 hex digits, e.g. 1DB954.")

    def _after_appearance_change(self) -> None:
        self._apply_theme()
        self._save_bg(self.settings.copy())

    # ------------------------------------------------------------------ settings
    def _collect_settings(self) -> Settings:
        s = self.settings.copy()
        s.whisper_model = self.model_dd.currentData() or "auto"
        s.language = self.lang_dd.currentData() or "auto"
        s.hf_token = self.token_field.text().strip()
        s.use_ai_fallback = self.sw_ai.isChecked()
        s.prefer_gpu = self.sw_gpu.isChecked()
        s.vad_filter = self.sw_vad.isChecked()
        s.recursive = self.sw_recursive.isChecked()
        s.overwrite_existing = self.sw_overwrite.isChecked()
        s.retry_failed = self.sw_retry.isChecked()
        s.save_plain_fallback = self.sw_plain.isChecked()
        s.use_forced_alignment = self.sw_align.isChecked()
        s.detect_mashups = self.sw_mashup.isChecked()
        s.identify_by_audio = self.sw_ident.isChecked()
        s.detect_language = self.sw_detect_lang.isChecked()
        s.isolate_vocals = self.sw_isolate.isChecked()
        s.genius_token = self.genius_field.text().strip()
        s.musixmatch_key = self.musixmatch_field.text().strip()
        s.enabled_providers = [pid for pid, cb in self.provider_checks.items() if cb.isChecked()]
        s.auto_tag = self.sw_autotag_settings.isChecked()
        s.tag_mode = self.tag_mode_settings.currentData() or "fill_missing"
        s.custom_providers = [p.normalised() for p in self.settings.custom_providers]
        s.last_folder = self.folder_field.text().strip()
        s.auto_check_updates = self.sw_updates.isChecked()
        s._normalise()
        return s

    def _apply_settings_to_controls(self) -> None:
        s = self.settings
        self.folder_field.setText(s.last_folder)
        index = self.model_dd.findData(s.whisper_model if s.whisper_model in MODEL_LABELS else "auto")
        self.model_dd.setCurrentIndex(max(0, index))
        index = self.lang_dd.findData(s.language if any(k == s.language for k, _ in LANGUAGES) else "auto")
        self.lang_dd.setCurrentIndex(max(0, index))
        self.token_field.setText(s.hf_token)
        self.genius_field.setText(s.genius_token)
        self.musixmatch_field.setText(s.musixmatch_key)
        self.sw_ai.setChecked(s.use_ai_fallback)
        self.sw_gpu.setChecked(s.prefer_gpu)
        self.sw_vad.setChecked(s.vad_filter)
        self.sw_recursive.setChecked(s.recursive)
        self.sw_overwrite.setChecked(s.overwrite_existing)
        self.sw_retry.setChecked(s.retry_failed)
        self.sw_plain.setChecked(s.save_plain_fallback)
        self.sw_align.setChecked(s.use_forced_alignment)
        self.sw_mashup.setChecked(s.detect_mashups)
        self.sw_ident.setChecked(s.identify_by_audio)
        self.sw_detect_lang.setChecked(s.detect_language)
        self.sw_isolate.blockSignals(True)
        self.sw_isolate.setChecked(s.isolate_vocals)
        self.sw_isolate.blockSignals(False)
        for pid, cb in self.provider_checks.items():
            cb.setChecked(pid in s.enabled_providers)
        self.sw_autotag_settings.blockSignals(True)
        self.sw_autotag_settings.setChecked(s.auto_tag)
        self.sw_autotag_settings.blockSignals(False)
        self.sw_autotag.blockSignals(True)
        self.sw_autotag.setChecked(s.auto_tag)
        self.sw_autotag.blockSignals(False)
        self.sw_updates.blockSignals(True)
        self.sw_updates.setChecked(s.auto_check_updates)
        self.sw_updates.blockSignals(False)
        if s.last_update_check:
            when = datetime.fromtimestamp(s.last_update_check).strftime("%d %b %Y %H:%M")
            self.update_status.setText(f"Last checked {when} · {s.update_repo}")
        else:
            self.update_status.setText(f"Updates are checked against {s.update_repo}.")
        index = self.tag_mode_settings.findData(s.tag_mode)
        self.tag_mode_settings.setCurrentIndex(max(0, index))
        index = self.tag_mode_dd.findData(s.tag_mode)
        self.tag_mode_dd.setCurrentIndex(max(0, index))
        self._sync_autotag_controls()
        self._refresh_provider_summary()
        self._refresh_output()

    def _save_settings(self) -> None:
        self.settings = self._collect_settings()
        for pid, cb in self.provider_checks.items():
            cb.setChecked(pid in self.settings.enabled_providers)
        self.settings.apply_environment()
        self._save_bg(self.settings.copy(), toast=True)

    def _apply_recommendation(self) -> None:
        if self.profile is None:
            return
        model = self.profile.recommendation.model
        index = self.model_dd.findData(model)
        if index >= 0:
            self.model_dd.setCurrentIndex(index)
            self._snack(f"Model set to {model} - press Save to keep it.")

    def _save_bg(self, settings: Settings, *, toast: bool = False) -> None:
        # Rapid appearance changes queue several saves; number them so an older
        # snapshot can never overwrite a newer one (independent pool threads finish
        # in arbitrary order).
        self._save_seq += 1
        sequence = self._save_seq

        def write() -> bool:
            with self._save_lock:
                if sequence < self._saved_seq:
                    return True
                ok = self.store.save(settings)
                if ok:
                    self._saved_seq = sequence
                return ok

        job = Job(write)
        if toast:
            job.signals.done.connect(
                lambda ok: self._snack("Settings saved." if ok else "Could not write the settings file."),
                Qt.QueuedConnection)
        job.signals.error.connect(lambda msg: self._snack(f"Could not save settings: {msg}"),
                                  Qt.QueuedConnection)
        self._start_job(job)

    # ------------------------------------------------------------------ batch
    def _start_batch(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        raw = self.folder_field.text().strip().strip('"').strip("'")
        folder = Path(os.path.expanduser(raw)) if raw else None

        queued = [path for path in self.queue_paths if path.is_file()]
        missing = len(self.queue_paths) - len(queued)
        target: Path | list[Path]
        if queued:
            target = queued
            source_label = f"{len(queued)} queued track(s)"
            last_folder = str(queued[0].parent)
            if missing:
                self._append_log("WARN", f"{missing} queued file(s) no longer exist and were "
                                         "skipped.", "warning")
        elif folder is not None and folder.is_dir():
            target = folder
            source_label = str(folder)
            last_folder = str(folder)
        elif folder is not None and folder.is_file() and folder.suffix.lower() in AUDIO_EXTENSIONS:
            target = folder
            source_label = str(folder)
            last_folder = str(folder.parent)
        else:
            self._snack("Choose an existing music folder or drop a few tracks first.")
            return

        settings = self._collect_settings()
        settings.last_folder = last_folder
        self.settings = settings
        self._save_bg(settings.copy())
        settings.apply_environment()

        self.cancel = threading.Event()
        self._run_id += 1
        self._status_after = "idle"
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting…")
        self.current_file.setText("")
        self.completion_card.setVisible(False)
        self._last_saved = None
        for label in self.stat_values.values():
            label.setText("0")
        self._set_running(True)
        self._append_log("SYS", f"Starting sync of {source_label}")
        self._launch(target, settings, None)

    def _launch(self, target: Path | list[Path], settings: Settings,
                lyrics: Optional[str]) -> None:
        worker = BatchWorker(target, settings, self.cancel, lyrics)
        worker.event.connect(self._on_event)
        worker.confirm.connect(self._on_confirm)
        worker.finished.connect(self._on_worker_finished)
        self.worker = worker
        worker.start()

    def _on_worker_finished(self) -> None:
        if self._running:
            self._set_running(False)

    def _stop(self) -> None:
        self.cancel.set()
        self.stop_btn.setEnabled(False)
        self.progress_stop_btn.setEnabled(False)
        self._append_log("SYS", "Stopping after the current step…")

    # ------------------------------------------------------------------ custom
    def _open_custom(self) -> None:
        if self._running:
            return
        dialog = CustomLyricsDialog(self, output_mode=self.settings.output_mode,
                                    isolate_on=self.sw_isolate.isChecked())
        dialog.submitted.connect(self._start_custom)
        self.custom_dialog = dialog
        dialog.show()

    def _start_custom(self, audio_path: str, lyrics: str) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        audio = Path(os.path.expanduser(audio_path)) if audio_path else None
        if audio is None or not audio.is_file():
            self._snack("Choose the audio file first.")
            return
        lyrics = (lyrics or "").strip()
        if len([line for line in lyrics.splitlines() if line.strip()]) < 2:
            self._snack("Paste at least two lines of lyrics first.")
            return

        settings = self._collect_settings()
        self.settings = settings
        self._save_bg(settings.copy())
        settings.apply_environment()

        self.cancel = threading.Event()
        self._run_id += 1
        self._status_after = "idle"
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting…")
        self.current_file.setText(audio.name)
        self.completion_card.setVisible(False)
        self._last_saved = None
        for label in self.stat_values.values():
            label.setText("0")
        self._set_running(True)
        self._append_log("SYS", f"Syncing {audio.name} with your own lyrics")
        self._launch(audio, settings, lyrics)

    # ------------------------------------------------------------------ cache
    def _clear_cache(self) -> None:
        answer = QMessageBox.question(
            self, "Clear the lyrics cache?",
            "Every track will be looked up again on the next run. Existing .lrc files "
            "are not touched.", QMessageBox.Cancel | QMessageBox.Yes, QMessageBox.Cancel)
        if answer != QMessageBox.Yes:
            return

        job = Job(lambda: clear_cache_file(database_path()))

        def done(removed: Any) -> None:
            self._append_log("SYS", f"Cache cleared ({removed} entries).")
            self._snack(f"Cache cleared ({removed} entries).")

        job.signals.done.connect(done, Qt.QueuedConnection)
        job.signals.error.connect(lambda msg: self._snack(f"Could not clear the cache: {msg}"),
                                  Qt.QueuedConnection)
        self._start_job(job)

    # ------------------------------------------------------------------ window
    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if self.toast.isVisible():
            root = self.centralWidget()
            if root is not None:
                self.toast.move(max(16, root.width() - self.toast.width() - 26),
                                max(16, root.height() - self.toast.height() - 26))

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.cancel.set()
        anim = self._view_anim
        if anim is not None:
            try:
                anim.stop()
            except RuntimeError:      # already deleted by Qt
                pass
            self._view_anim = None
        checker = self.update_checker
        if checker is not None and checker.isRunning():
            checker.wait(600)
            if checker.isRunning():      # last resort: never destroy a live QThread
                checker.terminate()
                checker.wait(1000)
        try:
            logging.getLogger("local_sync_lyrics").removeHandler(self._log_handler)
        except Exception:
            pass
        worker = self.worker
        if worker is not None and worker.isRunning():
            worker.wait(3000)
        self.pool.waitForDone(1500)
        super().closeEvent(event)


# --------------------------------------------------------------------------- entry
def run() -> int:
    """Create the QApplication, show the window and run the Qt event loop."""
    if sys.platform == "win32":
        try:  # taskbar grouping with the brand icon instead of the generic python.exe one
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                f"Lyricist.{__version__}")
        except Exception:
            pass
    QApplication.setApplicationName(APP_TITLE)
    QApplication.setOrganizationName(APP_TITLE)
    QApplication.setApplicationVersion(__version__)
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    app.setStyle("Fusion")
    try:
        app.setDesktopFileName("lyricist")
    except Exception:
        pass
    icon = _app_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)

    store = SettingsStore()
    settings = store.load()
    window = MainWindow(store, settings)
    window.show()
    return int(app.exec())
