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
import os
import sys
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from PySide6.QtCore import (
    QObject, QRunnable, Qt, QThread, QThreadPool, QTimer, Signal, Slot,
)
from PySide6.QtGui import (
    QColor, QFont, QGuiApplication, QIcon, QPixmap, QTextCharFormat, QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QFileDialog, QFrame,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget,
)

from . import __version__
from .app_paths import asset_path, data_dir, database_path, logs_dir
from .config import (
    ACCENT_PRESETS, CUSTOM_ACCENT_ID, OUTPUT_MODE_LABELS, OUTPUT_MODES, PROVIDER_IDS,
    TIER1_IDS, TIER2_IDS, Settings, SettingsStore, find_accent_preset,
)
from .database import LyricsDatabase, clear_cache_file
from .hardware_profiler import HardwareProfile, get_profile, run_preflight
from .models import Event
from .processor import AUDIO_EXTENSIONS, BatchProcessor
from .theme import (
    Palette, build_palette, build_qss, darken, lighten, normalise_hex, on_color, parse_hex,
    qss_color, qss_hex, resolve_dark, with_alpha,
)

logger = logging.getLogger(__name__)

APP_TITLE = "Lyricist"
MAX_LOG_LINES = 1500
SIDEBAR_WIDTH = 236

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

    def __init__(self, target: Path, settings: Settings, cancel: threading.Event,
                 lyrics: Optional[str] = None) -> None:
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
class MainWindow(QMainWindow):
    """The whole application window (sidebar + workspace + settings)."""

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

        self.palette = build_palette(resolve_dark(self.settings.theme_mode, _system_is_dark()),
                                     self.settings.accent_color())

        self.setWindowTitle(APP_TITLE)
        self.resize(1180, 820)
        self.setMinimumSize(1020, 700)
        icon = asset_path("logo.png")
        if icon is not None:
            self.setWindowIcon(QIcon(str(icon)))

        self._build_ui()
        self._apply_settings_to_controls()
        self._apply_theme()

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

        outer = QHBoxLayout(root)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(16)

        outer.addWidget(self._build_sidebar())

        self.stack = QStackedWidget()
        self.stack.setObjectName("Stack")
        self.workspace_view = self._build_workspace()
        self.settings_view = self._build_settings()
        self.stack.addWidget(self.workspace_view)
        self.stack.addWidget(self.settings_view)
        outer.addWidget(self.stack, 1)

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
        box.setSpacing(14)

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
        box.addSpacing(8)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_buttons: dict[str, QPushButton] = {}
        for key, text in (("workspace", "  Sync Workspace"), ("settings", "  Settings")):
            button = _button(text, "Nav", checkable=True)
            button.clicked.connect(lambda _=False, k=key: self._show_view(k))
            self.nav_group.addButton(button)
            self.nav_buttons[key] = button
            box.addWidget(button)
        self.nav_buttons["workspace"].setChecked(True)

        box.addStretch(1)

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

        # pre-flight banner ----------------------------------------------------
        self.banner = QFrame()
        self.banner.setObjectName("Banner")
        banner_layout = QHBoxLayout(self.banner)
        banner_layout.setContentsMargins(14, 10, 14, 10)
        self.banner_text = _label("", "BannerText", wrap=True)
        banner_layout.addWidget(self.banner_text, 1)
        self.banner.setVisible(False)
        box.addWidget(self.banner)

        # library card ---------------------------------------------------------
        self.folder_field = QLineEdit()
        self.folder_field.setPlaceholderText("Music folder - browse… or paste a path")
        self.folder_field.returnPressed.connect(self._start_batch)
        self.browse_btn = _button("Browse…", "Glass")
        self.browse_btn.clicked.connect(self._browse)
        self.start_btn = _button("Start sync", "Primary")
        self.start_btn.clicked.connect(self._start_batch)
        self.stop_btn = _button("Stop", "Danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)

        folder_row = QHBoxLayout()
        folder_row.setSpacing(10)
        folder_row.addWidget(self.folder_field, 1)
        folder_row.addWidget(self.browse_btn)
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

        library = self._card(_vbox(folder_row, output_row, isolate_row, spacing=12))
        box.addWidget(library)

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

        titles = _vbox(_label("Settings", "Title"),
                       _label("Appearance, local AI and the lyrics pipeline", "Caption"), spacing=2)
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
        column.addWidget(self._build_pipeline_card())
        column.addWidget(self._build_providers_card())
        column.addWidget(self._build_batch_card())
        column.addWidget(self._build_about_card())
        column.addStretch(1)
        scroll.setWidget(inner)
        box.addWidget(scroll, 1)
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

        body = _vbox(_label("Tier 1 - synced lyrics", "SectionLabel"), tier1,
                     _label("Tier 2 - plain text (aligned to the audio)", "SectionLabel"), tier2,
                     fields, spacing=10)
        return self._card(body, title="Lyrics providers",
                          caption="Tier 1 is searched first, in this order.")

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
        text = _vbox(_label(APP_TITLE, "Heading"),
                     _label(f"Version {__version__}  ·  Qt / Liquid Glass interface", "Caption"),
                     self.data_path_text, self.logs_path_text, spacing=3)
        row = QHBoxLayout()
        row.setSpacing(18)
        row.addWidget(logo)
        row.addLayout(text, 1)
        return self._card(row)

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
        self._refresh_status()
        self._refresh_appearance()
        self._refresh_output()
        self._refresh_stat_colors()
        self._rebuild_log()
        self._update_toast_style()

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
        elif event.kind == "finished":
            self._set_running(False)
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
        self.view_name = name
        self.stack.setCurrentWidget(self.workspace_view if name == "workspace" else self.settings_view)
        button = self.nav_buttons.get(name)
        if button is not None:
            button.setChecked(True)

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
        s.last_folder = self.folder_field.text().strip()
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
        if folder is None or not folder.is_dir():
            self._snack("Please choose an existing music folder first.")
            return

        settings = self._collect_settings()
        settings.last_folder = str(folder)
        self.settings = settings
        self._save_bg(settings.copy())
        settings.apply_environment()

        self.cancel = threading.Event()
        self._run_id += 1
        self._status_after = "idle"
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting…")
        self.current_file.setText("")
        for label in self.stat_values.values():
            label.setText("0")
        self._set_running(True)
        self._append_log("SYS", f"Starting sync of {folder}")
        self._launch(folder, settings, None)

    def _launch(self, target: Path, settings: Settings, lyrics: Optional[str]) -> None:
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
    QApplication.setApplicationName(APP_TITLE)
    QApplication.setOrganizationName(APP_TITLE)
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    app.setStyle("Fusion")
    icon = asset_path("logo.png")
    if icon is not None:
        app.setWindowIcon(QIcon(str(icon)))

    store = SettingsStore()
    settings = store.load()
    window = MainWindow(store, settings)
    window.show()
    return int(app.exec())
