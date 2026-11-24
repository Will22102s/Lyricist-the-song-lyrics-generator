"""Shared pytest fixtures.

The Qt front-end is exercised headless: ``QT_QPA_PLATFORM=offscreen`` lets a real
``QApplication`` and real widgets exist without a display server, so the tests cover
actual layout / styling / signal wiring instead of a mock control tree.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LYRICIST_NO_UPDATE_CHECK", "1")

import pytest


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtCore import QThreadPool
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app

    # Tear down explicitly.  Leaving windows/animations to the interpreter can make
    # PySide6 report "shared QObject was deleted directly" and segfault after pytest
    # has already printed its summary (exit code 139 on CI).
    for widget in QApplication.topLevelWidgets():
        try:
            widget.close()
            widget.deleteLater()
        except RuntimeError:
            pass
    app.processEvents()
    QThreadPool.globalInstance().waitForDone(3000)
    app.processEvents()
    shutdown = getattr(app, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception:
            pass


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    """A real MainWindow backed by a throw-away settings/cache folder."""
    monkeypatch.setenv("LYRICIST_HOME", str(tmp_path / "home"))
    from local_sync_lyrics import gui

    win = gui.MainWindow()
    win.show()
    qapp.processEvents()
    yield win
    win.cancel.set()
    if win.worker is not None and win.worker.isRunning():
        win.worker.wait(3000)
    win.close()
    qapp.processEvents()
    win.deleteLater()
    qapp.processEvents()
