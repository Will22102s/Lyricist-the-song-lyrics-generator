"""Shared pytest fixtures.

The Qt front-end is exercised headless: ``QT_QPA_PLATFORM=offscreen`` lets a real
``QApplication`` and real widgets exist without a display server, so the tests cover
actual layout / styling / signal wiring instead of a mock control tree.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


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
