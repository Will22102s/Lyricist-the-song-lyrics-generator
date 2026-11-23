"""Phase 2 - processor flows (output modes, Your Own Lyrics), GUI wiring and CLI options."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_appearance as ta  # noqa: E402
import test_pipeline as tp  # noqa: E402
from audio_fixtures import make_audio  # noqa: E402
from local_sync_lyrics import cli, metadata  # noqa: E402
from local_sync_lyrics.config import (  # noqa: E402
    CUSTOM_LYRICS_MIN_MATCH, OUTPUT_MODE_LABELS, OUTPUT_MODES, Settings,
)
from local_sync_lyrics.database import LyricsDatabase, fingerprint_file  # noqa: E402
from local_sync_lyrics.language import LanguageGuess  # noqa: E402
from local_sync_lyrics.models import Event, LyricsResult, Snippet, Status  # noqa: E402
from local_sync_lyrics.processor import BatchProcessor  # noqa: E402

SYNCED = LyricsResult("LRCLIB", synced_lrc=tp.lrc_of())


# ============================================================ settings & labels
def test_output_mode_settings():
    assert OUTPUT_MODES == ("lrc", "embed", "both")
    assert OUTPUT_MODE_LABELS == {"lrc": "Save Separate .lrc File", "embed": "Embed Into Audio File",
                                  "both": "Both (.lrc + Embedded)"}
    s = Settings()
    assert s.output_mode == "lrc" and s.wants_lrc_file and not s.wants_embedded
    assert Settings(output_mode="embed").wants_embedded and not Settings(output_mode="embed").wants_lrc_file
    both = Settings(output_mode="both")
    assert both.wants_embedded and both.wants_lrc_file
    assert Settings.from_dict({"output_mode": "nope"}).output_mode == "lrc"
    assert Settings().isolate_vocals is False and Settings().detect_language is True        # defaults per spec


def test_new_settings_survive_a_round_trip(tmp_path, monkeypatch):
    from local_sync_lyrics.config import SettingsStore

    monkeypatch.setenv("LYRICIST_HOME", str(tmp_path))
    store = SettingsStore()
    store.save(Settings(output_mode="both", isolate_vocals=True, detect_language=False))
    loaded = store.load()
    assert (loaded.output_mode, loaded.isolate_vocals, loaded.detect_language) == ("both", True, False)


# ===================================================================== output modes
def library(tmp_path, *names):
    lib = tmp_path / "lib"
    lib.mkdir(exist_ok=True)
    for name in names:
        path = lib / name
        if path.suffix == ".wav":
            tp.tone_wav(path, 3.0)
        elif not make_audio(path):
            pytest.skip(f"no encoder for {path.suffix}")
    return lib


def run_lib(tmp_path, lib, fetcher, tr=None, **settings):
    events = []
    with LyricsDatabase(tmp_path / "c.db") as db:
        summary = BatchProcessor(Settings(**settings), db, events.append, fetcher=fetcher,
                                 transcriber=tr or tp.ScriptedTranscriber()).run(lib)
    return summary, events


def log_texts(events, tag=None):
    return [e.message for e in events if e.kind == "log" and (tag is None or e.tag == tag)]


def test_default_mode_writes_only_the_lrc_file(tmp_path):
    lib = library(tmp_path, "A - B.mp3")
    summary, events = run_lib(tmp_path, lib, tp.ScriptedFetcher(tier1=SYNCED))
    assert summary.synced_online == 1 and (lib / "A - B.lrc").exists()
    assert not metadata.has_embedded_lyrics(lib / "A - B.mp3")


@pytest.mark.parametrize("name,expect_field", [("A - B.mp3", "USLT + SYLT"), ("A - B.flac", "LYRICS"),
                                               ("A - B.opus", "LYRICS"), ("A - B.m4a", "\xa9lyr")])
def test_embed_mode_writes_tags_and_no_lrc_file(tmp_path, name, expect_field):
    lib = library(tmp_path, name)
    summary, events = run_lib(tmp_path, lib, tp.ScriptedFetcher(tier1=SYNCED), output_mode="embed")
    assert summary.synced_online == 1 and not (lib / "A - B.lrc").exists()
    assert metadata.has_embedded_lyrics(lib / name)
    assert any(f"Embedded synced lyrics into {name} ({expect_field})" in m for m in log_texts(events, "OK"))
    assert any("Embed Into Audio File" in m for m in log_texts(events, "Output"))


def test_both_mode_writes_the_file_and_the_tags(tmp_path):
    lib = library(tmp_path, "A - B.mp3")
    summary, _ = run_lib(tmp_path, lib, tp.ScriptedFetcher(tier1=SYNCED), output_mode="both")
    assert summary.synced_online == 1 and (lib / "A - B.lrc").exists()
    assert metadata.read_embedded_lyrics(lib / "A - B.mp3").startswith("[00:10.00]")


def test_unsupported_format_falls_back_to_a_separate_lrc(tmp_path):
    lib = library(tmp_path, "A - B.wav")
    summary, events = run_lib(tmp_path, lib, tp.ScriptedFetcher(tier1=SYNCED), output_mode="embed")
    assert summary.synced_online == 1 and (lib / "A - B.lrc").exists()
    assert any("not supported" in m for m in log_texts(events, "WARN"))
    assert any("Saving a separate" in m for m in log_texts(events, "Output"))


def test_a_corrupt_file_that_cannot_be_tagged_still_gets_its_lrc(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "A - B.flac").write_bytes(b"not really flac")
    summary, events = run_lib(tmp_path, lib, tp.ScriptedFetcher(tier1=SYNCED), output_mode="embed")
    assert (lib / "A - B.lrc").exists() and summary.failed == 0 and (lib / "A - B.flac").read_bytes() == b"not really flac"


def test_existing_lrc_is_embedded_without_any_search(tmp_path):
    lib = library(tmp_path, "A - B.mp3")
    (lib / "A - B.lrc").write_text(tp.lrc_of(), encoding="utf-8")

    class NoSearch(tp.ScriptedFetcher):
        def fetch(self, *a, **k):
            raise AssertionError("must not search when an .lrc already exists")

    summary, events = run_lib(tmp_path, lib, NoSearch(), output_mode="embed")
    assert metadata.has_embedded_lyrics(lib / "A - B.mp3") and (lib / "A - B.lrc").exists()
    assert any("Embedded the existing A - B.lrc" in m for m in log_texts(events, "OK"))
    assert summary.skipped == 1 and summary.failed == 0


def test_both_mode_with_an_existing_lrc_only_adds_the_missing_embed(tmp_path):
    lib = library(tmp_path, "A - B.flac")
    original = tp.lrc_of(3)
    (lib / "A - B.lrc").write_text(original, encoding="utf-8")
    run_lib(tmp_path, lib, tp.ScriptedFetcher(), output_mode="both")
    assert (lib / "A - B.lrc").read_text(encoding="utf-8") == original            # untouched
    assert metadata.has_embedded_lyrics(lib / "A - B.flac")


def test_second_run_skips_files_whose_lyrics_are_already_embedded(tmp_path):
    lib = library(tmp_path, "A - B.mp3")
    run_lib(tmp_path, lib, tp.ScriptedFetcher(tier1=SYNCED), output_mode="embed")
    summary, events = run_lib(tmp_path, lib, tp.ScriptedFetcher(tier1=SYNCED), output_mode="embed")
    assert summary.skipped == 1 and summary.synced_online == 0
    assert any("cached" in m or "already embedded" in m for m in log_texts(events, "SKIP"))


def test_cache_key_is_recomputed_after_tagging(tmp_path):
    """Tagging rewrites the file head, so a stale fingerprint would orphan the DB record."""
    lib = library(tmp_path, "A - B.mp3")
    path = lib / "A - B.mp3"
    before = fingerprint_file(path)
    run_lib(tmp_path, lib, tp.ScriptedFetcher(tier1=SYNCED), output_mode="embed")
    after = fingerprint_file(path)
    assert before != after
    with LyricsDatabase(tmp_path / "c.db") as db:
        assert db.get(after) is not None and db.get(after).status == Status.SYNCED_ONLINE.value


def test_overwrite_replaces_embedded_lyrics(tmp_path):
    lib = library(tmp_path, "A - B.mp3")
    run_lib(tmp_path, lib, tp.ScriptedFetcher(tier1=SYNCED), output_mode="embed")
    newer = LyricsResult("LRCLIB", synced_lrc=tp.lrc_of(9, 20))
    summary, _ = run_lib(tmp_path, lib, tp.ScriptedFetcher(tier1=newer), output_mode="embed", overwrite_existing=True)
    assert summary.synced_online == 1
    assert metadata.read_embedded_lyrics(lib / "A - B.mp3").startswith("[00:20.00]")


def test_plain_text_fallback_is_embedded_unsynced(tmp_path):
    from mutagen.id3 import ID3

    lib = library(tmp_path, "A - B.mp3")
    summary, _ = run_lib(tmp_path, lib, tp.ScriptedFetcher(plain=tp.plain_hit()), output_mode="embed",
                         use_ai_fallback=False, save_plain_fallback=True)
    assert summary.plain_only == 1 and not (lib / "A - B.txt").exists()
    tags = ID3(str(lib / "A - B.mp3"))
    assert tags.getall("USLT") and not tags.getall("SYLT")


def test_aligned_results_embed_with_the_locked_language(tmp_path):
    from mutagen.id3 import ID3

    class JaTranscriber(tp.ScriptedTranscriber):
        def open_session(self, track):
            session = super().open_session(track)
            session.language_info = LanguageGuess("ja", 0.97)
            return session

    lib = library(tmp_path, "Pluto Artist - Understand.mp3")
    summary, _ = run_lib(tmp_path, lib, tp.ScriptedFetcher(plain=tp.plain_hit()),
                         JaTranscriber(align_results=[tp.guided_result(True)]), output_mode="embed")
    assert summary.synced_aligned == 1
    assert ID3(str(lib / "Pluto Artist - Understand.mp3")).getall("SYLT")[0].lang == "jpn"


def test_the_ui_is_told_the_language_and_the_badge_is_cleared_per_track(tmp_path):
    lib = library(tmp_path, "A - B.wav", "C - D.wav")
    _, events = run_lib(tmp_path, lib, tp.ScriptedFetcher(tier1=SYNCED))
    cleared = [e for e in events if e.kind == "language" and not e.data]
    assert len(cleared) == 2                                                      # one reset per track


# =========================================================== "Your Own Lyrics"
LYRICS_TEXT = tp.LYRICS


class P2Session(tp.ScriptedSession):
    language_info = None

    def lock_language(self, hint=None, *, multi=False):
        self.language_info = self.owner.language
        self.owner.hints.append(hint)
        return self.language_info

    def _language(self):
        return self.language_info.code if self.language_info else None

    def align(self, reference, progress=None, *, min_match=None):
        self.owner.references.append(reference)
        self.owner.min_matches.append(min_match)
        return self.owner.align_results.pop(0)

    def sample_snippets(self, fractions=None, seconds=None, *, language=None, tag="AI: Identify"):
        self.owner.snippet_calls.append((fractions, seconds, language, tag))
        if self.owner.snippet_error:
            raise self.owner.snippet_error
        return list(self.owner.snippets)


class P2Transcriber(tp.ScriptedTranscriber):
    def __init__(self, *a, language=LanguageGuess("en", 0.97), **k):
        super().__init__(*a, **k)
        self.language, self.hints, self.min_matches, self.snippet_calls = language, [], [], []

    def open_session(self, track):
        self.opened += 1
        return P2Session(self)


def heard(*chunks):
    return [Snippet(10.0 * i, text) for i, text in enumerate(chunks)]


MATCHING = heard("the river runs beneath the silver moon tonight", "i keep on walking through the quiet empty streets",
                 "every window tells me stories i forgot nothing here is waiting")
WRONG = heard("I will always love you and I know that you are the one for me tonight",
              "hold me close and never let me go because I need you here with me",
              "we could dance until the morning light comes up over the hill")


def custom(tmp_path, tr, *, lyrics=LYRICS_TEXT, ask=None, name="Pluto Artist - Understand.wav", make=True, **settings):
    path = tmp_path / name
    if make:
        tp.tone_wav(path, 3.0)
    events, asked = [], []

    def recorder(payload):
        asked.append(payload)
        return ask(payload) if callable(ask) else bool(ask)

    with LyricsDatabase(tmp_path / "c.db") as db:
        proc = BatchProcessor(Settings(**settings), db, events.append, transcriber=tr,
                              ask=recorder if ask is not None else None)
        summary = proc.run_custom(path, lyrics)
    return summary, events, asked, path


def test_matching_lyrics_are_aligned_without_asking(tmp_path):
    tr = P2Transcriber(align_results=[tp.guided_result(True)], snippets=MATCHING)
    summary, events, asked, path = custom(tmp_path, tr, ask=False)
    assert summary.synced_aligned == 1 and not asked
    assert path.with_suffix(".lrc").read_text().count("official line") == 6
    assert tr.min_matches == [None]                                               # normal threshold
    assert any("match the audio" in m for m in log_texts(events, "AI: Verify"))
    # the lyrics (not the audio) were used as the reference, language locked first, in the right place
    assert tr.references[0].components[0].lines[0].startswith("The river runs")
    assert tr.hints[0].startswith("The river runs") and tr.snippet_calls[0][2] == "en"
    assert tr.snippet_calls[0][3] == "AI: Verify" and len(tr.snippet_calls[0][0]) == 4


def test_mismatching_lyrics_halt_and_nothing_is_written_when_the_user_cancels(tmp_path):
    tr = P2Transcriber(align_results=[], snippets=WRONG, language=LanguageGuess("ja", 0.98))
    summary, events, asked, path = custom(tmp_path, tr, ask=False)
    assert len(asked) == 1 and asked[0]["kind"] == "mismatch"
    assert asked[0]["percent"] < CUSTOM_LYRICS_MIN_MATCH * 100 and asked[0]["threshold"] == CUSTOM_LYRICS_MIN_MATCH
    assert asked[0]["audio_language"] == "Japanese" and asked[0]["lyrics_language"] == "English"
    assert not tr.references and not path.with_suffix(".lrc").exists()           # alignment never started
    assert summary.skipped == 1 and summary.failed == 0
    assert any("do not match" in m for m in log_texts(events, "AI: Verify"))


def test_force_sync_anyway_aligns_with_the_threshold_disabled(tmp_path):
    tr = P2Transcriber(align_results=[tp.guided_result(True)], snippets=WRONG)
    summary, events, asked, path = custom(tmp_path, tr, ask=True)
    assert len(asked) == 1 and summary.synced_aligned == 1
    assert tr.min_matches == [0.0] and path.with_suffix(".lrc").exists()
    assert any("forced by you" in m for m in log_texts(events, "AI: Verify"))


def test_without_a_confirmation_callback_a_mismatch_is_refused(tmp_path):
    tr = P2Transcriber(snippets=WRONG)
    summary, _, _, path = custom(tmp_path, tr)                                    # ask=None (headless)
    assert summary.skipped == 1 and not path.with_suffix(".lrc").exists()


def test_inconclusive_check_does_not_block_the_sync(tmp_path):
    tr = P2Transcriber(align_results=[tp.guided_result(True)], snippets=[])       # nothing audible (instrumental?)
    summary, events, asked, _ = custom(tmp_path, tr, ask=False)
    assert summary.synced_aligned == 1 and not asked
    assert any("Could not verify" in m for m in log_texts(events, "WARN") + log_texts(events, "AI: Verify"))


def test_weak_full_alignment_asks_again_and_force_realigns(tmp_path):
    tr = P2Transcriber(align_results=[tp.guided_result(False), tp.guided_result(True)], snippets=MATCHING)
    summary, _, asked, path = custom(tmp_path, tr, ask=True)
    assert [a["kind"] for a in asked] == ["weak"] and tr.min_matches == [None, 0.0]
    assert summary.synced_aligned == 1 and path.with_suffix(".lrc").exists()


def test_weak_full_alignment_declined_writes_nothing(tmp_path):
    tr = P2Transcriber(align_results=[tp.guided_result(False)], snippets=MATCHING)
    summary, _, asked, path = custom(tmp_path, tr, ask=False)
    assert [a["kind"] for a in asked] == ["weak"] and summary.skipped == 1 and not path.with_suffix(".lrc").exists()


def test_custom_lyrics_follow_the_output_mode(tmp_path):
    path = tmp_path / "Pluto Artist - Understand.mp3"
    if not make_audio(path):
        pytest.skip("no mp3 encoder")
    tr = P2Transcriber(align_results=[tp.guided_result(True)], snippets=MATCHING)
    summary, _, _, _ = custom(tmp_path, tr, ask=False, make=False, name=path.name, output_mode="both")
    assert summary.synced_aligned == 1 and path.with_suffix(".lrc").exists()
    assert metadata.has_embedded_lyrics(path)


def test_custom_lyrics_overwrite_an_existing_lrc(tmp_path):
    (tmp_path / "Pluto Artist - Understand.lrc").write_text("[00:01.00]old stuff\n[00:05.00]more old\n[00:09.00]x y\n")
    tr = P2Transcriber(align_results=[tp.guided_result(True)], snippets=MATCHING)
    custom(tmp_path, tr, ask=False)
    assert "old stuff" not in (tmp_path / "Pluto Artist - Understand.lrc").read_text()


def test_custom_input_is_validated(tmp_path):
    tr = P2Transcriber()
    summary, events, _, _ = custom(tmp_path, tr, lyrics="just one line")
    assert summary.failed == 1 and any("at least two lines" in m for m in log_texts(events, "ERR"))
    summary, events, _, _ = custom(tmp_path, tr, make=False, name="missing.wav")
    assert summary.failed == 1 and any("audio file" in m for m in log_texts(events, "ERR"))
    (tmp_path / "notes.txt").write_text("hi")
    summary, _, _, _ = custom(tmp_path, tr, make=False, name="notes.txt")
    assert summary.failed == 1 and tr.opened == 0


def test_custom_run_reports_ai_problems_and_finishes_cleanly(tmp_path):
    tr = P2Transcriber(snippet_error=transcriber_error())
    summary, events, _, _ = custom(tmp_path, tr, ask=False)
    assert summary.failed == 1 and any("Could not listen" in m for m in log_texts(events, "AI: Verify"))
    assert [e.kind for e in events][-1] == "done"                                # the UI always gets 'done'


def transcriber_error():
    from local_sync_lyrics.transcriber import TranscriptionError

    return TranscriptionError("model could not be loaded")


def test_stop_during_a_custom_run_is_clean(tmp_path):
    from local_sync_lyrics.transcriber import OperationCancelled

    tr = P2Transcriber(snippet_error=OperationCancelled())
    summary, events, _, path = custom(tmp_path, tr, ask=False)
    assert summary.failed == 0 and not path.with_suffix(".lrc").exists() and events[-1].kind == "done"


# ===================================================================== Qt GUI
def test_gui_is_renamed_and_starts_with_the_phase2_defaults(window):
    assert window.windowTitle() == "Lyricist"
    assert window.settings.output_mode == "lrc" and not window.sw_isolate.isChecked()
    assert window.sw_isolate.text() == "Isolate Vocals (Strip Background Music)"
    assert list(window.output_buttons) == list(OUTPUT_MODES)
    assert [button.text() for button in window.output_buttons.values()] == [
        "Save Separate .lrc File", "Embed Into Audio File", "Both (.lrc + Embedded)"]


def test_output_pills_select_persist_and_lock_while_running(window):
    window._select_output("both")
    assert window.settings.output_mode == "both"
    assert window.output_buttons["both"].isChecked()
    assert ta._wait_for(lambda: json.loads(window.store.path.read_text())["output_mode"] == "both")
    assert window._collect_settings().output_mode == "both"
    window._set_running(True)
    window._select_output("embed")
    assert window.settings.output_mode == "both"                        # locked while running
    assert not window.output_buttons["embed"].isEnabled()
    window._set_running(False)
    assert window.output_buttons["both"].isEnabled()


def test_isolation_switch_persists_and_warns_when_the_backend_is_missing(window):
    window.isolation_ready = False
    window.sw_isolate.setChecked(True)                                  # triggers _toggle_isolate
    assert window.settings.isolate_vocals is True
    assert window._collect_settings().isolate_vocals is True
    assert ta._wait_for(lambda: json.loads(window.store.path.read_text())["isolate_vocals"] is True)
    assert "audio-separator" in window.toast_label.text()
    window._set_running(True)
    assert not window.sw_isolate.isEnabled()
    window._set_running(False)
    assert window.sw_isolate.isEnabled()


def test_startup_probe_reports_the_isolation_backend(window, monkeypatch):
    from local_sync_lyrics import gui
    from local_sync_lyrics.hardware_profiler import HardwareProfile, PreflightReport

    report = PreflightReport(ffmpeg_path="", audioread=False, audio_separator=False)
    monkeypatch.setattr(gui, "run_preflight", lambda: report)
    monkeypatch.setattr(gui, "get_profile",
                        lambda: HardwareProfile(physical_cores=4, ram_total_gb=8))
    window._startup()
    assert ta._wait_for(lambda: window.isolation_ready is False)
    assert "unavailable" in window.isolation_status.text()
    assert "audio-separator" in window.isolate_hint.text()
    assert not window.banner.isHidden()


def test_language_event_shows_and_clears_the_badge(window):
    window._set_running(True)
    window._handle_event(Event("language", message="Japanese (98%)",
                              data={"code": "ja", "name": "Japanese", "probability": 0.984,
                                    "source": "audio"}))
    assert not window.lang_chip.isHidden() and window.lang_text.text() == "Japanese · 98%"
    assert "Japanese" in window.lang_chip.toolTip()
    window._handle_event(Event("language", data={}))
    assert window.lang_chip.isHidden()
    window._handle_event(Event("language", data={"code": "ko", "name": "Korean", "probability": 0.5}))
    window._set_running(False)
    assert window.lang_chip.isHidden()                                  # cleared when the run ends


def test_detect_language_switch_is_collected(window):
    assert window.sw_detect_lang.isChecked() is True
    window.sw_detect_lang.setChecked(False)
    assert window._collect_settings().detect_language is False


def test_new_log_tags_have_colours_in_both_modes(window):
    for tag in ("Audio: Isolate", "AI: Language", "AI: Verify", "Output"):
        window._append_log(tag, "x")
        assert window.palette.tag_color(tag)
    dark_audio = window.palette.tags["Audio"]
    window._select_mode("light")
    assert window.palette.tags["Audio"] != dark_audio
    assert window.palette.tags["Audio"] != window.palette.tags["Output"]
    assert len(window.log_entries) >= 4
    assert [tag for _, tag, _, _ in window.log_entries][-4:] == [
        "Audio: Isolate", "AI: Language", "AI: Verify", "Output"]


def test_custom_lyrics_dialog_opens_and_is_inert_when_running(window):
    window._open_custom()
    assert window.custom_dialog is not None and not window.custom_dialog.isHidden()
    window.custom_dialog.reject()
    window._set_running(True)
    window._open_custom()
    assert window.custom_dialog is None or window.custom_dialog.isHidden()   # no dialog during a run
    window._set_running(False)


def test_lyrics_box_counts_lines(window):
    window._open_custom()
    dialog = window.custom_dialog
    dialog.lyrics.setPlainText("one\n\n two \nthree\n")
    assert dialog.counter.text() == "3 lines"
    dialog.lyrics.clear()
    assert dialog.counter.text() == "0 lines"
    dialog.reject()


def test_start_custom_validates_then_starts_a_worker_with_the_lyrics(window, tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(window, "_launch",
                        lambda target, settings, lyrics: started.append((target, settings, lyrics)))
    window._start_custom("", "a\nb")                                    # no audio
    assert not started and "Choose the audio" in window.toast_label.text()
    audio = tp.tone_wav(tmp_path / "s.wav", 2.0)
    window._start_custom(str(audio), "only one line")
    assert not started and "at least two lines" in window.toast_label.text()
    window._start_custom(str(audio), LYRICS_TEXT)
    assert started
    target, settings, lyrics = started[0]
    assert target == audio and lyrics == LYRICS_TEXT.strip() and isinstance(settings, Settings)
    assert window._running and window.status_text.text() == "Syncing"
    window._set_running(False)


def test_confirm_dialog_has_the_force_and_cancel_actions(window):
    from PySide6.QtWidgets import QLabel, QPushButton

    from local_sync_lyrics.gui import ConfirmDialog

    payload = {"kind": "mismatch", "title": "Idol", "percent": 12, "threshold": 0.35,
               "audio_language": "Japanese", "lyrics_language": "English"}
    dialog = ConfirmDialog(payload, window.palette, window)
    texts = " ".join(label.text() for label in dialog.findChildren(QLabel))
    assert "only 12% similar" in texts and "at least 35% expected" in texts
    assert "sounds Japanese, but the lyrics look English" in texts
    buttons = [button.text() for button in dialog.findChildren(QPushButton)]
    assert "Cancel" in buttons and "Force Sync Anyway" in buttons


def test_weak_alignment_dialog_uses_its_own_wording(window):
    from PySide6.QtWidgets import QLabel

    from local_sync_lyrics.gui import ConfirmDialog

    dialog = ConfirmDialog({"kind": "weak", "title": "Song", "percent": 22, "threshold": 0.4},
                           window.palette, window)
    texts = " ".join(label.text() for label in dialog.findChildren(QLabel))
    assert "does not fit these lyrics" in texts and "only 22%" in texts


def test_stop_unblocks_a_pending_question(window, tmp_path):
    from local_sync_lyrics.gui import BatchWorker

    worker = BatchWorker(tmp_path, Settings(), threading.Event())
    result = {}
    thread = threading.Thread(target=lambda: result.setdefault("a", worker._ask_user({"kind": "mismatch"})),
                              daemon=True)
    thread.start()
    time.sleep(0.1)
    worker.cancel.set()
    thread.join(3)
    assert result["a"] is False


def test_confirm_choice_unblocks_the_worker_once(window, monkeypatch):
    from PySide6.QtWidgets import QDialog

    from local_sync_lyrics import gui

    class FakeDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return QDialog.Accepted

    monkeypatch.setattr(gui, "ConfirmDialog", FakeDialog)
    box, done = {}, threading.Event()
    window._show_confirm({"kind": "mismatch", "percent": 5, "threshold": 0.35,
                          "_box": box, "_done": done})
    assert box["force"] is True and done.is_set()


# ========================================================================= CLI
def test_cli_parser_knows_the_phase2_options():
    args = cli.build_parser().parse_args(["song.mp3", "--lyrics", "l.txt", "--force", "--output", "both",
                                          "--isolate-vocals", "--no-detect-language"])
    assert (args.lyrics, args.force, args.output, args.isolate_vocals, args.no_detect_language) == (
        "l.txt", True, "both", True, True)
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["x", "--output", "sideways"])


def test_cli_validates_lyrics_mode_arguments(tmp_path, monkeypatch):
    monkeypatch.setenv("LYRICIST_HOME", str(tmp_path / "home"))
    folder, lyrics = tmp_path / "music", tmp_path / "l.txt"
    folder.mkdir()
    lyrics.write_text("a\nb\n")
    assert cli.run_cli([str(folder), "--lyrics", str(lyrics)]) == 2                # needs a FILE
    audio = tp.tone_wav(tmp_path / "s.wav", 1.0)
    assert cli.run_cli([str(audio), "--lyrics", str(tmp_path / "nope.txt")]) == 2


def test_cli_passes_the_options_through_and_answers_questions(tmp_path, monkeypatch):
    monkeypatch.setenv("LYRICIST_HOME", str(tmp_path / "home"))
    audio, lyrics = tp.tone_wav(tmp_path / "s.wav", 1.0), tmp_path / "l.txt"
    lyrics.write_text("first line\nsecond line\n")
    seen = {}

    class StubProcessor:
        def __init__(self, settings, db, emit, cancel, ask=None):
            seen.update(settings=settings, ask=ask)

        def run_custom(self, path, text):
            seen.update(path=path, text=text, forced=seen["ask"]({"kind": "mismatch", "percent": 9}))
            return type("S", (), {"cancelled": False, "failed": 0, "total": 1, "processed": 1,
                                  "synced_online": 0, "synced_aligned": 1, "synced_ai": 0, "skipped": 0,
                                  "not_found": 0, "plain_only": 0, "as_dict": lambda self: {}})()

    monkeypatch.setattr(cli, "BatchProcessor", StubProcessor)
    cli.run_cli([str(audio), "--lyrics", str(lyrics), "--force", "--output", "embed", "--isolate-vocals",
                 "--no-detect-language"])
    s = seen["settings"]
    assert (s.output_mode, s.isolate_vocals, s.detect_language) == ("embed", True, False)
    assert seen["path"] == audio and seen["text"].startswith("first line") and seen["forced"] is True
