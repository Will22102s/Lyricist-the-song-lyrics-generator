"""Offline tests: no network, no model downloads, no display required."""

from __future__ import annotations

import json
import math
import struct
import sys
import threading
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_sync_lyrics import api_providers, lrc_utils, text_utils, transcriber  # noqa: E402
from local_sync_lyrics.config import PROVIDER_IDS, Settings, SettingsStore  # noqa: E402
from local_sync_lyrics.database import LyricsDatabase, TrackRecord, fingerprint_file  # noqa: E402
from local_sync_lyrics.hardware_profiler import HardwareProfile, Recommendation, recommend_cpu  # noqa: E402
from local_sync_lyrics.models import Event, Status, TrackInfo, TranscriptionResult  # noqa: E402
from local_sync_lyrics.processor import BatchProcessor  # noqa: E402


# ----------------------------------------------------------------- helpers
def make_wav(path: Path, seconds: float = 3.0, rate: int = 44100, channels: int = 2, seed: int = 20) -> Path:
    n = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for i in range(n):
            sample = int(8000 * math.sin(i / seed))
            frames += struct.pack("<h", sample) * channels
        w.writeframes(bytes(frames))
    return path


GOOD_LRC = "\n".join(f"[00:{i * 5:02d}.00]line number {i}" for i in range(1, 8)) + "\n"


# --------------------------------------------------------------- text_utils
@pytest.mark.parametrize("stem,artist,title", [
    ("01 - Queen - Bohemian Rhapsody (Official Video) [dQw4w9WgXcQ]", "Queen", "Bohemian Rhapsody"),
    ("Artist_Name_-_Song_Title_[1080p]", "Artist Name", "Song Title"),
    ("1-05 Radiohead - Creep", "Radiohead", "Creep"),
    ("Jay-Z - Empire State of Mind (Remastered 2011)", "Jay-Z", "Empire State of Mind"),
    ("ABBA - Dancing Queen 4K", "ABBA", "Dancing Queen"),
])
def test_parse_filename(stem, artist, title):
    parsed = text_utils.parse_filename(stem)
    assert (parsed.artist, parsed.title) == (artist, title)


def test_real_parentheses_survive_and_numbers_not_stripped():
    assert text_utils.parse_filename("99 Problems").title == "99 Problems"
    assert "(New Orleans Remix)" in text_utils.parse_filename("The Weeknd - Blinding Lights (New Orleans Remix)").title


def test_redundant_artist_prefix_removed():
    assert text_utils.sanitize_title("Queen - Bohemian Rhapsody", "Queen") == "Bohemian Rhapsody"
    assert text_utils.sanitize_title("queen — Bohemian Rhapsody", "Queen") == "Bohemian Rhapsody"


def test_query_variants_dedupe_and_feat():
    variants = text_utils.query_variants("Daft Punk feat. Pharrell", "Get Lucky (feat. Pharrell)")
    assert ("Daft Punk", "Get Lucky") in variants
    assert len(variants) == len(set(variants))
    assert text_utils.query_variants("", "") == []


def test_youtube_id_detection():
    assert text_utils.looks_like_youtube_id("dQw4w9WgXcQ")
    assert not text_utils.looks_like_youtube_id("Remastered")


# ---------------------------------------------------------------- lrc_utils
def test_parse_multi_stamp_offset_and_enhanced():
    text = "[offset:500]\n[00:10.00][00:20.50]chorus <00:10.20>word\n[ti:x]\n[01:00]end"
    lines = lrc_utils.parse_lrc(text)
    assert [round(l.time, 2) for l in lines] == [10.5, 21.0, 60.5]
    assert lines[0].text == "chorus word"


def test_format_timestamp_rounding():
    assert lrc_utils.format_timestamp(59.999) == "[01:00.00]"
    assert lrc_utils.format_timestamp(0) == "[00:00.00]"
    assert lrc_utils.format_timestamp(3725.5) == "[62:05.50]"


def test_validate_synced():
    ok, _, _ = lrc_utils.validate_synced(GOOD_LRC, duration=60)
    assert ok
    assert not lrc_utils.validate_synced("just words\nno stamps", 60)[0]
    assert not lrc_utils.validate_synced(GOOD_LRC, duration=300)[0] is False  # fits a long track
    short = "\n".join(f"[10:{i:02d}.00]x line {i}" for i in range(6))
    assert not lrc_utils.validate_synced(short, duration=60)[0]  # starts after the song ends


def test_ttml_conversion_skips_translations():
    ttml = """<tt xmlns="http://www.w3.org/ns/ttml" xmlns:ttm="http://www.w3.org/ns/ttml#metadata"><body><div>
      <p begin="00:01.500" end="00:03.000"><span begin="00:01.500">Hello</span> <span begin="00:02.0">world</span>
        <span ttm:role="x-translation">Hola mundo</span></p>
      <p begin="1:02.250">Second line</p></div></body></tt>"""
    lines = lrc_utils.parse_ttml(ttml)
    assert [(l.time, l.text) for l in lines] == [(1.5, "Hello world"), (62.25, "Second line")]
    assert lrc_utils.parse_ttml("<not xml") == []


def test_atomic_write(tmp_path):
    target = tmp_path / "a.lrc"
    lrc_utils.write_text_atomic(target, "héllo\n")
    assert target.read_text(encoding="utf-8") == "héllo\n"
    assert not list(tmp_path.glob(".lsl-*"))


# ------------------------------------------------------------------- config
def test_settings_roundtrip_and_corrupt_file(tmp_path, monkeypatch):
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "LSL_WHISPER_MODEL", "LSL_LANGUAGE"):
        monkeypatch.delenv(var, raising=False)
    store = SettingsStore(tmp_path / "settings.json")
    s = store.load()
    s.whisper_model, s.hf_token = "small", "hf_secret"
    assert store.save(s)
    loaded = store.load()
    assert (loaded.whisper_model, loaded.hf_token) == ("small", "hf_secret")

    (tmp_path / "settings.json").write_text("{ not json", encoding="utf-8")
    fresh = store.load()  # must not raise
    assert fresh.whisper_model == "auto"
    assert list(tmp_path.glob("settings.corrupt-*.json"))


def test_env_fallback_and_token_injection(tmp_path, monkeypatch):
    monkeypatch.setenv("LSL_WHISPER_MODEL", "base")
    monkeypatch.setenv("HF_TOKEN", "env_token")
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    store = SettingsStore(tmp_path / "s.json")
    s = store.load()
    assert s.whisper_model == "base"                 # env used because file has no value
    assert s.token_from_environment()
    store.save(s)
    on_disk = json.loads((tmp_path / "s.json").read_text())
    assert on_disk["hf_token"] == ""                 # env token is never written to disk
    assert on_disk["whisper_model"] == "auto"
    monkeypatch.delenv("HF_TOKEN")
    s.hf_token = "from_file"
    s.apply_environment()
    import os
    assert os.environ["HF_TOKEN"] == "from_file"


def test_settings_coercion_and_clamping():
    s = Settings.from_dict({"request_timeout": "999", "min_match_score": "x", "enabled_providers": ["nope"],
                            "use_ai_fallback": "false", "theme_mode": "neon"})
    assert s.request_timeout == 60.0
    assert s.min_match_score == 72
    assert s.enabled_providers == list(PROVIDER_IDS)
    assert s.use_ai_fallback is False and s.theme_mode == "dark"


# ----------------------------------------------------------------- database
def test_database_cache_and_recovery(tmp_path):
    f = make_wav(tmp_path / "a.wav", 1)
    fp = fingerprint_file(f)
    renamed = f.rename(tmp_path / "b.wav")
    assert fingerprint_file(renamed) == fp           # survives renames

    db_path = tmp_path / "cache.db"
    with LyricsDatabase(db_path) as db:
        db.upsert(TrackRecord(fp, str(renamed), Status.SYNCED_ONLINE.value, artist="x"))
        db.upsert(TrackRecord(fp, str(renamed), Status.SYNCED_AI.value, ai_tried=True))
        rec = db.get(fp)
        assert rec and rec.status == "synced_ai" and rec.attempts == 2 and rec.ai_tried
        assert db.stats() == {"synced_ai": 1}

    db_path.write_bytes(b"this is definitely not sqlite" * 50)   # corrupt it
    with LyricsDatabase(db_path) as db2:                          # must self-heal
        assert db2.get(fp) is None
        db2.upsert(TrackRecord("z", "p", "failed"))
    assert list(tmp_path.glob("cache.db.corrupt-*"))


# ------------------------------------------------------------- audio decode
def test_decode_audio_safe_resamples_to_16k_mono(tmp_path):
    wav = make_wav(tmp_path / "t.wav", 3.0, rate=44100, channels=2)
    audio = transcriber.decode_audio_safe(wav)
    assert audio.dtype == np.float32 and audio.ndim == 1
    assert abs(len(audio) - 3 * 16000) < 800
    assert 0.1 < float(np.abs(audio).max()) < 1.0


def test_decode_survives_av_open_typeerror(tmp_path, monkeypatch):
    """Simulates the 'metadata_errors' TypeError of mismatched PyAV versions."""
    import av

    real_open = av.open
    calls = []

    def picky_open(file, mode="r", **kwargs):
        calls.append(kwargs)
        if "metadata_errors" in kwargs:
            raise TypeError("open() got an unexpected keyword argument 'metadata_errors'")
        return real_open(file, mode=mode, **kwargs)

    monkeypatch.setattr(av, "open", picky_open)
    audio = transcriber.decode_audio_safe(make_wav(tmp_path / "t.wav", 2.0))
    assert len(audio) > 16000
    assert any("metadata_errors" in c for c in calls)   # it tried, then recovered


def test_decode_garbage_raises_clean_error(tmp_path):
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"\x00\x01garbage" * 100)
    with pytest.raises(transcriber.AudioDecodeError):
        transcriber.decode_audio_safe(bad)


def test_decode_silence_is_reported(tmp_path):
    p = tmp_path / "silent.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 32000)
    with pytest.raises(transcriber.NoSpeechError):
        transcriber.decode_audio_safe(p)


def test_clean_segments_filters_junk():
    S = transcriber._Segment
    segs = [S(0, 1, "Thanks for watching!"), S(1, 2, "[Music]"), S(2, 3, "real lyric line"),
            *[S(3 + i, 4 + i, "la la la") for i in range(9)], S(20, 21, "ghost", 0.95, -2.0)]
    lines = transcriber.clean_segments(segs)
    texts = [l.text for l in lines]
    assert texts[0] == "real lyric line"
    assert texts.count("la la la") == 4              # endless loops are capped
    assert "ghost" not in texts and "Thanks for watching!" not in texts


# ----------------------------------------------------- GPU -> CPU fallback
def _cuda_profile() -> HardwareProfile:
    p = HardwareProfile(physical_cores=8, ram_total_gb=32, cuda_devices=1, cuda_libs_ok=True,
                        cuda_compute_types=["float16", "int8"], gpu_name="Fake GPU", gpu_vram_gb=12)
    p.recommendation = Recommendation("large-v3", "cuda", "float16", "test")
    p.cpu_recommendation = recommend_cpu(8, 32)
    return p


class FakeModel:
    def __init__(self, device, fail_on_inference):
        self.device, self.fail = device, fail_on_inference

    def transcribe(self, audio, **kwargs):
        info = SimpleNamespace(duration=3.0, language="en", language_probability=0.99)

        def gen():
            if self.fail:  # real CTranslate2 raises lazily, while iterating
                raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
            yield SimpleNamespace(start=0.5, end=1.5, text=" hello there ", no_speech_prob=0.0, avg_logprob=-0.2)
            yield SimpleNamespace(start=2.0, end=3.0, text=" general kenobi ", no_speech_prob=0.0, avg_logprob=-0.2)

        return gen(), info


def test_gpu_failure_falls_back_to_cpu_int8(tmp_path, monkeypatch):
    wav = make_wav(tmp_path / "Artist - Song.wav", 3.0)
    built = []

    def fake_make(name, device, compute_type, threads, local_only):
        built.append((name, device, compute_type))
        return FakeModel(device, fail_on_inference=(device == "cuda"))

    monkeypatch.setattr(transcriber.Transcriber, "_make_model", staticmethod(fake_make))
    messages = []
    t = transcriber.Transcriber(Settings(), _cuda_profile(), log=lambda *a: messages.append(a))
    result = t.transcribe(TrackInfo(wav, "Artist", "Song", duration=3.0))

    assert built[0][1] == "cuda" and built[-1][1:] == ("cpu", "int8")
    assert built[-1][0] != "large-v3"                  # auto mode downgrades the model on CPU
    assert result.device == "cpu" and result.line_count == 2
    assert "[00:00.50]hello there" in result.lrc_text
    assert any("falling back" in m[1].lower() or "retrying on cpu" in m[1].lower() for m in messages)


def test_gpu_init_error_falls_back_before_inference(tmp_path, monkeypatch):
    def fake_make(name, device, compute_type, threads, local_only):
        if device == "cuda":
            raise RuntimeError("CUDA failed with error out of memory")
        return FakeModel("cpu", False)

    monkeypatch.setattr(transcriber.Transcriber, "_make_model", staticmethod(fake_make))
    t = transcriber.Transcriber(Settings(), _cuda_profile())
    result = t.transcribe(TrackInfo(make_wav(tmp_path / "x.wav", 3.0), "A", "B", duration=3.0))
    assert result.device == "cpu"


def test_model_load_failure_is_remembered_not_retried(tmp_path, monkeypatch):
    attempts = []

    def fake_make(name, device, compute_type, threads, local_only):
        attempts.append(local_only)
        raise OSError("Name or service not known")      # offline, model not cached

    monkeypatch.setattr(transcriber.Transcriber, "_make_model", staticmethod(fake_make))
    t = transcriber.Transcriber(Settings(), HardwareProfile(physical_cores=4, ram_total_gb=8))
    track = TrackInfo(make_wav(tmp_path / "x.wav", 3.0), "A", "B", duration=3.0)
    for _ in range(3):
        with pytest.raises(transcriber.TranscriptionError):
            t.transcribe(track)
    assert attempts == [True, False]                    # cache probe + one download attempt, once


def test_hardware_recommendation_scales():
    assert recommend_cpu(2, 4).model == "tiny"
    assert recommend_cpu(4, 8).model == "base"
    assert recommend_cpu(16, 32).model == "small"


# ---------------------------------------------------------------- providers
class FakeHttp:
    """Stands in for HttpClient; maps URL fragments to canned payloads."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def _match(self, url):
        self.calls.append(url)
        for key, value in self.routes.items():
            if key in url:
                if isinstance(value, Exception):
                    raise value
                return value
        return None

    def get_json(self, url, params=None):
        return self._match(url)

    def get(self, url, params=None, headers=None):
        value = self._match(url)
        if value is None:
            return None
        return SimpleNamespace(text=value if isinstance(value, str) else json.dumps(value),
                               json=lambda: value if not isinstance(value, str) else json.loads(value))

    def close(self):
        pass


def _query(artist="Queen", title="Bohemian Rhapsody", duration=355.0):
    return api_providers.TrackQuery(text_utils.query_variants(artist, title), duration=duration)


def test_lrclib_picks_best_synced_match():
    lr = "\n".join(f"[{i // 60:02d}:{i % 60:02d}.00]line {i}" for i in range(10, 300, 20))
    http = FakeHttp({
        "/api/get": None,
        "/api/search": [
            {"trackName": "Bohemian Rhapsody (Live)", "artistName": "Other Band", "duration": 200, "syncedLyrics": lr},
            {"trackName": "Bohemian Rhapsody", "artistName": "Queen", "duration": 354, "syncedLyrics": lr, "plainLyrics": "x"},
        ],
    })
    hit = api_providers.LrclibProvider(http, Settings()).search(_query())
    assert hit and hit.artist == "Queen" and hit.has_synced and hit.score > 80


def test_lrclib_rejects_wrong_duration_and_reports_instrumental():
    http = FakeHttp({"/api/get": None, "/api/search": [
        {"trackName": "Bohemian Rhapsody", "artistName": "Queen", "duration": 100, "syncedLyrics": GOOD_LRC}]})
    assert api_providers.LrclibProvider(http, Settings()).search(_query()) is None

    http = FakeHttp({"/api/get": {"trackName": "Song", "artistName": "Band", "duration": 120,
                                  "instrumental": True, "syncedLyrics": None, "plainLyrics": None}})
    hit = api_providers.LrclibProvider(http, Settings()).search(_query("Band", "Song", 120))
    assert hit and hit.instrumental


def test_betterlyrics_parses_nested_ttml():
    body = "".join(f'<p begin="{i * 20}.0s">lyric {i}</p>' for i in range(1, 15))
    payload = {"ttml": f'<tt xmlns="http://www.w3.org/ns/ttml"><body><div>{body}</div></body></tt>'}
    hit = api_providers.BetterLyricsProvider(FakeHttp({"getLyrics": payload}), Settings()).search(_query())
    assert hit and hit.synced_lrc.count("[0") >= 10 and "[00:20.00]lyric 1" in hit.synced_lrc


def test_megalobiz_scrapes_search_and_detail_pages():
    search_html = '<a class="entity_name" href="/lrc/maker/queen-bohemian-rhapsody.12">Queen - Bohemian Rhapsody</a>'
    detail_html = '<div class="lyrics_details"><span>' + "<br>".join(
        f"[{i // 60:02d}:{i % 60:02d}.00] words {i}" for i in range(5, 340, 20)) + "</span></div>"
    http = FakeHttp({"/search/all": search_html, "/lrc/maker/": detail_html})
    hit = api_providers.MegalobizProvider(http, Settings()).search(_query())
    assert hit and hit.has_synced and "words 5" in hit.synced_lrc


def test_chain_continues_after_provider_failure(monkeypatch):
    settings = Settings()
    fetcher = api_providers.LyricsFetcher(settings)
    good = api_providers.LyricsResult("Fake", synced_lrc=GOOD_LRC, score=90)

    class Boom(api_providers.LyricsProvider):
        id = label = "boom"
        def search(self, q):
            raise api_providers.ProviderError("HTTP 503")

    class Crash(Boom):
        def search(self, q):
            raise ValueError("parser bug")

    class Fine(Boom):
        def search(self, q):
            return good

    fetcher.providers = [Boom(None, settings), Crash(None, settings), Fine(None, settings)]
    outcome = fetcher.fetch(TrackInfo(Path("x.mp3"), "A", "B", duration=40))
    assert outcome.synced is good and set(outcome.errors) == {"boom"}
    fetcher.close()


# ---------------------------------------------------------------- processor
class FakeFetcher:
    """Stands in for LyricsFetcher (Tier 1 only unless told otherwise)."""

    def __init__(self, behaviour):
        self.behaviour, self.seen = behaviour, []

    def fetch(self, track, tier2=True):
        self.seen.append(track.title)
        return self.behaviour(track)

    def fetch_tier2(self, track, outcome=None):
        return outcome

    def fetch_for(self, artist, title):
        from local_sync_lyrics.api_providers import FetchOutcome
        return FetchOutcome()

    def identify_by_lyrics(self, snippet):
        return []

    def close(self):
        pass


class FakeSession:
    def __init__(self, owner, track):
        self.owner, self.track = owner, track

    def blind(self, progress=None):
        self.owner.calls.append(self.track.title)
        if self.track.title in self.owner.fail_titles:
            raise transcriber.TranscriptionError("boom")
        return TranscriptionResult(GOOD_LRC, "en", 0.9, 7, "tiny", "cpu", 0.1)

    def sample_snippets(self, *a, **k):
        return []

    def align(self, reference, progress=None):
        raise AssertionError("alignment must not run without reference lyrics")

    def close(self):
        pass


class FakeTranscriber:
    def __init__(self, fail_titles=()):
        self.calls, self.fail_titles = [], fail_titles

    def open_session(self, track):
        return FakeSession(self, track)

    def close(self):
        pass


def _library(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    for n, name in enumerate(("Band - Online.wav", "Band - Needs AI.wav", "Band - Broken AI.wav", "Band - Corrupt.mp3")):
        if name.endswith(".mp3"):
            (tmp_path / name).write_bytes(b"not audio at all" * 20)
        else:
            make_wav(tmp_path / name, 3.0, seed=15 + n * 7)   # distinct audio => distinct fingerprints
    return tmp_path


def _online_only_for_online(track):
    from local_sync_lyrics.api_providers import FetchOutcome
    from local_sync_lyrics.models import LyricsResult

    if track.title == "Online":
        return FetchOutcome(synced=LyricsResult("LRCLIB", synced_lrc=GOOD_LRC, score=95))
    return FetchOutcome()


def test_batch_isolation_cache_and_rerun(tmp_path):
    lib = _library(tmp_path / "lib")
    events: list[Event] = []
    settings = Settings(use_ai_fallback=True)
    fetcher = FakeFetcher(_online_only_for_online)
    ai = FakeTranscriber(fail_titles=("Broken AI",))

    with LyricsDatabase(tmp_path / "c.db") as db:
        summary = BatchProcessor(settings, db, events.append, fetcher=fetcher, transcriber=ai).run(lib)
        assert summary.total == 4
        assert summary.synced_online == 1 and summary.synced_ai == 2   # Needs AI + Corrupt (fake AI ok)
        assert summary.failed == 1                                      # Broken AI - and the batch went on
        assert (lib / "Band - Online.lrc").exists() and (lib / "Band - Needs AI.lrc").exists()
        assert not (lib / "Band - Broken AI.lrc").exists()
        assert events[-1].kind == "done" and any(e.kind == "progress" for e in events)

        # second run: finished tracks are skipped; the failed one is NOT retried within a week
        fetcher2, ai2 = FakeFetcher(_online_only_for_online), FakeTranscriber()
        s2 = BatchProcessor(Settings(use_ai_fallback=True), db, lambda e: None, fetcher=fetcher2, transcriber=ai2).run(lib)
        assert s2.skipped == 4 and fetcher2.seen == [] and ai2.calls == []

        # "retry failed" gives the failed track another go
        s3 = BatchProcessor(Settings(retry_failed=True), db, lambda e: None,
                            fetcher=FakeFetcher(_online_only_for_online), transcriber=FakeTranscriber()).run(lib)
        assert s3.synced_ai == 1 and s3.skipped == 3


def test_deleted_lrc_is_regenerated_and_existing_respected(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    make_wav(lib / "Band - Online.wav", 3.0)
    with LyricsDatabase(tmp_path / "c.db") as db:
        run = lambda s: BatchProcessor(s, db, lambda e: None, fetcher=FakeFetcher(_online_only_for_online),
                                       transcriber=FakeTranscriber()).run(lib)
        assert run(Settings()).synced_online == 1
        (lib / "Band - Online.lrc").unlink()
        assert run(Settings()).synced_online == 1            # cache says synced, file is gone -> redo
        (lib / "Band - Online.lrc").write_text("[00:01.00]mine\n")
        assert run(Settings()).skipped == 1                   # a user's own .lrc is never touched
        assert "mine" in (lib / "Band - Online.lrc").read_text()
        assert run(Settings(overwrite_existing=True)).synced_online == 1


def test_cancel_stops_between_tracks(tmp_path):
    lib = _library(tmp_path / "lib")
    cancel = threading.Event()
    fetcher = FakeFetcher(lambda t: (cancel.set(), _online_only_for_online(t))[1])
    with LyricsDatabase(tmp_path / "c.db") as db:
        summary = BatchProcessor(Settings(use_ai_fallback=False), db, lambda e: None, cancel,
                                 fetcher=fetcher, transcriber=FakeTranscriber()).run(lib)
    assert summary.cancelled and summary.processed < 4


def test_unwritable_target_is_a_failure_not_a_crash(tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    lib.mkdir()
    make_wav(lib / "Band - Online.wav", 3.0)

    def deny(path, text):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr("local_sync_lyrics.processor.write_text_atomic", deny)
    with LyricsDatabase(tmp_path / "c.db") as db:
        summary = BatchProcessor(Settings(), db, lambda e: None, fetcher=FakeFetcher(_online_only_for_online),
                                 transcriber=FakeTranscriber()).run(lib)
    assert summary.failed == 1

# --------------------------------------------------------------------- Qt GUI
def test_gui_builds_and_pumps_events(window, tmp_path):
    from local_sync_lyrics import gui

    assert window.stack.count() == 2

    window._handle_event(Event("log", message="Searching LRCLIB…", tag="API"))
    window._handle_event(Event("progress", message="song.mp3", current=3, total=12))
    window._handle_event(Event("stats", data={"synced_online": 2, "failed": 1}))
    assert window.progress_bar.value() == 250 and "3 / 12" in window.progress_label.text()
    assert window.stat_values["synced_online"].text() == "2"
    assert "Searching LRCLIB" in window.log_entries[-1][2]

    for i in range(gui.MAX_LOG_LINES + 50):                  # log buffer is bounded
        window._append_log("INFO", f"line {i}")
    assert len(window.log_entries) <= gui.MAX_LOG_LINES

    window.token_field.setText("hf_abc")
    window.model_dd.setCurrentIndex(window.model_dd.findData("small"))
    for checkbox in window.provider_checks.values():
        checkbox.setChecked(False)                           # nothing selected -> falls back to all
    window._save_settings()
    assert window.settings.hf_token == "hf_abc"
    assert window.settings.enabled_providers == list(gui.PROVIDER_IDS)

    window.folder_field.setText(str(tmp_path / "does-not-exist"))
    window._start_batch()
    assert window.worker is None and "existing music folder" in window.toast_label.text()


def test_gui_full_run_through_worker_thread(window, tmp_path, monkeypatch):
    import time

    from PySide6.QtWidgets import QApplication

    lib = tmp_path / "lib"
    lib.mkdir()
    make_wav(lib / "Band - Online.wav", 3.0)
    monkeypatch.setattr("local_sync_lyrics.processor.LyricsFetcher",
                        lambda settings, log, cancel: FakeFetcher(_online_only_for_online))

    window.folder_field.setText(str(lib))
    window._start_batch()
    assert window.worker is not None and window.worker.isRunning()
    assert window.stop_btn.isEnabled() and not window.start_btn.isEnabled()

    end = time.time() + 15
    while time.time() < end:
        QApplication.processEvents()
        if window.worker is not None and not window.worker.isRunning():
            break
        time.sleep(0.02)
    QApplication.processEvents()

    assert (lib / "Band - Online.lrc").exists()
    assert window.start_btn.isEnabled() and not window.stop_btn.isEnabled()
    assert any("Saved Band - Online.lrc" in message for _, _, message, _ in window.log_entries)


def test_gui_switches_views_and_survives_theme_changes(window):
    window._show_view("settings")
    assert window.stack.currentWidget() is window.settings_view
    window._append_log("API", "hello")
    window._select_accent("apple_music")
    window._select_mode("light")
    assert window.palette.accent == "#FA243C" and not window.palette.dark
    window._show_view("workspace")
    assert window.stack.currentWidget() is window.workspace_view
    assert window.log_entries[-1][1] == "API"
