"""Phase 2 - audio layer: language, lyric matching, tag embedding, vocal isolation, sessions."""

from __future__ import annotations

import importlib.machinery
import random
import sys
import threading
import time
import types
import wave
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_pipeline as tp  # noqa: E402
from audio_fixtures import make_audio  # noqa: E402
from local_sync_lyrics import language, metadata, separation, transcriber  # noqa: E402
from local_sync_lyrics.config import Settings  # noqa: E402
from local_sync_lyrics.language import LanguageGuess  # noqa: E402
from local_sync_lyrics.lrc_utils import LrcLine, build_lrc, match_lyrics  # noqa: E402
from local_sync_lyrics.transcriber import OperationCancelled, Transcriber  # noqa: E402


# ===================================================================== language
@pytest.mark.parametrize("code,text", [
    ("ja", "君の名前を呼んで 夜空に消えてゆく 星のように輝いて"),
    ("ko", "너를 사랑해 영원히 함께 있을 거야 바람이 불어와"),
    ("zh", "我爱你到永远 月亮代表我的心 你问我爱你有多深"),
    ("th", "ฉันรักเธอมากที่สุดในโลกนี้ตลอดไป"),
    ("he", "אני אוהב אותך יותר מכל דבר בעולם הזה"),
    ("el", "Σ'αγαπώ περισσότερο από οτιδήποτε στον κόσμο"),
    ("en", "I will always love you and I know that you are the one for me so hold me close tonight"),
    ("es", "Yo te quiero mucho y no puedo vivir sin ti mi amor por favor no te vayas de mi lado"),
    ("fr", "Je ne regrette rien de tout ce que j ai fait pour toi mon amour dans la nuit"),
    ("de", "Ich liebe dich und ich will nicht ohne dich sein mit dir ist das Leben schön"),
    ("id", "Aku cinta kamu dan hatiku untuk dirimu yang tidak akan pernah pergi dari sini"),
])
def test_text_language_detection_works_without_any_extra_package(code, text, monkeypatch):
    monkeypatch.setitem(sys.modules, "langdetect", None)          # force the built-in fallback
    guess = language.detect_text_language(text)
    assert guess is not None and guess.code == code and guess.source == "lyrics"


def test_text_language_refuses_to_guess_on_tiny_or_symbolic_input():
    assert language.detect_text_language("") is None
    assert language.detect_text_language("la la la") is None
    assert language.detect_text_language("12345 67890 !!!! ???? ....") is None


def test_language_names_codes_and_guess_formatting():
    assert language.language_name("ja") == "Japanese" and language.language_name("JA") == "Japanese"
    assert language.language_name("xx") == "XX" and language.language_name(None) == "Unknown"
    assert language.iso639_2("ja") == "jpn" and language.iso639_2("en") == "eng"
    assert language.iso639_2("zz") == "XXX"
    assert LanguageGuess("ja", 0.984).describe() == "Japanese (98%)"
    assert len(language.LANGUAGE_NAMES) >= 99


def test_decision_rule_audio_wins_lyrics_break_ties():
    ja, en_lyrics = LanguageGuess("ja", 0.98), LanguageGuess("en", 0.9, "lyrics")
    guess, note = language.decide_language(ja, en_lyrics)            # confident audio beats the text
    assert guess.code == "ja" and "lyrics look English" in note
    guess, note = language.decide_language(LanguageGuess("ja", 0.45), en_lyrics)   # shaky audio loses
    assert guess.code == "en" and guess.source == "lyrics"
    guess, note = language.decide_language(LanguageGuess("en", 0.7), en_lyrics)    # agreement
    assert guess.code == "en" and guess.source == "audio+lyrics" and guess.probability >= 0.7
    assert language.decide_language(LanguageGuess("ja", 0.2), None)[0] is None
    assert language.decide_language(None, None)[0] is None
    assert language.decide_language(None, LanguageGuess("en", 0.9, "lyrics"))[0].code == "en"
    assert language.decide_language(None, LanguageGuess("en", 0.5, "lyrics"))[0] is None


def test_langdetect_is_used_when_installed(monkeypatch):
    fake = types.ModuleType("langdetect")
    fake.DetectorFactory = types.SimpleNamespace(seed=None)
    fake.detect_langs = lambda text: [types.SimpleNamespace(lang="pt", prob=0.93)]
    exc = types.ModuleType("langdetect.lang_detect_exception")
    exc.LangDetectException = type("LangDetectException", (Exception,), {})
    monkeypatch.setitem(sys.modules, "langdetect", fake)
    monkeypatch.setitem(sys.modules, "langdetect.lang_detect_exception", exc)
    guess = language.detect_text_language("uma frase razoavelmente longa para o detector funcionar bem")
    assert (guess.code, guess.probability) == ("pt", 0.93)


# ================================================================= lyric matching
COMMON = "the i you and to a me my it in of is that we love your be on all oh baby for so but just not".split()
SYLL = "ra lo min sha dow bur ning o cean whis per gol den high way thun der pa win sil ver moun tain dan cing".split()


def _vocab(n: int) -> list[str]:
    rng, out = random.Random(3), set()
    while len(out) < n:
        out.add("".join(rng.choice(SYLL) for _ in range(rng.randint(2, 3))))
    return sorted(out)


def _song(seed: int, rare: list[str], lines: int = 32) -> list[str]:
    r = random.Random(seed)
    return [" ".join(r.choice(COMMON) if r.random() < .5 else r.choice(rare) for _ in range(r.randint(5, 9)))
            for _ in range(lines)]


def _hear(lines: list[str], start: int, n: int, err: float, seed: int) -> str:
    r, out = random.Random(seed), []
    for w in " ".join(lines[start:start + n]).split():
        x = r.random()
        if x < err / 2:
            continue
        out.append(w[:-1] + "x" if x < err and len(w) > 3 else (r.choice(COMMON) if x < err else w))
    return " ".join(out)


def test_match_lyrics_accepts_the_right_song_even_when_recognition_is_noisy():
    rare = _vocab(400)
    for seed in range(40):
        song = _song(seed, rare)
        heard = [_hear(song, 2, 4, .30, seed), _hear(song, 12, 4, .30, seed + 1), _hear(song, 22, 4, .30, seed + 2)]
        report = match_lyrics(heard, song)
        assert report.verdict == "match" and report.ratio > 0.6, (seed, report.summary())


def test_match_lyrics_rejects_a_different_song_even_with_a_tiny_shared_vocabulary():
    for rare in (_vocab(33), _vocab(400)):                          # worst case and realistic case
        wrong = 0
        for seed in range(60):
            mine, other = _song(1000 + seed, rare), _song(5000 + seed, rare)
            heard = [_hear(other, 2, 4, .15, seed), _hear(other, 12, 4, .15, seed + 1), _hear(other, 22, 4, .15, seed + 2)]
            wrong += match_lyrics(heard, mine).verdict == "match"
        assert wrong == 0


def test_match_lyrics_lets_partial_lyrics_through_and_handles_other_languages():
    rare = _vocab(400)
    song, other = _song(1, rare), _song(2, rare)
    partial = [_hear(song, 2, 4, .15, 1), _hear(other, 12, 4, .15, 2), _hear(other, 22, 4, .15, 3)]
    assert match_lyrics(partial, song).verdict == "match"           # only one slice truly belongs
    ja = ["君の名前を呼んで", "夜空に消えてゆく", "星のように輝いて", "風に乗って飛んでいく"]
    assert match_lyrics(["君の名前を呼んで 夜空に消えて", "星のように輝いて 風に乗って"], ja).verdict == "match"
    wrong_language = match_lyrics(["I will always love you and I know that you are the one"], ja)
    assert wrong_language.verdict == "mismatch" and wrong_language.percent == 0


def test_match_lyrics_is_inconclusive_when_nothing_was_heard():
    report = match_lyrics(["la la", ""], ["some lyrics here", "and some more of them"])
    assert report.ratio is None and report.verdict == "inconclusive" and "not enough" in report.summary()
    assert match_lyrics([], ["a b c d e f g h"]).verdict == "inconclusive"


# ====================================================================== embedding
LRC = build_lrc([LrcLine(1.5, "Hello darkness"), LrcLine(4.25, "my old friend"), LrcLine(9.0, "日本語の行")],
                artist="A", title="T")


@pytest.fixture
def audio_file(tmp_path):
    def make(ext: str) -> Path:
        path = tmp_path / f"song{ext}"
        if not make_audio(path):
            pytest.skip(f"this PyAV build cannot encode {ext}")
        return path
    return make


def test_mp3_gets_uslt_and_sylt_with_millisecond_timestamps(audio_file):
    from mutagen.id3 import ID3

    path = audio_file(".mp3")
    assert not metadata.has_embedded_lyrics(path)
    result = metadata.embed_lyrics(path, LRC, language="ja")
    assert result.ok and result.fields == ["USLT", "SYLT"]
    tags = ID3(str(path))
    sylt, uslt = tags.getall("SYLT")[0], tags.getall("USLT")[0]
    assert sylt.text == [("Hello darkness", 1500), ("my old friend", 4250), ("日本語の行", 9000)]
    assert (sylt.lang, sylt.format, sylt.type) == ("jpn", 2, 1)
    assert uslt.lang == "jpn" and "[00:" not in str(uslt.text) and "Hello darkness" in str(uslt.text)
    assert metadata.has_embedded_lyrics(path)
    assert metadata.read_embedded_lyrics(path).splitlines()[1] == "[00:04.25]my old friend"


def test_embedding_twice_replaces_instead_of_stacking_and_keeps_id3v23(audio_file):
    from mutagen.id3 import ID3, TIT2

    path = audio_file(".mp3")
    tags = ID3()
    tags.add(TIT2(encoding=1, text="Kept"))
    tags.save(str(path), v2_version=3)
    assert metadata.embed_lyrics(path, LRC).ok
    assert metadata.embed_lyrics(path, LRC.replace("Hello", "Howdy")).ok
    tags = ID3(str(path))
    assert tags.version[:2] == (2, 3)                              # the file's own tag version survives
    assert len(tags.getall("USLT")) == 1 and len(tags.getall("SYLT")) == 1
    assert str(tags["TIT2"]) == "Kept" and "Howdy" in str(tags.getall("USLT")[0].text)


@pytest.mark.parametrize("ext", [".flac", ".opus", ".ogg"])
def test_vorbis_containers_get_a_lyrics_comment_holding_the_lrc(audio_file, ext):
    import mutagen

    path = audio_file(ext)
    result = metadata.embed_lyrics(path, LRC)
    assert result.ok and result.fields == ["LYRICS"]
    stored = mutagen.File(str(path)).tags["LYRICS"][0]
    assert stored.splitlines()[0] == "[00:01.50]Hello darkness" and "[ti:" not in stored   # no header lines
    assert metadata.embed_lyrics(path, LRC).ok and len(mutagen.File(str(path)).tags["LYRICS"]) == 1


def test_m4a_gets_the_lyrics_atom(audio_file):
    from mutagen.mp4 import MP4

    path = audio_file(".m4a")
    assert metadata.embed_lyrics(path, LRC).fields == ["\xa9lyr"]
    assert MP4(str(path)).tags["\xa9lyr"][0].startswith("[00:01.50]Hello darkness")


def test_plain_text_only_is_embedded_unsynced(audio_file):
    from mutagen.id3 import ID3

    path = audio_file(".mp3")
    result = metadata.embed_lyrics(path, None, plain_text="line one\nline two")
    assert result.ok and result.fields == ["USLT"]
    assert not ID3(str(path)).getall("SYLT") and metadata.read_embedded_lyrics(path) == "line one\nline two"


def test_embedding_reports_instead_of_raising(tmp_path, audio_file):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    unsupported = metadata.embed_lyrics(wav, LRC)
    assert not unsupported.ok and not unsupported.supported and ".wav" in unsupported.message
    assert not metadata.supports_embedding(wav) and metadata.supports_embedding(Path("x.M4A"))
    junk = tmp_path / "junk.flac"
    junk.write_bytes(b"this is not audio")
    broken = metadata.embed_lyrics(junk, LRC)
    assert not broken.ok and broken.supported and broken.message
    assert not metadata.embed_lyrics(audio_file(".mp3"), None).ok          # nothing to embed
    assert metadata.read_embedded_lyrics(tmp_path / "missing.mp3") is None


# ================================================================ vocal isolation
def _fake_audio_separator(monkeypatch, *, load_error=None, separate_error=None, calls=None):
    """Install a stand-in for the ``audio_separator`` package that writes a '(Vocals)' wav."""
    calls = calls if calls is not None else {"loads": 0, "separations": 0, "kwargs": None}

    class Separator:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs
            self.output_dir = kwargs["output_dir"]

        def load_model(self, model_filename):
            calls["loads"] += 1
            calls["model"] = model_filename
            if load_error:
                raise load_error

        def separate(self, path):
            calls["separations"] += 1
            if separate_error:
                raise separate_error
            out = Path(self.output_dir) / f"{Path(path).stem}_(Vocals)_UVR-MDX-NET-Inst_HQ_3.wav"
            tp.tone_wav(out, 1.0)
            return [out.name, f"{Path(path).stem}_(Instrumental).wav"]   # names only, one missing

    pkg = types.ModuleType("audio_separator")
    pkg.__spec__ = importlib.machinery.ModuleSpec("audio_separator", None)
    sub = types.ModuleType("audio_separator.separator")
    sub.Separator = Separator
    monkeypatch.setitem(sys.modules, "audio_separator", pkg)
    monkeypatch.setitem(sys.modules, "audio_separator.separator", sub)
    return calls


@pytest.fixture
def isolator_env(tmp_path, monkeypatch):
    monkeypatch.setattr(separation.tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    logs: list[tuple[str, str, str]] = []

    def make(**kw):
        return separation.VocalIsolator(lambda tag, msg, lvl: logs.append((tag, msg, lvl)),
                                        models_root=tmp_path / "models", free_ram=kw.pop("free_ram", lambda: 16.0), **kw)

    return make, logs, tmp_path / "tmp"


def test_isolator_produces_vocals_wav_and_deletes_it_afterwards(isolator_env, monkeypatch, tmp_path):
    make, logs, tmpdir = isolator_env
    calls = _fake_audio_separator(monkeypatch)
    src = tp.tone_wav(tmp_path / "song.wav", 2.0)
    iso = make()
    with iso.vocals(src, duration=120) as stem:
        assert stem is not None and stem.name == "vocals.wav" and stem.is_file()
        assert tmpdir in stem.parents and stem.parent.parent.name.startswith(separation.TEMP_PREFIX)
        assert wave.open(str(stem)).getnframes() > 0
        saved = stem
    assert not saved.exists() and not saved.parent.exists()          # stem + its folder are gone
    assert list((iso._workdir / "out").iterdir()) == []              # separator output wiped too
    assert calls["model"] == separation.MODEL_FILENAME
    assert calls["kwargs"]["output_single_stem"] == "Vocals" and calls["kwargs"]["mdx_params"]["batch_size"] == 1
    with iso.vocals(src) as again:                                   # second track: model is NOT reloaded
        assert again is not None
    assert calls["loads"] == 1 and calls["separations"] == 2
    messages = [m for t, m, _ in logs if t == "Audio: Isolate"]
    assert any("Isolating vocals" in m for m in messages) and any("Vocals isolated" in m for m in messages)
    workdir = iso._workdir
    iso.close()
    assert not workdir.exists()
    assert list(tmpdir.iterdir()) == []                              # nothing left in the temp folder


def test_isolator_cleans_up_even_when_the_caller_crashes(isolator_env, monkeypatch, tmp_path):
    make, _, tmpdir = isolator_env
    _fake_audio_separator(monkeypatch)
    iso = make()
    with pytest.raises(RuntimeError):
        with iso.vocals(tp.tone_wav(tmp_path / "s.wav", 1.0)) as stem:
            assert stem is not None
            raise RuntimeError("decode blew up")
    iso.close()
    assert list(tmpdir.iterdir()) == []


def test_missing_backend_disables_isolation_once_and_keeps_going(isolator_env, monkeypatch, tmp_path):
    make, logs, _ = isolator_env
    monkeypatch.setattr(separation, "available_backend", lambda: None)
    iso = make()
    src = tp.tone_wav(tmp_path / "s.wav", 1.0)
    for _ in range(2):
        with iso.vocals(src) as stem:
            assert stem is None
    warnings = [m for t, m, lvl in logs if lvl == "warning"]
    assert len(warnings) == 1 and "audio-separator[cpu]" in warnings[0] and iso.disabled_reason


def test_model_that_cannot_be_downloaded_switches_isolation_off_for_the_batch(isolator_env, monkeypatch, tmp_path):
    make, logs, _ = isolator_env
    calls = _fake_audio_separator(monkeypatch, load_error=ConnectionError("offline"))
    iso = make()
    src = tp.tone_wav(tmp_path / "s.wav", 1.0)
    for _ in range(3):
        with iso.vocals(src) as stem:
            assert stem is None
    assert calls["loads"] == 1                                       # no retry storm
    text = " ".join(m for _, m, lvl in logs if lvl == "warning")
    assert "could not be loaded" in text and "offline" in text and text.count("unavailable") == 1


def test_low_ram_skips_the_track_but_not_the_batch(isolator_env, monkeypatch, tmp_path):
    make, logs, _ = isolator_env
    calls = _fake_audio_separator(monkeypatch)
    free = {"gb": 0.7}
    iso = make(free_ram=lambda: free["gb"])
    src = tp.tone_wav(tmp_path / "s.wav", 1.0)
    with iso.vocals(src, duration=300) as stem:
        assert stem is None
    assert calls["separations"] == 0 and iso.disabled_reason is None
    assert any("0.7 GB RAM free" in m for _, m, _ in logs)
    free["gb"] = 16.0
    with iso.vocals(src, duration=300) as stem:
        assert stem is not None
    assert separation.required_ram_gb(600) > separation.required_ram_gb(60) > separation.BASE_RAM_GB


def test_runtime_errors_fall_back_per_track_and_repeated_failure_disables(isolator_env, monkeypatch, tmp_path):
    make, logs, _ = isolator_env
    _fake_audio_separator(monkeypatch, separate_error=RuntimeError("onnx exploded"))
    iso = make()
    src = tp.tone_wav(tmp_path / "s.wav", 1.0)
    for _ in range(separation.MAX_TRACK_FAILURES):
        with iso.vocals(src) as stem:
            assert stem is None
    assert iso.disabled_reason and "in a row" in iso.disabled_reason
    assert sum(m.startswith("Vocal isolation failed") for _, m, _ in logs) == separation.MAX_TRACK_FAILURES


def test_out_of_memory_errors_are_reported_as_such(isolator_env, monkeypatch, tmp_path):
    make, logs, _ = isolator_env
    _fake_audio_separator(monkeypatch, separate_error=RuntimeError("std::bad_alloc"))
    iso = make()
    with iso.vocals(tp.tone_wav(tmp_path / "s.wav", 1.0)) as stem:
        assert stem is None
    assert any("Not enough memory" in m for _, m, _ in logs)


def test_stop_button_cancels_isolation(isolator_env, monkeypatch, tmp_path):
    make, _, tmpdir = isolator_env
    _fake_audio_separator(monkeypatch)
    cancel = threading.Event()
    iso = make(cancel=cancel)
    cancel.set()
    with pytest.raises(separation.IsolationCancelled):
        with iso.vocals(tp.tone_wav(tmp_path / "s.wav", 1.0)):
            pass
    iso.close()
    assert list(tmpdir.iterdir()) == []


def test_demucs_is_the_fallback_backend(isolator_env, monkeypatch, tmp_path):
    make, logs, _ = isolator_env
    seen = {}

    def demucs_main(args):
        seen["args"] = args
        out = Path(args[args.index("-o") + 1]) / "mdx_extra_q" / "song"
        out.mkdir(parents=True)
        tp.tone_wav(out / "vocals.wav", 1.0)

    pkg = types.ModuleType("demucs")
    pkg.__spec__ = importlib.machinery.ModuleSpec("demucs", None)
    sub = types.ModuleType("demucs.separate")
    sub.main = demucs_main
    monkeypatch.setitem(sys.modules, "demucs", pkg)
    monkeypatch.setitem(sys.modules, "demucs.separate", sub)
    monkeypatch.setitem(sys.modules, "audio_separator", None)       # not installed
    assert separation.available_backend() == "demucs" and "mdx_extra_q" in separation.backend_summary()
    iso = make()
    with iso.vocals(tp.tone_wav(tmp_path / "song.wav", 1.0)) as stem:
        assert stem is not None and stem.name == "vocals.wav"
    assert seen["args"][:4] == ["-n", "mdx_extra_q", "--two-stems", "vocals"]

    sub.main = lambda args: (_ for _ in ()).throw(SystemExit("boom"))
    with iso.vocals(tp.tone_wav(tmp_path / "song2.wav", 1.0)) as stem:
        assert stem is None                                          # SystemExit is contained


def test_stale_temp_folders_from_a_crash_are_swept(monkeypatch, tmp_path):
    import os

    monkeypatch.setattr(separation.tempfile, "tempdir", str(tmp_path))
    old, fresh, other = (tmp_path / f"{separation.TEMP_PREFIX}old", tmp_path / f"{separation.TEMP_PREFIX}new",
                         tmp_path / "unrelated")
    for d in (old, fresh, other):
        d.mkdir()
    long_ago = time.time() - 24 * 3600
    os.utime(old, (long_ago, long_ago))
    assert separation.sweep_stale_temp_dirs() == 1
    assert not old.exists() and fresh.exists() and other.exists()


def test_backend_summary_mentions_the_install_command_when_nothing_is_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "audio_separator", None)
    monkeypatch.setitem(sys.modules, "demucs", None)
    assert separation.available_backend() is None
    assert "audio-separator[cpu]" in separation.backend_summary()


# ====================================================== sessions: isolation + language
class FakeIsolator:
    def __init__(self, tmp_path, *, stem=True, error=None):
        self.tmp_path, self.stem, self.error, self.stem_path, self.entered = tmp_path, stem, error, None, 0

    @contextmanager
    def vocals(self, source, *, duration=None):
        self.entered += 1
        if self.error:
            raise self.error
        if not self.stem:
            yield None
            return
        self.stem_path = tp.tone_wav(self.tmp_path / "vocals.wav", 6.0)
        try:
            yield self.stem_path
        finally:
            self.stem_path.unlink(missing_ok=True)


def isolating_session(tmp_path, monkeypatch, isolator, **settings):
    tp.fake_asr(monkeypatch, [])
    tr, session = tp.guided_session(tmp_path, 30.0, Settings(isolate_vocals=True, **settings))
    tr._isolator = isolator
    return tr, session


def test_session_decodes_the_isolated_stem_and_the_stem_is_deleted(tmp_path, monkeypatch):
    iso = FakeIsolator(tmp_path)
    tr, session = isolating_session(tmp_path, monkeypatch, iso)
    audio = session.audio
    assert session.vocals_isolated and iso.entered == 1
    assert len(audio) == pytest.approx(6.0 * transcriber.SAMPLE_RATE, abs=800)     # the 6 s stem, not the 30 s mix
    assert not iso.stem_path.exists()
    assert session.audio is audio and iso.entered == 1                              # decoded once


def test_session_falls_back_to_the_original_mix(tmp_path, monkeypatch):
    iso = FakeIsolator(tmp_path, stem=False)
    tr, session = isolating_session(tmp_path, monkeypatch, iso)
    assert len(session.audio) == pytest.approx(30.0 * transcriber.SAMPLE_RATE, abs=800)
    assert not session.vocals_isolated


def test_session_ignores_isolation_when_the_switch_is_off(tmp_path, monkeypatch):
    iso = FakeIsolator(tmp_path)
    tp.fake_asr(monkeypatch, [])
    tr, session = tp.guided_session(tmp_path, 10.0, Settings(isolate_vocals=False))
    tr._isolator = iso
    assert len(session.audio) > 0 and iso.entered == 0 and not session.vocals_isolated


def test_stop_during_isolation_becomes_operation_cancelled(tmp_path, monkeypatch):
    tr, session = isolating_session(tmp_path, monkeypatch, FakeIsolator(tmp_path, error=separation.IsolationCancelled()))
    with pytest.raises(OperationCancelled):
        _ = session.audio


def test_silent_vocal_stem_falls_back_to_the_original_audio(tmp_path, monkeypatch):
    class SilentIsolator(FakeIsolator):
        @contextmanager
        def vocals(self, source, *, duration=None):
            path = tmp_path / "vocals.wav"
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
                w.writeframes(np.zeros(16000 * 3, dtype="<i2").tobytes())
            try:
                yield path
            finally:
                path.unlink(missing_ok=True)

    logs = []
    tp.fake_asr(monkeypatch, [])
    tr, session = tp.guided_session(tmp_path, 12.0, Settings(isolate_vocals=True))
    tr._isolator = SilentIsolator(tmp_path)
    tr._log = lambda tag, msg, lvl="info": logs.append((tag, msg, lvl))
    assert len(session.audio) == pytest.approx(12.0 * transcriber.SAMPLE_RATE, abs=800)
    assert any("unusable" in m for _, m, _ in logs)


def lang_session(tmp_path, monkeypatch, detect, *, seconds=120.0, **settings):
    """Session whose Whisper language ID is scripted: ``detect(chunk)`` -> (code, prob, probs)."""
    tp.fake_asr(monkeypatch, [])
    monkeypatch.setattr(Transcriber, "detect_language", lambda self, chunk: detect(chunk))
    logs, heard = [], []
    tr, session = tp.guided_session(tmp_path, seconds, Settings(**settings))
    tr._log = lambda tag, msg, lvl="info": logs.append((tag, msg, lvl))
    tr.on_language = heard.append
    return tr, session, logs, heard


def test_language_is_detected_once_announced_and_locked(tmp_path, monkeypatch):
    chunks = []

    def detect(chunk):
        chunks.append(len(chunk))
        return "ja", 0.98, {"ja": 0.98, "en": 0.01}

    tr, session, logs, heard = lang_session(tmp_path, monkeypatch, detect)
    guess = session.lock_language()
    assert (guess.code, round(guess.probability, 2), guess.source) == ("ja", 0.98, "audio")
    assert session._language() == "ja" and len(heard) == 1 and heard[0].name == "Japanese"
    assert chunks == [30 * transcriber.SAMPLE_RATE]                                  # ONE 30 s sample was enough
    line = [m for t, m, _ in logs if t == "AI: Language"][0]
    assert "Japanese (98%)" in line and "locked" in line
    assert session.lock_language() is guess and len(chunks) == 1 and len(heard) == 1     # idempotent


def test_unsure_first_sample_triggers_a_second_one_and_votes_are_pooled(tmp_path, monkeypatch):
    answers = iter([("en", 0.40, {"en": 0.40, "ko": 0.35, "ja": 0.25}), ("ko", 0.80, {"ko": 0.80, "en": 0.10})])
    tr, session, _, _ = lang_session(tmp_path, monkeypatch, lambda c: next(answers))
    guess = session.lock_language()
    assert guess.code == "ko" and 0.5 < guess.probability < 0.7                     # (0.35 + 0.80) / 2


def test_sample_is_taken_mid_song_not_from_the_intro(tmp_path, monkeypatch):
    tp.fake_asr(monkeypatch, [])
    tr, session = tp.guided_session(tmp_path, 200.0)
    seen = []
    monkeypatch.setattr(Transcriber, "detect_language", lambda self, chunk: (seen.append(chunk.copy()) or ("en", .9, {"en": .9})))
    session.audio[:] = 0.0
    session.audio[int(60 * 16000):int(90 * 16000)] = 1.0                           # 30%-45% of 200 s
    session.lock_language()
    assert len(seen) == 1 and float(seen[0].mean()) == pytest.approx(1.0)


def test_explicit_language_setting_wins_and_skips_detection(tmp_path, monkeypatch):
    def boom(chunk):
        raise AssertionError("must not run")

    tr, session, logs, heard = lang_session(tmp_path, monkeypatch, boom, language="es")
    guess = session.lock_language("some english lyrics that should be ignored here ok")
    assert guess.code == "es" and guess.source == "settings" and session._language() == "es"
    assert any("set in Settings" in m for _, m, _ in logs) and heard[0].code == "es"


def test_language_detection_can_be_switched_off(tmp_path, monkeypatch):
    tr, session, _, heard = lang_session(tmp_path, monkeypatch, lambda c: 1 / 0, detect_language=False)
    assert session.lock_language() is None and session._language() is None and not heard


def test_mashups_are_never_locked_to_one_language(tmp_path, monkeypatch):
    tr, session, logs, _ = lang_session(tmp_path, monkeypatch, lambda c: 1 / 0)
    assert session.lock_language(multi=True) is None
    assert any("not locked" in m for _, m, _ in logs)


def test_detection_failure_is_never_fatal(tmp_path, monkeypatch):
    tr, session, logs, heard = lang_session(tmp_path, monkeypatch, lambda c: (_ for _ in ()).throw(RuntimeError("cuda!")))
    assert session.lock_language() is None and not heard
    assert any("Could not detect the language" in m for _, m, _ in logs)


def test_uncertain_audio_defers_to_the_lyrics_language(tmp_path, monkeypatch):
    tr, session, logs, _ = lang_session(tmp_path, monkeypatch, lambda c: ("ja", 0.42, {"ja": 0.42, "ko": 0.4}))
    guess = session.lock_language("너를 사랑해 영원히 함께 있을 거야 바람이 불어와")
    assert guess.code == "ko" and guess.source == "lyrics"
    assert any("shaky" in m for _, m, _ in logs)


def test_locked_language_reaches_every_transcribe_call(tmp_path, monkeypatch):
    """The point of the feature: Whisper is never left to guess (or flip) the language."""
    rng = random.Random(5)
    song = tp.make_song(14, 21)
    words, truth, _ = tp.sing(song, 3.0)
    seen = []

    def recognize(self, chunk, offset, *, prompt, vad, language, **_):
        seen.append(language)
        end = offset + len(chunk) / transcriber.SAMPLE_RATE
        heard = [w for w in tp.mishear(words, rng) if offset <= w.start and w.end <= end + 0.01]
        return transcriber._Window(tp.split_phrases(heard, max_gap=0.5), "xx", 0.4)   # a window that mis-detects

    monkeypatch.setattr(Transcriber, "_recognize", recognize)
    monkeypatch.setattr(Transcriber, "_load", lambda self: setattr(self, "model_name", "fake") or setattr(self, "device", "cpu"))
    monkeypatch.setattr(Transcriber, "detect_language", lambda self, chunk: ("ja", 0.97, {"ja": 0.97}))
    tr, session = tp.guided_session(tmp_path, 90.0)
    result = session.align(tp.reference_of(song))
    assert result.ok and seen and set(seen) == {"ja"}                                 # never None, never 'xx'
    assert session.language_info.code == "ja"


def test_forced_sync_threshold_override(tmp_path, monkeypatch):
    """``align(min_match=0)`` is what 'Force Sync Anyway' uses: wrong lyrics are then accepted."""
    rng = random.Random(1)
    sung, other = tp.make_song(14, 5), tp.make_song(14, 6)
    words, _, _ = tp.sing(sung, 3.0)
    tp.fake_asr(monkeypatch, tp.mishear(words, rng))
    monkeypatch.setattr(Transcriber, "detect_language", lambda self, chunk: ("en", .9, {"en": .9}))
    tr, session = tp.guided_session(tmp_path, 90.0)
    assert not session.align(tp.reference_of(other)).ok
    tr2, session2 = tp.guided_session(tmp_path, 90.0)
    forced = session2.align(tp.reference_of(other), min_match=0.0)
    assert forced.ok or forced.alignment.emitted_lines == 0                           # accepted unless nothing was timed


# ================================================ Whisper's detect_language contract
def test_installed_faster_whisper_still_has_the_detect_language_signature_we_call():
    """We call ``detect_language(audio=, vad_filter=, vad_parameters=, language_detection_segments=)``."""
    import inspect

    from faster_whisper import WhisperModel

    params = inspect.signature(WhisperModel.detect_language).parameters
    assert {"audio", "vad_filter", "vad_parameters", "language_detection_segments"} <= set(params)


class FakeWhisper:
    def __init__(self, *, fail_vad=False, error=None):
        self.calls, self.fail_vad, self.error = [], fail_vad, error

    def detect_language(self, audio=None, vad_filter=False, vad_parameters=None, language_detection_segments=1,
                        language_detection_threshold=0.5):
        self.calls.append({"vad": vad_filter, "segments": language_detection_segments, "n": len(audio)})
        if self.error:
            raise self.error
        if vad_filter and self.fail_vad:
            raise ValueError("no speech chunks")           # what faster-whisper does on a speech-free sample
        return "ja", 0.91, [("ja", 0.91), ("ko", 0.06), ("en", 0.03)]


def whisper_transcriber(model):
    tr = Transcriber(Settings(), profile=tp.HardwareProfile(physical_cores=4, ram_total_gb=16))
    tr._model, tr.device, tr.model_name = model, "cpu", "fake"
    tr._load = lambda: None
    return tr


def test_transcriber_detect_language_calls_whisper_with_vad_and_one_segment():
    model = FakeWhisper()
    code, prob, probs = whisper_transcriber(model).detect_language(np.zeros(16000 * 30, dtype="float32"))
    assert (code, prob) == ("ja", 0.91) and probs == {"ja": 0.91, "ko": 0.06, "en": 0.03}
    assert model.calls == [{"vad": True, "segments": 1, "n": 480000}]


def test_transcriber_detect_language_retries_without_vad_when_the_sample_has_no_speech():
    model = FakeWhisper(fail_vad=True)
    assert whisper_transcriber(model).detect_language(np.zeros(16000 * 30, dtype="float32"))[0] == "ja"
    assert [c["vad"] for c in model.calls] == [True, False]


def test_transcriber_detect_language_without_a_model_is_a_clean_error():
    tr = whisper_transcriber(None)
    with pytest.raises(transcriber.TranscriptionError):
        tr.detect_language(np.zeros(16000, dtype="float32"))


def test_real_isolator_and_real_session_end_to_end(tmp_path, monkeypatch):
    """audio-separator stand-in -> VocalIsolator -> AudioSession.audio -> temp files gone."""
    monkeypatch.setattr(separation.tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    _fake_audio_separator(monkeypatch)
    logs = []
    tp.fake_asr(monkeypatch, [])
    tr, session = tp.guided_session(tmp_path, 20.0, Settings(isolate_vocals=True))
    tr._log = lambda tag, msg, lvl="info": logs.append((tag, msg))
    tr._isolator = separation.VocalIsolator(tr._log, tr._cancel, models_root=tmp_path / "models",
                                            free_ram=lambda: 32.0)
    audio = session.audio
    assert session.vocals_isolated and len(audio) == pytest.approx(1.0 * transcriber.SAMPLE_RATE, abs=800)
    tags = [t for t, _ in logs]
    assert "Audio: Isolate" in tags and any("Decoding the isolated vocals" in m for _, m in logs)
    tr.close()                                                       # closes the isolator and its workdir
    assert list((tmp_path / "tmp").iterdir()) == []
