"""Offline tests for the Smart Search-First & Text-Guided Forced Alignment pipeline."""

from __future__ import annotations

import base64
import json
import random
import statistics
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_sync_lyrics import api_providers as ap  # noqa: E402
from local_sync_lyrics import lrc_utils, text_utils, transcriber  # noqa: E402
from local_sync_lyrics.config import Settings  # noqa: E402
from local_sync_lyrics.database import LyricsDatabase  # noqa: E402
from local_sync_lyrics.hardware_profiler import HardwareProfile  # noqa: E402
from local_sync_lyrics.lrc_utils import (  # noqa: E402
    AlignmentResult, ForcedAligner, LrcLine, RecWord, align_lines_to_words, build_lrc, split_phrases,
)
from local_sync_lyrics.models import (  # noqa: E402
    IdentifiedSong, LyricsResult, MatchKind, ReferenceComponent, ReferenceScript, Snippet,
    Status, TrackInfo,
)
from local_sync_lyrics.processor import BatchProcessor  # noqa: E402
from local_sync_lyrics.transcriber import GuidedResult, Transcriber, _Window, plan_windows  # noqa: E402

LYRICS = "\n".join([
    "The river runs beneath the silver moon tonight",
    "I keep on walking through the quiet empty streets",
    "Every window tells me stories I forgot",
    "Nothing here is waiting but the wind and me",
    "Carry me home carry me home",
    "Back to the place where the lanterns glow",
])


def lrc_of(n: int = 9, start: int = 10) -> str:
    return "\n".join(f"[{(start + 12 * i) // 60:02d}:{(start + 12 * i) % 60:02d}.00]words in line {i}"
                     for i in range(n)) + "\n"


def tone_wav(path: Path, seconds: float, rate: int = 16000) -> Path:
    t = np.arange(int(seconds * rate))
    data = (8000 * np.sin(t / 20.0)).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
        w.writeframes(data.tobytes())
    return path


# ============================================================ mashup detection
@pytest.mark.parametrize("title,names,conf_min", [
    ("Understand x Pluto Projector", ["Understand", "Pluto Projector"], 0.75),
    ("Song A vs Song B", ["Song A", "Song B"], 0.75),
    ("Song A & Song B (Mashup)", ["Song A", "Song B"], 0.85),
    ("Song A x Song B x Song C", ["Song A", "Song B", "Song C"], 0.75),
    ("Mashup of Hello and Rolling in the Deep", ["Hello", "Rolling in the Deep"], 0.85),
    ("Song A / Song B", ["Song A", "Song B"], 0.50),
    ("Song A + Song B [Bootleg]", ["Song A", "Song B"], 0.85),
    ("Dark Horse (feat. Juicy J) x Roar", ["Dark Horse (feat. Juicy J)", "Roar"], 0.75),
])
def test_mashup_titles_are_split(title, names, conf_min):
    info = text_utils.detect_mashup(title)
    assert info.is_mashup and [t for _, t in info.components] == names
    assert info.confidence >= conf_min


@pytest.mark.parametrize("title", [
    "Simon & Garfunkel", "Thunderstruck", "Don't Stop Believin'", "Lil Nas X",
    "Rock & Roll Part 2", "Montero (Call Me By Your Name)",
])
def test_ordinary_titles_are_not_mashups(title):
    assert not text_utils.detect_mashup(title).is_mashup


def test_mashup_with_artists_and_filename_fallback():
    info = text_utils.detect_mashup("Artist A - Song A x Artist B - Song B")
    assert info.components == (("Artist A", "Song A"), ("Artist B", "Song B"))
    # tag title is truncated, the file name keeps the full text
    info = text_utils.detect_mashup("Understand", filename_stem="DJ Foo - Understand x Pluto Projector (Mashup)")
    assert info.has_named_components and info.components[1][1] == "Pluto Projector"
    # keyword only: it IS a mashup but we do not know of what
    kw = text_utils.detect_mashup("Summer Mashup 2024")
    assert kw.is_mashup and not kw.has_named_components


def test_snippet_helpers():
    phrase = text_utils.best_query_phrase("la la\nI wanna be your lover tonight\nyeah yeah")
    assert phrase.startswith("i wanna be your lover")
    assert text_utils.snippet_match_score("wanna be your lover tonight", "x\nI just wanna be your lover tonight\ny" * 4) >= 90
    assert text_utils.snippet_match_score("completely different words here", LYRICS) < 60
    assert text_utils.fold("がぎ") == "がぎ"          # kana voiced marks survive folding


# ================================================================ providers
class FakeHttp:
    def __init__(self, routes=None, post_routes=None):
        self.routes, self.post_routes, self.calls = routes or {}, post_routes or {}, []

    @staticmethod
    def _pick(table, url):
        for key, value in table.items():
            if key in url:
                if isinstance(value, Exception):
                    raise value
                return value
        return None

    def get_json(self, url, params=None, headers=None):
        self.calls.append(("GET", url))
        return self._pick(self.routes, url)

    def get(self, url, params=None, headers=None):
        self.calls.append(("GET", url))
        value = self._pick(self.routes, url)
        return None if value is None else SimpleNamespace(
            text=value if isinstance(value, str) else json.dumps(value),
            json=lambda: value if not isinstance(value, str) else json.loads(value))

    def post(self, url, data=None, headers=None):
        self.calls.append(("POST", url))
        value = self._pick(self.post_routes, url)
        return None if value is None else SimpleNamespace(text=value)

    def close(self):
        pass


def q(artist="Pluto Artist", title="Understand", duration=None):
    return ap.TrackQuery(text_utils.query_variants(artist, title), duration=duration)


def test_rclyricsband_scrapes_form_search_and_lrc_block():
    search = '<a class="song_search" href="/song/understand-pluto">Understand - Pluto Artist</a>'
    page = '<div><p class="lrc_text_format">' + "<br>".join(
        f"[00:{10 + 7 * i:02d}.00]rc line {i}" for i in range(8)) + "</p></div>"
    http = FakeHttp({"/song/understand": page}, {"rclyricsband.com": search})
    hit = ap.RcLyricsBandProvider(http, Settings()).search(q())
    assert hit and hit.kind is MatchKind.SYNCED and "rc line 3" in hit.synced_lrc
    assert ("POST", "https://rclyricsband.com/") in http.calls


def test_netease_strips_credit_lines():
    lyric = "[00:00.000] 作词 : Someone\n[00:01.000] 作曲 : Other\n" + "\n".join(
        f"[00:{10 + 6 * i:02d}.500] ne line {i}" for i in range(8))
    http = FakeHttp({
        "api/search/get/web": {"result": {"songs": [{"id": 7, "name": "Understand",
                               "artists": [{"name": "Pluto Artist"}], "duration": 200000}]}},
        "api/song/lyric": {"lrc": {"lyric": lyric}},
    })
    hit = ap.NetEaseProvider(http, Settings()).search(q(duration=200))
    assert hit and "作词" not in hit.synced_lrc and "ne line 0" in hit.synced_lrc


def test_qqmusic_decodes_base64_lyrics():
    lrc = "\n".join(f"[00:{10 + 6 * i:02d}.00]qq line {i}" for i in range(8))
    http = FakeHttp({
        "client_search_cp": {"data": {"song": {"list": [{"songmid": "m1", "songname": "Understand",
                             "singer": [{"name": "Pluto Artist"}], "interval": 200}]}}},
        "fcg_query_lyric_new": {"lyric": base64.b64encode(lrc.encode()).decode()},
    })
    hit = ap.QQMusicProvider(http, Settings()).search(q(duration=200))
    assert hit and "qq line 7" in hit.synced_lrc


def test_jsonp_wrapper_is_parsed():
    def bad_json():
        raise ValueError
    resp = SimpleNamespace(json=bad_json, text='MusicJsonCallback({"a": [1, 2]})')
    assert ap._json_of(resp) == {"a": [1, 2]}


GENIUS_PAGE = ('<div data-lyrics-container="true">[Verse 1]<br/>' + "<br/>".join(LYRICS.split("\n"))
               + '<div data-exclude-from-selection="true">12 Contributors</div></div>')


def test_genius_public_search_skips_translations_and_scrapes_text():
    http = FakeHttp({
        "api/search/multi": {"response": {"sections": [{"type": "song", "hits": [
            {"result": {"title": "Understand (Traducción al Español)", "url": "https://genius.com/tr",
                        "primary_artist": {"name": "Genius Traducciones"}}},
            {"result": {"title": "Understand", "url": "https://genius.com/pluto-understand-lyrics",
                        "primary_artist": {"name": "Pluto Artist"}}}]}]}},
        "pluto-understand-lyrics": GENIUS_PAGE,
    })
    provider = ap.GeniusProvider(http, Settings())
    hit = provider.search(q())
    assert hit and hit.kind is MatchKind.PLAIN_TEXT and hit.tier == 2
    assert "[Verse 1]" not in hit.plain and "Contributors" not in hit.plain
    assert ("GET", "https://genius.com/tr") not in http.calls
    assert provider.identify("silver moon tonight")[0].title == "Understand"


def test_genius_uses_official_api_when_token_set():
    http = FakeHttp({"api.genius.com/search": {"response": {"hits": [
        {"result": {"title": "Understand", "url": "https://genius.com/x-lyrics",
                    "primary_artist": {"name": "Pluto Artist"}}}]}}, "x-lyrics": GENIUS_PAGE})
    assert ap.GeniusProvider(http, Settings(genius_token="tok")).search(q()) is not None
    assert any("api.genius.com" in url for _, url in http.calls)


def test_lyricsovh_direct_and_suggest_paths():
    text = "Paroles de la chanson Understand par Pluto Artist\r\n" + LYRICS.replace("\n", "\r\n")
    direct = FakeHttp({"/v1/Pluto%20Artist/Understand": {"lyrics": text}})
    hit = ap.LyricsOvhProvider(direct, Settings()).search(q())
    assert hit and not hit.plain.lower().startswith("paroles") and hit.kind is MatchKind.PLAIN_TEXT

    suggest = FakeHttp({
        "/suggest/": {"data": [{"title": "Understand", "artist": {"name": "Pluto Artist"}, "duration": 200}]},
        "/v1/Pluto%20Artist/Understand": {"lyrics": LYRICS},
    })
    # artist unknown in the file name -> only the suggestion index can supply it
    hit = ap.LyricsOvhProvider(suggest, Settings()).search(q(artist="", title="Understand"))
    assert hit and hit.artist == "Pluto Artist"


def test_musixmatch_needs_key_and_marks_excerpts_partial():
    assert not ap.MusixmatchProvider(FakeHttp(), Settings()).available()
    settings = Settings(musixmatch_key="k")
    body = LYRICS + "\n\n...\n\n******* This Lyrics is NOT for Commercial use *******"
    http = FakeHttp({
        "matcher.lyrics.get": {"message": {"header": {"status_code": 200}, "body": {"lyrics": {"lyrics_body": body}}}},
        "track.search": {"message": {"header": {"status_code": 200}, "body": {"track_list": [
            {"track": {"track_name": "Understand", "artist_name": "Pluto Artist"}}]}}},
    })
    provider = ap.MusixmatchProvider(http, settings)
    hit = provider.search(q())
    assert provider.available() and hit and hit.partial and "NOT for Commercial" not in hit.plain
    assert provider.identify("silver moon")[0].artist == "Pluto Artist"
    bad = ap.MusixmatchProvider(FakeHttp({"matcher": {"message": {"header": {"status_code": 401}}}}), settings)
    with pytest.raises(ap.ProviderError):
        bad.search(q())


class StubProvider(ap.LyricsProvider):
    def __init__(self, label, tier, result=None, boom=None, ident=None):
        self.label, self.tier, self._result, self._boom, self._ident = label, tier, result, boom, ident or []
        self.calls = 0

    def search(self, query):
        self.calls += 1
        if self._boom:
            raise self._boom
        return self._result

    def identify(self, snippet):
        return self._ident


def make_fetcher(providers, log=None):
    fetcher = ap.LyricsFetcher(Settings(), log=log)
    fetcher.providers = providers
    return fetcher


def test_chain_tier1_miss_then_plain_text_and_log_tags():
    logs = []
    full = LyricsResult("Genius", plain=LYRICS, score=90, tier=2)
    t1, t2a, t2b = StubProvider("LRCLIB", 1), StubProvider("Genius", 2, full), StubProvider("Lyrics.ovh", 2, full)
    fetcher = make_fetcher([t1, t2a, t2b], log=lambda *a: logs.append(a))
    outcome = fetcher.fetch(TrackInfo(Path("x.mp3"), "Pluto Artist", "Understand"))
    assert outcome.kind is MatchKind.PLAIN_TEXT and outcome.synced is None
    assert t2b.calls == 0                                   # a complete text ends Tier 2
    tags = {t for t, _, _ in logs}
    assert {"API: Tier 1", "API: Plain Text"} <= tags
    fetcher.close()


def test_chain_prefers_complete_text_over_excerpt_and_survives_crashes():
    excerpt = LyricsResult("Musixmatch", plain=LYRICS, score=75, tier=2, partial=True)
    full = LyricsResult("Lyrics.ovh", plain=LYRICS, score=85, tier=2)
    fetcher = make_fetcher([
        StubProvider("Boom", 1, boom=ValueError("parser bug")),
        StubProvider("Down", 1, boom=ap.ProviderError("HTTP 503")),
        StubProvider("Musixmatch", 2, excerpt), StubProvider("Lyrics.ovh", 2, full)])
    outcome = fetcher.fetch(TrackInfo(Path("x.mp3"), "A", "B"))
    assert outcome.plain.provider == "Lyrics.ovh" and not outcome.plain.partial
    assert set(outcome.errors) == {"Boom", "Down"}
    fetcher.close()


def test_chain_tier1_synced_stops_before_tier2_and_fetch_for_ignores_duration():
    synced = LyricsResult("LRCLIB", synced_lrc=lrc_of(), score=95)
    t2 = StubProvider("Genius", 2, LyricsResult("Genius", plain=LYRICS, tier=2))
    fetcher = make_fetcher([StubProvider("LRCLIB", 1, synced), t2])
    out = fetcher.fetch(TrackInfo(Path("x.mp3"), "A", "B", duration=130))
    assert out.kind is MatchKind.SYNCED and t2.calls == 0
    comp = fetcher.fetch_for("A", "B")                       # mashup component: no duration check
    assert comp.reference_text and "words in line 0" in comp.reference_text
    fetcher.close()


def test_identify_by_lyrics_merges_providers_and_dedupes():
    a = StubProvider("Genius", 2, ident=[IdentifiedSong("X", "One", "Genius"), IdentifiedSong("Y", "Two", "Genius")])
    b = StubProvider("Musixmatch", 2, ident=[IdentifiedSong("x", "one", "Musixmatch")])
    found = make_fetcher([a, b]).identify_by_lyrics("some sung words")
    assert [(s.artist, s.title) for s in found] == [("X", "One"), ("Y", "Two")]


def test_new_providers_are_in_the_default_chain_in_tier_order():
    from local_sync_lyrics.config import PROVIDER_IDS
    assert list(ap.PROVIDER_CLASSES) == list(PROVIDER_IDS)
    tiers = [c.tier for c in ap.PROVIDER_CLASSES.values()]
    assert tiers == sorted(tiers) and tiers.count(1) == 6 and tiers.count(2) == 3


# ============================================================ plain-text clean-up
def test_clean_plain_lyrics_and_validation():
    dirty = ("12 Contributors\nUnderstand Lyrics\n[Verse 1]\nFirst real line here (x2)\n"
             "Second real line here\n\n[Chorus]\n(ooh, ooh)\nThird real line goes here\n"
             "Fourth line in the song ends\nYou might also like\n123Embed")
    clean = lrc_utils.clean_plain_lyrics(dirty)
    assert "[Verse 1]" not in clean and "Contributors" not in clean and "Embed" not in clean
    assert "(x2)" not in clean and "(ooh, ooh)" in clean
    assert lrc_utils.looks_like_lyrics(clean)
    assert not lrc_utils.looks_like_lyrics("Sorry, page not found")
    assert not lrc_utils.looks_like_lyrics("<html><div>" * 30)


# ====================================================================== aligner
VOCAB = ("river moon silver night street window story forgot wind home lantern glow morning hill golden "
         "light road promise leaf heart fire dark neon rain dawn static lightning vein lose gain run "
         "carry place wander ocean thunder whisper shadow ember velvet crystal harbor meadow compass "
         "feather lantern marble anchor orchard violet").split()


def make_song(n_lines: int, seed: int, words_per_line: int = 6) -> list[str]:
    rng = random.Random(seed)
    return [" ".join(rng.choice(VOCAB) for _ in range(words_per_line)) for _ in range(n_lines)]


def sing(lines, t0, wdur=0.38, gap=0.9):
    t, words, truth = t0, [], []
    for line in lines:
        truth.append((line, t))
        for w in line.split():
            words.append(RecWord(w, t, t + wdur, 0.9))
            t += wdur + 0.04
        t += gap
    return words, truth, t


def mishear(words, rng, p_sub=0.12, p_drop=0.06, p_ins=0.03):
    out = []
    for w in words:
        r = rng.random()
        if r < p_drop:
            continue
        text = (w.text[:-1] + "x" if len(w.text) > 3 else w.text + "h") if r < p_drop + p_sub else w.text
        out.append(RecWord(text, w.start, w.end, 0.8))
        if rng.random() < p_ins:
            out.append(RecWord("thanks", w.end, w.end + 0.2, 0.3))
    return out


def accuracy(result, truth, tol=0.7):
    got: dict[str, list[float]] = {}
    for line in result.lines:
        got.setdefault(line.text, []).append(line.time)
    return sum(any(abs(t - x) <= tol for x in got.get(text, [])) for text, t in truth) / len(truth)


def test_aligner_snaps_official_spelling_and_timing_under_noise():
    scores = []
    for seed in range(12):
        lines = make_song(16, seed)
        words, truth, _ = sing(lines, 5.0)
        result = align_lines_to_words(lines, mishear(words, random.Random(seed)))
        scores.append(accuracy(result, truth))
        assert result.ok()
        assert {l.text for l in result.lines} <= set(lines)       # only OFFICIAL text is ever emitted
    assert statistics.mean(scores) >= 0.95, scores


def test_aligner_handles_chorus_repeated_but_written_once():
    verse, chorus = make_song(4, 1), make_song(2, 2)
    reference = verse + chorus + make_song(2, 3)
    audio_order = verse + chorus + reference[6:8] + chorus + chorus
    scores = []
    for seed in range(8):
        words, truth, _ = sing(audio_order, 3.0)
        result = align_lines_to_words(reference, mishear(words, random.Random(seed)))
        scores.append(accuracy(result, truth))
    assert statistics.mean(scores) >= 0.88, scores


def test_aligner_locates_mashup_transitions_between_two_songs():
    a, b = make_song(12, 11), make_song(8, 12)
    scores, switches = [], []
    for seed in range(8):
        t, words, truth = 2.0, [], []
        for part in (a[0:4], b[2:6], a[4:8], b[6:8]):
            w, tr, t = sing(part, t)
            words += w
            truth += tr
        aligner = ForcedAligner([a, b])
        aligner.feed(split_phrases(mishear(words, random.Random(seed))))
        result = aligner.finalize()
        scores.append(accuracy(result, truth))
        switches.append(result.switches)
    assert statistics.mean(scores) >= 0.92, scores
    assert min(switches) >= 2                                   # A->B->A->B transitions found


def test_aligner_does_not_invent_lines_for_cut_sections_or_wrong_lyrics():
    song = make_song(12, 21)
    words, truth, t = sing(song[0:4], 1.0)
    more, truth2, _ = sing(song[8:12], t)                       # lines 4-7 are not in the audio
    for seed in range(8):
        result = align_lines_to_words(song, mishear(words + more, random.Random(seed)))
        assert not [l for l in result.lines if l.text in song[4:8]]
    other, _, _ = sing(make_song(10, 99), 2.0)
    for seed in range(8):
        wrong = align_lines_to_words(song, mishear(other, random.Random(seed)))
        assert not wrong.ok()


def test_prompt_hint_follows_position_and_respects_budget():
    a, b = make_song(10, 31), make_song(10, 32)
    aligner = ForcedAligner([a, b])
    count = lambda text: len(text.split()) * 2
    first = aligner.prompt_hint(60, count)
    assert first.startswith(a[0]) and b[0] in first            # primes BOTH songs at the start
    words, _, _ = sing(a[0:5], 1.0)
    aligner.feed(split_phrases(words))
    later = aligner.prompt_hint(60, count)
    assert later.startswith(a[5]) and a[0] not in later         # now: the lines expected NEXT
    assert count(later) <= 60 + count(a[5])                     # budget (one line of slack)


def test_drop_credit_lines():
    lines = lrc_utils.parse_lrc("[00:00.00]作词 : A\n[00:01.00]Composed by: B\n[00:10.00]real one\n[00:20.00]作词 is a word")
    kept = lrc_utils.drop_credit_lines(lines)
    assert [l.text for l in kept] == ["real one", "作词 is a word"]


# ================================================= guided transcription (fake ASR)
def fake_asr(monkeypatch, schedule, *, drop=None, calls=None, language="en"):
    """Replace Whisper with a recogniser that 'hears' a scheduled list of RecWords."""

    def recognize(self, chunk, offset, *, prompt, vad, language, **_):
        end = offset + len(chunk) / transcriber.SAMPLE_RATE
        heard = [w for w in schedule if offset <= w.start and w.end <= end + 0.01]
        if drop:
            heard = [w for w in heard if not drop(w, vad, prompt)]
        if calls is not None:
            calls.append({"offset": offset, "len": len(chunk) / transcriber.SAMPLE_RATE, "vad": vad, "prompt": prompt})
        return _Window(split_phrases(heard, max_gap=0.5), "en", 0.95)

    def load(self):
        self.model_name, self.device = "fake", "cpu"

    monkeypatch.setattr(Transcriber, "_recognize", recognize)
    monkeypatch.setattr(Transcriber, "_load", load)


def guided_session(tmp_path, seconds, settings=None):
    tr = Transcriber(settings or Settings(), HardwareProfile(physical_cores=4, ram_total_gb=8))
    track = TrackInfo(tone_wav(tmp_path / "t.wav", seconds), "Artist", "Song", duration=seconds)
    return tr, tr.open_session(track)


def reference_of(*songs):
    return ReferenceScript([ReferenceComponent("", f"Song {i}", s, "test") for i, s in enumerate(songs)],
                           is_mashup=len(songs) > 1)


def test_plan_windows_cover_the_track_with_overlap():
    wins = plan_windows(100)
    assert wins[0][0] == 0 and wins[-1][1] == 100
    assert all(b - a <= transcriber.CHUNK_SEC + 1e-6 for a, b in wins)
    assert all(w2[0] < w1[1] for w1, w2 in zip(wins, wins[1:]))          # overlapping seams
    assert plan_windows(20) == [(0.0, 20)]


def test_guided_alignment_end_to_end_with_chunked_prompts(tmp_path, monkeypatch):
    lines = make_song(26, 5)
    words, truth, end = sing(lines, 4.0)
    calls: list[dict] = []
    rng = random.Random(3)
    noisy = {id(w): m for w, m in zip(words, mishear(words, rng, 0.1, 0.0, 0.0)) }   # same length, substitutions only
    fake_asr(monkeypatch, [noisy.get(id(w), w) for w in words], calls=calls)
    tr, session = guided_session(tmp_path, end + 3)
    progress = []
    guided = session.align(reference_of(lines), progress.append)

    assert guided.ok and accuracy(guided.alignment, truth) >= 0.95
    assert all(l.text in lines for l in guided.alignment.lines)           # official spelling only
    assert progress[-1] == pytest.approx(1.0)
    main = [c for c in calls if c["vad"]]
    assert len(main) == len(plan_windows(end + 3)) and all(c["len"] <= 30.01 for c in main)
    assert main[0]["prompt"].startswith(lines[0])
    assert any(not p["prompt"].startswith(lines[0]) for p in main[2:])     # prompt moved with the singer
    assert all(tr.count_tokens(c["prompt"]) <= transcriber.PROMPT_TOKEN_BUDGET + 20 for c in main)
    assert "[ti:Song]" in guided.lrc_text and "[00:" in guided.lrc_text


def test_guided_alignment_relistens_to_unheard_lines(tmp_path, monkeypatch):
    lines = make_song(14, 8)
    words, truth, end = sing(lines, 3.0)
    missing = set(map(id, words[sum(len(l.split()) for l in lines[:6]):sum(len(l.split()) for l in lines[:7])]))
    calls = []
    # VAD "drops" line 6 in the main pass; a focused re-listen without VAD hears it.
    fake_asr(monkeypatch, words, calls=calls, drop=lambda w, vad, prompt: id(w) in missing and vad)
    tr, session = guided_session(tmp_path, end + 3)
    guided = session.align(reference_of(lines), None)
    assert guided.refined_gaps >= 1 and any(not c["vad"] for c in calls)
    target = dict(truth)[lines[6]]
    assert any(abs(l.time - target) <= 0.7 for l in guided.alignment.lines if l.text == lines[6])


def test_guided_alignment_retries_without_vad_when_vad_swallows_the_singing(tmp_path, monkeypatch):
    lines = make_song(10, 9)
    words, _, end = sing(lines, 3.0)
    fake_asr(monkeypatch, words, drop=lambda w, vad, prompt: vad)       # Silero thinks it is all music
    _, session = guided_session(tmp_path, end + 3)
    guided = session.align(reference_of(lines), None)
    assert guided.ok and guided.used_vad is False


def test_guided_alignment_rejects_the_wrong_lyrics(tmp_path, monkeypatch):
    sung, _, end = sing(make_song(12, 40), 3.0)
    fake_asr(monkeypatch, sung)
    _, session = guided_session(tmp_path, end + 3)
    assert not session.align(reference_of(make_song(12, 41)), None).ok


def test_guided_alignment_two_song_mashup(tmp_path, monkeypatch):
    a, b = make_song(12, 51), make_song(10, 52)
    t, words, truth = 2.0, [], []
    for part in (a[0:4], b[3:7], a[4:7]):
        w, tr_, t = sing(part, t)
        words += w
        truth += tr_
    fake_asr(monkeypatch, words)
    _, session = guided_session(tmp_path, t + 3)
    guided = session.align(reference_of(a, b), None)
    assert guided.ok and guided.alignment.switches >= 2
    assert accuracy(guided.alignment, truth) >= 0.9


def test_sample_snippets_listen_across_the_track(tmp_path, monkeypatch):
    schedule = []
    for start in (1.0, 30.0, 60.0, 90.0):
        w, _, _ = sing(make_song(3, int(start)), start)
        schedule += w
    calls = []
    fake_asr(monkeypatch, schedule, calls=calls)
    _, session = guided_session(tmp_path, 120)
    snippets = session.sample_snippets()
    assert len(snippets) >= 3 and snippets == sorted(snippets, key=lambda s: s.start)
    assert snippets[-1].start - snippets[0].start >= 60              # spread over the track, not just the intro
    assert all(c["prompt"] == "" for c in calls)                      # blind: no reference given


# ====================================================== processor decision flow
def guided_result(ok: bool = True) -> GuidedResult:
    lines = [LrcLine(5.0 + 4 * i, f"official line {i}") for i in range(6)]
    result = AlignmentResult(
        lines=lines if ok else [], match_ratio=0.9 if ok else 0.1, coverage=1.0,
        emitted_lines=6 if ok else 0, total_lines=6, phrases_total=6, phrases_matched=6, switches=1)
    return GuidedResult(build_lrc(lines), result, "tiny", "cpu", "en", 1.0, True, 0, 0.40)


class ScriptedSession:
    def __init__(self, owner):
        self.owner = owner

    def align(self, reference, progress=None):
        self.owner.references.append(reference)
        return self.owner.align_results.pop(0)

    def blind(self, progress=None):
        self.owner.blind_calls += 1
        if self.owner.blind_error:
            raise self.owner.blind_error
        return transcriber.TranscriptionResult(lrc_of(), "en", 0.9, 9, "tiny", "cpu", 0.1)

    def sample_snippets(self, *a, **k):
        if self.owner.snippet_error:
            raise self.owner.snippet_error
        return list(self.owner.snippets)

    def close(self):
        pass


class ScriptedTranscriber:
    def __init__(self, align_results=(), snippets=(), blind_error=None, snippet_error=None):
        self.align_results, self.snippets, self.blind_error = list(align_results), snippets, blind_error
        self.snippet_error = snippet_error
        self.references, self.blind_calls, self.opened = [], 0, 0

    def open_session(self, track):
        self.opened += 1
        return ScriptedSession(self)

    def close(self):
        pass


class ScriptedFetcher:
    def __init__(self, tier1=None, plain=None, components=None, identify=None):
        self.tier1, self.plain, self.components, self.identify = tier1, plain, components or {}, identify or []
        self.tier2_calls, self.component_calls = 0, []

    def fetch(self, track, tier2=True):
        return ap.FetchOutcome(synced=self.tier1)

    def fetch_tier2(self, track, outcome=None):
        self.tier2_calls += 1
        outcome = outcome or ap.FetchOutcome()
        outcome.plain = self.plain
        return outcome

    def fetch_for(self, artist, title):
        self.component_calls.append((artist, title))
        hit = self.components.get(title)
        return ap.FetchOutcome(plain=hit)

    def identify_by_lyrics(self, snippet):
        return list(self.identify)

    def close(self):
        pass


def plain_hit(text=LYRICS, provider="Genius", partial=False, title="Understand"):
    return LyricsResult(provider, plain=text, score=90, tier=2, partial=partial, artist="Pluto Artist", title=title)


def run_one(tmp_path, name, fetcher, tr, **settings):
    lib = tmp_path / "lib"
    lib.mkdir(exist_ok=True)
    tone_wav(lib / name, 3.0)
    events = []
    with LyricsDatabase(tmp_path / "c.db") as db:
        summary = BatchProcessor(Settings(**settings), db, events.append, fetcher=fetcher, transcriber=tr).run(lib)
    tags = [(e.tag, e.message) for e in events if e.kind == "log"]
    return summary, tags, lib


def test_flow_tier1_hit_never_touches_ai(tmp_path):
    tr = ScriptedTranscriber()
    summary, _, lib = run_one(tmp_path, "A - B.wav", ScriptedFetcher(tier1=LyricsResult("LRCLIB", synced_lrc=lrc_of())), tr)
    assert summary.synced_online == 1 and tr.opened == 0 and (lib / "A - B.lrc").exists()


def test_flow_plain_text_is_force_aligned_not_blindly_transcribed(tmp_path):
    tr = ScriptedTranscriber(align_results=[guided_result(True)])
    fetcher = ScriptedFetcher(plain=plain_hit())
    summary, tags, lib = run_one(tmp_path, "Pluto Artist - Understand.wav", fetcher, tr)
    assert summary.synced_aligned == 1 and tr.blind_calls == 0
    assert (lib / "Pluto Artist - Understand.lrc").read_text().count("official line") == 6
    assert tr.references[0].components[0].lines[0].startswith("The river runs")
    assert {"AI: Forced Alignment", "OK"} <= {t for t, _ in tags}


def test_flow_weak_alignment_falls_back_to_blind(tmp_path):
    tr = ScriptedTranscriber(align_results=[guided_result(False)])
    summary, tags, _ = run_one(tmp_path, "A - B.wav", ScriptedFetcher(plain=plain_hit()), tr, identify_by_audio=False)
    assert summary.synced_ai == 1 and tr.blind_calls == 1
    assert any(t == "AI: Blind" and "last resort" in m for t, m in tags)


def test_flow_no_text_anywhere_is_the_only_time_blind_runs_first(tmp_path):
    tr = ScriptedTranscriber()
    summary, tags, _ = run_one(tmp_path, "A - B.wav", ScriptedFetcher(), tr)
    assert summary.synced_ai == 1 and tr.blind_calls == 1 and not tr.references


def test_flow_alignment_switch_off_uses_blind_even_with_text(tmp_path):
    tr = ScriptedTranscriber()
    summary, _, _ = run_one(tmp_path, "A - B.wav", ScriptedFetcher(plain=plain_hit()), tr, use_forced_alignment=False)
    assert summary.synced_ai == 1 and not tr.references


def test_flow_ai_disabled_saves_plain_text_when_asked(tmp_path):
    tr = ScriptedTranscriber()
    summary, _, lib = run_one(tmp_path, "A - B.wav", ScriptedFetcher(plain=plain_hit()), tr,
                              use_ai_fallback=False, save_plain_fallback=True)
    assert summary.plain_only == 1 and (lib / "A - B.txt").exists() and tr.opened == 0


def test_flow_mashup_components_are_fetched_and_aligned_together(tmp_path):
    tr = ScriptedTranscriber(align_results=[guided_result(True)])
    fetcher = ScriptedFetcher(components={
        "Understand": plain_hit(LYRICS),
        "Pluto Projector": plain_hit("\n".join(make_song(8, 77)), "Lyrics.ovh", title="Pluto Projector")})
    summary, tags, _ = run_one(tmp_path, "Understand x Pluto Projector.wav", fetcher, tr)
    assert summary.synced_aligned == 1 and fetcher.tier2_calls == 0           # components, not the full title
    ref = tr.references[0]
    assert ref.is_mashup and [c.title for c in ref.components] == ["Understand", "Pluto Projector"]
    mashup_msgs = [m for t, m in tags if t == "Mashup Detected"]
    assert mashup_msgs and "Understand" in mashup_msgs[0]


def test_flow_ambiguous_slash_title_tries_full_title_text_first(tmp_path):
    tr = ScriptedTranscriber(align_results=[guided_result(True)])
    fetcher = ScriptedFetcher(plain=plain_hit())
    summary, _, _ = run_one(tmp_path, "Sunshine - Day + Night.wav", fetcher, tr)
    assert fetcher.tier2_calls == 1 and fetcher.component_calls == []          # a real single title
    assert not tr.references[0].is_mashup


def test_flow_unresolved_component_is_found_by_listening_and_verified(tmp_path):
    sung = "velvet harbor whispers keep calling my name tonight"
    tr = ScriptedTranscriber(align_results=[guided_result(True)],
                             snippets=[Snippet(0.0, "river runs beneath the silver moon\nlalala"), Snippet(60.0, sung)])
    real_text = ("Intro\nvelvet harbor whispers keep calling my name tonight\nsilent marble anchors hold the shore\n"
                 "more words to make this long enough to look real\nand one more line here")
    fetcher = ScriptedFetcher(
        components={                                                      # "Pluto Projector": not found by name
            "Understand": plain_hit(LYRICS),
            "Not It": plain_hit("totally different words about something else entirely here\n" * 6, title="Not It"),
            "Real Song": plain_hit(real_text, title="Real Song")},
        identify=[IdentifiedSong("Wrong", "Not It", "Genius"), IdentifiedSong("Real", "Real Song", "Genius")])
    summary, tags, _ = run_one(tmp_path, "Understand x Pluto Projector.wav", fetcher, tr)
    titles = [c.title for c in tr.references[0].components]
    assert titles == ["Understand", "Real Song"]                          # wrong candidate rejected by verification
    assert any(t == "AI: Identify" and "Identified" in m for t, m in tags)
    assert any("already explained" in m for t, m in tags if t == "AI: Identify")   # snippet 1 matched known lyrics


def test_flow_weak_alignment_triggers_one_identification_pass_then_realigns(tmp_path):
    tr = ScriptedTranscriber(align_results=[guided_result(False), guided_result(True)],
                             snippets=[Snippet(30.0, "totally unrelated hidden second song words appear here now")])
    fetcher = ScriptedFetcher(plain=plain_hit(), identify=[IdentifiedSong("Other", "Hidden Song", "Genius")],
                              components={"Hidden Song": plain_hit(
                                  "totally unrelated hidden second song words appear here now\nand some more lines to follow\n"
                                  "so the text is long enough to count\nas real lyrics for the test")})
    summary, tags, _ = run_one(tmp_path, "Plain Name.wav", fetcher, tr)
    assert summary.synced_aligned == 1 and len(tr.references) == 2
    assert len(tr.references[1].components) == 2 and tr.blind_calls == 0
    assert any("Re-aligning" in m for _, m in tags)


def test_status_and_stats_include_aligned():
    from local_sync_lyrics.models import BatchSummary
    s = BatchSummary()
    s.bump(Status.SYNCED_ALIGNED)
    assert s.as_dict()["synced_aligned"] == 1 and s.processed == 1


def test_flow_unloadable_model_switches_ai_off_for_the_batch_and_allows_retry(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    for name in ("A - One.wav", "A - Two.wav"):
        tone_wav(lib / name, 3.0 + len(name) / 100)             # different audio => different fingerprints
    tr = ScriptedTranscriber(snippet_error=transcriber.TranscriptionError(
        "could not load model “tiny”: LocalEntryNotFoundError: offline"))
    events, fetcher = [], ScriptedFetcher()
    with LyricsDatabase(tmp_path / "c.db") as db:
        summary = BatchProcessor(Settings(), db, events.append, fetcher=fetcher, transcriber=tr).run(lib)
        assert summary.not_found == 2 and tr.blind_calls == 0 and not tr.references
        assert tr.opened == 1                                    # second track never even opened a session
        downs = [e for e in events if e.kind == "log" and "unavailable" in e.message]
        assert len(downs) == 1                                   # reported once, not once per track
        # AI never really looked at these tracks, so a later run (online again) must retry them
        with LyricsDatabase(tmp_path / "c.db") as db2:
            again = BatchProcessor(Settings(), db2, lambda e: None, fetcher=ScriptedFetcher(),
                                   transcriber=ScriptedTranscriber()).run(lib)
        assert again.skipped == 0 and again.synced_ai == 2


def test_installed_faster_whisper_accepts_every_option_we_pass():
    """Guards against upstream renames: these kwargs are used by the guided pass."""
    import inspect

    from faster_whisper import WhisperModel
    from faster_whisper.vad import VadOptions

    params = inspect.signature(WhisperModel.transcribe).parameters
    for name in ("language", "task", "beam_size", "condition_on_previous_text", "initial_prompt",
                 "word_timestamps", "vad_filter", "vad_parameters", "no_speech_threshold"):
        assert name in params, name
    VadOptions(**transcriber.VAD_SINGING)                      # valid VAD keys
    assert hasattr(WhisperModel, "__init__")
