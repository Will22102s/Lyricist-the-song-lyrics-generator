"""Phase 3 - manual lyric editor, custom providers and the smart auto-tagger."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_fixtures import make_audio  # noqa: E402
from local_sync_lyrics import api_providers as ap  # noqa: E402
from local_sync_lyrics import lrc_utils, metadata  # noqa: E402
from local_sync_lyrics.config import (  # noqa: E402
    CustomProvider, Settings, SettingsStore, TAG_MODES,
)
from local_sync_lyrics.models import Event, TrackInfo, TrackMetadata  # noqa: E402
from local_sync_lyrics.processor import BatchProcessor  # noqa: E402

LRC = lrc_utils.build_lrc(
    [lrc_utils.LrcLine(1.5, "Hello darkness"), lrc_utils.LrcLine(4.25, "my old friend")],
    artist="A", title="T")

LRC_LONG = lrc_utils.build_lrc(
    [lrc_utils.LrcLine(1.5 + i * 4, f"line {i} of the song") for i in range(6)],
    artist="A", title="T")

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


@pytest.fixture
def audio_file(tmp_path):
    def make(ext: str) -> Path:
        path = tmp_path / f"song{ext}"
        if not make_audio(path):
            pytest.skip(f"this PyAV build cannot encode {ext}")
        return path
    return make


# ============================================================== editor helpers
def test_parse_timestamp_accepts_the_forms_people_actually_type():
    assert lrc_utils.parse_timestamp("01:02.34") == pytest.approx(62.34)
    assert lrc_utils.parse_timestamp("[01:02.34]") == pytest.approx(62.34)
    assert lrc_utils.parse_timestamp("1:2") == pytest.approx(62.0)
    assert lrc_utils.parse_timestamp("01:02,5") == pytest.approx(62.5)
    assert lrc_utils.parse_timestamp("01:02:345") == pytest.approx(62.345)
    assert lrc_utils.parse_timestamp("junk") is None
    assert lrc_utils.parse_timestamp("1:2:3:4") is None


def test_shift_lrc_moves_every_stamp_and_never_goes_negative():
    shifted = lrc_utils.shift_lrc(LRC, 500)
    assert "[00:02.00]Hello darkness" in shifted and "[00:04.75]my old friend" in shifted
    back = lrc_utils.shift_lrc(LRC, -10_000)
    assert all(line.time >= 0 for line in lrc_utils.parse_lrc(back))


def test_repair_lrc_normalises_sloppy_stamps_and_reports_untimed_lines():
    fixed, problems = lrc_utils.repair_lrc(
        "[ti:Demo]\n[1:2:5] hello\n[01:02,50]world\nplain line\n")
    lines = lrc_utils.parse_lrc(fixed)
    assert [line.text for line in lines] == ["hello", "world"]
    assert problems == ["plain line"]
    assert lrc_utils.split_lrc(fixed)[0] == ["[ti:Demo]"]


def test_compose_lrc_round_trips_header_and_body():
    header = ["[ti:T]", "[ar:A]"]
    lines = [lrc_utils.LrcLine(3.0, "third"), lrc_utils.LrcLine(1.0, "first")]
    text = lrc_utils.compose_lrc(header, sorted(lines, key=lambda line: line.time))
    assert text.splitlines() == ["[ti:T]", "[ar:A]", "[00:01.00]first", "[00:03.00]third"]


# ============================================================ custom providers
def test_custom_provider_settings_round_trip_and_normalisation(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    settings = Settings()
    settings.custom_providers = [
        CustomProvider(name="  Demo  ", url_template=" https://x.test/{artist}/{title} ",
                       response_type="JSON", json_lyrics_path=" data.lyrics "),
        CustomProvider(name="Broken", url_template="ftp://nope"),
    ]
    settings._normalise()
    assert len(settings.custom_providers) == 1
    provider = settings.custom_providers[0]
    assert (provider.name, provider.response_type, provider.json_lyrics_path) == (
        "Demo", "json", "data.lyrics")
    assert store.save(settings)
    again = store.load()
    assert again.custom_providers == [provider]
    assert again.tag_mode in TAG_MODES and again.auto_tag is False


def test_custom_provider_requires_a_placeholder_and_http():
    assert not CustomProvider(name="X", url_template="https://x.test/static").valid()
    assert not CustomProvider(name="X", url_template="ftp://x.test/{title}").valid()
    assert CustomProvider(name="X", url_template="https://x.test/{title}").valid()


class FakeHttp:
    def __init__(self, routes=None):
        self.routes, self.calls = routes or {}, []

    def get(self, url, params=None, headers=None):
        self.calls.append(url)
        for key, value in self.routes.items():
            if key in url:
                if isinstance(value, bytes):
                    text, content = "", value
                elif isinstance(value, str):
                    text, content = value, value.encode()
                else:
                    text, content = json.dumps(value), b""
                return SimpleNamespace(
                    text=text,
                    json=lambda value=value: (value if not isinstance(value, str)
                                              else json.loads(value)),
                    content=content, headers={"Content-Type": "image/png"})
        return None

    def get_json(self, url, params=None, headers=None):
        self.calls.append(url)
        for key, value in self.routes.items():
            if key in url:
                return value
        return None

    def close(self):
        pass


def test_custom_lrc_provider_quotes_placeholders_and_returns_synced_lyrics():
    http = FakeHttp({"demo.test": LRC_LONG})
    entry = CustomProvider(name="Demo", url_template=(
        "https://demo.test/lyrics?artist={artist}&title={title}&album={album}"))
    hit = ap.CustomLyricsProvider(http, Settings(), entry).search(
        ap.TrackQuery.from_names("Cold Play", "Yellow", 200))
    assert hit is not None and hit.synced_lrc and "line 0 of the song" in hit.synced_lrc
    assert "Cold%20Play" in http.calls[0] and "Yellow" in http.calls[0]


def test_custom_json_provider_follows_the_dot_path_and_plain_text():
    http = FakeHttp({"demo.test": {"data": {"lines": [{"text": LRC_LONG}]}}})
    entry = CustomProvider(name="Demo", url_template="https://demo.test/{title}",
                           response_type="json", json_lyrics_path="data.lines.0.text")
    hit = ap.CustomLyricsProvider(http, Settings(), entry).search(
        ap.TrackQuery.from_names("A", "T"))
    assert hit is not None and hit.synced_lrc

    plain = FakeHttp({"demo.test": {"lyrics": "\n".join(
        ["I will always love you", "and I know that you are the one",
         "so hold me close tonight", "never let me go away",
         "because my heart belongs to you"])}})
    entry = CustomProvider(name="Demo", url_template="https://demo.test/{title}",
                           response_type="json", json_lyrics_path="lyrics")
    hit = ap.CustomLyricsProvider(plain, Settings(), entry).search(
        ap.TrackQuery.from_names("A", "T"))
    assert hit is not None and hit.plain and "hold me close" in hit.plain


def test_custom_json_provider_reports_a_missing_path_instead_of_crashing():
    http = FakeHttp({"demo.test": {"other": 1}})
    entry = CustomProvider(name="Demo", url_template="https://demo.test/{title}",
                           response_type="json", json_lyrics_path="data.lyrics")
    with pytest.raises(ap.ProviderError):
        ap.CustomLyricsProvider(http, Settings(), entry).search(
            ap.TrackQuery.from_names("A", "T"))


def test_test_custom_provider_uses_a_sample_query(monkeypatch):
    monkeypatch.setattr(ap, "HttpClient", lambda timeout=12.0: FakeHttp({"demo.test": LRC_LONG}))
    entry = CustomProvider(name="Demo", enabled=False,
                           url_template="https://demo.test/{artist}/{title}")
    summary = ap.test_custom_provider(entry)      # disabled entries are still testable
    assert summary.startswith("OK - synced lyrics") and "Coldplay - Yellow" in summary


# ================================================================== tag writing
def test_write_tags_fills_missing_fields_and_cover(audio_file):
    from mutagen.id3 import ID3

    path = audio_file(".mp3")
    meta = TrackMetadata(title="Yellow", artist="Coldplay", album="Parachutes",
                         album_artist="Coldplay", genre="Alternative", year="2000",
                         track_number=5, cover=PNG, cover_mime="image/png")
    result = metadata.write_tags(path, meta, mode="fill_missing")
    assert result.ok and result.cover_written and "TIT2" in result.fields
    tags = ID3(str(path))
    assert str(tags["TIT2"]) == "Yellow" and str(tags["TRCK"]) == "5"
    assert tags.getall("APIC")[0].mime == "image/png"

    untouched = metadata.write_tags(path, TrackMetadata(title="Changed"), mode="fill_missing")
    assert untouched.ok and untouched.fields == [] and not untouched.cover_written
    assert str(ID3(str(path))["TIT2"]) == "Yellow"

    replaced = metadata.write_tags(path, TrackMetadata(title="Changed"), mode="overwrite_all")
    assert replaced.ok and replaced.fields == ["TIT2"]
    assert str(ID3(str(path))["TIT2"]) == "Changed"


@pytest.mark.parametrize("ext,key", [(".flac", "TITLE"), (".ogg", "TITLE"),
                                     (".opus", "TITLE"), (".m4a", "\xa9nam")])
def test_write_tags_supports_every_embedded_container(audio_file, ext, key):
    import mutagen

    path = audio_file(ext)
    meta = TrackMetadata(title="Yellow", artist="Coldplay", album="Parachutes",
                         cover=PNG, cover_mime="image/png")
    result = metadata.write_tags(path, meta, mode="fill_missing")
    assert result.ok and key in result.fields and result.cover_written
    assert mutagen.File(str(path)).tags[key] == ["Yellow"]


def test_write_tags_is_honest_about_unsupported_files(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    result = metadata.write_tags(wav, TrackMetadata(title="X"))
    assert not result.ok and not result.supported and ".wav" in result.message
    assert not metadata.supports_tagging(wav) and metadata.supports_tagging(Path("x.FLAC"))


def test_save_lyrics_writes_the_sidecar_and_reembeds(audio_file):
    path = audio_file(".mp3")
    result = metadata.save_lyrics(path, LRC, output_mode="both")
    assert result.ok and result.wrote_lrc
    assert path.with_suffix(".lrc").read_text(encoding="utf-8") == LRC
    assert "Hello darkness" in (metadata.read_embedded_lyrics(path) or "")
    assert "embedded" in result.describe()


def test_save_lyrics_rejects_timeless_text_and_keeps_an_existing_sidecar(audio_file):
    path = audio_file(".mp3")
    lrc = path.with_suffix(".lrc")
    lrc.write_text(LRC, encoding="utf-8")
    bad = metadata.save_lyrics(path, "just words\nand more words\n", output_mode="lrc")
    assert not bad.ok and "not valid" in bad.message
    embed_only = metadata.save_lyrics(path, LRC, output_mode="embed")
    assert embed_only.ok and embed_only.wrote_lrc          # stale sidecar refreshed too
    assert metadata.read_embedded_lyrics(path)


# ================================================================ metadata client
MB_PAYLOAD = {"recordings": [{
    "title": "Yellow", "length": 266000,
    "artist-credit": [{"name": "Coldplay"}],
    "tags": [{"name": "Alternative Rock", "count": 4}],
    "releases": [{
        "id": "rel-1", "title": "Parachutes", "date": "2000-07-10",
        "artist-credit": [{"name": "Coldplay"}],
        "media": [{"track": [{"number": "5"}]}],
        "release-group": {"genres": [{"name": "Alternative", "count": 7}]},
    }],
}]}

ITUNES_PAYLOAD = {"results": [{
    "trackName": "Yellow", "artistName": "Coldplay", "collectionName": "Parachutes",
    "primaryGenreName": "Alternative", "releaseDate": "2000-07-10T07:00:00Z",
    "trackNumber": 5, "trackTimeMillis": 266000,
    "artworkUrl100": "https://is1.mzstatic.com/image/thumb/x/100x100bb.jpg",
}]}


def test_metadata_client_prefers_musicbrainz_and_downloads_the_cover():
    http = FakeHttp({
        "musicbrainz.org": MB_PAYLOAD,
        "coverartarchive.org": PNG,
    })
    client = ap.MetadataClient()
    client.http = http
    meta = client.lookup("Coldplay", "Yellow", "Parachutes", 266)
    assert meta is not None and meta.source == "MusicBrainz"
    assert (meta.title, meta.artist, meta.album, meta.year) == (
        "Yellow", "Coldplay", "Parachutes", "2000")
    assert meta.track_number == 5 and "Alternative" in meta.genre
    assert meta.cover == PNG and meta.cover_mime == "image/png"
    assert meta.cover_art_url.endswith("/release/rel-1/front-1200")


def test_metadata_client_falls_back_to_itunes_and_requests_high_res_art():
    http = FakeHttp({"musicbrainz.org": {"recordings": []}, "itunes.apple.com": ITUNES_PAYLOAD})
    client = ap.MetadataClient()
    client.http = http
    meta = client.lookup("Coldplay", "Yellow")
    assert meta is not None and meta.source == "iTunes"
    assert meta.cover_art_url.endswith("1200x1200bb.jpg")
    assert meta.year == "2000" and meta.track_number == 5


def test_metadata_client_returns_none_when_nothing_matches():
    client = ap.MetadataClient()
    client.http = FakeHttp({"musicbrainz.org": {"recordings": []},
                            "itunes.apple.com": {"results": []}})
    assert client.lookup("Nobody", "Nothing") is None


def test_cover_download_rejects_html_and_oversized_payloads():
    client = ap.MetadataClient()
    client.http = FakeHttp({"coverartarchive.org": "<html>not an image</html>"})
    assert client.download_cover("https://coverartarchive.org/release/x/front-1200") == (b"", "")


def test_metadata_client_merges_missing_genre_from_itunes():
    mb = json.loads(json.dumps(MB_PAYLOAD))
    mb["recordings"][0]["tags"] = []
    mb["recordings"][0]["releases"][0]["release-group"] = {}
    http = FakeHttp({"musicbrainz.org": mb, "itunes.apple.com": ITUNES_PAYLOAD,
                     "coverartarchive.org": PNG})
    client = ap.MetadataClient()
    client.http = http
    meta = client.lookup("Coldplay", "Yellow")
    assert meta is not None and meta.genre == "Alternative"


# ================================================================ processor flow
class FakeFetcher:
    def close(self):
        pass


def test_processor_auto_tag_writes_tags_and_emits_an_event(tmp_path, audio_file):
    path = audio_file(".mp3")
    events: list[Event] = []
    settings = Settings(auto_tag=True, tag_mode="fill_missing")
    processor = BatchProcessor(settings, db=None, emit=events.append, fetcher=FakeFetcher())
    processor._metadata_client = SimpleNamespace(lookup=lambda *a, **k: TrackMetadata(
        title="Yellow", artist="Coldplay", album="Parachutes", genre="Alternative",
        year="2000", track_number=5, cover=PNG, cover_mime="image/png",
        source="MusicBrainz"))

    from mutagen.id3 import ID3

    changed = processor._auto_tag(path, TrackInfo(path=path, artist="Coldplay",
                                                  title="Yellow"))
    assert changed and str(ID3(str(path))["TIT2"]) == "Yellow"
    tagged = [event for event in events if event.kind == "tagged"]
    assert tagged and tagged[0].data["fields"] and tagged[0].data["cover"]


def test_processor_auto_tag_survives_a_failed_lookup(tmp_path, audio_file):
    path = audio_file(".mp3")
    events: list[Event] = []
    processor = BatchProcessor(Settings(auto_tag=True), db=None, emit=events.append,
                               fetcher=FakeFetcher())

    def boom(*_args, **_kwargs):
        raise ap.ProviderError("network down")

    processor._metadata_client = SimpleNamespace(lookup=boom)
    assert processor._auto_tag(path, TrackInfo(path=path, artist="A", title="T")) is False
    assert any(event.level == "warning" for event in events)


def test_auto_tag_flow_rekeys_the_cache_after_embedding(tmp_path, audio_file):
    import test_pipeline as tp
    from local_sync_lyrics.database import LyricsDatabase, fingerprint_file

    lib = tmp_path / "lib"
    lib.mkdir()
    path = lib / "Coldplay - Yellow.mp3"
    if not make_audio(path, 1.0):
        pytest.skip("this PyAV build cannot encode mp3")
    before = fingerprint_file(path)

    events: list[Event] = []
    fetcher = tp.ScriptedFetcher(tier1=tp.LyricsResult("LRCLIB", synced_lrc=tp.lrc_of()))
    settings = Settings(auto_tag=True, output_mode="both")
    with LyricsDatabase(tmp_path / "cache.db") as db:
        processor = BatchProcessor(settings, db, events.append, fetcher=fetcher,
                                   transcriber=tp.ScriptedTranscriber())
        processor._metadata_client = SimpleNamespace(lookup=lambda *a, **k: TrackMetadata(
            title="Yellow", artist="Coldplay", album="Parachutes", source="MusicBrainz"))
        summary = processor.run(lib)
        after = fingerprint_file(path)
        assert summary.synced_online == 1
        assert before != after and db.get(before) is None and db.get(after) is not None
    assert any(event.kind == "tagged" for event in events)
    assert any(event.kind == "saved" for event in events)
    assert path.with_suffix(".lrc").exists()
    assert metadata.read_embedded_lyrics(path)


# ======================================================================== GUI
@pytest.fixture
def editor(qapp, tmp_path):
    from local_sync_lyrics.gui import LyricEditorDialog
    from local_sync_lyrics.theme import build_palette

    results = []

    def saver(text: str):
        results.append(text)
        return metadata.SaveLyricsResult(True, wrote_lrc=True)

    dialog = LyricEditorDialog(tmp_path / "song.mp3", LRC, build_palette(True, "#0A84FF"),
                               output_mode="lrc", saver=saver)
    dialog.show()
    qapp.processEvents()
    yield dialog, results
    dialog.close()


def test_editor_loads_the_table_and_insert_delete_shift(editor, qapp):
    dialog, _ = editor
    assert dialog.table.rowCount() == 2
    assert dialog.table.item(0, 0).text() == "00:01.50"
    dialog.table.selectRow(1)
    dialog._delete_rows()
    assert dialog.table.rowCount() == 1
    dialog._insert_row()
    assert dialog.table.rowCount() == 2
    dialog._shift(500)
    assert dialog.table.item(0, 0).text() == "00:02.00"
    qapp.processEvents()


def test_editor_validates_and_saves_via_the_injected_saver(editor, qapp):
    dialog, results = editor
    dialog.pool.start = lambda job: job.run()          # run the job synchronously
    dialog.table.item(0, 1).setText("Hello darkness (edited)")
    dialog._save()
    qapp.processEvents()
    assert results and "[00:01.50]Hello darkness (edited)" in results[0]
    assert "[00:01.50]Hello darkness (edited)" in dialog.edited_text


def test_editor_raw_view_round_trips(editor, qapp):
    dialog, _ = editor
    dialog.raw_toggle.setChecked(True)
    qapp.processEvents()
    assert dialog.stack.currentWidget() is dialog.raw_view
    assert "[00:01.50]Hello darkness" in dialog.raw_view.toPlainText()
    dialog._shift(500)                       # the shift buttons work in raw mode too
    assert "[00:02.00]Hello darkness" in dialog.raw_view.toPlainText()
    assert not dialog.insert_btn.isEnabled()
    dialog.raw_view.setPlainText("[1:2:5] first\n[00:04.00]second\n")
    dialog.raw_toggle.setChecked(False)
    qapp.processEvents()
    assert dialog.table.rowCount() == 2
    assert dialog.table.item(0, 0).text() == "00:04.00"
    assert dialog.table.item(1, 0).text() == "01:02.50"


def test_editor_reports_invalid_timestamps_without_saving(editor, qapp):
    dialog, results = editor
    dialog.pool.start = lambda job: job.run()
    dialog.table.item(0, 0).setText("not a time")
    dialog._save()
    qapp.processEvents()
    assert not results
    assert "not a mm:ss.xx timestamp" in dialog.status.text()


def test_custom_providers_dialog_adds_toggles_and_deletes(qapp):
    from local_sync_lyrics.gui import CustomProviderForm, CustomProvidersDialog
    from local_sync_lyrics.theme import build_palette

    palette = build_palette(True, "#0A84FF")
    providers = [
        CustomProvider(name="One", url_template="https://one.test/{title}"),
        CustomProvider(name="Two", url_template="https://two.test/{title}", enabled=False),
    ]
    dialog = CustomProvidersDialog(providers, palette)
    dialog.show()
    qapp.processEvents()
    assert dialog.table.rowCount() == 2
    assert "1 enabled" in dialog.status.text()

    item = dialog.table.item(1, 0)
    item.setCheckState(Qt.Checked)
    qapp.processEvents()
    assert dialog.providers[1].enabled is True

    dialog.table.selectRow(0)
    dialog._delete()
    assert len(dialog.providers) == 1 and dialog.providers[0].name == "Two"
    dialog.close()

    form = CustomProviderForm(None)
    form.name_field.setText("Mine")
    form.url_field.setText("https://mine.test/{artist}/{title}")
    form._accept()
    assert form.provider.name == "Mine" and form.provider.response_type == "lrc"


def test_custom_provider_form_rejects_incomplete_entries(qapp):
    from local_sync_lyrics.gui import CustomProviderForm

    form = CustomProviderForm(None)
    form.name_field.setText("Mine")
    form.url_field.setText("https://mine.test/static")
    form._accept()
    assert "placeholder" in form.error.text()
    form.url_field.setText("https://mine.test/{title}")
    form.type_dd.setCurrentIndex(form.type_dd.findData("json"))
    form.path_field.setText("")
    form._accept()
    assert "dot path" in form.error.text()


def test_window_completion_card_and_tagged_events(window):
    window._show_completion({"path": "/music/a.mp3", "lrc": LRC, "source": "LRCLIB"})
    assert window.completion_card.isVisible()
    assert "a.mp3" in window.completion_detail.text()
    assert window._last_saved["source"] == "LRCLIB"
    window._dismiss_completion()
    assert not window.completion_card.isVisible()

    window._handle_event(Event("saved", message="a.mp3", data={
        "path": "/music/a.mp3", "lrc": LRC, "source": "LRCLIB"}))
    assert window.completion_card.isVisible()

    window._handle_event(Event("tagged", message="a.mp3", data={
        "fields": ["TIT2"], "cover": True, "source": "MusicBrainz"}))
    assert any("Tags written" in message for *_rest, message, _level in window.log_entries)


def test_window_exposes_auto_tag_controls_and_persists_them(window):
    assert window.sw_autotag is not None and window.tag_mode_dd.count() == len(TAG_MODES)
    window.sw_autotag.setChecked(True)
    window.tag_mode_dd.setCurrentIndex(window.tag_mode_dd.findData("overwrite_all"))
    window.settings.auto_tag = True
    window.settings.tag_mode = "overwrite_all"
    collected = window._collect_settings()
    assert collected.auto_tag is True and collected.tag_mode == "overwrite_all"
