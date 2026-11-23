# Lyricist

*(formerly “Local Sync Lyrics” - the Python package is still called `local_sync_lyrics`, and the first start of this
version moves your old `LocalSyncLyrics` settings / cache / model folder to `Lyricist` automatically.)*

Desktop app (Windows + Linux) that writes **synced lyrics** - as `.lrc` files, embedded in the audio file's own tags, or both.
It searches first, listens second and guesses last:

```
Tier 1  synced providers   LRCLIB → BetterLyrics → RCLyricsBand → Megalobiz → NetEase → QQ Music   → save .lrc
Mashup  "A x B", "A / B", "A & B (Mashup)", "A vs B"  → split into songs, fetch each one's lyrics
Tier 2  plain-text lyrics  Genius → Lyrics.ovh → Musixmatch (key)                                   → official WORDS
Listen  nothing found / a song unnamed → blind-listen to slices, search lyric DBs, VERIFY the match
Align   official words + Whisper word timestamps → forced alignment → .lrc with the OFFICIAL spelling
Blind   only if no text exists anywhere (or the alignment is too weak to trust) → plain Whisper
```

```
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # GUI + online lyrics + Whisper (~200 MB)
python main.py                                          # GUI (native Qt 6 window)
python main.py --cli ~/Music --no-ai                    # headless (rich progress)
python main.py --cli ~/Music --output both              # .lrc + embedded tags
python main.py --cli song.mp3 --lyrics my_lyrics.txt    # sync ONE file with your own lyrics

# Optional: vocal isolation ("Isolate Vocals"). Pick ONE - it pulls in PyTorch:
pip install -r requirements-cpu.txt                     # CPU wheels (recommended)
pip install -r requirements-gpu.txt                     # CUDA 12 wheels
```

Python 3.10–3.13 recommended. The GUI is **native Qt 6** (`PySide6`) - no Flutter/Flet
runtime, no embedded browser. The base install is deliberately lean; vocal isolation is
an opt-in extra (`audio-separator` + PyTorch + `audioread`/`librosa`/`soundfile`). If the
extra is missing, the *Isolate Vocals* switch shows a banner and the original audio is
used instead. FFmpeg must be on `PATH` for decoding. Pillow is needed only to *build* the
icons (see Packaging).

## How the upgraded pipeline works

**1. Provider chain (`api_providers.py`).** Nine isolated providers in two tiers. Every hit is a
`LyricsResult` whose `kind` is `SYNCED` (save it) or `PLAIN_TEXT` (align it). Candidates are fuzzy-scored
on artist/title/duration and validated (does the LRC fit the track? does the text look like lyrics?).
A provider that is offline, rate-limited or has changed its layout is logged and skipped.

**2. Mashup detection (`text_utils.detect_mashup`, `processor.py`).** The title (then the file name) is split
on `x`, `×`, `vs`, `/`, `+`, `→`, and on `&`/`and` only when a mashup keyword is present
(`mashup`, `bootleg`, `megamix`, `medley`…), so `Simon & Garfunkel` stays whole. Each match carries a
confidence: ≥75 % goes straight to the component pipeline after the full-title Tier 1 search; 50–75 %
(`Day / Night`) first tries the full title as a normal song. Components that cannot be resolved by name are
found by listening: Whisper transcribes ~30 s slices at 2 %, 30 %, 55 % and 80 % of the track, the most
distinctive words go to Genius / Musixmatch lyric search, and **a candidate is accepted only if the sung
words really appear in its lyrics** (fuzzy `partial_ratio` ≥ 70).

**3. Forced alignment (`lrc_utils.ForcedAligner`, `transcriber.AudioSession.align`).**
* Audio is processed in 30 s windows (3 s overlap). One window per call is deliberate:
  faster-whisper applies `initial_prompt` only to the first window of a call, and keeps only ~223 prompt
  tokens. Each call therefore gets the *lines expected next* (tracked position per song), trimmed to a token
  budget, plus `word_timestamps=True` and a singing-tuned `vad_filter=True`.
* Recognised words are matched to the official lines with `difflib.SequenceMatcher` after snapping misheard
  words to the song's own vocabulary (`thefuzz`). Matching *tracks* position per song, so repeated choruses,
  sections that were cut from the audio, and interleaved songs of a mashup all resolve; each change of song is a
  located transition.
* Output lines are the official text with the time of their first word. Unheard words are placed at the
  singer's tempo; stretches whose timing is still a guess are re-listened to (no VAD, missing lines as the
  prompt). If VAD swallows soft vocals the whole pass is repeated without it.
* Quality gate: the lyrics must explain ≥ 40 % of the sung words (`min_alignment_match`), otherwise the result
  is not trusted. Wrong lyrics (a different song, a different version) are rejected rather than written.

**4. Preserved safeguards.** PyAV-version-proof decoding (`metadata_errors` TypeError handled; works with
`av==12.3.0` *and* current releases), CUDA→CPU int8 fallback during inference, SQLite cache, atomic writes,
per-file error isolation, Qt-signal event delivery to the GUI, and: if the Whisper model cannot be loaded
(offline first run) every AI step is skipped once for the whole batch and the tracks are retried next run.

GUI log tags: `[API: Tier 1]`, `[API: Plain Text]`, `[Mashup Detected]`, `[AI: Identify]`,
`[AI: Forced Alignment]`, `[AI: Blind]`.

## Interface — Liquid Glass (Qt 6)

Layered translucent glass over an ambient gradient: a dark (or frosted) plane, `rgba(...)`
cards with a 1 px specular rim, 14–28 px corners and one accent colour. The whole look is
one generated Qt Style Sheet (`theme.build_qss`), so a sidebar switches between **Sync
Workspace** (library card, output-mode pills, stat tiles, activity log) and **Settings**
with no restart; while a batch runs a tinted **floating progress card** appears above the
stats with a gradient bar that follows each file. Hover / checked / disabled states come
from QSS `:hover`, `:checked`, `:disabled` rules; the window is a native `QMainWindow`
(no embedded browser, no Flutter runtime).

**Settings → Appearance & Theme** (applies instantly, no restart, saved to `settings.json`):

| Appearance | Accent presets |
|---|---|
| Dark (Liquid Obsidian) · Light (Frosted Pearl) · System | Apple Blue `#0A84FF` (default) · Spotify Green `#1DB954` · Apple Music Pink/Red `#FA243C` · Midnight Purple `#BF5AF2` · Sunset Amber `#FF9F0A` · **Custom** (any hex) |

One accent + dark/light derive the whole palette (`theme.py`): buttons, progress bar, active
pills, checkboxes, focus rings, the tinted background and the logo. Label colour flips to
near-black on light accents (Spotify green, amber) to keep contrast. `System` follows the OS
colour scheme. New settings keys: `theme_mode`, `accent_preset`, `custom_accent`
(`settings_version` is unchanged, so older files load with defaults and keep their provider
choices).

Code map: `theme.py` (colour maths, palette + QSS generator, no Qt import) · `branding.py`
(logo geometry → SVG string, Pillow renderer for PNG/ICO) · `gui.py` (PySide6 window, QThread
batch worker, `QtLogHandler`, accent/theme repaint).

**UI-thread rules.** No handler blocks: the batch runs in a `QThread` (`BatchWorker`) that
emits `Event` objects through a Qt signal; settings saves, the cache clear and the startup
hardware/FFmpeg probe run as `QRunnable` jobs on the global thread pool; a thread-safe
`logging.Handler` forwards warnings from any thread to the console through a Qt signal.
Every widget update happens on the UI thread (queued connections).

## Audio pipeline & output (Phase 2)

**Isolate Vocals (Strip Background Music)** - a switch in the Sync Workspace (default **off**: it costs minutes of CPU
per song; turn it on for dense or noisy mixes). The track goes through `audio-separator` with the ONNX model
**UVR-MDX-NET-Inst_HQ_3** (downloaded once, ~65 MB, into the models folder) or, if only `demucs` is installed,
`mdx_extra_q`. The stem is written as `vocals.wav` into a private temp folder, decoded into memory for
Whisper and **deleted immediately**; leftovers of a crash are swept on the next start. Progress shows as
`[Audio: Isolate] Isolating vocals … 40s elapsed`. It never blocks a sync: no backend installed, a model that cannot be
downloaded (offline first run), too little free RAM (about `1.6 GB + 0.08 GB/min of audio` is required) or any runtime
error just logs a warning and the original audio is used - problems that would repeat on every track switch isolation off
for the rest of the batch after one message. *Stop* takes effect when the current separation finishes.

**Language lock.** Before any transcription or alignment, ~30 s from mid-song (a second sample if the first is unsure)
goes through Whisper's own `detect_language`; with lyrics text available, its language (script analysis, then
`langdetect`, then a stop-word model) only breaks ties - what is sung wins. The result is passed explicitly to every
`transcribe()` call, so Whisper never guesses or flips language mid-song. Shown as a badge on the progress card
(`Japanese · 98%`) and in the log (`[AI: Language] Detected language -> Japanese (98%) · locked for transcription`). A language
chosen in Settings overrides detection; *Settings → Smart pipeline → Detect language* turns it off; several songs mixed in one
track (mashups) are never locked.

**Your Own Lyrics** (button in the workspace header → glass dialog). Pick one audio file, paste the lyrics, *Match & Sync*:
1. a quick speech-to-text pass over four slices is compared with the text (`lrc_utils.match_lyrics`: recognised words are
   snapped to the lyrics' vocabulary and credited only inside ONE contiguous stretch of lyrics, so common words scattered
   over a different song do not add up; partial lyrics still pass);
2. below **35 %** similarity (wrong song, wrong language) nothing is written - a dialog offers **Cancel** or **Force Sync Anyway**;
   too little singing to judge does not block;
3. otherwise text-guided forced alignment maps real timestamps onto *your* lines. If the full pass finds the lyrics do not fit
   after all, the same question is asked again. Forcing accepts whatever the alignment produced.

**Output mode** (segmented control, default *Save Separate .lrc File*; also `--output lrc|embed|both`):

| Format | Written |
|---|---|
| MP3 | `USLT` (plain text) **and** `SYLT` (millisecond timestamps), UTF-16, the file's ID3 v2.3/v2.4 is kept |
| FLAC · OGG · Opus | `LYRICS` Vorbis comment holding the time-stamped LRC |
| M4A · MP4 · M4B | `©lyr` atom holding the time-stamped LRC |
| anything else (WAV ...) | cannot be tagged → a separate `.lrc` is saved instead, with a warning |

Tags are written only after the lyrics passed validation, are read back to verify the write, and a failure is reported
instead of raised. An existing `.lrc` is embedded without any new search; files that already carry lyrics are skipped
unless *Overwrite* is on; plain-text results are embedded unsynced. Tagging rewrites the head of the file, so the cache key is
recomputed afterwards (otherwise every embedded track would be re-processed on the next run).

All heavy steps (separation, language ID, alignment, embedding) run on the worker thread; the UI only receives events.
Log tags: `[Audio: Isolate]`, `[AI: Language]`, `[AI: Verify]`, `[Output]`.

## Providers

| Tier | Provider | Needs | Notes |
|---|---|---|---|
| 1 | LRCLIB | – | JSON API, exact + fuzzy search |
| 1 | BetterLyrics | – | TTML → LRC |
| 1 | RCLyricsBand | – | form search + HTML scrape |
| 1 | Megalobiz | – | HTML scrape |
| 1 | NetEase | – | public web API; credit lines (`作词 :`) stripped |
| 1 | QQ Music | – | public web API; base64/JSONP handled |
| 2 | Genius | optional token | official search API with a token, public search without; page scrape for text |
| 2 | Lyrics.ovh | – | direct lookup + Deezer-backed suggestions |
| 2 | Musixmatch | API key | free plan returns excerpts → flagged partial; also identifies songs by lyric snippet |

**Custom providers (Phase 3).** Settings → *Lyrics providers* → **Custom providers…** adds your own
endpoints. Each entry has a name, an enable toggle, a URL template (`{artist}`, `{title}` and `{album}`
are URL-encoded and substituted) and a response type: `lrc` (the body is an LRC document) or `json`
(the lyrics are extracted with a dot path such as `data.lyrics` or `items.0.text`). Custom endpoints are
queried **after** the built-ins, in list order, through the same timeout/exception-safe pipeline — a broken
endpoint is logged and skipped and can never abort a batch. **Test** runs the endpoint against a sample
query. The list lives in `settings.json` under `custom_providers`.

## Manual editor, tags & artwork (Phase 3)

**Lyric editor.** After a track is written, the workspace shows a non-blocking completion card
(*View / Edit Lyrics* · *Done*). The editor is a modal `QDialog` with a two-column table
(`Timestamp [mm:ss.xx]` · `Lyric line`) and a **Raw .lrc view** toggle that shares the same document.
Toolbar: **Insert Line**, **Delete Line**, **Shift All Timestamps ±ms** (never below 00:00.00) and
**Validate Timestamps** (normalises `[1:2:5]`, `[01:02,5]`, missing brackets; reports untimed text
instead of inventing a time). **Save & Apply** writes the `.lrc` and/or re-embeds the lyrics
(`USLT`/`SYLT` for MP3, `LYRICS` for FLAC/OGG/Opus, `©lyr` for M4A) according to the output mode, and
refreshes an existing sidecar even in embed-only mode. The write runs on a thread-pool job, so the
dialog never freezes.

**Smart auto-tagger.** Toggle *Auto-tag & cover art* in the workspace (or Settings → *Tags & artwork*),
then choose **Fill Missing** (default) or **Overwrite**. Lookups use **MusicBrainz** (1 request/s) with
**Cover Art Archive** artwork and fall back to the **iTunes Search API** (artwork upgraded from
`100x100bb` to `1200x1200bb`). No API key is needed. Written fields: title, artist, album, album artist,
genre, year and track number, plus an embedded cover (`APIC` for MP3, `Picture` for FLAC, the
`metadata_block_picture` comment for OGG/Opus, `covr` for M4A). *Fill Missing* keeps every value that
already exists; *Overwrite* replaces them. Network and disk work happen on the existing background
thread; a failed lookup or an unsupported container is logged, never fatal. Because writing tags changes
the file's fingerprint, the cache entry is re-keyed afterwards. CLI: `--auto-tag [--tag-mode fill_missing|overwrite_all]`.

## Settings & environment variables

Stored in `%APPDATA%\Lyricist\settings.json` / `~/.config/Lyricist/settings.json`
(cache, models, logs in the data folder; `LYRICIST_HOME` - or the older `LSL_HOME` - makes everything portable). Older settings files are
upgraded automatically (new providers are enabled).

| Variable | Used when settings.json has no value |
|---|---|
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | Hugging Face token |
| `GENIUS_ACCESS_TOKEN` / `GENIUS_TOKEN` | Genius API token |
| `MUSIXMATCH_API_KEY` | Musixmatch key |
| `LYRICIST_WHISPER_MODEL`, `LYRICIST_LANGUAGE` (or the older `LSL_*`) | model / language |

Tokens are never copied from the environment into the file and are never logged. New switches:
*Forced alignment*, *Detect mashups*, *Name unknown songs by listening*.

## Packaging

```
pip install -r requirements-dev.txt
python build.py            # makes assets/logo.* if missing/stale, then PyInstaller -> ./dist
python build.py --icons    # force-regenerate the icons first
python make_icons.py --accent 1DB954   # (optional) the same logo in another colour
```

`build.py` runs
`python -m PyInstaller --noconfirm --clean --windowed --onedir --name Lyricist --add-data "assets:assets" --collect-all …`
(`;` instead of `:` on Windows). If `audio-separator` / `langdetect` / `librosa` / `soundfile`
are installed in the build environment they are bundled too (the first drags in PyTorch -
expect a much larger package; leave it out of the build environment to keep the app small).
`--icon` stamps the executable (Explorer, taskbar, title bar); `--add-data` ships `assets/`
so the running app finds `logo.png` and its window icon. Output is `dist/Lyricist/`.
`assets/logo.svg|png|ico` are committed; they are drawn by `local_sync_lyrics/branding.py`,
so edit the geometry there and re-run `python main.py --make-icons`.

## Verification status — please read

Verified in a Linux sandbox (Python 3.13, PySide6 6.12, faster-whisper 1.2.1, PyAV 19, mutagen): **261 tests** (95 original
pipeline/GUI + 29 Liquid Glass UI + 104 for Phase 2 + 33 for Phase 3), the real Qt window built under the `offscreen` platform through
both views, all five accents, dark/light/system, the "Your Own Lyrics" dialog, the mismatch dialog and a simulated
batch, the PyInstaller command line assembled by `build.py`, a 60-seed noise sweep of the aligner, and an end-to-end
offline run of the real pipeline (log tags, graceful degradation). The Phase 2 tests were mutation-checked (breaking the
mismatch dialog, the `SYLT` frame, the language lock and the stem clean-up each makes a test fail).

Phase 2 specifics: tags were written to **real MP3 / FLAC / Opus / M4A files** generated with PyAV and read back with mutagen
(`SYLT` timestamps and language, ID3 v2.3 preserved, replace-not-stack, Unicode); the lyric matcher was run over 300 synthetic
wrong-song and 300 right-song comparisons per vocabulary size (0 false accepts, 0 false rejects, even at 30 % simulated
recognition errors); `VocalIsolator` was exercised against stand-ins that follow the `audio-separator` and `demucs`
call contracts (stem naming, temp clean-up on every path, no model reload per track, offline / low-RAM / OOM / repeated-failure /
Stop handling); language locking was verified to reach every `transcribe()` call, and the installed faster-whisper's
`detect_language` signature is pinned by a test.

Phase 3 specifics: tag and cover writing was exercised on **real MP3 / FLAC / OGG / Opus / M4A files** generated with PyAV
(fill-missing vs overwrite, no-op second pass, unsupported WAV reported instead of raised, read-back of every field and
the embedded cover); the editor's table/raw view round-trip, insert/delete/shift/validate and Save & Apply were driven
through real widgets (a stubbed saver proved the signal path, and a real MP3 proved the `.lrc` + `USLT`/`SYLT` rewrite);
custom providers were tested for URL quoting, LRC and JSON dot-path extraction, missing-path errors and the settings
round-trip; the MusicBrainz → Cover Art Archive → iTunes fallback, high-res artwork rewrite, cover sniffing (HTML and
oversized payloads rejected) and genre merge were covered with canned payloads. A live CLI run (`--cli --auto-tag`)
filled `TPE2`/`TRCK` from MusicBrainz and embedded the lyrics.

Aligner accuracy on **simulated** recognition errors (12 % misheard, 6 % dropped, 3 % hallucinated words),
share of lines timestamped within 0.7 s of the truth, 80 random seeds:
plain song 99.6 % · chorus repeated but written once 98.6 % · interleaved two-song mashup 98.7 % ·
section cut from the audio 99.8 % · fully unheard line 99.8 % · wrong lyrics wrongly accepted 0 / 80.
At 25 % misheard / 12 % dropped: 98 % / 93 % / 93.5 % / 99.8 % / 99.7 %.

**Not verified** (no network access to the providers, no GPU, no model weights, no Windows in the sandbox): **real vocal
separation** (`audio-separator` needs PyTorch and a model download, neither available here - only its call contract is tested), real
Whisper `detect_language` / transcription on audio, OGG *Vorbis* tag writing (this PyAV build has no Vorbis encoder; Opus and FLAC
share the code path), real-world accuracy of the 35 % match threshold on actual Whisper output (the numbers above are
synthetic), live
responses of every provider (NetEase, QQ Music, RCLyricsBand, Megalobiz and Genius are unofficial
endpoints/scrapes and can change), real Whisper word-timestamp quality on singing, real mashup recordings,
a full PyInstaller run, and everything Windows-specific: the title-bar / taskbar icon (`--icon`), true
GPU-accelerated blur performance (Qt stylesheets use flat translucent fills, not live blur), and
the hover animations (the offscreen platform has no pointer). The simulated
numbers measure the *aligner*, not Whisper: real accuracy depends on how well Whisper hears the vocals
(heavy effects, rap and layered mashups are harder). The ≥ 40 % quality gate exists so that a poor result
falls back instead of being written.
