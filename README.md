# Lyricist

*(Formerly “Local Sync Lyrics”. The Python package is still named `local_sync_lyrics`. On first launch this version automatically migrates your old `LocalSyncLyrics` settings, cache, and model folders to `Lyricist`.)*

Desktop application for **Windows, Linux and macOS** that generates high-quality **synced lyrics** — as standalone `.lrc` files, embedded tags inside the audio file, or both.

It follows a clear priority order: **search first → listen second → guess last**.

```
Tier 1   Synced providers     LRCLIB → BetterLyrics → RCLyricsBand → Megalobiz → NetEase → QQ Music  → save .lrc
Mashup   "A x B", "A / B", "A & B (Mashup)", "A vs B"  → split into songs and fetch lyrics for each
Tier 2   Plain-text lyrics    Genius → Lyrics.ovh → Musixmatch (API key)                             → official words
Listen   Nothing found / unnamed song → blind-listen to slices, search lyric DBs, verify the match
Align    Official words + Whisper word timestamps → forced alignment → .lrc with official spelling
Blind    Only if no text exists or alignment confidence is too low → plain Whisper transcription
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install -r requirements.txt            # GUI + online lyrics + Whisper (~200 MB)
```

**Optional – Vocal Isolation** (pulls in PyTorch):

```bash
pip install -r requirements-cpu.txt        # CPU wheels (recommended)
# or
pip install -r requirements-gpu.txt        # CUDA 12 wheels
```

Python 3.10–3.14 recommended.  
The GUI is **native Qt 6** (PySide6) — no Flutter/Flet runtime and no embedded browser.  
FFmpeg must be available on `PATH` for audio decoding.  
Pillow is optional: it sharpens history thumbnails and is required when building icons.

---

## Usage

```bash
python main.py                                      # Launch GUI
python main.py --cli ~/Music --no-ai                # Headless batch (rich progress)
python main.py --cli ~/Music --output both          # Write both .lrc and embedded tags
python main.py --cli song.mp3 --lyrics my_lyrics.txt # Sync one file using your own lyrics
python main.py --cli ~/Music --auto-tag             # Also auto-tag + fetch cover art
```

---

## How the Pipeline Works

### 1. Provider Chain
Nine isolated providers are tried in two tiers. Every result is a `LyricsResult` marked as either `SYNCED` (saved immediately) or `PLAIN_TEXT` (sent to forced alignment). Candidates are fuzzy-scored on artist, title and duration, then validated. Offline, rate-limited or broken providers are logged and skipped.

### 2. Mashup Detection
Titles (and filenames) are split on `x`, `×`, `vs`, `/`, `+`, `→`, and on `&`/`and` only when mashup keywords (`mashup`, `bootleg`, `megamix`, `medley`…) are present — so artists such as “Simon & Garfunkel” stay intact.

- ≥ 75 % confidence → process components after a full-title Tier-1 search  
- 50–75 % confidence → first try the full title as a normal song  

Unresolved components are identified by listening: Whisper transcribes ~30 s slices at 2 %, 30 %, 55 % and 80 % of the track. Distinctive words are searched on Genius / Musixmatch. A candidate is accepted **only** if the sung words appear in its lyrics (`partial_ratio ≥ 70`).

### 3. Forced Alignment
- Audio is processed in 30-second windows with 3 s overlap.
- Official lyric lines are supplied as prompts to faster-whisper (`word_timestamps=True`, singing-tuned VAD).
- Recognised words are matched to the official text using sequence matching after vocabulary snapping.
- Repeated choruses, cut sections and interleaved mashups are handled correctly.
- Unheard words are placed according to the singer’s tempo; uncertain stretches are re-listened.
- **Quality gate**: lyrics must explain ≥ 40 % of the sung words. Wrong or poorly matching lyrics are rejected rather than written.

### 4. Safeguards
- PyAV version-proof decoding  
- CUDA → CPU int8 fallback  
- SQLite cache + atomic writes  
- Per-file error isolation  
- If the Whisper model cannot be loaded (e.g. offline first run), AI steps are skipped for the whole batch and retried on the next run  

GUI log tags: `[API: Tier 1]`, `[API: Plain Text]`, `[Mashup Detected]`, `[AI: Identify]`, `[AI: Forced Alignment]`, `[AI: Blind]`.

---

## Interface — Liquid Glass (Qt 6)

Layered translucent glass over an ambient gradient. The window is **frameless** with a custom translucent title bar (native drag, double-click to maximise, minimise / maximise / close buttons, drop shadow and OS-aware rounded corners).

The sidebar switches between four icon-driven views with a short cross-fade:

| View                  | Contents                                                                 |
|-----------------------|--------------------------------------------------------------------------|
| **Sync Workspace**    | Drag-and-drop queue, library card, output-mode pills, live stat tiles, floating progress card, activity log |
| **History & Library** | Searchable cover-art grid of everything ever synced, with per-track actions |
| **Providers & Rules** | Smart pipeline, provider toggles, custom endpoints, tagging and batch rules |
| **Settings & About**  | Appearance, local AI, updates, support links and hardware profile        |

**Settings → Appearance & Theme** (applied instantly, no restart):

| Appearance                  | Accent Presets                                      |
|-----------------------------|-----------------------------------------------------|
| Dark (Liquid Obsidian)      | Apple Blue `#0A84FF` (default)                      |
| Light (Frosted Pearl)       | Spotify Green `#1DB954`                             |
| System                      | Apple Music Pink/Red `#FA243C` · Midnight Purple `#BF5AF2` · Sunset Amber `#FF9F0A` · **Custom** |

One accent colour + dark/light mode generates the entire palette.

### Drag-and-drop Queue
Drop folders or individual audio files (`.mp3`, `.flac`, `.m4a`, `.wav`, `.ogg`, `.opus`) onto the workspace, or use **Add files…**. Folders are scanned recursively on a worker thread. Each row shows a live status badge and the language Whisper locked onto. Double-click opens the lyric editor.

### History & Library
The SQLite cache stores full history (path, album, genre, year, language, output mode, provider, timestamp and thumbnail). The view shows a searchable / filterable cover-art grid with actions: Open in Lyric Editor, Open File Location, Re-Sync Track, Export `.lrc`.

### Updates & Support
Automatic or manual update checks against GitHub Releases. A newer version shows a non-blocking banner with changelog and download links.  
**About & Support** displays version, hardware summary, third-party credits and configurable support links (GitHub Sponsors / Ko-fi / Buy Me a Coffee).

### Keyboard Shortcuts

| Shortcut          | Action                              |
|-------------------|-------------------------------------|
| `Ctrl+O`          | Choose a music folder               |
| `Ctrl+Shift+O`    | Add individual audio files          |
| `Ctrl+R`          | Start / resume the queue            |
| `Ctrl+1` … `Ctrl+4` | Switch between the four main views |
| `Ctrl+E`          | Edit the selected track’s lyrics    |
| `Ctrl+F`          | Focus library search / folder field |
| `Ctrl+U`          | Check for updates now               |
| `Ctrl+,`          | Open Settings                       |
| `Esc`             | Stop the current batch              |
| `Ctrl+Q`          | Quit                                |

---

## Audio Pipeline & Output

### Isolate Vocals
Optional switch (default **off**). Uses `audio-separator` with UVR-MDX-NET-Inst_HQ_3 (~65 MB) or falls back to `demucs`. Failures gracefully fall back to the original audio.

### Language Lock
~30 s from mid-track is analysed with Whisper’s language detector. The detected language is locked for the whole song (mashups are never locked).

### Your Own Lyrics
Paste lyrics for a single file → **Match & Sync**. A similarity check runs first; below 35 % a confirmation dialog appears. On success, text-guided forced alignment produces timestamps for your lines.

### Output Modes

| Format            | What is written                                      |
|-------------------|------------------------------------------------------|
| MP3               | `USLT` (plain) + `SYLT` (millisecond timestamps)     |
| FLAC / OGG / Opus | `LYRICS` Vorbis comment containing timed LRC         |
| M4A / MP4 / M4B   | `©lyr` atom containing timed LRC                     |
| Other (WAV …)     | Separate `.lrc` file (with warning)                  |

---

## Phase 3 Features

### Custom Providers
Add your own endpoints (name, URL template with `{artist}` / `{title}` / `{album}`, response type `lrc` or `json`). Queried after built-in providers; failures never abort a batch.

### Lyric Editor
Two-column table + Raw `.lrc` view. Insert / Delete line, Shift timestamps, Validate timestamps. **Save & Apply** rewrites the `.lrc` and/or re-embeds the lyrics.

### Smart Auto-Tagger & Cover Art
**Fill Missing** (default) or **Overwrite**. Sources: MusicBrainz + Cover Art Archive, with iTunes Search API fallback (high-resolution artwork). Writes title, artist, album, album artist, genre, year, track number and embedded cover.

CLI: `--auto-tag [--tag-mode fill_missing|overwrite_all]`

---

## Providers

| Tier | Provider       | Requirements     | Notes                                              |
|------|----------------|------------------|----------------------------------------------------|
| 1    | LRCLIB         | –                | JSON API, exact + fuzzy search                     |
| 1    | BetterLyrics   | –                | TTML → LRC                                         |
| 1    | RCLyricsBand   | –                | Form search + HTML scrape                          |
| 1    | Megalobiz      | –                | HTML scrape                                        |
| 1    | NetEase        | –                | Public web API (credit lines stripped)             |
| 1    | QQ Music       | –                | Public web API                                     |
| 2    | Genius         | Optional token   | Official API or public search + page scrape        |
| 2    | Lyrics.ovh     | –                | Direct lookup + suggestions                        |
| 2    | Musixmatch     | API key          | Free plan returns excerpts (flagged as partial)    |

---

## Settings & Environment Variables

Settings location:
- **Windows**: `%APPDATA%\Lyricist\settings.json`
- **Linux/macOS**: `~/.config/Lyricist/settings.json`

Use `LYRICIST_HOME` (or the older `LSL_HOME`) for a portable data folder.

| Variable                                      | Purpose                              |
|-----------------------------------------------|--------------------------------------|
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`         | Hugging Face token                   |
| `GENIUS_ACCESS_TOKEN` / `GENIUS_TOKEN`        | Genius API token                     |
| `MUSIXMATCH_API_KEY`                          | Musixmatch key                       |
| `LYRICIST_WHISPER_MODEL` / `LYRICIST_LANGUAGE`| Whisper model / language             |
| `LYRICIST_NO_UPDATE_CHECK`                    | Skip the startup update check        |

Tokens are never written into the settings file or logged.

---

## Packaging

```bash
pip install -r requirements-dev.txt
python build.py                  # one-folder build → dist/Lyricist/
python build.py --onefile        # single executable
python build.py --clean          # wipe PyInstaller cache first
python build.py --icons          # force-regenerate icons
python build.py --dry-run        # print the exact command only
python make_icons.py --accent 1DB954
```

Build on the target OS (PyInstaller does not cross-compile). Optional extras such as `audio-separator` are only bundled if present in the build environment.

### Automated Release Builds

Push a version tag to trigger `.github/workflows/release.yml`. It runs the test suite, builds Windows and Linux packages, and attaches them to the GitHub Release:

```bash
git tag -a v2.3.0 -m "Lyricist 2.3.0"
git push origin v2.3.0
```

---

## Project Status

**278 automated tests** covering the full pipeline, GUI (all four views), theme system, forced alignment, tag writing, vocal isolation, language locking, “Your Own Lyrics”, lyric editor, custom providers, auto-tagging + cover art, history/library, updater and packaging.

**Not verified in the test environment**
- Live responses from every external provider  
- Real vocal separation with actual model weights  
- Real-world Whisper accuracy on heavily produced tracks  
- Native Windows/macOS packaging and window-icon behaviour  
- Live GitHub release download / in-app self-update  

The ≥ 40 % alignment quality gate and the 35 % “Your Own Lyrics” similarity threshold exist specifically to prevent low-confidence results from being written.

---

## License

This project is licensed under the **Apache License 2.0**.

See the [LICENSE](LICENSE) file for the full text.
```
