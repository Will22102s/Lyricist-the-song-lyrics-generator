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

Layered translucent glass over an ambient gradient: a dark (or frosted) plane, `rgba(...)` cards with a 1 px specular rim, 14–28 px corners and one accent colour. The entire look is one generated Qt Style Sheet (`theme.build_qss`), so the appearance can change with no restart. The window is **frameless** with a custom translucent title bar: native drag and double-click-to-maximise, minimise / maximise / close buttons, a drop shadow and OS-aware rounded corners. The sidebar switches between four icon-driven views with a short cross-fade (`QPropertyAnimation`):

| View | Contents |
|---|---|
| **Sync Workspace** | drag-and-drop queue, library card, output-mode pills, live stat tiles, floating progress card, activity log |
| **History & Library** | searchable cover-art grid of everything ever synced, with per-track actions |
| **Providers & Rules** | smart pipeline, provider toggles, custom endpoints, tagging and batch rules |
| **Settings & About** | appearance, local AI, updates, support links and hardware profile |

While a batch runs a tinted **floating progress card** appears above the stats with a gradient bar that follows each file.

**Settings → Appearance & Theme** (applied instantly, no restart):

| Appearance                  | Accent Presets                                      |
|-----------------------------|-----------------------------------------------------|
| Dark (Liquid Obsidian)      | Apple Blue `#0A84FF` (default)                      |
| Light (Frosted Pearl)       | Spotify Green `#1DB954`                             |
| System                      | Apple Music Pink/Red `#FA243C` · Midnight Purple `#BF5AF2` · Sunset Amber `#FF9F0A` · **Custom** |

One accent colour + dark/light mode generates the entire palette (buttons, progress bar, active pills, focus rings, background and logo). Label colour automatically switches to near-black on light accents for contrast.

### Drag-and-drop queue

Drop folders or individual `.mp3` / `.flac` / `.m4a` / `.wav` / `.ogg` / `.opus` files on the workspace, or use **Add files…**. Folders are scanned recursively on a worker thread (never the UI thread). Every row shows a live badge — *Queued*, *Isolating Vocals*, *Syncing*, *Auto-Tagging*, *Done*, *Error*, *Not Found* — plus the language Whisper locked onto for that track. Select a row for **Edit Lyrics**, **Open Location** or **Remove**; double-click opens the lyric editor. The queue is cleared or extended at any time, and **Start sync** processes exactly the queued files (or the folder in the field when the queue is empty).

### History & Library

The SQLite cache is upgraded in place (`SCHEMA_VERSION = 2`, idempotent `ALTER TABLE` migration — no existing record is lost) to store `file_path`, album, genre, year, language, output mode, provider, `synced_at` and a cached thumbnail. The **History & Library** view loads rows, thumbnails and decoded artwork on the thread pool, then shows a searchable / filterable grid. Real embedded artwork (`APIC`, FLAC/OGG `Picture` / `metadata_block_picture`, MP4 `covr`) is downscaled with Pillow when available; otherwise a branded placeholder is drawn. Select a tile for **Open in Lyric Editor**, **Open File Location** (Explorer `/select` on Windows, Finder `-R` on macOS, `xdg-open` elsewhere), **Re-Sync Track** or **Export .lrc**.

### Updates & support

`updater.py` checks `https://api.github.com/repos/<owner>/<repo>/releases/latest` on a background `QThread` at startup (when *Automatically check for updates* is on) or from **Check Now**. Versions are compared with `packaging.version` against the single `__version__` in `config.py`. A newer release shows a non-blocking banner with its changelog; the dialog opens the latest `.exe` / `.msi` / `.zip` / AppImage asset or the release page in the browser. Set `LYRICIST_NO_UPDATE_CHECK=1` to disable the check for a single run, or point `DEFAULT_UPDATE_REPO` / the `update_repo` setting at your own fork.

**About & Support** shows the version, CPU / GPU / RAM summary, third-party credits (`faster-whisper`, CTranslate2, PySide6, mutagen, MusicBrainz, LRCLIB…) and configurable **GitHub Sponsors / Ko-fi / Buy Me a Coffee** buttons (`SUPPORT_LINKS` in `config.py`).

### Keyboard shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl+O` | choose a music folder |
| `Ctrl+Shift+O` | add individual audio files |
| `Ctrl+R` | start / resume the queue |
| `Ctrl+1` … `Ctrl+4` | Workspace · History · Providers · Settings |
| `Ctrl+E` | edit the selected track's lyrics |
| `Ctrl+F` | focus the library search (or the folder field) |
| `Ctrl+U` | check for updates now |
| `Ctrl+,` | open Settings |
| `Esc` | stop the current batch |
| `Ctrl+Q` | quit |

---

## Audio Pipeline & Output

### Isolate Vocals
Optional switch in the Sync Workspace (default **off**). Uses `audio-separator` with the UVR-MDX-NET-Inst_HQ_3 model (~65 MB, downloaded once) or falls back to `demucs`. The vocal stem is written to a private temp folder, used for Whisper, then deleted. Failures (missing backend, offline, low RAM, etc.) simply fall back to the original audio and continue.

### Language Lock
Before transcription/alignment, ~30 s from the middle of the track is analysed with Whisper’s language detector. The detected language is locked for the entire song (mashups are never locked). A language set in Settings overrides detection.

### Your Own Lyrics
Button in the workspace header. Paste lyrics for a single file and click **Match & Sync**:
1. A quick speech-to-text pass checks similarity.
2. Below 35 % similarity a confirmation dialog appears.
3. On success (or forced), text-guided forced alignment produces timestamps for *your* lines.

### Output Modes
| Format            | What is written                                      |
|-------------------|------------------------------------------------------|
| MP3               | `USLT` (plain) + `SYLT` (millisecond timestamps)     |
| FLAC / OGG / Opus | `LYRICS` Vorbis comment containing timed LRC         |
| M4A / MP4 / M4B   | `©lyr` atom containing timed LRC                     |
| Other (WAV …)     | Separate `.lrc` file (with warning)                  |

Tags are written only after validation, read back for verification, and failures are reported cleanly. Existing lyrics are skipped unless **Overwrite** is enabled.

---

## Phase 3 Features

### Custom Providers
Settings → *Lyrics providers* → **Custom providers…** lets you add your own endpoints.  
Each entry has a name, enable toggle, URL template (`{artist}`, `{title}`, `{album}` are substituted), and response type (`lrc` or `json` with a dot path). Custom providers are queried after the built-ins and never abort a batch if they fail.

### Lyric Editor
After a track is written, a completion card offers **View / Edit Lyrics**. The editor provides:
- Two-column table (Timestamp · Lyric line) + Raw `.lrc` view
- Insert / Delete line, Shift all timestamps, Validate timestamps
- **Save & Apply** rewrites the `.lrc` and/or re-embeds the lyrics according to the current output mode

### Smart Auto-Tagger & Cover Art
Toggle *Auto-tag & cover art* in the workspace (or Settings).  
Supports **Fill Missing** (default) or **Overwrite**.  
Sources: **MusicBrainz** + Cover Art Archive, with fallback to the **iTunes Search API** (artwork upgraded to high resolution).  
Writes title, artist, album, album artist, genre, year, track number, and embedded cover art for supported formats.  
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

Settings are stored in:
- **Windows**: `%APPDATA%\Lyricist\settings.json`
- **Linux/macOS**: `~/.config/Lyricist/settings.json`

Use `LYRICIST_HOME` (or the older `LSL_HOME`) to make the entire data folder portable.

| Variable                                      | Purpose                     |
|-----------------------------------------------|-----------------------------|
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`         | Hugging Face token          |
| `GENIUS_ACCESS_TOKEN` / `GENIUS_TOKEN`        | Genius API token            |
| `MUSIXMATCH_API_KEY`                          | Musixmatch key              |
| `LYRICIST_WHISPER_MODEL` / `LYRICIST_LANGUAGE`| Whisper model / language    |
| `LYRICIST_NO_UPDATE_CHECK`                    | skip the startup update check for this run |

Tokens are never written into the settings file or logged.

---

## Packaging

```bash
pip install -r requirements-dev.txt
python build.py                  # one-folder build (instant start) -> dist/Lyricist/
python build.py --onefile        # single executable (slower first launch)
python build.py --clean          # wipe PyInstaller's cache first
python build.py --icons          # force-regenerate the icons first
python build.py --dry-run        # print the exact PyInstaller command, build nothing
python make_icons.py --accent 1DB954   # (optional) the same logo in another colour
```

`build.py` composes
`python -m PyInstaller --noconfirm --windowed --onedir --name Lyricist --add-data "assets:assets" --collect-all …`
(`;` instead of `:` on Windows) and adds:

* **hidden imports** for `mutagen` (and its per-format modules), `audioread` and `PySide6`;
* **`--collect-all`** for `faster_whisper`, `ctranslate2`, `onnxruntime` and `av` so their
  native libraries / data files are found by static analysis;
* **excludes** for the Qt modules the app never uses (`QtWebEngine*`, `Qt3D*`, `QtQuick`,
  `QtQml`, `QtMultimedia`, `QtCharts`, `QtDesigner`, `QtTest`), which keeps the bundle
  ~150–300 MB smaller and keeps Chromium out;
* the right icon per platform: `assets/icon.ico` (Windows), `assets/icon.icns` (macOS),
  `assets/icon.png` (Linux). `assets/logo.*` keeps the historical in-app names.

If `audio-separator` / `langdetect` / `librosa` / `soundfile` are installed in the build
environment they are bundled too (the first drags in PyTorch — expect a much larger
package; leave it out of the build environment to keep the app small). Output is
`dist/Lyricist/` (`--onedir`) or `dist/Lyricist` (`--onefile`). Build on the OS you are
targeting; PyInstaller cannot cross-compile. `assets/logo.*` and `assets/icon.*` are
committed; they are drawn by `local_sync_lyrics/branding.py`, so edit the geometry there
and re-run `python main.py --make-icons`.

### Automated release builds

`.github/workflows/release.yml` builds both desktop targets on GitHub runners. Push a
version tag and the workflow runs the test suite, builds on `windows-latest` and
`ubuntu-latest`, packages `dist/Lyricist/` as
`Lyricist-<version>-windows-x64.zip` / `Lyricist-<version>-linux-x86_64.tar.gz`, and
attaches both files to that tag's GitHub Release:

```bash
git tag -a v2.3.2 -m "Lyricist 2.3.2"
git push origin main
git push origin v2.3.2
```

Use **Run workflow** (manual dispatch) to produce the two archives as downloadable
artifacts without creating a release. The tag must point at a commit that contains the
workflow file.


---

## Project Status

Verified in a Linux sandbox (Python 3.14, PySide6 6.12, faster-whisper 1.2.1, PyAV 19, mutagen):
**287 automated tests** covering the full pipeline, GUI, theme system, forced alignment, tag
writing, vocal isolation, language locking, “Your Own Lyrics” matching, lyric editor, custom
providers, auto-tagging + cover art, and the Phase 4 history / updater / packaging work. The
real Qt window was built under the `offscreen` platform through all four views, the queue and
library grids, all five accents, dark/light/system, the lyric editor dialogs and a simulated
batch. The PyInstaller command line is checked for `--onedir`, `--onefile`, `--clean`,
`--dry-run`, hidden imports, Qt excludes and the `assets` data argument; a frozen onedir build
also launches.

Phase 4 specifics: a synthetic **v1 cache** is opened by the new `LyricsDatabase` and checked
for row preservation, `file_path` / `synced_at` backfill and the added history columns;
search/filter/count and “a later failure must not erase an earlier sync's artwork/metadata”
are covered; embedded artwork is written to a **real MP3**, read back byte-for-byte and
thumbnailed; the updater is exercised with a mocked GitHub payload (platform asset choice,
changelog, semantic-version comparison) and a mocked HTTP 404; the four-view window, queue
badges/language labels and the thumbnail grid are driven through real widgets; and a real
drag-and-drop event is delivered to the drop zone.

**Not verified in the test environment**
- Live responses from every external provider  
- Real vocal separation with actual model weights  
- Real-world Whisper accuracy on heavily produced or layered tracks  
- Windows/macOS packaging and native window-icon behaviour (the Linux build was verified)  
- A real GitHub release download / in-app self-update (the check and payload handling are mocked)  

The ≥ 40 % alignment quality gate and the 35 % “Your Own Lyrics” similarity threshold exist specifically to prevent low-confidence results from being written.

---

## License

This project is licensed under the **Apache License 2.0**.

See the [License](LICENSE) file for the full text.
