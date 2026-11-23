# Lyricist

*(Formerly “Local Sync Lyrics”. The Python package is still named `local_sync_lyrics`. On first launch this version automatically migrates your old `LocalSyncLyrics` settings, cache, and model folders to `Lyricist`.)*

Desktop application for **Windows and Linux** that generates high-quality **synced lyrics** — as standalone `.lrc` files, embedded tags inside the audio file, or both.

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

Python 3.10–3.13 recommended.  
The GUI is **native Qt 6** (PySide6) — no Flutter/Flet runtime and no embedded browser.  
FFmpeg must be available on `PATH` for audio decoding.  
Pillow is only required when building icons.

---

## Usage

```bash
python main.py                                      # Launch GUI
python main.py --cli ~/Music --no-ai                # Headless batch (rich progress)
python main.py --cli ~/Music --output both          # Write both .lrc and embedded tags
python main.py --cli song.mp3 --lyrics my_lyrics.txt # Sync one file using your own lyrics
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

Translucent glass panels over an ambient gradient, driven by a single generated Qt Style Sheet. The window is a native `QMainWindow`.

**Settings → Appearance & Theme** (applied instantly, no restart):

| Appearance                  | Accent Presets                                      |
|-----------------------------|-----------------------------------------------------|
| Dark (Liquid Obsidian)      | Apple Blue `#0A84FF` (default)                      |
| Light (Frosted Pearl)       | Spotify Green `#1DB954`                             |
| System                      | Apple Music Pink/Red `#FA243C` · Midnight Purple `#BF5AF2` · Sunset Amber `#FF9F0A` · **Custom** |

One accent colour + dark/light mode generates the entire palette (buttons, progress bar, active pills, focus rings, background and logo). Label colour automatically switches to near-black on light accents for contrast.

---

## Audio Pipeline & Output (Phase 2)

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
| Format          | What is written                                      |
|-----------------|------------------------------------------------------|
| MP3             | `USLT` (plain) + `SYLT` (millisecond timestamps)     |
| FLAC / OGG / Opus | `LYRICS` Vorbis comment containing timed LRC       |
| M4A / MP4 / M4B | `©lyr` atom containing timed LRC                     |
| Other (WAV …)   | Separate `.lrc` file (with warning)                  |

Tags are written only after validation, read back for verification, and failures are reported cleanly. Existing lyrics are skipped unless **Overwrite** is enabled.

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

Tokens are never written into the settings file or logged.

---

## Packaging

```bash
pip install -r requirements-dev.txt
python build.py                 # Generate icons if needed → PyInstaller → ./dist
python build.py --icons         # Force icon regeneration
python make_icons.py --accent 1DB954   # Optional: logo in another colour
```

Output is placed in `dist/Lyricist/`. Icons are generated from geometry defined in `local_sync_lyrics/branding.py`.

---

## Project Status

**228 automated tests** covering the pipeline, GUI, theme system, forced alignment, tag writing, vocal isolation contract, language locking and “Your Own Lyrics” matching. Real MP3/FLAC/Opus/M4A files were written and verified with mutagen. Alignment accuracy under simulated recognition errors remains extremely high (≥ 98 % of lines within 0.7 s on typical cases).

**Not verified in the test environment**
- Live responses from every external provider  
- Real vocal separation with actual model weights  
- Real-world Whisper accuracy on heavily produced or layered tracks  
- Full Windows packaging and native window-icon behaviour  

The ≥ 40 % alignment quality gate and the 35 % “Your Own Lyrics” similarity threshold exist specifically to prevent low-confidence results from being written.

---

## License

[Add your license here]
```
