# Lyricist

**Synced `.lrc` lyrics generator for your music library**

Lyricist is a desktop application for Windows and Linux that automatically finds or generates high-quality synchronized lyrics for your audio files. It prioritizes official synced sources, handles mashups intelligently, and falls back to forced alignment or blind transcription only when necessary.

---

## What It Does

Lyricist follows a clear priority pipeline:

1. **Synced providers** → Save ready-to-use `.lrc` files  
2. **Mashup detection** → Split titles like “A x B”, “A / B”, “A & B (Mashup)”, “A vs B” and process each song  
3. **Plain-text providers** → Fetch official lyrics  
4. **Forced alignment** → Align official text to the audio using Whisper word timestamps  
5. **Blind transcription** → Only when no reliable text source exists

The result is a properly timed `.lrc` file written next to your music, using official spelling whenever possible.

---

## Features

- Multi-tier lyrics pipeline with automatic fallbacks
- Intelligent mashup detection and component resolution
- Forced alignment of official lyrics to audio
- Blind listening mode for unidentified tracks
- Quality gates that reject low-confidence results
- Modern Liquid Glass UI (dark / light / system themes + multiple accent colors)
- Headless CLI mode for batch processing
- SQLite caching, atomic writes, and per-file error isolation
- Portable mode via `LSL_HOME`

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10–3.13 recommended.

---

## Usage

**GUI**
```bash
python main.py
```

**CLI / Headless**
```bash
python main.py --cli ~/Music --no-ai
```

The `--no-ai` flag disables all Whisper-based steps.

---

## How the Pipeline Works

### 1. Provider Chain
Nine providers are tried in two tiers. Results are classified as `SYNCED` (saved directly) or `PLAIN_TEXT` (sent to alignment). Candidates are fuzzy-scored on artist, title, and duration, then validated.

### 2. Mashup Detection
Titles and filenames are split on common separators (`x`, `×`, `vs`, `/`, `+`, `→`, and `&`/`and` when mashup keywords are present). High-confidence matches go straight to the component pipeline. Unresolved components can be identified by listening to short audio slices and searching lyric databases (only accepted if sung words match the candidate lyrics).

### 3. Forced Alignment
- Audio is processed in overlapping 30-second windows
- Official lyric lines are used as prompts for faster-whisper (with word timestamps)
- Recognised words are matched back to the official text
- Handles repeated choruses, cut sections, and interleaved mashups
- Quality gate requires lyrics to explain ≥ 40 % of sung words

### 4. Blind Mode
Used only when no text source can be found or alignment confidence is too low. Whisper transcribes the track and writes a plain `.lrc`.

---

## Interface – Liquid Glass

Frosted-glass design with ambient lighting, smooth animations, and a sidebar switching between **Sync Workspace** and **Settings**.

**Appearance options** (applied instantly, no restart required):

| Mode                  | Accent Presets                                      |
|-----------------------|-----------------------------------------------------|
| Dark (Liquid Obsidian)| Apple Blue (default) · Spotify Green · Apple Music Pink/Red |
| Light (Frosted Pearl) | Midnight Purple · Sunset Amber · Custom hex         |
| System                | Follows OS theme                                    |

All colors are derived from a single accent. Settings are stored in `settings.json` and survive upgrades.

---

## Providers

| Tier | Provider       | Requirements     | Notes                                      |
|------|----------------|------------------|--------------------------------------------|
| 1    | LRCLIB         | –                | JSON API, exact + fuzzy search             |
| 1    | BetterLyrics   | –                | TTML → LRC                                 |
| 1    | RCLyricsBand   | –                | Form search + HTML scrape                  |
| 1    | Megalobiz      | –                | HTML scrape                                |
| 1    | NetEase        | –                | Public web API                             |
| 1    | QQ Music       | –                | Public web API                             |
| 2    | Genius         | Optional token   | Official API or public search + scrape     |
| 2    | Lyrics.ovh     | –                | Direct lookup + suggestions                |
| 2    | Musixmatch     | API key          | Free plan returns excerpts (flagged)       |

---

## Configuration

Settings location:
- **Windows**: `%APPDATA%\LocalSyncLyrics\settings.json`
- **Linux/macOS**: `~/.config/LocalSyncLyrics/settings.json`

Environment variables (used when not set in settings):

| Variable                              | Purpose                  |
|---------------------------------------|--------------------------|
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | Hugging Face token       |
| `GENIUS_ACCESS_TOKEN` / `GENIUS_TOKEN`| Genius API token         |
| `MUSIXMATCH_API_KEY`                  | Musixmatch key           |
| `LSL_WHISPER_MODEL`                   | Whisper model name       |
| `LSL_LANGUAGE`                        | Forced language          |
| `LSL_HOME`                            | Portable data directory  |

Tokens are never written to the settings file or logged.

Available toggles:
- Forced alignment
- Detect mashups
- Name unknown songs by listening

---

## Building & Packaging

```bash
pip install -r requirements-dev.txt
python build.py                 # Generate icons if needed, then package
python build.py --icons         # Force icon regeneration
```

Icons are generated from geometry defined in `local_sync_lyrics/branding.py`. Create a differently colored logo with:

```bash
python make_icons.py --accent 1DB954
```

---

## Project Status

Core pipeline, GUI, theme system, and forced-alignment logic have been extensively tested (124 automated tests + manual desktop client verification).

**Verified**
- Full provider fallback chain
- Mashup detection and component resolution
- Forced alignment under simulated recognition errors
- Theme system (all presets, dark/light/system, live switching)
- Settings persistence and off-UI-thread operations
- Graceful degradation when models or network are unavailable

**Limitations of the test environment**
- Live responses from every external provider were not tested
- Real-world Whisper accuracy on heavily produced tracks depends on audio quality
- Windows-specific window icon behavior not verified in the sandbox

The ≥ 40 % alignment quality gate prevents low-confidence results from being written.

---

## License

[Add your license here]
```