# Local Sync Lyrics

Desktop app (Windows + Linux) that finds **synced `.lrc` lyrics** for a local music
folder. Per track it tries **LRCLIB → BetterLyrics → Megalobiz**, and if none has a
believable match it transcribes the song locally with **faster-whisper**. Results are
cached in SQLite so re-runs only touch new files.

```
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py                                          # GUI
python main.py --cli ~/Music --no-ai                    # headless (rich progress)
```

Python 3.10–3.13 recommended (3.12 has the widest wheel coverage for CTranslate2/ONNX).

## Project layout

| File | Responsibility |
|---|---|
| `main.py` | Entry point; safe stdout for `--noconsole` builds; GUI or `--cli` |
| `local_sync_lyrics/config.py` | Typed `Settings` ⇄ JSON in the user's app-data dir, `os.environ` fallback, runtime `HF_TOKEN` injection |
| `local_sync_lyrics/database.py` | SQLite cache keyed by a rename-proof file fingerprint; self-heals if corrupt |
| `local_sync_lyrics/text_utils.py` | Filename/tag sanitiser (YouTube `[ID]`, `(Official Video)`, track numbers, `Artist - ` prefixes) |
| `local_sync_lyrics/api_providers.py` | Provider chain, fuzzy scoring (`thefuzz`), duration checks, retries/throttling |
| `local_sync_lyrics/lrc_utils.py` | LRC parse/validate/build, TTML→LRC, atomic writes |
| `local_sync_lyrics/transcriber.py` | Whisper fallback: defensive PyAV decoding + GPU→CPU(int8) fallback |
| `local_sync_lyrics/hardware_profiler.py` | RAM/CPU/GPU/cuBLAS probe → recommended model |
| `local_sync_lyrics/processor.py` | Batch runner with a `try/except` boundary around every file |
| `local_sync_lyrics/gui.py` | Flet 1.x UI (worker thread → queue → async UI pump) |
| `local_sync_lyrics/cli.py` | Headless runner using `rich` |
| `build.py` | `flet pack` (PyInstaller) with the right `--collect-all` flags |
| `tests/test_core.py` | 39 offline tests |

## Behaviour worth knowing

* **Existing `.lrc` files are never overwritten** unless you enable *Overwrite*.
* A cached "not found/failed" track is retried automatically after 7 days, or right
  away if you newly enable the AI fallback or tick *Retry failed*.
* Online candidates are rejected if their last timestamp is longer than the track, if
  duration differs by >20 s, or if fuzzy artist/title score is below the threshold.
* Whisper output is cleaned of typical hallucinations ("Thanks for watching",
  `[Music]`, endless repeated lines).
* If the Whisper model can't be loaded (e.g. offline on first run) it is reported once
  and the rest of the batch fails fast instead of retrying the download per file.

## Settings & environment variables

Stored in `%APPDATA%\LocalSyncLyrics\settings.json` (Windows) or
`~/.config/LocalSyncLyrics/settings.json` (Linux). Cache, models and logs live in
`%LOCALAPPDATA%\LocalSyncLyrics` / `~/.local/share/LocalSyncLyrics`.
Set `LSL_HOME=/some/folder` for a fully portable install.

| Variable | Used when settings.json has no value |
|---|---|
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | Hugging Face token (never written back to the file) |
| `LSL_WHISPER_MODEL` | Whisper model (`auto`, `tiny` … `large-v3`) |
| `LSL_LANGUAGE` | `auto` or an ISO code such as `en`, `es`, `ja` |

The token is stored in plain text in `settings.json` (chmod 600 on Linux). Leave the
field empty to rely on the environment variable instead.

## Packaging

```
pip install -r requirements-dev.txt
python build.py            # runs `flet pack` -> ./dist
```

Build on the OS you target. On Windows it creates a one-folder build (a one-file build
would unpack several hundred MB on every launch). Expect a large bundle: Flet's client,
CTranslate2, ONNX Runtime and FFmpeg (PyAV) are included. Whisper models are **not**
bundled; they download once into the data folder.

## Troubleshooting

* **"Folder dialog unavailable" on Linux** – install `zenity` (or paste the path).
* **GPU not used** – the log shows why. Usually CUDA 12 cuBLAS is missing: install the
  NVIDIA CUDA 12 runtime or `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`. The app
  falls back to CPU automatically either way.
* **`metadata_errors` TypeError** – that was faster-whisper's own decoder disagreeing
  with the installed PyAV. This app decodes audio itself, so no `av` pin is needed.
* **Megalobiz returns nothing** – it is HTML scraping and may block or change layout;
  uncheck it in Settings, the other providers are unaffected.

## What has and hasn't been verified

Verified in a Linux sandbox with Flet 1.1.2, faster-whisper 1.2.1, PyAV 19.0.2:
39 offline tests (sanitiser, LRC/TTML, settings, cache corruption recovery, real audio
decoding incl. the `metadata_errors` TypeError, simulated cuBLAS failure → CPU retry,
provider parsing against canned payloads, batch isolation/skip/cancel, GUI controls
built and serialised with Flet's wire encoder, worker-thread → UI event flow) and a
real `flet pack` build whose executable ran the pipeline in `--cli` mode.

**Not** verified (no network or display in the sandbox): live responses from LRCLIB,
BetterLyrics and Megalobiz, real Whisper inference and model downloads, a CUDA GPU, and
the rendered desktop window / Windows build. Expect to tweak provider details if a
service has changed its API.
