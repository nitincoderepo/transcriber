# 🎙️ Gemini Audio Transcription Engine

A scalable, modular, secure, and fault-tolerant audio transcription engine powered by **Google Gemini AI (Gemini 3.7 / 2.5)** and **FFmpeg**. Transcribes large or small audio and video recordings into beautifully formatted, structured Microsoft Word (`.docx`) documents and text files with verbatim accuracy, speaker labels, continuous minute markers, and exact acoustic grounding.

---

## 🌟 Key Features

- **🚀 Powered by Gemini 3.7 Flash**: Uses `gemini-3.7-flash` (or `gemini-2.5-flash-lite`) for fast, accurate, and cost-effective verbatim transcriptions.
- **⏱️ Custom Time-Range Transcription (`--start`, `--end`, `--range`)**: Transcribe specific intervals (e.g. `21:00` to `35:00` or `21m-35m`) while preserving absolute in-text timestamps (`[21:00]`, `[23:00]`...) aligned with the original recording.
- **🔇 Local Silence Pre-Filtering ($0 API Cost)**: Automatically inspects audio volume peak/mean dB locally using FFmpeg before uploading to Gemini. Silent or non-vocal sections are marked as `[સંગીત / મૌન / Silence]` without making paid API calls.
- **🎯 Zero-Temperature Anti-Hallucination Guardrails**: Deterministic decoding (`temperature=0.0`) and strict prompt grounding rules prevent the model from autocompleting or altering Sanskrit shlokas, stotrams, and mantras from memory.
- **🔪 Smart 10-Minute Slicing**: Automatically probes audio length via `ffprobe` and slices files longer than 10 minutes into consecutive parts via `ffmpeg`.
- **⚡ Slice Caching**: Reuses existing slice files on disk—no redundant `ffmpeg` slicing operations.
- **💾 Fault-Tolerant Resuming (Checkpointing)**: If interrupted mid-way (e.g. at Part 7 of 10), re-running the script automatically **loads completed parts from disk without re-calling Gemini or wasting tokens**, resuming directly from where it stopped.
- **⏩ Smart Deduplication**: Automatically skips audio files that have already been transcribed into `.docx` (use `--force` to overwrite).
- **🎵 Wide Audio & Video Format Support**: Works natively with `.mp3`, `.m4a`, `.mp4`, `.wav`, `.aac`, `.flac`, `.ogg`, `.wma`, `.opus`, `.m4b`, `.webm`, `.m4v`, `.mov`, `.mkv`, `.avi`, `.3gp`.
- **💰 Token & Cost Tracking**: Displays real-time token breakdown (prompt/candidate) and exact USD billing estimates per slice and batch total.
- **⚙️ 100% Configurable via `.env`**: Configure API keys, models, chunk durations, languages, and output directories in a single `.env` file without modifying code.
- **🔒 Secure by Design**: Zero hardcoded API keys; `.gitignore` is pre-configured to keep secrets, temporary slices, transcripts, logs, and media files safe.

---

## 📁 Modular Package Structure

```
Audacity/
├── transcriber/
│   ├── __init__.py          # Package exports & public API
│   ├── config.py            # .env loader, API key resolver, TranscriberConfig dataclass
│   ├── checkpoint.py        # CheckpointManager for fault-tolerant state resuming
│   ├── audio.py             # Audio probing (ffprobe), volume detection, parsing, and slicing (ffmpeg)
│   ├── engine.py            # Gemini API client, prompt generator, zero-temp grounding, retry handler
│   ├── document.py          # Word (.docx) and plain text (.txt) document generator
│   ├── discovery.py         # Audio file discovery and filename title normalizer
│   ├── logger.py            # Transcript file logging and visual progress bar
│   ├── cost.py              # Token tracking and USD pricing calculation
│   └── pipeline.py          # Orchestration pipeline for single & batch audio processing
│
├── tests/                   # Comprehensive automated unit test suite (19 tests)
│   ├── test_audio.py        # Duration, volume detection, time parser, discovery tests
│   ├── test_config.py       # .env and CLI argument override tests
│   ├── test_checkpoint.py   # Checkpoint caching and recovery tests
│   └── test_logger_and_resilience.py # Retry handling and cost calculation tests
│
├── transcribe.py            # Main CLI entrypoint
├── .env.example             # Configuration template
├── .env                     # Active configuration (git-ignored)
├── .gitignore               # Git safety rules
└── README.md                # Full documentation
```

---

## 📋 Prerequisites & Setup

### 1. System Requirements
- **Python 3.10+** (`python3 --version` or `python --version`)
- **FFmpeg & FFprobe**: Required for audio length probing, volume detection, and slicing.

---

### 2. Platform Setup Guides

#### 🍎 macOS (Apple Silicon M1/M2/M3/M4 & Intel)

1. **Install FFmpeg via Homebrew**:
   ```bash
   brew install ffmpeg

   # Verify installation
   ffmpeg -version
   ffprobe -version
   ```

2. **Set Up Python Virtual Environment (`venv`)**:
   ```bash
   # Create virtual environment
   python3 -m venv .venv

   # Activate virtual environment
   source .venv/bin/activate

   # Install dependencies inside venv
   pip3 install -r requirements.txt
   ```

---

#### 🪟 Windows Setup

1. **Install FFmpeg**:
   - Using **winget**: `winget install Gyan.FFmpeg`
   - Or using **Chocolatey**: `choco install ffmpeg`
   - Or download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) and add the `bin` folder to your system `PATH`.

2. **Set Up Virtual Environment**:
   ```powershell
   # In PowerShell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

---

#### 🐧 Linux Setup

```bash
sudo apt update && sudo apt install -y ffmpeg python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Quick Start Guide

### Step 1: Configure `.env`
Copy `.env.example` to create your active `.env` file:
```bash
# macOS / Linux
cp .env.example .env

# Windows (PowerShell)
Copy-Item .env.example .env
```

Open `.env` in any text editor and set your **Gemini API Key** (get one free at [Google AI Studio](https://aistudio.google.com/)):
```env
GEMINI_API_KEY=your_actual_api_key_here
GEMINI_MODEL=gemini-3.7-flash
CHUNK_MINUTES=10
```

---

### Step 2: Run Transcription

```bash
# 1. Preview planned slices and file discovery without calling API (Dry Run):
python transcribe.py --dir "C:\Users\name\Documents\audioFiles" --dry-run

# 2. Transcribe all audio files in a folder:
python transcribe.py --dir "C:\Users\name\Documents\audioFiles"

# 3. Transcribe a specific audio file:
python transcribe.py "C:\Users\name\Documents\audioFiles\AudioFile.mp3"
```

---

## ⏱️ Custom Time-Range Transcription (`--start`, `--end`, `--range`)

You can transcribe a specific time segment from a long recording. Timestamps inside the transcript will preserve the real timeline (e.g. `[21:00]`, `[23:00]`...):

```powershell
# Using explicit start and end times (supports MM:SS, HH:MM:SS, minutes '21m', or seconds '1260s'):
python transcribe.py "AudioFile.mp3" --start "21:00" --end "35:00"

# Using shorthand range:
python transcribe.py "AudioFile.mp3" --range "21:00-35:00"
python transcribe.py "AudioFile.mp3" --range "21m-35m"

# Preview time-range slice without calling API:
python transcribe.py "AudioFile.mp3" --range "21:00-35:00" --dry-run
```

Output for time segments is uniquely labeled so it will never overwrite a full-file transcript:
- **Title:** `AudioFile (Segment 21:00 - 35:00)`
- **Output File:** `AudioFile_Segment_21-00_to_35-00_Transcript.docx`

---

## ⚙️ Configuration Reference (`.env`)

You can customize defaults by editing your `.env` file:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | *(Required)* | Your Google Gemini API key. |
| `GEMINI_MODEL` | `gemini-3.7-flash` | Gemini model name (`gemini-3.7-flash`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.5-pro`). |
| `CHUNK_MINUTES` | `10` | Duration in minutes for each slice. Set to `0` to disable slicing. |
| `LANGUAGE` | `Gujarati script (ગુજરાતી લિપિ)` | Target language / script for verbatim transcription. |
| `TIMESTAMP_INTERVAL` | `2` | Interval in minutes for minute markers (e.g. `2`, `5`). Set to `0` to disable. |
| `OUTPUT_DIR` | *(Blank)* | Folder to save `.docx` files (defaults to `transcript/` in audio directory). |
| `SAVE_TXT` | `false` | Set to `true` to also export a `.txt` file alongside the `.docx`. |
| `KEEP_CHUNKS` | `false` | Set to `true` to retain temporary sliced audio files on disk. |
| `MAX_RETRY_MINUTES` | `30` | Configurable maximum retry wait duration in minutes for temporary demand spikes (503). |
| `CUSTOM_INSTRUCTIONS` | *(Blank)* | Optional custom prompt instructions / glossary added to the Gemini prompt. |

---

## 💻 Command-Line Reference & CLI Switches

```bash
usage: transcribe.py [-h] [-d INPUT_DIR] [-o OUTPUT_DIR] [-r]
                     [-p PATTERN] [-f FILTER] [--force]
                     [-s START_TIME] [-e END_TIME] [--range TIME_RANGE]
                     [-c CHUNK_MINUTES] [--no-slice] [--keep-chunks]
                     [-w MAX_RETRY_MINUTES]
                     [-l LANGUAGE] [-m MODEL] [-t TIMESTAMP_INTERVAL]
                     [--no-timestamps] [--instructions INSTRUCTIONS]
                     [-k API_KEY] [--save-txt] [--dry-run]
                     [files ...]
```

### Complete List of Switches and Arguments

| Switch / Option | Aliases | Type / Format | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `-h` | `--help` | Flag | `None` | Display help message, usage syntax, and exit. |
| `files` | *(Positional)* | File path(s) | `None` | Optional specific audio/video file(s) to transcribe. |
| `-d` | `--dir`, `--input-dir` | Directory path | Current folder | Folder to search for audio files. |
| `-o` | `--output-dir` | Directory path | `transcript/` | Folder to save generated Word docs & text files. |
| `-r` | `--recursive` | Flag | `False` | Search subdirectories recursively for audio files. |
| `-p` | `--pattern` | Glob string | `*` | Glob pattern to match files (e.g. `*.m4a`, `*Part*.mp3`). |
| `-f` | `--filter` | Substring | `None` | Case-insensitive substring filter for filenames (e.g. `'Part 1'`). |
| `--force` | `--overwrite` | Flag | `False` | Force re-transcription even if output `.docx` already exists. |
| **`-s`** | **`--start`**, **`--start-time`** | `MM:SS` / `21m` / `sec` | `None` | **Start time offset** for custom segment transcription. |
| **`-e`** | **`--end`**, **`--end-time`** | `MM:SS` / `35m` / `sec` | `None` | **End time offset** for custom segment transcription. |
| **`--range`** | **`--time-range`** | `21:00-35:00` / `21m-35m` | `None` | **Shorthand time range** for segment transcription. |
| `-c` | `--chunk-minutes` | Number (minutes) | `10` | Slicing chunk length in minutes. Set to `0` to disable slicing. |
| `--no-slice` | | Flag | `False` | Disable audio slicing (upload entire file regardless of length). |
| `--keep-chunks` | | Flag | `False` | Retain local temporary sliced audio files and checkpoints. |
| `-w` | `--max-retry-minutes` | Number (minutes) | `30` | Maximum retry wait budget for temporary API demand spikes (503). |
| `-l` | `--language` | String | `.env` default | Target language/script prompt (e.g. `Gujarati`, `Hindi`, `English`). |
| `-m` | `--model` | String | `gemini-3.7-flash` | Gemini model name (`gemini-3.7-flash`, `gemini-2.5-flash-lite`, etc.). |
| `-t` | `--timestamp-interval` | Integer (minutes) | `2` | Interval in minutes for timeline minute markers (e.g. `2`, `5`). |
| `--no-timestamps` | | Flag | `False` | Disable minute markers in the generated transcript. |
| `--instructions` | | String | `None` | Additional custom instructions or glossary for the Gemini prompt. |
| `-k` | `--api-key` | String | `.env` key | Override Gemini API Key directly from CLI. |
| `--save-txt` | | Flag | `False` | Also export transcript as `.txt` file alongside the `.docx`. |
| `--starting-balance` | `--credits` | Float ($ USD) | `None` | Optional initial credit balance to track remaining budget in summary. |
| `--dry-run` | | Flag | `False` | Preview planned slices, file discovery, and paths without API calls. |

---

## 📊 Run Summary & Local Usage Ledger

On **normal completion** or if you interrupt with **`Ctrl + C`** (SIGINT), the engine displays a comprehensive summary table with exact processing times, token metrics, and all-time usage:

```
========================================================================================================================
📊 TRANSCRIPTION RUN SUMMARY (🎉 COMPLETED / 🛑 INTERRUPTED)
========================================================================================================================
📁 Target Folder:      C:\Users\name\Downloads\audioFiles
📅 Session Started:   2026-09-01 22:55:10
📅 Session Ended:     2026-09-01 22:58:34 (Total Runtime: 3m 24s)
🤖 Gemini Model:       gemini-3.7-flash (temperature=0.0)

------------------------------------------------------------------------------------------------------------------------
#   File / Segment                   Clip Start  Clip End   Duration  Time Taken   Tokens     Cost ($)   Status
------------------------------------------------------------------------------------------------------------------------
1   Track 11 (Part 01)               00:00       10:00      10:00     27.2s        20,137     $0.0141    ✅ Completed
2   Track 11 (Part 02)               10:00       20:00      10:00     28.5s        20,450     $0.0143    ✅ Completed
3   AudioFile (Segment)              21:00       35:00      14:00     31.2s        28,300     $0.0198    ✅ Completed
4   Track 12.mp3                     00:00       45:00      45:00      0.1s             0     $0.0000    ⚡ Skipped (Exists)
5   Silence_Track.mp3                00:00       10:00      10:00      0.3s             0     $0.0000    🔇 Silent ($0)
------------------------------------------------------------------------------------------------------------------------

📈 CURRENT RUN TOTALS:
  • Audio Processed:         34:00 (34 mins 00 secs)
  • Active Processing Time:  1m 27s (5 item(s) processed)
  • Tokens Consumed:         68,887 (64,200 input + 4,687 output)
  • Current Run Cost:        ~$0.0482 USD

💰 ALL-TIME CUMULATIVE USAGE (Auto-Tracked in Local Ledger):
  • Lifetime Audio Done:     1h 45m (105 mins)
  • Lifetime Total Tokens:   215,400 tokens
  • Lifetime Incurred Cost:  ~$0.1508 USD
========================================================================================================================
```

> **🔒 Git Privacy:** The central `.usage_ledger.json` file is kept locally on your machine and is automatically ignored by git.

---

### Practical Command Examples

- **Transcribe an entire directory**:
  ```powershell
  python transcribe.py --dir "C:\Users\name\Downloads\audioFiles"
  ```
- **Filter files containing specific text**:
  ```powershell
  python transcribe.py --dir "C:\Users\name\Downloads\audioFiles" --filter "Part 1"
  ```
- **Transcribe a custom segment (21m to 35m)**:
  ```powershell
  python transcribe.py "AudioFile.mp3" --range "21:00-35:00"
  ```
- **Transcribe with explicit start and end offsets**:
  ```powershell
  python transcribe.py "AudioFile.mp3" --start 21m --end 35m
  ```
- **Dry-run preview before executing API calls**:
  ```powershell
  python transcribe.py --dir "C:\Users\name\Downloads\audioFiles" --range "10m-25m" --dry-run
  ```
- **Force re-transcribe an existing file**:
  ```powershell
  python transcribe.py "AudioFile.mp3" --force
  ```
- **Export both `.docx` and `.txt` with 5-minute minute markers**:
  ```powershell
  python transcribe.py --timestamp-interval 5 --save-txt "AudioFile.mp3" 
  ```

---

## 🧪 Running Automated Tests

Run the test suite anytime to verify all components (audio probing, volume detection, time parser, resilience, cost calculations):

```powershell
python -m unittest discover tests
```

---

## ❓ Frequently Asked Questions (FAQ)

### How does the Checkpoint System work?
All successfully transcribed slices are saved in `_temp_chunks/<file_name>/checkpoint.json`. If execution is stopped or interrupted, re-running the script immediately loads completed parts from local cache, saving 100% of the API tokens and cost for finished slices.

### How does Local Silence Detection save money?
Before uploading any slice to Gemini, FFmpeg runs a local `volumedetect` check. If a slice has no audible vocal speech (e.g. ambient background or silence below -45 dB), it tags the slice as `[સંગીત / મૌન / Silence]` directly on disk at **$0 API cost**.

### How do I prevent Sanskrit shloka hallucinations?
The engine uses **`temperature=0.0`** for deterministic acoustic decoding, combined with explicit prompt rules preventing the model from autocompleting known verses from memory.

---

## 🔒 Security & Git Best Practices
- Never commit your `.env` file to any repository.
- The repository's [`.gitignore`](.gitignore) is pre-configured to ignore `.env`, `_temp_chunks/`, `transcript/`, audio/video files, and python caches.
