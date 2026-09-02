#!/usr/bin/env python3
"""
Scalable & Secure Audio Transcription Engine with Smart FFmpeg Slicing & Resuming
Powered by Google Gemini & Python-Docx

CLI Entry Point.
"""

import sys
import argparse
from pathlib import Path

# Ensure UTF-8 console output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

from transcriber import TranscriberConfig, run_pipeline, load_env_file


def parse_arguments() -> argparse.Namespace:
    load_env_file()
    
    parser = argparse.ArgumentParser(
        description="Scalable & Secure Audio Transcription Engine with Smart FFmpeg Slicing & Gemini AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe all audio files in current folder (uses settings from .env):
  python transcribe.py

  # Transcribe files in a specific directory:
  python transcribe.py --dir "C:/Users/name/Documents/audioFiles"

  # Transcribe specific audio file(s):
  python transcribe.py "AudioFile.mp3"

  # Force re-transcribing even if Word doc exists:
  python transcribe.py --force

  # Dry run to preview file durations, slices, and skip status:
  python transcribe.py --dry-run
        """
    )

    # File / Folder Arguments
    parser.add_argument(
        "files",
        nargs="*",
        help="Optional specific audio file path(s) to transcribe."
    )
    parser.add_argument(
        "-d", "--dir", "--input-dir",
        dest="input_dir",
        type=str,
        default=None,
        help="Directory to search for audio files (defaults to current working directory)."
    )
    parser.add_argument(
        "-o", "--output-dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Directory to save generated Word documents (defaults to input directory or .env)."
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Scan subdirectories recursively for audio files."
    )
    parser.add_argument(
        "-p", "--pattern",
        type=str,
        default=None,
        help="Glob pattern to match audio files (e.g. '*.mp3', '*Day*.wav')."
    )
    parser.add_argument(
        "-f", "--filter",
        type=str,
        default=None,
        help="Case-insensitive substring filter for filenames (e.g. 'Day 1')."
    )
    parser.add_argument(
        "--force", "--overwrite",
        action="store_true",
        help="Force re-transcription even if the output transcript document already exists."
    )

    # Time Range / Segment Arguments
    parser.add_argument(
        "-s", "--start", "--start-time",
        dest="start_time",
        type=str,
        default=None,
        help="Start time offset for transcription (e.g. '21:00', '01:21:00', '21m', '1260s')."
    )
    parser.add_argument(
        "-e", "--end", "--end-time",
        dest="end_time",
        type=str,
        default=None,
        help="End time offset for transcription (e.g. '35:00', '35m', '2100s')."
    )
    parser.add_argument(
        "--range", "--time-range",
        dest="time_range",
        type=str,
        default=None,
        help="Time range for transcription (e.g. '21:00-35:00', '21m-35m')."
    )

    # Slicing Arguments
    parser.add_argument(
        "-c", "--chunk-minutes",
        type=float,
        default=None,
        help="Slice long audio files into chunks of this duration in minutes (default from .env: 10). Set to 0 to disable slicing."
    )
    parser.add_argument(
        "--no-slice",
        action="store_true",
        help="Disable audio slicing (upload entire files regardless of length)."
    )
    parser.add_argument(
        "--keep-chunks",
        action="store_true",
        default=None,
        help="Retain local temporary sliced audio files and checkpoints."
    )
    parser.add_argument(
        "-w", "--max-retry-minutes",
        type=float,
        default=None,
        help="Maximum retry wait budget in minutes for temporary demand spikes (default from .env: 30)."
    )

    # Gemini & Transcription Options
    parser.add_argument(
        "-l", "--language",
        type=str,
        default=None,
        help="Target language / script for verbatim transcription (default from .env)."
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Gemini model name to use (default from .env: 'gemini-3.7-flash')."
    )
    parser.add_argument(
        "-t", "--timestamp-interval",
        type=int,
        default=None,
        help="Interval in minutes for embedding timestamps / minute markers (default from .env: 2). Set to 0 to disable."
    )
    parser.add_argument(
        "--no-timestamps",
        action="store_true",
        help="Disable timestamps in the generated transcript."
    )
    parser.add_argument(
        "--instructions",
        type=str,
        default=None,
        help="Additional custom prompt instructions for Gemini."
    )
    parser.add_argument(
        "-k", "--api-key",
        type=str,
        default=None,
        help="Gemini API Key (overrides GEMINI_API_KEY environment variable and .env file)."
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        default=None,
        help="Also export transcript as a plain .txt file alongside the .docx document."
    )
    parser.add_argument(
        "--starting-balance", "--credits",
        dest="starting_balance",
        type=float,
        default=None,
        help="Initial API credit balance in USD (e.g. 50.00) to track remaining balance in run summary."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview files to process, durations, slices, and output paths without calling Gemini API."
    )

    return parser.parse_args()


def sanitize_arg_path(p: Optional[str]) -> Optional[str]:
    """Strip smart quotes, regular quotes, and extra whitespace from path arguments."""
    if not p:
        return None
    return p.strip(" \t\n\r'\"“”‘’")


def main():
    args = parse_arguments()
    args_dict = vars(args)

    raw_input_dir = sanitize_arg_path(args.input_dir)
    target_files = [sanitize_arg_path(f) for f in args.files] if args.files else []

    # Auto-heal: If smart quotes in shell broke a path with spaces into multiple arguments
    # (e.g., --dir ”/Users/name/Path with spaces”)
    if raw_input_dir and target_files:
        candidate_dir_str = " ".join([args.input_dir] + args.files)
        candidate_dir_str = sanitize_arg_path(candidate_dir_str)
        candidate_path = Path(candidate_dir_str).expanduser() if candidate_dir_str else None
        if candidate_path and candidate_path.exists():
            raw_input_dir = candidate_dir_str
            target_files = []

    config = TranscriberConfig.from_env_and_args(args_dict)
    input_path = Path(raw_input_dir).expanduser().resolve() if raw_input_dir else Path.cwd()

    run_pipeline(
        target_files=target_files if target_files else None,
        input_dir=input_path,
        pattern=args.pattern,
        name_filter=args.filter,
        recursive=args.recursive,
        config=config
    )


if __name__ == "__main__":
    main()
