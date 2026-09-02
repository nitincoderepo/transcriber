"""
Gemini Audio Transcriber Package.

Modular audio transcription engine with smart FFmpeg slicing, checkpoint resuming,
deduplication, and Word/text document generation.
"""

import sys

# Ensure UTF-8 console output across all platforms
if hasattr(sys.stdout, "reconfigure") and getattr(sys.stdout, "encoding", "") != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

from .config import TranscriberConfig, resolve_api_key, load_env_file
from .audio import (
    AudioSlice, get_audio_duration, slice_audio_file, format_duration,
    parse_time_str, detect_audio_volume, is_audio_silent, SUPPORTED_AUDIO_EXTENSIONS
)
from .checkpoint import CheckpointManager
from .engine import transcribe_slice, build_prompt, print_quota_exhausted_guide, is_daily_limit_error
from .document import save_document
from .discovery import discover_audio_files, clean_title_from_filename
from .pipeline import process_audio_file, run_pipeline
from .logger import TranscriptLogger, get_logger, format_wait_time, get_progress_bar
from .cost import calculate_cost, format_cost, TokenUsageSummary, SessionTracker, FileRunRecord, LocalUsageLedger

__all__ = [
    "TranscriberConfig",
    "resolve_api_key",
    "load_env_file",
    "AudioSlice",
    "get_audio_duration",
    "slice_audio_file",
    "format_duration",
    "parse_time_str",
    "detect_audio_volume",
    "is_audio_silent",
    "SUPPORTED_AUDIO_EXTENSIONS",
    "CheckpointManager",
    "transcribe_slice",
    "build_prompt",
    "print_quota_exhausted_guide",
    "is_daily_limit_error",
    "save_document",
    "discover_audio_files",
    "clean_title_from_filename",
    "process_audio_file",
    "run_pipeline",
    "TranscriptLogger",
    "get_logger",
    "format_wait_time",
    "get_progress_bar",
    "calculate_cost",
    "format_cost",
    "TokenUsageSummary",
    "SessionTracker",
    "FileRunRecord",
    "LocalUsageLedger",
]
