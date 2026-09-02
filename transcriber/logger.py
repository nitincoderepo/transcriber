"""
Logging, progress formatting, and clean console output management for audio transcription.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Optional


def format_wait_time(seconds: float) -> str:
    """
    Format wait seconds into human-readable representation:
    - Less than 60s: '45 seconds' (or '1 second')
    - 60s or more: '2 minutes 15 seconds' or '3 minutes'
    """
    secs = int(round(seconds))
    if secs < 60:
        return f"{secs} second" if secs == 1 else f"{secs} seconds"
    
    mins = secs // 60
    rem_secs = secs % 60
    
    min_str = f"{mins} minute" if mins == 1 else f"{mins} minutes"
    if rem_secs == 0:
        return min_str
    return f"{mins} min {rem_secs} sec"


def get_progress_bar(current: int, total: int, width: int = 16) -> str:
    """Generate a clean ASCII progress bar with percentage."""
    if total <= 0:
        return ""
    pct = min(100.0, max(0.0, (current / total) * 100.0))
    filled = int(round((current / total) * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct:5.1f}%"


class TranscriptLogger:
    """
    Dual-destination logger:
    - Writes technical details, HTTP payloads, and tracebacks to '<output_dir>/transcription.log'
    - Prints clean, structured, user-friendly status updates to console
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir
        self.log_file: Optional[Path] = None
        self._file_logger: Optional[logging.Logger] = None
        self._handler: Optional[logging.FileHandler] = None
        
        if output_dir:
            self.setup_file_logging(output_dir)

    def setup_file_logging(self, output_dir: Path):
        """Set up file logging inside the transcript directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "transcription.log"

        # Unique logger name based on output dir path
        logger_name = f"transcription_logger_{abs(hash(str(self.log_file)))}"
        self._file_logger = logging.getLogger(logger_name)
        self._file_logger.setLevel(logging.DEBUG)
        self._file_logger.propagate = False

        # Clear existing handlers if any
        if self._file_logger.handlers:
            for h in list(self._file_logger.handlers):
                h.close()
                self._file_logger.removeHandler(h)

        try:
            self._handler = logging.FileHandler(str(self.log_file), mode="a", encoding="utf-8")
            formatter = logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            self._handler.setFormatter(formatter)
            self._file_logger.addHandler(self._handler)
        except Exception as e:
            print(f"  ⚠️ Warning: Could not initialize log file at '{self.log_file}': {e}")

    def log(self, level: int, msg: str, exc: Optional[Exception] = None):
        """Write to file logger with timestamp and level."""
        if self._file_logger:
            if exc:
                self._file_logger.log(level, msg, exc_info=exc)
            else:
                self._file_logger.log(level, msg)

    def info(self, msg: str, console: bool = True):
        """Log info message to file and optionally console."""
        self.log(logging.INFO, msg)
        if console:
            print(msg)

    def debug(self, msg: str):
        """Log debug message to file only."""
        self.log(logging.DEBUG, msg)

    def warning(self, msg: str, console: bool = True):
        """Log warning message to file and optionally console."""
        self.log(logging.WARNING, msg)
        if console:
            print(msg)

    def error(self, msg: str, exc: Optional[Exception] = None, console: bool = True):
        """Log error message & exception traceback to file, and print clean message to console."""
        self.log(logging.ERROR, msg, exc=exc)
        if console:
            print(msg)
            if self.log_file:
                print(f"   📋 Detailed logs saved in: {self.log_file.name}")

    def close(self):
        """Flush and close file handlers."""
        if self._handler:
            try:
                self._handler.flush()
                self._handler.close()
            except Exception:
                pass
        if self._file_logger and self._handler:
            self._file_logger.removeHandler(self._handler)


# Global default logger instance
_default_logger = TranscriptLogger()


def get_logger() -> TranscriptLogger:
    """Get the active global transcript logger."""
    return _default_logger
