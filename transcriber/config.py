"""
Configuration and Environment Management for Gemini Audio Transcriber.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


def load_env_file(search_paths: Optional[List[Path]] = None) -> Dict[str, str]:
    """
    Search for and load variables from a .env file without requiring external dependencies.
    Also supports python-dotenv if already installed.
    """
    env_vars: Dict[str, str] = {}
    
    # Try python-dotenv first if available
    try:
        from dotenv import load_dotenv, find_dotenv
        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path:
            load_dotenv(dotenv_path)
            return dict(os.environ)
    except ImportError:
        pass

    # Built-in zero-dependency .env reader
    if search_paths is None:
        search_paths = [
            Path.cwd(),
            Path(__file__).resolve().parent.parent,
            Path(__file__).resolve().parent,
            Path.home()
        ]

    for directory in search_paths:
        env_file = directory / ".env"
        if env_file.is_file():
            try:
                with open(env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        if line.startswith("export "):
                            line = line[7:].strip()
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip("'\"")
                        if key and key not in os.environ:
                            os.environ[key] = val
                            env_vars[key] = val
                break
            except Exception:
                pass

    return env_vars


def parse_bool_env(key: str, default: bool = False) -> bool:
    """Helper to parse boolean values from environment variables."""
    val = os.environ.get(key, "").strip().lower()
    if not val:
        return default
    return val in ("true", "1", "yes", "on")


def resolve_api_key(cli_key: Optional[str] = None) -> str:
    """
    Resolve Gemini API key securely from CLI argument, .env file, or environment variables.
    Exits with actionable instructions if no key is found.
    """
    load_env_file()
    
    api_key = (
        (cli_key or "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )

    if not api_key:
        print("\n" + "=" * 70)
        print("❌ ERROR: Gemini API Key not found!")
        print("=" * 70)
        print("Please configure your API key in the '.env' file:\n")
        print("  1. Open or create '.env' in this directory")
        print("  2. Add your key:  GEMINI_API_KEY=your_actual_api_key_here\n")
        print("Or set as an environment variable in your terminal:")
        print("  - Windows (PowerShell):  $env:GEMINI_API_KEY=\"your_key\"")
        print("  - Windows (CMD):         set GEMINI_API_KEY=your_key")
        print("  - Linux / macOS:         export GEMINI_API_KEY=\"your_key\"\n")
        print("Get your API key at: https://aistudio.google.com/")
        print("=" * 70 + "\n")
        sys.exit(1)

    return api_key


from .audio import parse_time_str


@dataclass
class TranscriberConfig:
    """Encapsulates all runtime configuration parameters for the transcription engine."""
    api_key: Optional[str] = None
    model_name: str = "gemini-3.7-flash"
    chunk_minutes: float = 10.0
    language: str = "Gujarati script (ગુજરાતી લિપિ)"
    timestamp_interval: int = 2
    output_dir: Optional[Path] = None
    save_txt: bool = False
    keep_chunks: bool = False
    force: bool = False
    dry_run: bool = False
    max_retry_minutes: float = 30.0
    custom_instructions: Optional[str] = None
    start_time_sec: Optional[float] = None
    end_time_sec: Optional[float] = None
    starting_balance: Optional[float] = None

    @classmethod
    def from_env_and_args(cls, args_dict: Optional[Dict[str, Any]] = None) -> "TranscriberConfig":
        """Build a TranscriberConfig combining .env defaults with CLI argument overrides."""
        load_env_file()
        args = args_dict or {}

        # 1. Defaults from .env
        env_model = os.environ.get("GEMINI_MODEL", "gemini-3.7-flash").strip() or "gemini-3.7-flash"
        env_chunk_mins = float(os.environ.get("CHUNK_MINUTES", "10.0").strip() or "10.0")
        env_language = os.environ.get("LANGUAGE", "Gujarati script (ગુજરાતી લિપિ)").strip() or "Gujarati script (ગુજરાતી લિપિ)"
        env_timestamp_interval = int(os.environ.get("TIMESTAMP_INTERVAL", "2").strip() or "2")
        env_output_dir = os.environ.get("OUTPUT_DIR", "").strip() or None
        env_save_txt = parse_bool_env("SAVE_TXT", False)
        env_keep_chunks = parse_bool_env("KEEP_CHUNKS", False)
        env_max_retry_mins = float(os.environ.get("MAX_RETRY_MINUTES", "30.0").strip() or "30.0")
        env_instructions = os.environ.get("CUSTOM_INSTRUCTIONS", "").strip() or None

        # 2. CLI Overrides (falling back to .env defaults if None)
        model_name = args.get("model") or env_model
        
        chunk_mins_arg = args.get("chunk_minutes")
        chunk_minutes = float(chunk_mins_arg) if chunk_mins_arg is not None else env_chunk_mins
        if args.get("no_slice"):
            chunk_minutes = 0.0

        language = args.get("language") or env_language
        
        ts_arg = args.get("timestamp_interval")
        timestamp_interval = int(ts_arg) if ts_arg is not None else env_timestamp_interval
        if args.get("no_timestamps"):
            timestamp_interval = 0

        out_dir_str = args.get("output_dir") or env_output_dir
        if out_dir_str:
            out_dir_str = out_dir_str.strip(" \t\n\r'\"“”‘’")
        output_dir = Path(out_dir_str).expanduser().resolve() if out_dir_str else None

        save_txt = args.get("save_txt") if args.get("save_txt") is not None else env_save_txt
        keep_chunks = args.get("keep_chunks") if args.get("keep_chunks") is not None else env_keep_chunks
        force = bool(args.get("force", False))
        dry_run = bool(args.get("dry_run", False))
        
        retry_arg = args.get("max_retry_minutes")
        max_retry_minutes = float(retry_arg) if retry_arg is not None else env_max_retry_mins

        custom_instructions = args.get("instructions") or env_instructions
        api_key = args.get("api_key")

        # Handle time-range arguments: --range, --start, --end
        start_time_sec = None
        end_time_sec = None

        time_range = args.get("range") or args.get("time_range")
        if time_range:
            import re
            parts = re.split(r"\s*(?:-|to)\s*", str(time_range).strip(), maxsplit=1)
            if len(parts) == 2:
                start_time_sec = parse_time_str(parts[0])
                end_time_sec = parse_time_str(parts[1])
            elif len(parts) == 1:
                start_time_sec = parse_time_str(parts[0])

        if args.get("start") is not None or args.get("start_time") is not None:
            raw_start = args.get("start") if args.get("start") is not None else args.get("start_time")
            start_time_sec = parse_time_str(raw_start)

        if args.get("end") is not None or args.get("end_time") is not None:
            raw_end = args.get("end") if args.get("end") is not None else args.get("end_time")
            end_time_sec = parse_time_str(raw_end)

        # Handle starting balance: STARTING_BALANCE or API_CREDITS in .env or CLI arg
        env_balance_str = os.environ.get("STARTING_BALANCE") or os.environ.get("API_CREDITS") or os.environ.get("INITIAL_CREDIT_BALANCE")
        starting_balance = None
        if env_balance_str:
            try:
                starting_balance = float(env_balance_str.strip().replace("$", ""))
            except ValueError:
                pass

        cli_balance = args.get("starting_balance") or args.get("credits")
        if cli_balance is not None:
            try:
                starting_balance = float(str(cli_balance).strip().replace("$", ""))
            except ValueError:
                pass

        return cls(
            api_key=api_key,
            model_name=model_name,
            chunk_minutes=chunk_minutes,
            language=language,
            timestamp_interval=timestamp_interval,
            output_dir=output_dir,
            save_txt=save_txt,
            keep_chunks=keep_chunks,
            force=force,
            dry_run=dry_run,
            max_retry_minutes=max_retry_minutes,
            custom_instructions=custom_instructions,
            start_time_sec=start_time_sec,
            end_time_sec=end_time_sec,
            starting_balance=starting_balance
        )
