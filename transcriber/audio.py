"""
Audio probing, formatting, and smart FFmpeg slicing with disk caching.
"""

import shutil
import subprocess
import re
from pathlib import Path
from typing import List, Tuple, Set

SUPPORTED_AUDIO_EXTENSIONS: Set[str] = {
    ".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg", ".wma", ".opus", ".m4b", ".webm",
    ".mp4", ".m4v", ".mov", ".mkv", ".avi", ".3gp"
}


def check_ffmpeg_available() -> Tuple[bool, bool]:
    """Check if ffmpeg and ffprobe are available in the system PATH."""
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    ffprobe_ok = shutil.which("ffprobe") is not None
    return ffmpeg_ok, ffprobe_ok


def detect_audio_volume(file_path: Path) -> Tuple[float, float]:
    """
    Detect peak (max) and mean volume in dB using FFmpeg volumedetect filter.
    Returns: (max_volume_db, mean_volume_db)
    If volume detection fails, returns (0.0, 0.0).
    """
    cmd = [
        "ffmpeg", "-nostats",
        "-i", str(file_path),
        "-af", "volumedetect",
        "-vn", "-sn", "-dn",
        "-f", "null",
        "-"
    ]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
        )
        output = result.stderr or ""
        
        max_vol = 0.0
        mean_vol = 0.0
        
        max_match = re.search(r"max_volume:\s*(-?[\d.]+)\s*dB", output)
        if max_match:
            max_vol = float(max_match.group(1))
            
        mean_match = re.search(r"mean_volume:\s*(-?[\d.]+)\s*dB", output)
        if mean_match:
            mean_vol = float(mean_match.group(1))
            
        return max_vol, mean_vol
    except Exception:
        return 0.0, 0.0


def is_audio_silent(file_path: Path, max_db_threshold: float = -45.0) -> Tuple[bool, float]:
    """
    Determine if an audio file or slice is silent / below audible speech threshold.
    Returns: (is_silent, max_volume_db)
    - Pure digital silence is typically max_volume = -91.0 dB (or <= -90.0 dB).
    - Extremely quiet/empty background noise is typically <= -45.0 dB.
    """
    if not file_path.exists() or file_path.stat().st_size == 0:
        return True, -91.0

    max_vol, _ = detect_audio_volume(file_path)
    # If peak volume is at or below threshold, consider silent
    if max_vol <= max_db_threshold:
        return True, max_vol
    return False, max_vol


def parse_time_str(time_val: Any) -> Optional[float]:
    """
    Parse a flexible time string into seconds as a float.
    Supported formats:
      - "21:00" -> 1260.0 (MM:SS)
      - "01:21:00" -> 4860.0 (HH:MM:SS)
      - "21m", "21min", "21mins" -> 1260.0
      - "90s", "90sec" -> 90.0
      - "1h", "1.5h" -> 3600.0, 5400.0
      - "1260" / 1260 / 1260.0 -> 1260.0
    Returns None if input is empty or invalid.
    """
    if time_val is None:
        return None
    if isinstance(time_val, (int, float)):
        return max(0.0, float(time_val))
        
    s = str(time_val).strip().lower()
    if not s:
        return None

    # Handle colon format HH:MM:SS or MM:SS
    if ":" in s:
        parts = s.split(":")
        try:
            if len(parts) == 2:
                mins, secs = float(parts[0]), float(parts[1])
                return max(0.0, mins * 60.0 + secs)
            elif len(parts) == 3:
                hrs, mins, secs = float(parts[0]), float(parts[1]), float(parts[2])
                return max(0.0, hrs * 3600.0 + mins * 60.0 + secs)
        except ValueError:
            pass

    # Handle unit suffixes: 21m, 21min, 90s, 1h
    m_match = re.match(r"^([\d.]+)\s*(h|hr|hrs|hours?)$", s)
    if m_match:
        return max(0.0, float(m_match.group(1)) * 3600.0)

    m_match = re.match(r"^([\d.]+)\s*(m|min|mins|minutes?)$", s)
    if m_match:
        return max(0.0, float(m_match.group(1)) * 60.0)

    m_match = re.match(r"^([\d.]+)\s*(s|sec|secs|seconds?)$", s)
    if m_match:
        return max(0.0, float(m_match.group(1)))

    # Plain number (treated as seconds)
    try:
        return max(0.0, float(s))
    except ValueError:
        return None


def format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS string."""
    seconds = int(round(seconds))
    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def get_audio_duration(file_path: Path) -> float:
    """
    Get duration of audio file in seconds using ffprobe.
    Fallback to ffmpeg duration parse if needed.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path)
    ]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        duration = float(result.stdout.strip())
        return duration
    except Exception:
        # Fallback to ffmpeg -i
        try:
            cmd = ["ffmpeg", "-i", str(file_path)]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", result.stderr)
            if match:
                hrs, mins, secs = match.groups()
                return float(hrs) * 3600 + float(mins) * 60 + float(secs)
        except Exception as e:
            print(f"  [Warning] Unable to probe duration for {file_path.name}: {e}")
    return 0.0


class AudioSlice:
    """Represents a segment / slice of an audio file."""
    def __init__(self, file_path: Path, start_sec: float, end_sec: float, part_num: int, total_parts: int, is_temp: bool):
        self.file_path = file_path
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.part_num = part_num
        self.total_parts = total_parts
        self.is_temp = is_temp

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    @property
    def start_time_str(self) -> str:
        return format_duration(self.start_sec)

    @property
    def end_time_str(self) -> str:
        return format_duration(self.end_sec)


def slice_audio_file(
    input_path: Path,
    chunk_minutes: float,
    temp_dir: Path,
    start_time_sec: Optional[float] = None,
    end_time_sec: Optional[float] = None,
    dry_run: bool = False
) -> List[AudioSlice]:
    """
    Smart Audio Slicing with Caching & Optional Time-Range:
    - Slices audio within [start_time_sec, end_time_sec].
    - If range duration <= chunk_minutes (or chunk_minutes <= 0), extracts a single slice.
    - If range duration > chunk_minutes, partitions into consecutive slices.
    - Preserves absolute start_sec and end_sec timestamps for transcription alignment.
    """
    total_duration = get_audio_duration(input_path)
    if total_duration <= 0:
        return []

    # Bound custom range
    range_start = max(0.0, start_time_sec if start_time_sec is not None else 0.0)
    range_end = min(total_duration, end_time_sec if end_time_sec is not None else total_duration)
    
    if range_start >= total_duration or range_start >= range_end:
        return []

    range_duration = range_end - range_start
    chunk_duration_sec = chunk_minutes * 60.0 if chunk_minutes > 0 else range_duration

    stem = input_path.stem
    ext = input_path.suffix.lower()

    # Case 1: Full original file with no range and no slicing needed
    is_full_file = (start_time_sec is None or start_time_sec == 0.0) and (end_time_sec is None or end_time_sec >= total_duration)
    if is_full_file and (chunk_minutes <= 0 or total_duration <= (chunk_duration_sec + 10.0)):
        return [AudioSlice(
            file_path=input_path,
            start_sec=0.0,
            end_sec=total_duration,
            part_num=1,
            total_parts=1,
            is_temp=False
        )]

    # Case 2: Custom range or chunked file
    total_parts = int((range_duration + chunk_duration_sec - 1) // chunk_duration_sec) if chunk_duration_sec > 0 else 1
    total_parts = max(1, total_parts)

    if not dry_run:
        range_str = f"{format_duration(range_start)} - {format_duration(range_end)}"
        print(f"  🔪 Checking slices ({range_str}, duration: {format_duration(range_duration)}) -> {total_parts} part(s)...")
        temp_dir.mkdir(parents=True, exist_ok=True)

    slices: List[AudioSlice] = []
    for part_idx in range(1, total_parts + 1):
        part_start = range_start + (part_idx - 1) * chunk_duration_sec
        part_end = min(part_start + chunk_duration_sec, range_end)
        part_duration = part_end - part_start

        slice_filename = f"{stem}_part{part_idx:02d}{ext}"
        slice_path = temp_dir / slice_filename

        if not dry_run:
            # Check if valid slice already exists on disk (Slice Caching)
            if slice_path.exists() and slice_path.stat().st_size > 1024:
                print(f"     -> Part {part_idx:02d}: {format_duration(part_start)} to {format_duration(part_end)} (⚡ Reusing existing slice: {slice_path.name})")
            else:
                # FFmpeg slice command (fast stream copy first, fallback to re-encode)
                cmd_copy = [
                    "ffmpeg", "-y",
                    "-ss", str(part_start),
                    "-i", str(input_path),
                    "-t", str(part_duration),
                    "-c", "copy",
                    str(slice_path)
                ]
                
                try:
                    subprocess.run(
                        cmd_copy,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True
                    )
                except subprocess.CalledProcessError:
                    cmd_reencode = [
                        "ffmpeg", "-y",
                        "-ss", str(part_start),
                        "-i", str(input_path),
                        "-t", str(part_duration),
                        "-c:a", "libmp3lame" if ext == ".mp3" else "aac",
                        "-q:a", "2",
                        str(slice_path)
                    ]
                    subprocess.run(
                        cmd_reencode,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True
                    )

                print(f"     -> Part {part_idx:02d}: {format_duration(part_start)} to {format_duration(part_end)} ({slice_path.name})")
        
        slices.append(AudioSlice(
            file_path=slice_path,
            start_sec=part_start,
            end_sec=part_end,
            part_num=part_idx,
            total_parts=total_parts,
            is_temp=True
        ))

    return slices
