"""
Audio file and directory discovery, pattern matching, and title normalization.
"""

import re
from pathlib import Path
from typing import List, Optional

from .audio import SUPPORTED_AUDIO_EXTENSIONS


def discover_audio_files(
    target_files: Optional[List[str]] = None,
    input_dir: Optional[Path] = None,
    pattern: Optional[str] = None,
    name_filter: Optional[str] = None,
    recursive: bool = False
) -> List[Path]:
    """Find audio files matching criteria."""
    audio_files: List[Path] = []
    
    if target_files:
        for f in target_files:
            clean_f = f.strip(" \t\n\r'\"“”‘’") if isinstance(f, str) else str(f)
            p = Path(clean_f).expanduser().resolve()
            if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                audio_files.append(p)
            elif p.is_file():
                print(f"  [Notice] File '{p.name}' does not have standard audio extension, but including as requested.")
                audio_files.append(p)
            else:
                print(f"  ⚠️ Warning: Specified file not found: {f}")
        return audio_files

    scan_dir = input_dir if input_dir else Path.cwd()
    if not scan_dir.exists():
        print(f"❌ Error: Input directory '{scan_dir}' does not exist.")
        return []

    glob_pattern = pattern if pattern else "*"
    iterator = scan_dir.rglob(glob_pattern) if recursive else scan_dir.glob(glob_pattern)

    for p in iterator:
        if not p.is_file():
            continue
        # Skip macOS AppleDouble metadata files (._*) and hidden files
        if p.name.startswith("."):
            continue
        if p.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue
        if "_temp_chunks" in p.parts or "_temp_slices" in p.parts:
            continue
        if name_filter and name_filter.lower() not in p.name.lower():
            continue
        
        audio_files.append(p)

    audio_files.sort(key=lambda x: x.name.lower())
    return audio_files


def clean_title_from_filename(filename: str) -> str:
    """Generate a clean, human-readable document title from an audio filename."""
    stem = Path(filename).stem
    for ext in SUPPORTED_AUDIO_EXTENSIONS:
        if stem.lower().endswith(ext):
            stem = stem[:-len(ext)]
            break
    clean = re.sub(r"_+", " ", stem)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean
