"""
Fault-Tolerant Checkpoint and State Management for Audio Slices.
"""

import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple


class CheckpointManager:
    """Manages saving and resuming slice transcripts and token metrics from a JSON checkpoint file."""
    
    def __init__(self, temp_dir: Path, audio_filename: str):
        self.checkpoint_file = temp_dir / "checkpoint.json"
        self.data: Dict[str, Any] = {
            "audio_file": audio_filename,
            "created_at": time.time(),
            "slices": {}
        }
        self.load()

    def load(self):
        """Load checkpoint data from disk if it exists."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception as e:
                print(f"  [Notice] Could not load checkpoint file: {e}")

    def save(self):
        """Save current checkpoint state to disk."""
        try:
            self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  [Warning] Could not save checkpoint: {e}")

    def get_slice_transcript(self, part_num: int) -> Optional[str]:
        """Retrieve cached transcript for a specific slice part if present and valid."""
        slice_entry = self.data.get("slices", {}).get(str(part_num))
        if slice_entry and isinstance(slice_entry, dict):
            text = slice_entry.get("transcript", "")
            if text and len(text.strip()) > 50:
                return text
        return None

    def get_slice_tokens(self, part_num: int) -> Tuple[int, int, float]:
        """Retrieve cached token usage (prompt_tokens, candidate_tokens, cost_usd)."""
        slice_entry = self.data.get("slices", {}).get(str(part_num), {})
        return (
            slice_entry.get("prompt_tokens", 0),
            slice_entry.get("candidate_tokens", 0),
            slice_entry.get("cost_usd", 0.0)
        )

    def store_slice_transcript(
        self,
        part_num: int,
        start_sec: float,
        end_sec: float,
        transcript: str,
        prompt_tokens: int = 0,
        candidate_tokens: int = 0,
        cost_usd: float = 0.0
    ):
        """Store transcript and token metrics for a completed slice and immediately flush to disk."""
        if "slices" not in self.data:
            self.data["slices"] = {}
        self.data["slices"][str(part_num)] = {
            "start_sec": start_sec,
            "end_sec": end_sec,
            "transcript": transcript,
            "prompt_tokens": prompt_tokens,
            "candidate_tokens": candidate_tokens,
            "cost_usd": cost_usd,
            "completed_at": time.time()
        }
        self.save()
