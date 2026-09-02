import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

# Official Google Gemini API Pricing (USD per 1,000,000 tokens)
# (Input Multimodal/Audio rate, Output Text rate)
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # Gemini 3.7 / 2.5 / 2.0 Flash family ($0.70 / 1M audio input, $0.40 / 1M output)
    "gemini-3.7-flash": (0.70, 0.40),
    "gemini-3.6-flash": (0.70, 0.40),
    "gemini-3.5-flash": (0.70, 0.40),
    "gemini-2.5-flash": (0.70, 0.40),
    "gemini-2.0-flash": (0.70, 0.40),
    "gemini-flash-latest": (0.70, 0.40),
    "gemini-2.5-flash-lite": (0.35, 0.20),
    
    # Gemini Pro family ($1.25 / 1M input, $5.00 / 1M output)
    "gemini-2.5-pro": (1.25, 5.00),
    "gemini-pro-latest": (1.25, 5.00),
    "gemini-3.1-pro-preview": (1.25, 5.00),
}

DEFAULT_FLASH_PRICING: Tuple[float, float] = (0.70, 0.40)


def calculate_cost(model_name: str, prompt_tokens: int, candidate_tokens: int) -> float:
    """Calculate the estimated USD cost based on token counts and model pricing."""
    clean_model = model_name.lower().replace("models/", "")
    input_rate_per_m, output_rate_per_m = MODEL_PRICING.get(clean_model, DEFAULT_FLASH_PRICING)
    
    input_cost = (prompt_tokens / 1_000_000.0) * input_rate_per_m
    output_cost = (candidate_tokens / 1_000_000.0) * output_rate_per_m
    return input_cost + output_cost


def format_cost(cost_usd: float) -> str:
    """Format USD cost in a clean, human-readable string."""
    if cost_usd <= 0.0:
        return "$0.0000"
    if cost_usd < 0.0001:
        return "< $0.0001"
    if cost_usd < 0.01:
        return f"${cost_usd:.4f}"
    return f"${cost_usd:.3f}"


@dataclass
class TokenUsageSummary:
    """Aggregated token usage and cost metrics for a slice or full file."""
    prompt_tokens: int = 0
    candidate_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def add(self, prompt_tok: int, candidate_tok: int, cost: float):
        self.prompt_tokens += prompt_tok
        self.candidate_tokens += candidate_tok
        self.total_tokens += (prompt_tok + candidate_tok)
        self.estimated_cost_usd += cost

    def format_summary(self) -> str:
        cost_str = format_cost(self.estimated_cost_usd)
        return (
            f"Input Tokens: {self.prompt_tokens:,} | "
            f"Output Tokens: {self.candidate_tokens:,} | "
            f"Total Tokens: {self.total_tokens:,} | "
            f"Estimated Cost: ~{cost_str} USD"
        )


@dataclass
class FileRunRecord:
    """Record of a single file or audio slice processed during the session."""
    file_name: str
    segment_label: str
    audio_duration_sec: float
    elapsed_processing_sec: float
    clip_start_str: str = "00:00"
    clip_end_str: str = "00:00"
    prompt_tokens: int = 0
    candidate_tokens: int = 0
    cost_usd: float = 0.0
    status: str = "Completed"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.candidate_tokens


class LocalUsageLedger:
    """
    Central persistent ledger storing cumulative audio duration, token consumption,
    estimated USD cost, and session history across all transcription runs.
    Saved locally (e.g. .usage_ledger.json) and excluded from git.
    """
    DEFAULT_LEDGER_FILE = Path(".usage_ledger.json")

    def __init__(self, ledger_path: Optional[Path] = None):
        self.ledger_path = ledger_path or self.DEFAULT_LEDGER_FILE
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.ledger_path.exists():
            try:
                import json
                with open(self.ledger_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "lifetime_audio_seconds": 0.0,
            "lifetime_prompt_tokens": 0,
            "lifetime_candidate_tokens": 0,
            "lifetime_total_tokens": 0,
            "lifetime_cost_usd": 0.0,
            "lifetime_runs_count": 0,
            "lifetime_items_processed": 0,
            "history": []
        }

    def record_session(
        self,
        audio_sec: float,
        prompt_tokens: int,
        candidate_tokens: int,
        cost_usd: float,
        target_dir: str,
        items_count: int,
        status: str = "Completed"
    ):
        """Update and persist all-time cumulative usage and session history."""
        import json
        self.data["lifetime_audio_seconds"] = self.data.get("lifetime_audio_seconds", 0.0) + audio_sec
        self.data["lifetime_prompt_tokens"] = self.data.get("lifetime_prompt_tokens", 0) + prompt_tokens
        self.data["lifetime_candidate_tokens"] = self.data.get("lifetime_candidate_tokens", 0) + candidate_tokens
        self.data["lifetime_total_tokens"] = self.data.get("lifetime_total_tokens", 0) + (prompt_tokens + candidate_tokens)
        self.data["lifetime_cost_usd"] = self.data.get("lifetime_cost_usd", 0.0) + cost_usd
        self.data["lifetime_runs_count"] = self.data.get("lifetime_runs_count", 0) + 1
        self.data["lifetime_items_processed"] = self.data.get("lifetime_items_processed", 0) + items_count

        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "target_dir": target_dir,
            "audio_seconds": round(audio_sec, 1),
            "prompt_tokens": prompt_tokens,
            "candidate_tokens": candidate_tokens,
            "total_tokens": prompt_tokens + candidate_tokens,
            "cost_usd": round(cost_usd, 6),
            "items_count": items_count,
            "status": status
        }
        history = self.data.get("history", [])
        history.append(entry)
        self.data["history"] = history[-50:]

        try:
            with open(self.ledger_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            pass

    @property
    def lifetime_cost_usd(self) -> float:
        return float(self.data.get("lifetime_cost_usd", 0.0))

    @property
    def lifetime_total_tokens(self) -> int:
        return int(self.data.get("lifetime_total_tokens", 0))

    @property
    def lifetime_audio_seconds(self) -> float:
        return float(self.data.get("lifetime_audio_seconds", 0.0))

    @property
    def lifetime_runs_count(self) -> int:
        return int(self.data.get("lifetime_runs_count", 0))


class SessionTracker:
    """
    Tracks all items processed in the current execution run, active processing times,
    token usage, overall cost, and all-time cumulative usage ledger.
    Generates a structured report on normal completion or Ctrl+C interruption.
    """
    def __init__(
        self,
        target_dir: str = "",
        model_name: str = "gemini-3.7-flash",
        starting_balance: Optional[float] = None,
        ledger_path: Optional[Path] = None
    ):
        self.target_dir = target_dir
        self.model_name = model_name
        self.starting_balance = starting_balance
        self.start_epoch = time.time()
        self.start_dt = datetime.now()
        self.records: List[FileRunRecord] = []
        self.ledger = LocalUsageLedger(ledger_path=ledger_path)
        self._recorded_to_ledger = False

    def add_record(self, record: FileRunRecord):
        self.records.append(record)

    def format_time_taken(self, seconds: float) -> str:
        if seconds < 60.0:
            return f"{seconds:.1f}s"
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs:02d}s"

    def generate_summary(self, interrupted: bool = False) -> str:
        end_epoch = time.time()
        end_dt = datetime.now()
        total_wall_sec = max(0.0, end_epoch - self.start_epoch)
        total_wall_str = self.format_time_taken(total_wall_sec)

        title_status = "🛑 INTERRUPTED BY USER (Ctrl+C)" if interrupted else "🎉 COMPLETED"
        header_line = "=" * 108
        sub_line = "-" * 108

        lines = [
            "",
            header_line,
            f"📊 TRANSCRIPTION RUN SUMMARY ({title_status})",
            header_line,
            f"📁 Target Folder:      {self.target_dir or 'Current Directory'}",
            f"📅 Session Started:   {self.start_dt.strftime('%Y-%m-%d %H:%M:%S')}",
            f"📅 Session Ended:     {end_dt.strftime('%Y-%m-%d %H:%M:%S')} (Total Runtime: {total_wall_str})",
            f"🤖 Gemini Model:       {self.model_name}",
            "",
            sub_line,
            f"{'#':<3} {'File / Segment':<32} {'Clip Start':<11} {'Clip End':<10} {'Duration':<9} {'Time Taken':<12} {'Tokens':<10} {'Cost ($)':<10} {'Status'}",
            sub_line,
        ]

        total_audio_sec = 0.0
        total_processing_sec = 0.0
        total_prompt_tok = 0
        total_cand_tok = 0
        total_cost = 0.0

        if not self.records:
            lines.append("   (No items were processed in this session)")
        else:
            for idx, r in enumerate(self.records, start=1):
                total_audio_sec += r.audio_duration_sec
                total_processing_sec += r.elapsed_processing_sec
                total_prompt_tok += r.prompt_tokens
                total_cand_tok += r.candidate_tokens
                total_cost += r.cost_usd

                display_label = r.segment_label if r.segment_label else r.file_name
                if len(display_label) > 30:
                    display_label = display_label[:27] + "..."

                mins = int(round(r.audio_duration_sec) // 60)
                secs = int(round(r.audio_duration_sec) % 60)
                audio_str = f"{mins:02d}:{secs:02d}"

                time_taken_str = self.format_time_taken(r.elapsed_processing_sec)
                tokens_str = f"{r.total_tokens:,}" if r.total_tokens > 0 else "0"
                cost_str = f"${r.cost_usd:.4f}" if r.cost_usd > 0 else "$0.0000"

                lines.append(
                    f"{idx:<3} {display_label:<32} {r.clip_start_str:<11} {r.clip_end_str:<10} {audio_str:<9} {time_taken_str:<12} {tokens_str:<10} {cost_str:<10} {r.status}"
                )

        lines.append(sub_line)

        total_audio_mins = int(round(total_audio_sec) // 60)
        total_audio_secs = int(round(total_audio_sec) % 60)
        total_audio_str = f"{total_audio_mins:02d}:{total_audio_secs:02d}"
        total_active_str = self.format_time_taken(total_processing_sec)
        total_tok_sum = total_prompt_tok + total_cand_tok

        # Record to local ledger if not already recorded
        if not self._recorded_to_ledger and (total_tok_sum > 0 or len(self.records) > 0):
            self.ledger.record_session(
                audio_sec=total_audio_sec,
                prompt_tokens=total_prompt_tok,
                candidate_tokens=total_cand_tok,
                cost_usd=total_cost,
                target_dir=self.target_dir,
                items_count=len(self.records),
                status="Interrupted" if interrupted else "Completed"
            )
            self._recorded_to_ledger = True

        lines.extend([
            "",
            "📈 CURRENT RUN TOTALS:",
            f"  • Audio Processed:         {total_audio_str} ({total_audio_mins} mins {total_audio_secs} secs)",
            f"  • Active Processing Time:  {total_active_str} ({len(self.records)} item(s) processed)",
            f"  • Tokens Consumed:         {total_tok_sum:,} ({total_prompt_tok:,} in + {total_cand_tok:,} out)",
            f"  • Current Run Cost:        ~{format_cost(total_cost)} USD",
        ])

        # All-Time Cumulative Ledger Section
        life_audio_sec = self.ledger.lifetime_audio_seconds
        life_mins = int(round(life_audio_sec) // 60)
        life_hours = life_mins // 60
        life_rem_mins = life_mins % 60
        life_audio_str = f"{life_hours}h {life_rem_mins:02d}m" if life_hours > 0 else f"{life_mins} mins"

        lines.extend([
            "",
            "💰 ALL-TIME CUMULATIVE USAGE (Auto-Tracked in Local Ledger):",
            f"  • Lifetime Audio Done:     {life_audio_str} ({self.ledger.lifetime_runs_count} total run(s))",
            f"  • Lifetime Total Tokens:   {self.ledger.lifetime_total_tokens:,} tokens",
            f"  • Lifetime Incurred Cost:  ~{format_cost(self.ledger.lifetime_cost_usd)} USD",
        ])

        if self.starting_balance is not None:
            end_balance = max(0.0, self.starting_balance - total_cost)
            lines.extend([
                "",
                "💳 OPTIONAL BUDGET / CREDIT BALANCE:",
                f"  • Starting Balance:        ${self.starting_balance:.4f} USD",
                f"  • Run Cost Incurred:      -${total_cost:.4f} USD",
                f"  • Ending Balance:          ${end_balance:.4f} USD (Remaining)",
            ])

        lines.append(header_line)
        if interrupted:
            lines.append("💾 Checkpoints safely saved! You can resume anytime by re-running the script.")
            lines.append(header_line)

        lines.append("")
        return "\n".join(lines)
