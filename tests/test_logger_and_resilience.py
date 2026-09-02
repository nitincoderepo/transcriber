import unittest
import tempfile
import shutil
from pathlib import Path

from transcriber.logger import format_wait_time, get_progress_bar, TranscriptLogger
from transcriber.engine import is_daily_limit_error


class MockAPIError(Exception):
    def __init__(self, message: str, code: int = 429):
        super().__init__(message)
        self.code = code


class TestLoggerAndResilience(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_format_wait_time(self):
        # Less than a minute -> show seconds
        self.assertEqual(format_wait_time(1.0), "1 second")
        self.assertEqual(format_wait_time(15.0), "15 seconds")
        self.assertEqual(format_wait_time(45.0), "45 seconds")
        self.assertEqual(format_wait_time(59.0), "59 seconds")

        # 1 minute or more -> show minutes & seconds
        self.assertEqual(format_wait_time(60.0), "1 minute")
        self.assertEqual(format_wait_time(120.0), "2 minutes")
        self.assertEqual(format_wait_time(135.0), "2 min 15 sec")
        self.assertEqual(format_wait_time(1800.0), "30 minutes")
        self.assertEqual(format_wait_time(1835.0), "30 min 35 sec")

    def test_get_progress_bar(self):
        bar_0 = get_progress_bar(0, 4, width=10)
        self.assertIn("0.0%", bar_0)

        bar_half = get_progress_bar(2, 4, width=10)
        self.assertIn("50.0%", bar_half)
        self.assertIn("█████", bar_half)

        bar_full = get_progress_bar(4, 4, width=10)
        self.assertIn("100.0%", bar_full)
        self.assertEqual(bar_full.count("█"), 10)

    def test_transcript_logger_writes_to_file(self):
        transcript_dir = self.test_dir / "transcript"
        logger = TranscriptLogger(output_dir=transcript_dir)
        
        logger.info("Test info message", console=False)
        logger.debug("Test debug message")
        logger.warning("Test warning message", console=False)
        logger.error("Test error message", console=False)
        logger.close()

        log_file = transcript_dir / "transcription.log"
        self.assertTrue(log_file.exists())
        content = log_file.read_text(encoding="utf-8")
        
        self.assertIn("[INFO] Test info message", content)
        self.assertIn("[DEBUG] Test debug message", content)
        self.assertIn("[WARNING] Test warning message", content)
        self.assertIn("[ERROR] Test error message", content)

    def test_is_daily_limit_error(self):
        # 1. Prepayment credits depleted
        e1 = MockAPIError("Your prepayment credits are depleted. Please go to AI Studio.", code=429)
        is_daily, reason = is_daily_limit_error(e1)
        self.assertTrue(is_daily)
        self.assertIn("Prepayment credits", reason)

        # 2. Daily limit in text
        e2 = MockAPIError("Rate limit exceeded: 50 requests per day limit reached.", code=429)
        is_daily, reason = is_daily_limit_error(e2)
        self.assertTrue(is_daily)
        self.assertIn("Daily", reason)

        # 3. Transient 503 error
        e3 = MockAPIError("This model is currently experiencing high demand.", code=503)
        is_daily, reason = is_daily_limit_error(e3)
        self.assertFalse(is_daily)
        self.assertEqual(reason, "")

        # 4. Standard transient 429 TPM burst
        e4 = MockAPIError("Rate limit exceeded. Please wait a moment.", code=429)
        is_daily, reason = is_daily_limit_error(e4)
        self.assertFalse(is_daily)

    def test_cost_calculation(self):
        from transcriber.cost import calculate_cost, format_cost, TokenUsageSummary
        
        # 1 hour audio (~115,200 input tokens) + 5,000 output tokens on gemini-3.7-flash
        cost = calculate_cost("gemini-3.7-flash", prompt_tokens=115200, candidate_tokens=5000)
        # 115200 * 0.70 / 1M = 0.08064
        # 5000 * 0.40 / 1M = 0.002
        # Total = 0.08264
        self.assertAlmostEqual(cost, 0.08264, places=5)
        self.assertEqual(format_cost(cost), "$0.083")

        # Zero cost
        self.assertEqual(format_cost(0.0), "$0.0000")

        # Micro cost
        self.assertEqual(format_cost(0.00005), "< $0.0001")

        # TokenUsageSummary accumulation
        summary = TokenUsageSummary()
        summary.add(prompt_tok=100000, candidate_tok=4000, cost=0.0716)
        summary.add(prompt_tok=100000, candidate_tok=4000, cost=0.0716)
        self.assertEqual(summary.prompt_tokens, 200000)
        self.assertEqual(summary.candidate_tokens, 8000)
        self.assertEqual(summary.total_tokens, 208000)
        self.assertAlmostEqual(summary.estimated_cost_usd, 0.1432, places=4)

    def test_transcribe_slice_empty_response_resilience(self):
        from unittest.mock import MagicMock
        from transcriber.engine import transcribe_slice
        from transcriber.audio import AudioSlice

        mock_client = MagicMock()
        mock_file = MagicMock()
        mock_file.name = "files/test_audio_123"
        mock_file.state = "ACTIVE"
        mock_client.files.upload.return_value = mock_file

        mock_response = MagicMock()
        mock_response.text = ""  # Gemini returns empty response (silence/music)
        mock_response.usage_metadata = MagicMock(prompt_token_count=1000, candidates_token_count=0)
        candidate = MagicMock()
        candidate.finish_reason = "STOP"
        mock_response.candidates = [candidate]
        mock_client.models.generate_content.return_value = mock_response

        slice_info = AudioSlice(
            file_path=self.test_dir / "slice.mp3",
            start_sec=0.0,
            end_sec=600.0,
            part_num=1,
            total_parts=1,
            is_temp=False
        )
        slice_info.file_path.write_text("fake audio")

        logger = TranscriptLogger(output_dir=self.test_dir)
        text, p_tok, c_tok, cost = transcribe_slice(
            client=mock_client,
            slice_info=slice_info,
            audio_title="Test Audio",
            language="Gujarati",
            model_name="gemini-3.7-flash",
            logger=logger
        )
        logger.close()

        # Should verify once and gracefully return silence marker without infinite retrying
        self.assertEqual(mock_client.models.generate_content.call_count, 2)
        self.assertIn("Silence", text)
        self.assertEqual(p_tok, 1000)

    def test_session_tracker_and_credit_balance(self):
        from transcriber.cost import SessionTracker, FileRunRecord, LocalUsageLedger

        test_ledger_file = self.test_dir / "test_ledger.json"
        tracker = SessionTracker(
            target_dir="C:/test/audio",
            model_name="gemini-3.7-flash",
            starting_balance=50.0,
            ledger_path=test_ledger_file
        )

        record1 = FileRunRecord(
            file_name="track1.mp3",
            segment_label="track1 (Part 01)",
            audio_duration_sec=600.0,
            elapsed_processing_sec=25.4,
            clip_start_str="00:00",
            clip_end_str="10:00",
            prompt_tokens=19000,
            candidate_tokens=1200,
            cost_usd=0.01378,
            status="✅ Completed"
        )
        record2 = FileRunRecord(
            file_name="track2.mp3",
            segment_label="track2.mp3",
            audio_duration_sec=300.0,
            elapsed_processing_sec=0.2,
            clip_start_str="21:00",
            clip_end_str="26:00",
            prompt_tokens=0,
            candidate_tokens=0,
            cost_usd=0.0,
            status="⚡ Skipped (Exists)"
        )
        record3 = FileRunRecord(
            file_name="track3.mp3",
            segment_label="track3.mp3",
            audio_duration_sec=600.0,
            elapsed_processing_sec=0.15,
            clip_start_str="00:00",
            clip_end_str="10:00",
            prompt_tokens=0,
            candidate_tokens=0,
            cost_usd=0.0,
            status="🔇 Silent ($0)"
        )

        tracker.add_record(record1)
        tracker.add_record(record2)
        tracker.add_record(record3)

        # Test normal summary
        summary = tracker.generate_summary(interrupted=False)
        self.assertIn("TRANSCRIPTION RUN SUMMARY (🎉 COMPLETED)", summary)
        self.assertIn("Clip Start", summary)
        self.assertIn("Clip End", summary)
        self.assertIn("track1 (Part 01)", summary)
        self.assertIn("00:00", summary)
        self.assertIn("10:00", summary)
        self.assertIn("25.4s", summary)
        self.assertIn("ALL-TIME CUMULATIVE USAGE", summary)
        self.assertIn("Lifetime Total Tokens", summary)

        # Verify ledger on disk
        self.assertTrue(test_ledger_file.exists())
        ledger_reloaded = LocalUsageLedger(ledger_path=test_ledger_file)
        self.assertEqual(ledger_reloaded.lifetime_runs_count, 1)
        self.assertEqual(ledger_reloaded.lifetime_total_tokens, 20200)

        # Test interrupted summary (Ctrl+C)
        tracker2 = SessionTracker(
            target_dir="C:/test/audio",
            model_name="gemini-3.7-flash",
            ledger_path=test_ledger_file
        )
        interrupted_summary = tracker2.generate_summary(interrupted=True)
        self.assertIn("INTERRUPTED BY USER (Ctrl+C)", interrupted_summary)
        self.assertIn("Checkpoints safely saved", interrupted_summary)


if __name__ == "__main__":
    unittest.main()
