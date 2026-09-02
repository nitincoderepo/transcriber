import unittest
from pathlib import Path
import tempfile
import shutil

from transcriber.checkpoint import CheckpointManager


class TestCheckpoint(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_checkpoint_store_and_reload(self):
        # 1. Initialize checkpoint manager
        mgr1 = CheckpointManager(self.test_dir, "test_audio.mp3")
        self.assertIsNone(mgr1.get_slice_transcript(1))

        # 2. Store slice 1 transcript
        transcript_1 = "This is a valid long enough sample transcript for testing checkpoint storage."
        mgr1.store_slice_transcript(1, 0.0, 1800.0, transcript_1)
        self.assertEqual(mgr1.get_slice_transcript(1), transcript_1)

        # 3. Simulate process restart by reloading from disk in a fresh instance
        mgr2 = CheckpointManager(self.test_dir, "test_audio.mp3")
        self.assertEqual(mgr2.get_slice_transcript(1), transcript_1)
        self.assertIsNone(mgr2.get_slice_transcript(2))


if __name__ == "__main__":
    unittest.main()
