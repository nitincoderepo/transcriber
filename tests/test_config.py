import os
import unittest
from pathlib import Path
import tempfile
import shutil

from transcriber.config import load_env_file, parse_bool_env, TranscriberConfig


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_parse_bool_env(self):
        os.environ["TEST_TRUE_1"] = "true"
        os.environ["TEST_TRUE_2"] = "1"
        os.environ["TEST_FALSE_1"] = "false"
        os.environ["TEST_FALSE_2"] = "0"
        
        self.assertTrue(parse_bool_env("TEST_TRUE_1"))
        self.assertTrue(parse_bool_env("TEST_TRUE_2"))
        self.assertFalse(parse_bool_env("TEST_FALSE_1"))
        self.assertFalse(parse_bool_env("TEST_FALSE_2"))
        self.assertTrue(parse_bool_env("NON_EXISTENT", default=True))
        self.assertFalse(parse_bool_env("NON_EXISTENT", default=False))

    def test_load_env_file(self):
        env_file = self.test_dir / ".env"
        env_file.write_text(
            "GEMINI_MODEL=gemini-3.7-flash\nCHUNK_MINUTES=45\nSAVE_TXT=true\n",
            encoding="utf-8"
        )
        loaded = load_env_file([self.test_dir])
        self.assertEqual(loaded.get("CHUNK_MINUTES"), "45")
        self.assertEqual(loaded.get("SAVE_TXT"), "true")

    def test_transcriber_config_overrides(self):
        args = {
            "model": "gemini-2.5-pro",
            "chunk_minutes": 20.0,
            "no_timestamps": True,
            "save_txt": True,
            "force": True,
            "max_retry_minutes": 45.0
        }
        config = TranscriberConfig.from_env_and_args(args)
        self.assertEqual(config.model_name, "gemini-2.5-pro")
        self.assertEqual(config.chunk_minutes, 20.0)
        self.assertEqual(config.timestamp_interval, 0)
        self.assertTrue(config.save_txt)
        self.assertTrue(config.force)
        self.assertEqual(config.max_retry_minutes, 45.0)

    def test_time_range_config(self):
        # 1. Using explicit start and end
        args1 = {"start": "21:00", "end": "35:00"}
        cfg1 = TranscriberConfig.from_env_and_args(args1)
        self.assertEqual(cfg1.start_time_sec, 1260.0)
        self.assertEqual(cfg1.end_time_sec, 2100.0)

        # 2. Using range string
        args2 = {"range": "21:00-35:00"}
        cfg2 = TranscriberConfig.from_env_and_args(args2)
        self.assertEqual(cfg2.start_time_sec, 1260.0)
        self.assertEqual(cfg2.end_time_sec, 2100.0)

        # 3. Using minute notation
        args3 = {"range": "21m to 35m"}
        cfg3 = TranscriberConfig.from_env_and_args(args3)
        self.assertEqual(cfg3.start_time_sec, 1260.0)
        self.assertEqual(cfg3.end_time_sec, 2100.0)


if __name__ == "__main__":
    unittest.main()
