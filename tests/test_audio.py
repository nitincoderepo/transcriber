import unittest
from pathlib import Path
from transcriber.audio import format_duration, AudioSlice
from transcriber.discovery import clean_title_from_filename


class TestAudioAndDiscovery(unittest.TestCase):
    def test_format_duration(self):
        self.assertEqual(format_duration(0.0), "00:00")
        self.assertEqual(format_duration(65.0), "01:05")
        self.assertEqual(format_duration(1800.0), "30:00")
        self.assertEqual(format_duration(3600.0), "01:00:00")
        self.assertEqual(format_duration(5460.55), "01:31:01")

    def test_parse_time_str(self):
        from transcriber.audio import parse_time_str
        self.assertIsNone(parse_time_str(None))
        self.assertIsNone(parse_time_str(""))
        self.assertEqual(parse_time_str("21:00"), 1260.0)
        self.assertEqual(parse_time_str("01:21:00"), 4860.0)
        self.assertEqual(parse_time_str("35:00"), 2100.0)
        self.assertEqual(parse_time_str("21m"), 1260.0)
        self.assertEqual(parse_time_str("21min"), 1260.0)
        self.assertEqual(parse_time_str("90s"), 90.0)
        self.assertEqual(parse_time_str("1.5h"), 5400.0)
        self.assertEqual(parse_time_str("1260"), 1260.0)
        self.assertEqual(parse_time_str(1260), 1260.0)

    def test_audio_slice_properties(self):
        s = AudioSlice(
            file_path=Path("sample_part01.mp3"),
            start_sec=1800.0,
            end_sec=3600.0,
            part_num=2,
            total_parts=4,
            is_temp=True
        )
        self.assertEqual(s.start_time_str, "30:00")
        self.assertEqual(s.end_time_str, "01:00:00")
        self.assertEqual(s.duration_sec, 1800.0)

    def test_clean_title_from_filename(self):
        name1 = "Sample_Audio_Track_Day_1.mp3"
        self.assertEqual(clean_title_from_filename(name1), "Sample Audio Track Day 1")
        name2 = "Audio - File - 2026.wav"
        self.assertEqual(clean_title_from_filename(name2), "Audio - File - 2026")

    def test_discover_audio_files_filters_hidden_and_appledouble(self):
        import tempfile
        from transcriber.discovery import discover_audio_files

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "01 Real Audio.mp3").write_text("dummy audio")
            (tmppath / "._01 Real Audio.mp3").write_text("apple metadata")
            (tmppath / ".DS_Store").write_text("mac junk")
            (tmppath / "subfolder").mkdir()
            (tmppath / "subfolder" / "._02 Sub.mp3").write_text("apple metadata")
            (tmppath / "subfolder" / "02 Sub.mp3").write_text("dummy audio 2")

            files = discover_audio_files(input_dir=tmppath, recursive=True)
            filenames = [f.name for f in files]
            self.assertIn("01 Real Audio.mp3", filenames)
            self.assertIn("02 Sub.mp3", filenames)
            self.assertNotIn("._01 Real Audio.mp3", filenames)
            self.assertNotIn("._02 Sub.mp3", filenames)
            self.assertNotIn(".DS_Store", filenames)


    def test_save_document_to_transcript_folder(self):
        import tempfile
        from transcriber.document import save_document
        from transcriber.audio import AudioSlice

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir)
            transcript_dir = input_dir / "transcript"
            output_docx = transcript_dir / "Sample_Recording_Transcript.docx"
            
            slice_mock = AudioSlice(
                file_path=input_dir / "Sample_Recording.mp3",
                start_sec=0.0,
                end_sec=60.0,
                part_num=1,
                total_parts=1,
                is_temp=False
            )
            slices_data = [(slice_mock, "આ એક ટેસ્ટ ટ્રાન્સક્રિપ્ટ છે.")]

            # Verify that save_document creates the transcript directory and files automatically
            save_document(
                output_docx_path=output_docx,
                title="Sample Recording",
                source_filename="Sample_Recording.mp3",
                total_duration_str="01:00",
                language="Gujarati",
                model_name="gemini-3.7-flash",
                timestamp_interval=2,
                slices_data=slices_data,
                save_txt=True
            )

            self.assertTrue(transcript_dir.exists())
            self.assertTrue(transcript_dir.is_dir())
            self.assertTrue(output_docx.exists())
            self.assertGreater(output_docx.stat().st_size, 0)
            
            output_txt = transcript_dir / "Sample_Recording_Transcript.txt"
            self.assertTrue(output_txt.exists())
            self.assertIn("આ એક ટેસ્ટ ટ્રાન્સક્રિપ્ટ છે.", output_txt.read_text(encoding="utf-8"))


    def test_is_audio_silent_and_volume_detect(self):
        import tempfile
        from transcriber.audio import is_audio_silent, detect_audio_volume

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Non-existent or empty file
            empty_file = tmppath / "empty.mp3"
            is_silent, vol = is_audio_silent(empty_file)
            self.assertTrue(is_silent)
            self.assertEqual(vol, -91.0)

            # File that does not exist
            missing_file = tmppath / "non_existent.mp3"
            is_silent, vol = is_audio_silent(missing_file)
            self.assertTrue(is_silent)

    def test_sanitize_arg_path(self):
        from transcribe import sanitize_arg_path
        self.assertIsNone(sanitize_arg_path(None))
        self.assertEqual(sanitize_arg_path(""), None)
        self.assertEqual(sanitize_arg_path('"/path/to/dir"'), "/path/to/dir")
        self.assertEqual(sanitize_arg_path("'/path/to/dir'"), "/path/to/dir")
        self.assertEqual(sanitize_arg_path("“/path/to/dir”"), "/path/to/dir")
        self.assertEqual(sanitize_arg_path("‘/path/to/dir’"), "/path/to/dir")
        self.assertEqual(sanitize_arg_path("  ”/Users/username/Transcript”  "), "/Users/username/Transcript")


if __name__ == "__main__":
    unittest.main()
