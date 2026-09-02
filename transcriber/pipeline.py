"""
Transcription pipeline orchestrator for single file and batch audio processing.
Provides clean console progress tracking, token usage / cost metrics, and transcript logging.
"""

import time
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
from google import genai

from .config import TranscriberConfig, resolve_api_key
from .audio import (
    AudioSlice, get_audio_duration, format_duration, slice_audio_file,
    check_ffmpeg_available, is_audio_silent, SUPPORTED_AUDIO_EXTENSIONS
)
from .checkpoint import CheckpointManager
from .engine import transcribe_slice
from .document import save_document
from .discovery import discover_audio_files, clean_title_from_filename
from .logger import TranscriptLogger, get_progress_bar
from .cost import TokenUsageSummary, format_cost, SessionTracker, FileRunRecord


def process_audio_file(
    file_path: Path,
    client: Optional[genai.Client],
    config: TranscriberConfig,
    tracker: Optional[SessionTracker] = None
) -> TokenUsageSummary:
    """Process a single audio file with deduplication, slice caching, cost tracking, and checkpoint resuming."""
    file_name = file_path.name
    total_duration = get_audio_duration(file_path)
    total_duration_str = format_duration(total_duration)
    base_title = clean_title_from_filename(file_name)
    output_dir = config.output_dir or (file_path.parent / "transcript")
    output_dir.mkdir(parents=True, exist_ok=True)

    file_tokens = TokenUsageSummary()

    # Dedicated transcript file logger
    logger = TranscriptLogger(output_dir)

    if total_duration <= 0.0:
        logger.warning(f"⚠️ Skipping '{file_name}': Audio file duration is 0.0s or unreadable by FFmpeg.")
        if tracker:
            tracker.add_record(FileRunRecord(
                file_name=file_name,
                segment_label=file_name,
                audio_duration_sec=0.0,
                elapsed_processing_sec=0.01,
                clip_start_str="00:00",
                clip_end_str="00:00",
                status="⚠️ Skipped (0s)"
            ))
        return file_tokens

    # Check if start offset exceeds file duration
    if config.start_time_sec is not None and config.start_time_sec >= total_duration:
        logger.warning(
            f"⚠️ Skipping '{file_name}': File duration ({total_duration_str}) is shorter "
            f"than requested start offset ({format_duration(config.start_time_sec)})."
        )
        if tracker:
            tracker.add_record(FileRunRecord(
                file_name=file_name,
                segment_label=file_name,
                audio_duration_sec=total_duration,
                elapsed_processing_sec=0.01,
                clip_start_str=format_duration(config.start_time_sec or 0.0),
                clip_end_str=format_duration(total_duration),
                status="⚠️ Out of Range"
            ))
        return file_tokens

    # Determine custom time range vs full file
    is_custom_range = (
        (config.start_time_sec is not None and config.start_time_sec > 0) or
        (config.end_time_sec is not None and config.end_time_sec < total_duration)
    )

    seg_start = max(0.0, config.start_time_sec if config.start_time_sec is not None else 0.0)
    seg_end = min(total_duration, config.end_time_sec if config.end_time_sec is not None else total_duration)
    seg_duration = max(0.0, seg_end - seg_start)
    seg_duration_str = format_duration(seg_duration)

    if is_custom_range:
        start_tag = format_duration(seg_start).replace(":", "-")
        end_tag = format_duration(seg_end).replace(":", "-")
        doc_title = f"{base_title} (Segment {format_duration(seg_start)} - {format_duration(seg_end)})"
        output_docx_name = f"{base_title}_Segment_{start_tag}_to_{end_tag}_Transcript.docx"
        temp_dir = output_dir / "_temp_chunks" / f"{file_path.stem}_range_{start_tag}_to_{end_tag}"
        display_duration_str = f"{seg_duration_str} (Range: {format_duration(seg_start)} - {format_duration(seg_end)} | Original File: {total_duration_str})"
    else:
        doc_title = base_title
        output_docx_name = f"{base_title}_Transcript.docx"
        temp_dir = output_dir / "_temp_chunks" / file_path.stem
        display_duration_str = total_duration_str

    output_docx_path = output_dir / output_docx_name

    # 1. Deduplication: Check if already transcribed
    if output_docx_path.exists() and output_docx_path.stat().st_size > 1024 and not config.force and not config.dry_run:
        logger.info(f"⏩ Skipping '{file_name}': Transcript document already exists.")
        logger.info(f"   -> {output_docx_path.name} (Use --force to re-transcribe)")
        if tracker:
            tracker.add_record(FileRunRecord(
                file_name=file_name,
                segment_label=file_name,
                audio_duration_sec=seg_duration,
                elapsed_processing_sec=0.01,
                clip_start_str=format_duration(seg_start),
                clip_end_str=format_duration(seg_end),
                status="⚡ Skipped (Exists)"
            ))
        return file_tokens

    logger.info("\n" + "=" * 78)
    logger.info(f"🎵 Audio:    {file_name}")
    logger.info(f"⏱️ Duration: {display_duration_str}")
    logger.info(f"📁 Output:   {output_dir}")
    logger.info(f"📋 Log File: {output_dir / 'transcription.log'}")
    logger.info(f"⏳ Retry:    Up to {config.max_retry_minutes:.0f} mins per slice")
    logger.info("=" * 78)

    # 2. Slice audio file if needed (with slice caching)
    slices = slice_audio_file(
        input_path=file_path,
        chunk_minutes=config.chunk_minutes,
        temp_dir=temp_dir,
        start_time_sec=config.start_time_sec,
        end_time_sec=config.end_time_sec,
        dry_run=config.dry_run
    )

    if not slices:
        logger.warning(f"⚠️ No valid audio slices generated for '{file_name}' in range {format_duration(seg_start)} to {format_duration(seg_end)}.")
        return file_tokens

    # Dry-run mode: show preview
    if config.dry_run:
        print(f"\n[DRY RUN PREVIEW]")
        print(f"  - Document Title:      {doc_title}")
        print(f"  - Target Time Range:   {format_duration(seg_start)} -> {format_duration(seg_end)} ({seg_duration_str})")
        print(f"  - Planned Slices:      {len(slices)}")
        for s in slices:
            print(f"    * Part {s.part_num}/{s.total_parts}: {s.start_time_str} -> {s.end_time_str} ({s.file_path.name})")
        print(f"  - Target Word Doc:     {output_docx_path}")
        print(f"  - Language Prompt:     {config.language}")
        print(f"  - Minute Markers:      {'Every ' + str(config.timestamp_interval) + ' min' if config.timestamp_interval > 0 else 'Disabled'}")
        print(f"  - Gemini Model:        {config.model_name}")
        print(f"  - Max Retry Budget:    {config.max_retry_minutes:.0f} minutes")
        print(f"  - Already Transcribed: {'Yes (would be skipped unless --force)' if output_docx_path.exists() else 'No'}")
        if tracker:
            tracker.add_record(FileRunRecord(
                file_name=file_name,
                segment_label=doc_title,
                audio_duration_sec=seg_duration,
                elapsed_processing_sec=0.01,
                clip_start_str=format_duration(seg_start),
                clip_end_str=format_duration(seg_end),
                status="🔍 Dry Run"
            ))
        return file_tokens

    # 3. Checkpoint Manager for fault-tolerant resuming
    checkpoint = CheckpointManager(temp_dir=temp_dir, audio_filename=file_name)

    # 4. Transcribe each slice (or resume from checkpoint)
    slices_data: List[Tuple[AudioSlice, str]] = []
    retry_budget_sec = config.max_retry_minutes * 60.0

    try:
        total_slices = len(slices)
        for idx, slice_info in enumerate(slices, start=1):
            progress_str = get_progress_bar(idx - 1, total_slices)
            print(f"\n--- 🎙️ [{idx}/{total_slices}] Part {idx:02d} ({slice_info.start_time_str} - {slice_info.end_time_str}) {progress_str} ---")
            
            slice_start_time = time.time()
            slice_status = "✅ Completed"

            # Check if this slice is already in checkpoint
            cached_text = checkpoint.get_slice_transcript(slice_info.part_num)
            if cached_text and not config.force:
                prompt_tok, cand_tok, cost_val = checkpoint.get_slice_tokens(slice_info.part_num)
                file_tokens.add(prompt_tok, cand_tok, cost_val)
                logger.info(f"  ⚡ [Checkpoint Resumed] Part {idx} loaded from local cache! (Skipped API call & saved tokens)")
                transcript_text = cached_text
                slice_status = "⚡ Cached"
            else:
                # Local silence pre-check: skip Gemini upload if audio slice has no audible vocal speech ($0 API cost)
                is_silent, peak_db = is_audio_silent(slice_info.file_path, max_db_threshold=-45.0)
                if is_silent and not config.force:
                    transcript_text = "[સંગીત / મૌન / Silence / Instrumental]"
                    prompt_tok, cand_tok, cost_val = 0, 0, 0.0
                    slice_status = "🔇 Silent ($0)"
                    logger.info(
                        f"  🔇 [Local Silence Detected] Part {idx} contains no audible speech "
                        f"(Peak: {peak_db:.1f} dB). Skipped Gemini API upload ($0.00 saved)."
                    )
                else:
                    transcript_text, prompt_tok, cand_tok, cost_val = transcribe_slice(
                        client=client,
                        slice_info=slice_info,
                        audio_title=doc_title,
                        language=config.language,
                        model_name=config.model_name,
                        timestamp_interval=config.timestamp_interval,
                        custom_instructions=config.custom_instructions,
                        max_retry_budget_sec=retry_budget_sec,
                        logger=logger
                    )
                    file_tokens.add(prompt_tok, cand_tok, cost_val)
                    slice_status = "✅ Completed"

                # Store in checkpoint immediately for recovery
                checkpoint.store_slice_transcript(
                    part_num=slice_info.part_num,
                    start_sec=slice_info.start_sec,
                    end_sec=slice_info.end_sec,
                    transcript=transcript_text,
                    prompt_tokens=prompt_tok,
                    candidate_tokens=cand_tok,
                    cost_usd=cost_val
                )

            slice_elapsed = time.time() - slice_start_time
            if tracker:
                slice_label = f"{file_path.stem} (Part {idx:02d})"
                tracker.add_record(FileRunRecord(
                    file_name=file_name,
                    segment_label=slice_label,
                    audio_duration_sec=slice_info.duration_sec,
                    elapsed_processing_sec=slice_elapsed,
                    clip_start_str=slice_info.start_time_str,
                    clip_end_str=slice_info.end_time_str,
                    prompt_tokens=prompt_tok,
                    candidate_tokens=cand_tok,
                    cost_usd=cost_val,
                    status=slice_status
                ))

            slices_data.append((slice_info, transcript_text))

            if idx < total_slices and not cached_text:
                time.sleep(4)

        # 5. Assemble and save the final document
        overall_progress = get_progress_bar(total_slices, total_slices)
        print(f"\n💾 Assembling Word Document... {overall_progress}")
        save_document(
            output_docx_path=output_docx_path,
            title=doc_title,
            source_filename=file_name,
            total_duration_str=total_duration_str,
            language=config.language,
            model_name=config.model_name,
            timestamp_interval=config.timestamp_interval,
            slices_data=slices_data,
            save_txt=config.save_txt
        )
        
        # 6. Display cost & token metrics
        cost_str = format_cost(file_tokens.estimated_cost_usd)
        logger.info(
            f"💰 Cost Summary for '{file_name}': "
            f"{file_tokens.total_tokens:,} tokens (~{cost_str} USD | "
            f"{file_tokens.prompt_tokens:,} in + {file_tokens.candidate_tokens:,} out)"
        )
        logger.info(f"✅ Successfully completed transcription for '{file_name}'")

    finally:
        all_completed = len(slices_data) == len(slices)
        if all_completed and not config.keep_chunks and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info(f"  🧹 Cleaned up temporary slices folder ({temp_dir.name})", console=True)
            except Exception:
                pass
        elif not all_completed:
            logger.info(f"\n💾 Progress saved in checkpoint! You can resume anytime by re-running the script.", console=True)
        
        logger.close()

    return file_tokens


def run_pipeline(
    target_files: Optional[List[str]] = None,
    input_dir: Optional[Path] = None,
    pattern: Optional[str] = None,
    name_filter: Optional[str] = None,
    recursive: bool = False,
    config: Optional[TranscriberConfig] = None
):
    """High-level batch pipeline execution with clean console reporting, session tracker, and cost totals."""
    cfg = config or TranscriberConfig.from_env_and_args()

    # 1. Check FFmpeg availability
    ffmpeg_ok, ffprobe_ok = check_ffmpeg_available()
    if not (ffmpeg_ok and ffprobe_ok):
        print("⚠️ Warning: 'ffmpeg' or 'ffprobe' was not found in PATH.")
        print("   Audio duration probing and slicing may be limited.")
        if cfg.chunk_minutes > 0:
            print("   To enable audio slicing, please install FFmpeg and add it to your system PATH.")

    # 2. Discover audio files
    search_dir = input_dir if input_dir else Path.cwd()
    audio_files = discover_audio_files(
        target_files=target_files,
        input_dir=search_dir,
        pattern=pattern,
        name_filter=name_filter,
        recursive=recursive
    )

    if not audio_files:
        print(f"\n❌ No matching audio files found in: {search_dir}")
        print(f"Supported formats: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}")
        return

    print("\n" + "=" * 78)
    print(f"🚀 Audio Transcription Engine")
    print("=" * 78)
    print(f"📂 Found {len(audio_files)} audio file(s) to process:")
    for i, af in enumerate(audio_files, 1):
        print(f"   {i}. {af.name}")
    print("=" * 78)

    # 3. Initialize Gemini Client
    client = None
    if not cfg.dry_run:
        api_key = resolve_api_key(cfg.api_key)
        client = genai.Client(api_key=api_key)

    # 4. Initialize Run Session Tracker
    tracker = SessionTracker(
        target_dir=str(search_dir),
        model_name=cfg.model_name,
        starting_balance=cfg.starting_balance
    )

    # 5. Process all files with graceful KeyboardInterrupt (Ctrl+C) handling
    total_batch_tokens = TokenUsageSummary()
    interrupted = False

    try:
        for idx, audio_file in enumerate(audio_files, start=1):
            batch_progress = get_progress_bar(idx - 1, len(audio_files))
            print(f"\n[{idx}/{len(audio_files)}] Batch Item: {audio_file.name} {batch_progress}")
            file_tokens = process_audio_file(
                file_path=audio_file,
                client=client,
                config=cfg,
                tracker=tracker
            )
            if file_tokens:
                total_batch_tokens.add(
                    file_tokens.prompt_tokens,
                    file_tokens.candidate_tokens,
                    file_tokens.estimated_cost_usd
                )
    except KeyboardInterrupt:
        interrupted = True
        print("\n\n🛑 Process interrupted by user (Ctrl+C). Checkpoints are safely preserved.")

    # 6. Display comprehensive session summary report
    summary_text = tracker.generate_summary(interrupted=interrupted)
    print(summary_text)

    # Also record summary to transcription.log if an output directory was created
    log_dir = cfg.output_dir if cfg.output_dir else (audio_files[0].parent / "transcript" if audio_files else None)
    if log_dir and log_dir.exists():
        try:
            log_path = log_dir / "transcription.log"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(summary_text + "\n")
        except Exception:
            pass
