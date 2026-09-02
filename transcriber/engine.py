"""
Gemini AI transcription engine, prompt builder, resilient configurable retry handler,
token cost tracker, and daily quota limit guard.
"""

import time
from pathlib import Path
from typing import Optional, Tuple
from google import genai
from google.genai import types
from google.genai.errors import APIError

from .audio import AudioSlice, format_duration
from .logger import TranscriptLogger, get_logger, format_wait_time, get_progress_bar
from .cost import calculate_cost, format_cost


def build_prompt(
    slice_info: AudioSlice,
    audio_title: str,
    language: str,
    timestamp_interval: int,
    custom_instructions: Optional[str] = None
) -> str:
    """Construct an accurate prompt including time offsets and timestamp minute markers."""
    
    part_context = ""
    if slice_info.total_parts > 1:
        part_context = f"This audio slice covers time range [{slice_info.start_time_str} - {slice_info.end_time_str}] (Part {slice_info.part_num} of {slice_info.total_parts}) from '{audio_title}'."
    else:
        part_context = f"This is the complete audio recording for '{audio_title}'."

    timestamp_instruction = ""
    if timestamp_interval > 0:
        example_start = slice_info.start_time_str
        example_next = format_duration(slice_info.start_sec + timestamp_interval * 60)
        timestamp_instruction = f"3. Accurate Minute Markers: Include timestamps every {timestamp_interval} minutes (e.g. [{example_start}], [{example_next}], etc.) matching the timeline of the segment."
    else:
        timestamp_instruction = "3. Plain Transcript: Do not include timestamps."

    prompt = f"""Transcribe this audio file completely and verbatim in {language}.
{part_context}

Requirements:
1. Complete Verbatim Transcription: Transcribe the entire speech verbatim from start to finish without summarizing, condensing, omitting, or guessing any section.
2. Strict Verse & Quotation Fidelity: Transcribe verses, hymns, chants, shlokas, stotrams, mantras, and quotations strictly as pronounced and chanted in the audio. DO NOT autocomplete verses from memory, DO NOT substitute standard/canonical versions, and DO NOT insert unstated lines.
3. Acoustic Grounding: Every transcribed word must be grounded strictly in the spoken audio. If a word or phrase is unclear, transcribe phonetically as heard in {language} rather than replacing it with other texts.
4. Speaker Labels: Include clear speaker labels (e.g. **Speaker 1:**, **Speaker 2:**, or **વક્તા:**) whenever the speaker changes.
{timestamp_instruction}
5. Clean Output: Do not include conversational AI introductory preambles, meta-commentary, or apologies. Begin directly with the transcript."""

    if custom_instructions:
        prompt += f"\n\nAdditional Instructions:\n{custom_instructions}"

    return prompt


def is_daily_limit_error(e: Exception) -> Tuple[bool, str]:
    """
    Determine if an exception is a permanent daily quota limit or depleted credits.
    Returns (is_daily_limit, reason_description).
    """
    err_str = str(e).lower()
    
    if "prepayment credits are depleted" in err_str or "prepayment credits" in err_str:
        return True, "Prepayment credits depleted on your Google AI project."
    
    if "per day" in err_str or "daily limit" in err_str:
        return True, "Daily free-tier request/token limit reached."
        
    code = getattr(e, "code", None)
    if code == 429 and ("free_tier_requests" in err_str or "daily" in err_str):
        return True, "Daily API quota exhausted (429 RESOURCE_EXHAUSTED)."
        
    return False, ""


def print_quota_exhausted_guide(model_name: str, reason: str = "", logger: Optional[TranscriptLogger] = None):
    """Print a clear, highlighted guide on console and log file when daily quota is reached."""
    log = logger or get_logger()
    
    reason_text = reason or "Your Gemini API key reached the daily limit for model."
    log.error(f"Daily quota limit reached for model '{model_name}': {reason_text}", console=False)

    print("\n" + "=" * 78)
    print("🛑 GEMINI API DAILY LIMIT REACHED (429 RESOURCE_EXHAUSTED)")
    print("=" * 78)
    print(f"📌 Reason: {reason_text}")
    print(f"   Model:  {model_name}")
    print("\n⏰ WHEN TO RETURN (Daily Quota Reset):")
    print("   • Google resets daily free quotas every day at 12:00 AM Midnight Pacific Time (PT).")
    print("   • You can resume tomorrow by simply re-running the script.")
    print("\n💾 YOUR PROGRESS IS SAFELY SAVED:")
    print("   • All finished slices are saved in the local checkpoint cache.")
    print("   • Tomorrow's run will resume directly from the remaining slice without repeating.")
    print("\n💳 HOW TO CONTINUE IMMEDIATELY:")
    print("   • Option 1: Add a secondary API key into your '.env' file.")
    print("   • Option 2: Enable pay-as-you-go billing at https://aistudio.google.com/")
    print("=" * 78 + "\n")


def transcribe_slice(
    client: genai.Client,
    slice_info: AudioSlice,
    audio_title: str,
    language: str,
    model_name: str,
    timestamp_interval: int = 2,
    custom_instructions: Optional[str] = None,
    max_retry_budget_sec: float = 1800.0,
    logger: Optional[TranscriptLogger] = None
) -> Tuple[str, int, int, float]:
    """
    Upload audio slice to Gemini, generate transcription, handle resilient configurable
    retries for temporary pauses/high demand, and exit cleanly on daily rate limits.
    Returns: (transcript_text, prompt_tokens, candidate_tokens, cost_usd)
    """
    log = logger or get_logger()
    file_name = slice_info.file_path.name
    part_label = f"Part {slice_info.part_num}/{slice_info.total_parts}"

    log.info(f"  [1/2] ☁️ Uploading {file_name} ({slice_info.start_time_str} - {slice_info.end_time_str}) to Gemini...")
    
    upload_start = time.time()
    try:
        audio_file = client.files.upload(file=str(slice_info.file_path))
    except Exception as e:
        log.error(f"Failed to upload audio slice '{file_name}' to Gemini: {e}", exc=e)
        raise

    upload_sec = time.time() - upload_start
    log.debug(f"Audio uploaded successfully in {upload_sec:.1f}s. URI: {audio_file.name}, State: {audio_file.state}")
    
    prompt = build_prompt(
        slice_info=slice_info,
        audio_title=audio_title,
        language=language,
        timestamp_interval=timestamp_interval,
        custom_instructions=custom_instructions
    )
    
    thinking_config = None
    if "3.7" in model_name or "thinking" in model_name:
        thinking_config = types.ThinkingConfig(thinking_budget=0)

    gen_config = types.GenerateContentConfig(
        temperature=0.0,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        thinking_config=thinking_config
    )

    log.info(f"  [2/2] ✍️ Transcribing with model '{model_name}'...")
    
    start_retry_time = time.time()
    attempt = 0
    short_retry_count = 0
    backoff_intervals = [15, 30, 45, 60, 90, 120, 180, 240, 300]

    try:
        while True:
            attempt += 1

            try:
                log.debug(f"Calling generate_content (Attempt {attempt}, {part_label})")
                call_start = time.time()
                
                response = client.models.generate_content(
                    model=model_name,
                    contents=[audio_file, prompt],
                    config=gen_config
                )
                
                text = (response.text or "").strip()
                duration_call = time.time() - call_start
                
                # Extract token usage metadata from Gemini response if available
                prompt_tok = 0
                candidate_tok = 0
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    prompt_tok = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                    candidate_tok = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
                
                cost_usd = calculate_cost(model_name, prompt_tok, candidate_tok)
                cost_str = format_cost(cost_usd)

                # Differentiate silence/instrumental from errors and cap verification retries to max 1
                if (len(text) == 0 or (len(text) < 40 and slice_info.duration_sec > 180)) and short_retry_count < 1:
                    short_retry_count += 1
                    log.warning(
                        f"  ⚠️ Warning: Response seems unusually short ({len(text)} chars) on attempt {attempt}. "
                        f"Verifying once ({short_retry_count}/1 retry)..."
                    )
                    time.sleep(3)
                    continue

                # If still empty or filtered after 1 verification attempt, handle gracefully without looping
                if not text:
                    candidates = getattr(response, "candidates", None) or []
                    finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
                    if finish_reason and "SAFETY" in str(finish_reason).upper():
                        text = "[સામગ્રી ફિલ્ટર કરવામાં આવી / Content Filtered by Safety Guard]"
                        log.warning(f"  ⚠️ Part {part_label} was flagged by safety filter ({finish_reason}).")
                    else:
                        text = "[સંગીત / મૌન / Silence / Instrumental]"
                        log.info(f"  ℹ️ No vocal speech detected for {part_label}. Marked as silence/instrumental.")

                log.info(
                    f"        ✅ Succeeded in {duration_call:.1f}s! "
                    f"Received {len(text):,} chars (~{cost_str} USD | {prompt_tok + candidate_tok:,} tokens)"
                )
                log.debug(
                    f"Token details for {part_label}: {prompt_tok:,} in + {candidate_tok:,} out "
                    f"= {prompt_tok + candidate_tok:,} tokens (${cost_usd:.5f} USD)"
                )
                return text, prompt_tok, candidate_tok, cost_usd

            except APIError as e:
                # 1. Check if Daily Limit reached
                is_daily, daily_reason = is_daily_limit_error(e)
                if is_daily:
                    log.error(f"Daily rate limit reached during {part_label}: {e}", exc=e, console=False)
                    print_quota_exhausted_guide(model_name=model_name, reason=daily_reason, logger=log)
                    raise

                # 2. Transient errors (503 UNAVAILABLE, rate spike, etc.)
                err_str = str(e)
                code = getattr(e, "code", None)
                is_503 = code == 503 or "UNAVAILABLE" in err_str or "503" in err_str or "high demand" in err_str.lower()
                
                log.debug(f"APIError on attempt {attempt} for {part_label}: Code {code}, Details: {err_str}")

                # Determine backoff duration
                backoff_idx = min(attempt - 1, len(backoff_intervals) - 1)
                sleep_sec = backoff_intervals[backoff_idx]
                if not is_503 and sleep_sec > 60:
                    sleep_sec = 60

                total_time_if_wait = (time.time() - start_retry_time) + sleep_sec
                if total_time_if_wait > max_retry_budget_sec:
                    log.error(f"❌ Exceeded configured retry budget ({format_wait_time(max_retry_budget_sec)}) for {part_label}.", exc=e)
                    raise

                wait_str = format_wait_time(sleep_sec)
                elapsed_str = format_wait_time(time.time() - start_retry_time)
                budget_str = format_wait_time(max_retry_budget_sec)

                if is_503:
                    print(f"        ⏳ Model is experiencing high demand (503). Waiting {wait_str} before retrying {part_label}...")
                    print(f"           (Attempt {attempt} | {elapsed_str} elapsed of {budget_str} retry budget. Please check later.)")
                else:
                    print(f"        ⏳ Temporary API pause ({code or 'notice'}). Waiting {wait_str} before retrying {part_label}...")
                    print(f"           (Attempt {attempt} | {elapsed_str} elapsed of {budget_str} retry budget. Please check later.)")

                time.sleep(sleep_sec)

            except Exception as e:
                log.error(f"Unexpected error on attempt {attempt} for {part_label}: {e}", exc=e, console=False)
                
                is_daily, daily_reason = is_daily_limit_error(e)
                if is_daily:
                    print_quota_exhausted_guide(model_name=model_name, reason=daily_reason, logger=log)
                    raise

                backoff_idx = min(attempt - 1, len(backoff_intervals) - 1)
                sleep_sec = backoff_intervals[backoff_idx]
                
                total_time_if_wait = (time.time() - start_retry_time) + sleep_sec
                if total_time_if_wait > max_retry_budget_sec:
                    log.error(f"❌ Exceeded configured retry budget for {part_label}: {e}", exc=e)
                    raise

                wait_str = format_wait_time(sleep_sec)
                print(f"        ⏳ Connection/Server notice: {type(e).__name__}. Waiting {wait_str} before retrying {part_label}... Please check later.")
                time.sleep(sleep_sec)

    finally:
        try:
            client.files.delete(name=audio_file.name)
            log.debug(f"Cleaned up remote Gemini file: {audio_file.name}")
        except Exception as e:
            log.debug(f"Could not delete remote Gemini file {getattr(audio_file, 'name', '')}: {e}")
