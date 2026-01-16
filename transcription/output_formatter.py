"""
Output formatting module for transcription results.

This module provides functions to export transcription results to various formats
including SRT subtitle files and plain text files.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import textwrap

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def seconds_to_srt_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).
    
    Args:
        seconds: Time in seconds (can be float)
        
    Returns:
        Formatted timestamp string in SRT format
        
    Example:
        >>> seconds_to_srt_timestamp(65.5)
        '00:01:05,500'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def _wrap_subtitle_text(text: str, max_width: int = 42) -> List[str]:
    """
    Wrap subtitle text without breaking words, targeting max_width characters.
    
    Args:
        text: Text to wrap
        max_width: Maximum characters per line
        
    Returns:
        List of wrapped text lines
    """
    if not text:
        return [""]
    
    wrapped = textwrap.wrap(
        text.strip(),
        width=max_width,
        break_long_words=False,
        break_on_hyphens=False
    )
    
    return wrapped if wrapped else [text.strip()]


def write_srt(transcription: Dict[str, Any], output_path: Path) -> Path:
    """
    Write transcription to SRT subtitle file format.
    
    Args:
        transcription: Dictionary containing 'timestamps' list with timing info
        output_path: Path where the SRT file should be written
        
    Returns:
        Path to the created SRT file
        
    Raises:
        ValueError: If timestamps are missing or empty
        IOError: If file cannot be written
    """
    timestamps = transcription.get("timestamps", [])
    
    if not timestamps:
        logger.warning("No timestamps available in transcription data")
        raise ValueError("Cannot generate SRT: timestamps are missing or empty")
    
    logger.info(f"Generating SRT file with {len(timestamps)} subtitle entries")
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for idx, chunk in enumerate(timestamps, start=1):
                timestamp_data = chunk.get("timestamp", [0.0, 0.0])
                text = chunk.get("text", "").strip()
                
                if not text:
                    continue
                
                if isinstance(timestamp_data, (list, tuple)) and len(timestamp_data) >= 2:
                    start_time = timestamp_data[0]
                    end_time = timestamp_data[1]
                else:
                    logger.warning(f"Invalid timestamp format at chunk {idx}: {timestamp_data}")
                    start_time = 0.0
                    end_time = 0.0
                
                start_ts = seconds_to_srt_timestamp(start_time)
                end_ts = seconds_to_srt_timestamp(end_time)
                
                wrapped_lines = _wrap_subtitle_text(text)
                
                f.write(f"{idx}\n")
                f.write(f"{start_ts} --> {end_ts}\n")
                for line in wrapped_lines:
                    f.write(f"{line}\n")
                f.write("\n")
        
        logger.info(f"SRT file written successfully: {output_path}")
        return output_path
        
    except PermissionError as e:
        logger.error(f"Permission denied writing to {output_path}: {e}")
        raise
    except OSError as e:
        logger.error(f"I/O error writing SRT file {output_path}: {e}")
        raise


def write_txt(transcription: Dict[str, Any], output_path: Path) -> Path:
    """
    Write transcription to plain text file with header.
    
    Args:
        transcription: Dictionary containing 'text' field with full transcription
        output_path: Path where the text file should be written
        
    Returns:
        Path to the created text file
        
    Raises:
        IOError: If file cannot be written
    """
    text = transcription.get("text", "")
    duration = transcription.get("duration_seconds", 0.0)
    
    logger.info(f"Generating plain text file: {output_path}")
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("TRANSCRIPTION\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            if output_path.stem:
                f.write(f"Source: {output_path.stem}\n")
            
            if duration > 0:
                f.write(f"Duration: {duration:.2f} seconds\n")
            
            f.write("="*80 + "\n\n")
            f.write(text)
            f.write("\n")
        
        logger.info(f"Text file written successfully: {output_path}")
        return output_path
        
    except PermissionError as e:
        logger.error(f"Permission denied writing to {output_path}: {e}")
        raise
    except OSError as e:
        logger.error(f"I/O error writing text file {output_path}: {e}")
        raise


def write_transcript(
    transcription: Dict[str, Any],
    original_audio_path: Path,
    output_dir: Path,
    preferred_format: str = "srt"
) -> Path:
    """
    Write transcription to file with automatic format fallback.
    
    This function orchestrates the output process, creating the output directory
    if needed and deriving the output filename from the original audio file.
    If SRT format is preferred but timestamps are unavailable, it automatically
    falls back to plain text format.
    
    Args:
        transcription: Dictionary containing transcription results
        original_audio_path: Path to the original audio file
        output_dir: Directory where output file should be written
        preferred_format: Preferred output format ('srt' or 'txt')
        
    Returns:
        Path to the created output file
        
    Raises:
        ValueError: If preferred_format is not 'srt' or 'txt'
        IOError: If file cannot be written
    """
    if preferred_format not in ("srt", "txt"):
        raise ValueError(f"Invalid format: {preferred_format}. Must be 'srt' or 'txt'")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory ensured: {output_dir}")
    
    base_name = original_audio_path.stem
    
    if preferred_format == "srt":
        timestamps = transcription.get("timestamps", [])
        
        if not timestamps:
            logger.warning(
                f"SRT format requested but no timestamps available. "
                f"Falling back to TXT format for {base_name}"
            )
            output_path = output_dir / f"{base_name}.txt"
            return write_txt(transcription, output_path)
        
        try:
            output_path = output_dir / f"{base_name}.srt"
            return write_srt(transcription, output_path)
        except ValueError as e:
            logger.warning(f"Failed to write SRT, falling back to TXT: {e}")
            output_path = output_dir / f"{base_name}.txt"
            return write_txt(transcription, output_path)
    
    else:
        output_path = output_dir / f"{base_name}.txt"
        return write_txt(transcription, output_path)


def main() -> None:
    """
    Demo script showing usage of output formatting functions.
    """
    logger.info("Running output formatter demo")
    
    demo_output_dir = Path("./_demo_output")
    demo_output_dir.mkdir(exist_ok=True)
    
    mock_transcription = {
        "text": "你好世界 歡迎使用廣東話語音識別系統 這是一個測試句子 希望一切順利",
        "timestamps": [
            {"text": "你好世界", "timestamp": [0.0, 1.5]},
            {"text": "歡迎使用廣東話語音識別系統", "timestamp": [1.5, 4.2]},
            {"text": "這是一個測試句子", "timestamp": [4.2, 6.0]},
            {"text": "希望一切順利", "timestamp": [6.0, 7.5]}
        ],
        "duration_seconds": 7.5,
        "error": None
    }
    
    mock_audio_path = Path("demo_audio.mp3")
    
    print("\n" + "="*80)
    print("DEMO: Writing SRT format")
    print("="*80)
    srt_path = write_transcript(
        mock_transcription,
        mock_audio_path,
        demo_output_dir,
        preferred_format="srt"
    )
    print(f"✓ SRT file created: {srt_path}")
    
    print("\n" + "="*80)
    print("DEMO: Writing TXT format")
    print("="*80)
    txt_path = write_transcript(
        mock_transcription,
        mock_audio_path,
        demo_output_dir,
        preferred_format="txt"
    )
    print(f"✓ TXT file created: {txt_path}")
    
    print("\n" + "="*80)
    print("DEMO: Fallback behavior (no timestamps)")
    print("="*80)
    mock_no_timestamps = {
        "text": "這是沒有時間戳記的轉錄文本",
        "timestamps": [],
        "duration_seconds": 3.0,
        "error": None
    }
    fallback_path = write_transcript(
        mock_no_timestamps,
        Path("no_timestamps.mp3"),
        demo_output_dir,
        preferred_format="srt"
    )
    print(f"✓ Fallback file created: {fallback_path}")
    
    print("\n" + "="*80)
    print(f"Demo complete! Check {demo_output_dir.absolute()} for output files.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
