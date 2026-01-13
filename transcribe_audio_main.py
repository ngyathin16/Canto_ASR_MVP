"""
Main orchestration script for Cantonese audio transcription pipeline.

This script coordinates the transcription service and output formatting to provide
a complete end-to-end transcription workflow.

Usage Examples (with activated venv):
    # Windows PowerShell:
    .venv\\Scripts\\Activate.ps1
    python transcribe_audio_main.py site_01.mp3
    
    # Unix/macOS:
    source .venv/bin/activate
    python transcribe_audio_main.py site_01.mp3
    
    # With custom output directory and verbose logging:
    python transcribe_audio_main.py site_01.mp3 --output-dir ./transcripts --verbose
    
    # Generate plain text instead of SRT:
    python transcribe_audio_main.py site_01.mp3 --output-format txt
    
    # Dry run (transcribe but don't write output):
    python transcribe_audio_main.py site_01.mp3 --dry-run --verbose
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from transcription_service import TranscriptionService
from output_formatter import write_transcript

logger = logging.getLogger(__name__)


def configure_logging(verbose: bool = False) -> None:
    """
    Configure logging for the application.
    
    Args:
        verbose: If True, set log level to DEBUG; otherwise INFO
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )


def check_venv() -> bool:
    """
    Check if running inside a virtual environment.
    
    Returns:
        True if inside a venv, False otherwise
    """
    return (
        hasattr(sys, 'real_prefix') or
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    )


def validate_input_file(file_path: Path) -> bool:
    """
    Validate that the input file exists and is an MP3.
    
    Args:
        file_path: Path to the input file
        
    Returns:
        True if valid, False otherwise
    """
    if not file_path.exists():
        logger.error(f"Input file does not exist: {file_path}")
        return False
    
    if not file_path.is_file():
        logger.error(f"Input path is not a file: {file_path}")
        return False
    
    if file_path.suffix.lower() not in ['.mp3', '.wav']:
        logger.warning(
            f"Input file extension is '{file_path.suffix}'. "
            "Expected .mp3 or .wav. Proceeding anyway..."
        )
    
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    logger.info(f"Input file validated: {file_path} ({file_size_mb:.2f} MB)")
    
    return True


def run(args: argparse.Namespace) -> int:
    """
    Execute the transcription pipeline.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0=success, 1=transcription error, 2=I/O error, 3=unexpected error)
    """
    configure_logging(args.verbose)
    
    logger.info("="*80)
    logger.info("Cantonese Audio Transcription Pipeline v0.1.0-mvp")
    logger.info("="*80)
    logger.info(f"Python version: {sys.version.split()[0]}")
    logger.info(f"Running in virtual environment: {check_venv()}")
    logger.info(f"Dry run mode: {args.dry_run}")
    logger.info("")
    logger.info("VAD Configuration:")
    logger.info(f"  Energy threshold: {args.vad_energy_threshold}")
    logger.info(f"  Min segment duration: {args.vad_min_segment_duration}s")
    logger.info(f"  Merge gap: {args.vad_merge_gap}s")
    logger.info(f"  Max segment length: {args.max_segment_length}s")
    logger.info(f"  Timestamp mode: {args.return_timestamps}")
    logger.info("")
    logger.info("Decoding Configuration:")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Compression ratio threshold: {args.compression_ratio_threshold}")
    logger.info(f"  Log probability threshold: {args.logprob_threshold}")
    logger.info(f"  No-speech threshold: {args.no_speech_threshold}")
    logger.info("="*80)
    
    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    
    if not validate_input_file(input_path):
        return 1
    
    try:
        logger.info("Initializing transcription service...")
        service = TranscriptionService(
            temperature=args.temperature,
            compression_ratio_threshold=args.compression_ratio_threshold,
            logprob_threshold=args.logprob_threshold,
            no_speech_threshold=args.no_speech_threshold,
            vad_energy_threshold=args.vad_energy_threshold,
            vad_min_segment_duration=args.vad_min_segment_duration,
            vad_merge_gap=args.vad_merge_gap,
            max_segment_length=args.max_segment_length,
            return_timestamps=args.return_timestamps
        )
        
        logger.info(f"Starting transcription: {input_path}")
        start_time = time.time()
        
        transcription = service.transcribe(str(input_path))
        
        elapsed_time = time.time() - start_time
        logger.info(f"Transcription completed in {elapsed_time:.2f} seconds")
        
        if transcription.get("error"):
            logger.error(f"Transcription failed: {transcription['error']}")
            return 1
        
        logger.info(f"Transcript length: {len(transcription.get('text', ''))} characters")
        logger.info(f"Timestamps: {len(transcription.get('timestamps', []))} chunks")
        
        if args.dry_run:
            logger.info("Dry run mode: skipping output file generation")
            logger.info("Transcription preview (first 200 chars):")
            logger.info(transcription.get('text', '')[:200])
            return 0
        
        logger.info(f"Writing output to: {output_dir}")
        output_path = write_transcript(
            transcription=transcription,
            original_audio_path=input_path,
            output_dir=output_dir,
            preferred_format=args.output_format
        )
        
        logger.info("="*80)
        logger.info(f"✓ SUCCESS: Output written to {output_path}")
        logger.info(f"✓ Total elapsed time: {elapsed_time:.2f} seconds")
        logger.info("="*80)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return 1
    except (PermissionError, OSError) as e:
        logger.error(f"I/O error: {e}")
        return 2
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 3


def main() -> None:
    """
    Main entry point for the transcription pipeline.
    
    Parses command-line arguments and executes the pipeline with appropriate
    error handling and exit codes.
    """
    parser = argparse.ArgumentParser(
        description="Transcribe Cantonese audio files to text with timestamps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcribe_audio_main.py site_01.mp3
  python transcribe_audio_main.py site_01.mp3 --output-dir ./transcripts --verbose
  python transcribe_audio_main.py site_01.mp3 --output-format txt
  python transcribe_audio_main.py site_01.mp3 --dry-run
        """
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input audio file (MP3 or WAV)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./transcripts",
        help="Directory for output files (default: ./transcripts)"
    )
    
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["srt", "txt"],
        default="srt",
        help="Output format: srt (subtitles) or txt (plain text) (default: srt)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Transcribe but don't write output file"
    )
    
    advanced_group = parser.add_argument_group(
        "Advanced Options",
        "These options control VAD and decoding behavior. Most users can ignore these."
    )
    
    advanced_group.add_argument(
        "--vad-energy-threshold",
        type=float,
        default=0.015,
        help="Energy/RMS threshold for VAD speech detection (relative to max energy). "
             "Lower values detect more speech but may include noise. (default: 0.015)"
    )
    
    advanced_group.add_argument(
        "--vad-min-segment-duration",
        type=float,
        default=0.3,
        help="Minimum duration (seconds) for a speech segment to be kept. "
             "Shorter segments are filtered as noise. (default: 0.3)"
    )
    
    advanced_group.add_argument(
        "--vad-merge-gap",
        type=float,
        default=0.3,
        help="Maximum gap (seconds) between speech segments to merge them. "
             "Segments closer than this are combined. (default: 0.3)"
    )
    
    advanced_group.add_argument(
        "--max-segment-length",
        type=float,
        default=30.0,
        help="Maximum length (seconds) for a single segment. Longer segments are split "
             "to prevent memory issues during transcription. (default: 30.0)"
    )
    
    advanced_group.add_argument(
        "--return-timestamps",
        type=str,
        choices=["word", "chunk"],
        default="chunk",
        help="Timestamp granularity. 'chunk' is 3-5x faster than 'word'. "
             "Use 'word' only if you need precise word-level timing. (default: chunk)"
    )
    
    advanced_group.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for generation. 0.0 = deterministic (greedy). "
             "Higher values increase randomness. (default: 0.0)"
    )
    
    advanced_group.add_argument(
        "--compression-ratio-threshold",
        type=float,
        default=2.4,
        help="Threshold for detecting repetitive/compressed output. "
             "Lower values are more strict. (default: 2.4)"
    )
    
    advanced_group.add_argument(
        "--logprob-threshold",
        type=float,
        default=-1.0,
        help="Minimum average log probability for generated tokens. "
             "Segments below this may be filtered. (default: -1.0)"
    )
    
    advanced_group.add_argument(
        "--no-speech-threshold",
        type=float,
        default=0.6,
        help="Threshold for detecting non-speech segments. "
             "Higher values are more strict. (default: 0.6)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="0.1.0-mvp"
    )
    
    args = parser.parse_args()
    
    exit_code = run(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
