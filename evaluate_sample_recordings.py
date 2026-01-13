"""
Evaluation script for quick feedback on VAD and decoding configurations.

This script processes multiple audio files from a directory to help tune
VAD and decoding parameters before formal dataset evaluation.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

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


def find_audio_files(input_dir: Path, max_files: int = None) -> List[Path]:
    """
    Find all audio files in the input directory.
    
    Args:
        input_dir: Directory to search for audio files
        max_files: Maximum number of files to return (None = no limit)
        
    Returns:
        List of audio file paths
    """
    audio_extensions = {'.mp3', '.wav'}
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(input_dir.glob(f'*{ext}'))
    
    audio_files.sort()
    
    if max_files is not None and max_files > 0:
        audio_files = audio_files[:max_files]
    
    return audio_files


def run_evaluation(args: argparse.Namespace) -> int:
    """
    Execute the evaluation pipeline on multiple audio files.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0=success, 1=partial failure, 2=complete failure)
    """
    configure_logging(args.verbose)
    
    logger.info("="*80)
    logger.info("Sample Recordings Evaluation - VAD/Decoding Configuration Feedback")
    logger.info("="*80)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Max files: {args.max_files if args.max_files else 'unlimited'}")
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
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 2
    
    if not input_dir.is_dir():
        logger.error(f"Input path is not a directory: {input_dir}")
        return 2
    
    audio_files = find_audio_files(input_dir, args.max_files)
    
    if not audio_files:
        logger.warning(f"No audio files found in {input_dir}")
        return 1
    
    logger.info(f"Found {len(audio_files)} audio file(s) to process")
    logger.info("")
    
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
        logger.info("Service initialized successfully")
        logger.info("="*80)
    except Exception as e:
        logger.exception(f"Failed to initialize transcription service: {e}")
        return 2
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results: List[Dict[str, Any]] = []
    total_audio_duration = 0.0
    total_processing_time = 0.0
    successful_files = 0
    failed_files = 0
    
    for idx, audio_file in enumerate(audio_files, 1):
        logger.info(f"Processing file {idx}/{len(audio_files)}: {audio_file.name}")
        
        file_result = {
            "file": audio_file.name,
            "success": False,
            "duration_seconds": 0.0,
            "vad_segments": 0,
            "processing_time": 0.0,
            "error": None
        }
        
        try:
            start_time = time.time()
            
            transcription = service.transcribe(str(audio_file))
            
            processing_time = time.time() - start_time
            file_result["processing_time"] = processing_time
            
            if transcription.get("error"):
                logger.error(f"  ✗ Transcription failed: {transcription['error']}")
                file_result["error"] = transcription["error"]
                failed_files += 1
            else:
                duration = transcription.get("duration_seconds", 0.0)
                num_segments = len(transcription.get("timestamps", []))
                
                file_result["duration_seconds"] = duration
                file_result["vad_segments"] = num_segments
                file_result["success"] = True
                
                total_audio_duration += duration
                total_processing_time += processing_time
                successful_files += 1
                
                logger.info(f"  ✓ Duration: {duration:.2f}s")
                logger.info(f"  ✓ VAD segments: {num_segments}")
                logger.info(f"  ✓ Processing time: {processing_time:.2f}s")
                logger.info(f"  ✓ Real-time factor: {processing_time/duration:.2f}x")
                
                if not args.dry_run:
                    try:
                        output_path = write_transcript(
                            transcription=transcription,
                            original_audio_path=audio_file,
                            output_dir=output_dir,
                            preferred_format=args.output_format
                        )
                        logger.info(f"  ✓ Output written to: {output_path}")
                    except Exception as e:
                        logger.error(f"  ✗ Failed to write output: {e}")
                        file_result["error"] = f"Output write failed: {e}"
                        file_result["success"] = False
                        failed_files += 1
                        successful_files -= 1
                else:
                    logger.info(f"  ✓ Dry run mode: skipping output file")
                
        except KeyboardInterrupt:
            logger.warning("Process interrupted by user")
            results.append(file_result)
            break
        except Exception as e:
            logger.exception(f"  ✗ Unexpected error processing {audio_file.name}: {e}")
            file_result["error"] = str(e)
            failed_files += 1
        
        results.append(file_result)
        logger.info("")
    
    logger.info("="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Total files processed: {len(results)}")
    logger.info(f"  Successful: {successful_files}")
    logger.info(f"  Failed: {failed_files}")
    logger.info("")
    
    if successful_files > 0:
        logger.info(f"Total audio duration: {total_audio_duration:.2f}s ({total_audio_duration/60:.2f} min)")
        logger.info(f"Total processing time: {total_processing_time:.2f}s ({total_processing_time/60:.2f} min)")
        logger.info(f"Average real-time factor: {total_processing_time/total_audio_duration:.2f}x")
        logger.info(f"Average VAD segments per file: {sum(r['vad_segments'] for r in results if r['success'])/successful_files:.1f}")
    
    if failed_files > 0:
        logger.info("")
        logger.info("Failed files:")
        for result in results:
            if not result["success"]:
                logger.info(f"  - {result['file']}: {result['error']}")
    
    logger.info("="*80)
    
    if failed_files == len(results):
        return 2
    elif failed_files > 0:
        return 1
    else:
        return 0


def main() -> None:
    """
    Main entry point for the evaluation script.
    
    Parses command-line arguments and executes the evaluation pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate VAD and decoding configurations on sample audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_sample_recordings.py
  python evaluate_sample_recordings.py --input-dir ./samples --max-files 5
  python evaluate_sample_recordings.py --vad-energy-threshold 0.02 --temperature 0.2
  python evaluate_sample_recordings.py --dry-run --verbose
        """
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./samples",
        help="Directory containing audio files to evaluate (default: ./samples)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./transcripts_eval",
        help="Directory for output files (default: ./transcripts_eval)"
    )
    
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (default: unlimited)"
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
        help="Transcribe but don't write output files"
    )
    
    advanced_group = parser.add_argument_group(
        "Advanced Options",
        "VAD and decoding parameters for configuration tuning"
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
    
    exit_code = run_evaluation(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
