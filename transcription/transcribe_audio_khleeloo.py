"""
Main orchestration script for Cantonese audio transcription using khleeloo/whisper-large-v3-cantonese.

This script coordinates the transcription service and output formatting to provide
a complete end-to-end transcription workflow using the khleeloo Whisper model.

OPTIMIZED VERSION: Uses float16, flash attention, and batch processing for faster inference.

Usage Examples (with activated venv):
    # Windows PowerShell:
    .venv\\Scripts\\Activate.ps1
    python transcribe_audio_khleeloo.py site_01.mp3
    
    # Unix/macOS:
    source .venv/bin/activate
    python transcribe_audio_khleeloo.py site_01.mp3
    
    # With custom output directory and verbose logging:
    python transcribe_audio_khleeloo.py site_01.mp3 --output-dir ./transcripts --verbose
    
    # Generate plain text instead of SRT:
    python transcribe_audio_khleeloo.py site_01.mp3 --output-format txt
    
    # Dry run (transcribe but don't write output):
    python transcribe_audio_khleeloo.py site_01.mp3 --dry-run --verbose
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa
from scipy import signal
from silero_vad import load_silero_vad, get_speech_timestamps

from output_formatter import write_transcript

TERMS_FILE = Path(__file__).parent / "terms.json"

logger = logging.getLogger(__name__)

MODEL_NAME = "khleeloo/whisper-large-v3-cantonese"


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


def load_terms_prompt() -> str:
    """
    Load terms from terms.json and convert to a comma-separated string for initial_prompt.
    
    Returns:
        Comma-separated string of terms, or empty string if file not found.
    """
    if not TERMS_FILE.exists():
        return ""
    
    try:
        data = json.loads(TERMS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    
    terms = data.get("terms", {})
    if not terms:
        return ""
    
    all_terms = []
    for category_items in terms.values():
        if isinstance(category_items, list):
            for item in category_items:
                # Extract just the Chinese term (before the parenthesis)
                term = item.split(" (")[0] if " (" in item else item
                all_terms.append(term)
    
    return ", ".join(all_terms)


class OptimizedTranscriptionService:
    """
    Optimized transcription service using float16, flash attention, and batched inference.
    """
    
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        temperature: float = 0.0,
        compression_ratio_threshold: Optional[float] = 1.8,
        logprob_threshold: Optional[float] = -0.5,
        no_speech_threshold: Optional[float] = 0.5,
        vad_energy_threshold: float = 0.008,
        vad_min_segment_duration: float = 0.15,
        vad_merge_gap: float = 0.5,
        max_segment_length: float = 30.0,
        return_timestamps: str = "chunk",
        batch_size: int = 8,
        initial_prompt: str = "",
        skip_vad: bool = False,
        use_silero_vad: bool = True
    ) -> None:
        self.model_name = model_name
        self.skip_vad = skip_vad
        self.use_silero_vad = use_silero_vad
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.temperature = temperature
        self.compression_ratio_threshold = compression_ratio_threshold
        self.logprob_threshold = logprob_threshold
        self.no_speech_threshold = no_speech_threshold
        self.vad_energy_threshold = vad_energy_threshold
        self.vad_min_segment_duration = vad_min_segment_duration
        self.vad_merge_gap = vad_merge_gap
        self.max_segment_length = max_segment_length
        self.return_timestamps = return_timestamps
        # On CPU, batch processing causes memory issues - use sequential processing
        self.batch_size = batch_size if self.device == "cuda" else 1
        self.initial_prompt = initial_prompt
        
        logger.info(f"Initializing OptimizedTranscriptionService with model: {model_name}")
        logger.info(f"Using device: {self.device}, dtype: {self.torch_dtype}")
        logger.info(f"Batch size: {self.batch_size} {'(forced to 1 for CPU)' if self.device == 'cpu' else ''}")
        
        try:
            # Set number of threads for CPU inference
            if self.device == "cpu":
                import os
                num_threads = os.cpu_count() or 4
                torch.set_num_threads(num_threads)
                logger.info(f"CPU threads set to: {num_threads}")
            
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                attn_implementation="sdpa"
            )
            model.to(self.device)
            
            self.processor = AutoProcessor.from_pretrained(model_name)
            
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            logger.info("Model loaded successfully with optimizations")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_audio(self, file_path: str) -> Tuple[np.ndarray, float]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        logger.info(f"Loading audio file: {file_path}")
        audio_array, sample_rate = librosa.load(file_path, sr=16000, mono=True)
        duration = len(audio_array) / sample_rate
        
        if duration == 0 or len(audio_array) == 0:
            raise ValueError("Audio file is empty or contains no data")
        
        logger.info(f"Audio loaded: {duration:.2f}s")
        return audio_array, duration
    
    def preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        audio_processed = audio.copy()
        
        nyquist = sr / 2
        cutoff_hz = 80
        if cutoff_hz < nyquist:
            sos = signal.butter(4, cutoff_hz / nyquist, btype='high', output='sos')
            audio_processed = signal.sosfilt(sos, audio_processed)
        
        rms = np.sqrt(np.mean(audio_processed ** 2))
        if rms > 1e-6:
            target_rms = 0.1
            audio_processed = audio_processed * (target_rms / rms)
        
        peak = np.abs(audio_processed).max()
        if peak > 1.0:
            audio_processed = audio_processed / peak
        
        return audio_processed
    
    def detect_speech_segments_silero(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """Use Silero VAD (neural network) for robust speech detection."""
        logger.info("Running Silero VAD to detect speech segments")
        
        # Load Silero VAD model (cached after first load)
        vad_model = load_silero_vad()
        
        # Convert to torch tensor (Silero expects float32)
        audio_tensor = torch.from_numpy(audio).float()
        
        # Get speech timestamps from Silero
        # threshold: speech probability threshold (lower = more sensitive)
        # min_speech_duration_ms: minimum speech segment duration
        # min_silence_duration_ms: minimum silence to split segments
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            vad_model,
            sampling_rate=sr,
            threshold=0.25,  # Balance: not too sensitive to noise (was 0.2)
            min_speech_duration_ms=150,  # Filter out very short non-speech sounds (was 50ms)
            min_silence_duration_ms=250,  # Moderate silence gap (was 200ms)
            speech_pad_ms=150,  # Moderate padding (was 200ms)
        )
        
        if not speech_timestamps:
            logger.warning("Silero VAD detected no speech")
            return []
        
        # Convert sample indices to time in seconds
        segments = []
        for ts in speech_timestamps:
            start_time = ts['start'] / sr
            end_time = ts['end'] / sr
            segments.append((start_time, end_time))
        
        # Split segments longer than max_segment_length
        final_segments = []
        for start, end in segments:
            segment_duration = end - start
            if segment_duration > self.max_segment_length:
                num_splits = int(np.ceil(segment_duration / self.max_segment_length))
                split_duration = segment_duration / num_splits
                for i in range(num_splits):
                    split_start = start + i * split_duration
                    split_end = min(start + (i + 1) * split_duration, end)
                    final_segments.append((split_start, split_end))
            else:
                final_segments.append((start, end))
        
        total_speech = sum(e - s for s, e in final_segments)
        logger.info(f"Silero VAD detected {len(final_segments)} speech segments, total: {total_speech:.2f}s")
        return final_segments
    
    def detect_speech_segments_energy(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """Legacy energy-based VAD (fallback)."""
        logger.info("Running energy-based VAD to detect speech segments")
        
        frame_length = int(0.03 * sr)
        hop_length = int(0.015 * sr)
        
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        max_energy = rms.max()
        if max_energy < 1e-6:
            return []
        
        threshold = self.vad_energy_threshold * max_energy
        is_speech = rms > threshold
        
        segments = []
        in_segment = False
        segment_start = 0
        
        for i, speech in enumerate(is_speech):
            time_val = i * hop_length / sr
            if speech and not in_segment:
                segment_start = time_val
                in_segment = True
            elif not speech and in_segment:
                segments.append((segment_start, time_val))
                in_segment = False
        
        if in_segment:
            segments.append((segment_start, len(audio) / sr))
        
        segments = [(s, e) for s, e in segments if (e - s) >= self.vad_min_segment_duration]
        
        merged_segments = []
        for start, end in segments:
            if merged_segments and (start - merged_segments[-1][1]) < self.vad_merge_gap:
                merged_segments[-1] = (merged_segments[-1][0], end)
            else:
                merged_segments.append((start, end))
        
        final_segments = []
        for start, end in merged_segments:
            segment_duration = end - start
            if segment_duration > self.max_segment_length:
                num_splits = int(np.ceil(segment_duration / self.max_segment_length))
                split_duration = segment_duration / num_splits
                for i in range(num_splits):
                    split_start = start + i * split_duration
                    split_end = min(start + (i + 1) * split_duration, end)
                    final_segments.append((split_start, split_end))
            else:
                final_segments.append((start, end))
        
        total_speech = sum(e - s for s, e in final_segments)
        logger.info(f"Energy VAD detected {len(final_segments)} speech segments, total: {total_speech:.2f}s")
        return final_segments
    
    def detect_speech_segments(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """Detect speech segments using Silero VAD (default) or energy-based VAD."""
        if self.use_silero_vad:
            return self.detect_speech_segments_silero(audio, sr)
        else:
            return self.detect_speech_segments_energy(audio, sr)
    
    def _is_repetitive(self, text: str, char_threshold: float = 0.6, word_threshold: float = 0.7) -> bool:
        if len(text) < 10:
            return False
        
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        max_char_count = max(char_counts.values())
        if max_char_count / len(text) > char_threshold:
            return True
        
        words = text.split()
        if len(words) < 3:
            return False
        
        word_counts = {}
        for word in words:
            if len(word) > 0:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        if word_counts:
            max_word_count = max(word_counts.values())
            if max_word_count / len(words) > word_threshold and max_word_count >= 3:
                return True
        
        return False
    
    def transcribe(self, file_path: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "text": "",
            "timestamps": [],
            "duration_seconds": 0.0,
            "error": None
        }
        
        try:
            audio_array, duration = self._load_audio(file_path)
            result["duration_seconds"] = duration
            
            logger.info(f"Starting transcription for: {file_path}")
            
            audio_preprocessed = self.preprocess_audio(audio_array, sr=16000)
            
            if self.skip_vad:
                logger.info("VAD disabled, processing full audio")
                speech_segments = [(0.0, duration)]
            else:
                speech_segments = self.detect_speech_segments(audio_preprocessed, sr=16000)
                if not speech_segments:
                    logger.warning("No speech segments detected, falling back to full audio")
                    speech_segments = [(0.0, duration)]
            
            generate_kwargs = {"language": "cantonese"}
            if self.temperature is not None:
                generate_kwargs["temperature"] = self.temperature
            if self.compression_ratio_threshold is not None:
                generate_kwargs["compression_ratio_threshold"] = self.compression_ratio_threshold
            if self.logprob_threshold is not None:
                generate_kwargs["logprob_threshold"] = self.logprob_threshold
            if self.no_speech_threshold is not None:
                generate_kwargs["no_speech_threshold"] = self.no_speech_threshold
            if self.initial_prompt:
                prompt_ids = self.processor.get_prompt_ids(self.initial_prompt, return_tensors="pt")
                generate_kwargs["prompt_ids"] = prompt_ids.squeeze().to(self.device)
                logger.info(f"Using initial prompt with {len(self.initial_prompt)} characters")
            
            segment_audios = []
            for seg_start, seg_end in speech_segments:
                start_sample = int(seg_start * 16000)
                end_sample = int(seg_end * 16000)
                segment_audios.append(audio_preprocessed[start_sample:end_sample])
            
            all_text_parts = []
            all_timestamps = []
            
            logger.info(f"Processing {len(segment_audios)} segments in batches of {self.batch_size}")
            
            for batch_start in range(0, len(segment_audios), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(segment_audios))
                batch_audios = segment_audios[batch_start:batch_end]
                batch_segments = speech_segments[batch_start:batch_end]
                
                logger.info(f"Processing batch {batch_start//self.batch_size + 1}/{(len(segment_audios) + self.batch_size - 1)//self.batch_size}")
                
                with torch.no_grad():
                    outputs = self.pipe(
                        batch_audios,
                        return_timestamps=self.return_timestamps,
                        generate_kwargs=generate_kwargs,
                        batch_size=len(batch_audios)
                    )
                
                if not isinstance(outputs, list):
                    outputs = [outputs]
                
                for idx, (output, (seg_start, seg_end)) in enumerate(zip(outputs, batch_segments)):
                    segment_text = output.get("text", "")
                    
                    if self._is_repetitive(segment_text):
                        logger.warning(f"Skipping repetitive segment {batch_start + idx + 1}")
                        continue
                    
                    all_text_parts.append(segment_text)
                    
                    if "chunks" in output:
                        for chunk in output["chunks"]:
                            chunk_timestamp = chunk.get("timestamp", [0.0, 0.0])
                            chunk_text = chunk.get("text", "")
                            
                            chunk_start = chunk_timestamp[0] if chunk_timestamp[0] is not None else 0.0
                            global_start = seg_start + chunk_start
                            
                            if chunk_timestamp[1] is not None:
                                global_end = seg_start + chunk_timestamp[1]
                            else:
                                segment_duration = seg_end - seg_start
                                estimated_chunk_duration = max(len(chunk_text) * 0.15, 0.5)
                                global_end = min(global_start + estimated_chunk_duration, seg_start + segment_duration)
                            
                            all_timestamps.append({
                                "text": chunk_text,
                                "timestamp": [global_start, global_end]
                            })
            
            result["text"] = " ".join(all_text_parts)
            result["timestamps"] = all_timestamps
            
            logger.info(f"Transcription completed. {len(result['text'])} chars, {len(all_timestamps)} chunks")
            
        except FileNotFoundError as e:
            result["error"] = str(e)
            logger.error(f"File not found: {e}")
        except ValueError as e:
            result["error"] = str(e)
            logger.error(f"Audio error: {e}")
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"
            logger.exception(f"Unexpected error: {e}")
        
        return result


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
    logger.info("Cantonese Audio Transcription Pipeline (khleeloo model) v0.1.0-mvp")
    logger.info(f"Model: {MODEL_NAME}")
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
        logger.info("Loading terminology for initial prompt...")
        initial_prompt = load_terms_prompt()
        if initial_prompt:
            logger.info(f"Loaded {len(initial_prompt.split(', '))} terms from terms.json")
        else:
            logger.info("No terms.json found or empty, proceeding without initial prompt")
        
        logger.info("Initializing optimized transcription service...")
        service = OptimizedTranscriptionService(
            model_name=MODEL_NAME,
            temperature=args.temperature,
            compression_ratio_threshold=args.compression_ratio_threshold,
            logprob_threshold=args.logprob_threshold,
            no_speech_threshold=args.no_speech_threshold,
            vad_energy_threshold=args.vad_energy_threshold,
            vad_min_segment_duration=args.vad_min_segment_duration,
            vad_merge_gap=args.vad_merge_gap,
            max_segment_length=args.max_segment_length,
            return_timestamps=args.return_timestamps,
            batch_size=args.batch_size,
            initial_prompt=initial_prompt,
            skip_vad=args.no_vad,
            use_silero_vad=not args.energy_vad
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
        description=f"Transcribe Cantonese audio files using {MODEL_NAME}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcribe_audio_khleeloo.py site_01.mp3
  python transcribe_audio_khleeloo.py site_01.mp3 --output-dir ./transcripts --verbose
  python transcribe_audio_khleeloo.py site_01.mp3 --output-format txt
  python transcribe_audio_khleeloo.py site_01.mp3 --dry-run
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
        default="./transcripts_khleeloo",
        help="Directory for output files (default: ./transcripts_khleeloo)"
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
        "--no-vad",
        action="store_true",
        help="Disable Voice Activity Detection and process full audio. "
             "Use this if VAD is dropping speech segments mixed with loud sounds."
    )
    
    advanced_group.add_argument(
        "--energy-vad",
        action="store_true",
        help="Use legacy energy-based VAD instead of Silero VAD (neural network). "
             "Silero VAD is more accurate but slightly slower."
    )
    
    advanced_group.add_argument(
        "--vad-energy-threshold",
        type=float,
        default=0.008,
        help="Energy/RMS threshold for VAD speech detection (relative to max energy). "
             "Lower values detect more speech but may include noise. (default: 0.008)"
    )
    
    advanced_group.add_argument(
        "--vad-min-segment-duration",
        type=float,
        default=0.15,
        help="Minimum duration (seconds) for a speech segment to be kept. "
             "Shorter segments are filtered as noise. (default: 0.15)"
    )
    
    advanced_group.add_argument(
        "--vad-merge-gap",
        type=float,
        default=0.5,
        help="Maximum gap (seconds) between speech segments to merge them. "
             "Segments closer than this are combined. (default: 0.5)"
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
        default=1.8,
        help="Threshold for detecting repetitive/compressed output. "
             "Lower values are more strict. (default: 1.8)"
    )
    
    advanced_group.add_argument(
        "--logprob-threshold",
        type=float,
        default=-0.5,
        help="Minimum average log probability for generated tokens. "
             "Segments below this may be filtered. (default: -0.5)"
    )
    
    advanced_group.add_argument(
        "--no-speech-threshold",
        type=float,
        default=0.5,
        help="Threshold for detecting non-speech segments. "
             "Higher values are more strict. (default: 0.5)"
    )
    
    advanced_group.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of audio segments to process in parallel. "
             "Higher values use more GPU memory but are faster. (default: 8)"
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
