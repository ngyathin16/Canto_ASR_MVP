"""
Production-grade Cantonese transcription service using Whisper Large V3 Turbo.

This module provides a TranscriptionService class for transcribing Cantonese audio
files using the fine-tuned Whisper model from Hugging Face.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import pipeline
import librosa
from scipy import signal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TranscriptionService:
    """
    Service for transcribing Cantonese audio files using Whisper Large V3 Turbo.
    
    This class handles model loading, audio preprocessing, VAD-based chunking,
    and transcription with automatic device detection (CUDA/CPU).
    
    Decoding parameters:
        - temperature: Controls randomness in generation (0.0 = deterministic)
        - compression_ratio_threshold: Threshold for detecting repetitive output
        - logprob_threshold: Minimum log probability threshold for tokens
        - no_speech_threshold: Threshold for detecting non-speech segments
    
    VAD parameters:
        - vad_energy_threshold: Energy threshold for speech detection (relative to max)
    """
    
    def __init__(
        self, 
        model_name: str = "JackyHoCL/whisper-large-v3-turbo-cantonese-yue-english",
        temperature: float = 0.0,
        compression_ratio_threshold: Optional[float] = 2.4,
        logprob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        vad_energy_threshold: float = 0.015,
        vad_min_segment_duration: float = 0.3,
        vad_merge_gap: float = 0.3,
        max_segment_length: float = 30.0,
        return_timestamps: str = "word"
    ) -> None:
        """
        Initialize the transcription service with the specified model.
        
        Args:
            model_name: Hugging Face model identifier for the Whisper model
            temperature: Sampling temperature for generation (0.0 = deterministic)
            compression_ratio_threshold: Threshold for detecting repetitive output
            logprob_threshold: Minimum log probability threshold for tokens
            no_speech_threshold: Threshold for detecting non-speech segments
            vad_energy_threshold: Energy threshold for VAD (relative to max energy)
            vad_min_segment_duration: Minimum duration (seconds) for speech segments
            vad_merge_gap: Maximum gap (seconds) to merge adjacent segments
            max_segment_length: Maximum length (seconds) for a single segment to prevent memory issues
            return_timestamps: Timestamp granularity - 'word' (slow, detailed) or 'chunk' (fast)
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.temperature = temperature
        self.compression_ratio_threshold = compression_ratio_threshold
        self.logprob_threshold = logprob_threshold
        self.no_speech_threshold = no_speech_threshold
        self.vad_energy_threshold = vad_energy_threshold
        self.vad_min_segment_duration = vad_min_segment_duration
        self.vad_merge_gap = vad_merge_gap
        self.max_segment_length = max_segment_length
        self.return_timestamps = return_timestamps
        
        logger.info(f"Initializing TranscriptionService with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Decoding params: temp={temperature}, compression_ratio_threshold={compression_ratio_threshold}, "
                   f"logprob_threshold={logprob_threshold}, no_speech_threshold={no_speech_threshold}")
        logger.info(f"VAD params: energy_threshold={vad_energy_threshold}, "
                   f"min_segment_duration={vad_min_segment_duration}s, merge_gap={vad_merge_gap}s, "
                   f"max_segment_length={max_segment_length}s")
        logger.info(f"Timestamp mode: {return_timestamps}")
        
        try:
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=self.device
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_audio(self, file_path: str) -> tuple[Any, float]:
        """
        Load and preprocess audio file to 16kHz mono PCM format.
        
        Args:
            file_path: Path to the audio file (MP3 or WAV)
            
        Returns:
            Tuple of (audio_array, duration_seconds)
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If audio is empty or corrupted
        """
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        try:
            logger.info(f"Loading audio file: {file_path}")
            audio_array, sample_rate = librosa.load(file_path, sr=16000, mono=True)
            
            duration = len(audio_array) / sample_rate
            
            if duration == 0 or len(audio_array) == 0:
                logger.warning(f"Empty audio file: {file_path}")
                raise ValueError("Audio file is empty or contains no data")
            
            rms_energy = librosa.feature.rms(y=audio_array)[0].mean()
            if rms_energy < 1e-6:
                logger.warning(f"Near-silent audio detected (RMS: {rms_energy:.2e}): {file_path}")
            
            if duration > 1800:
                logger.warning(
                    f"Audio duration ({duration/60:.1f} minutes) exceeds 30 minutes. "
                    "Processing may take significant time."
                )
            
            logger.info(f"Audio loaded successfully: {duration:.2f}s, sample_rate={sample_rate}Hz")
            return audio_array, duration
            
        except (RuntimeError, ValueError) as e:
            logger.error(f"Failed to decode audio file (corrupted or unsupported format?): {file_path} - {e}")
            raise ValueError(f"Corrupted or invalid audio file: {e}")
        except Exception as e:
            logger.error(f"Error loading audio file: {file_path} - {e}")
            raise
    
    def preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply denoising and normalization to the audio signal.
        
        Args:
            audio: Input audio array
            sr: Sample rate in Hz
            
        Returns:
            Preprocessed audio array
        """
        logger.info("Applying audio preprocessing (normalization and high-pass filtering)")
        
        audio_processed = audio.copy()
        
        nyquist = sr / 2
        cutoff_hz = 80
        if cutoff_hz < nyquist:
            sos = signal.butter(4, cutoff_hz / nyquist, btype='high', output='sos')
            audio_processed = signal.sosfilt(sos, audio_processed)
            logger.debug(f"Applied high-pass filter at {cutoff_hz} Hz")
        
        rms = np.sqrt(np.mean(audio_processed ** 2))
        if rms > 1e-6:
            target_rms = 0.1
            audio_processed = audio_processed * (target_rms / rms)
            logger.debug(f"Normalized RMS from {rms:.4f} to {target_rms:.4f}")
        
        peak = np.abs(audio_processed).max()
        if peak > 1.0:
            audio_processed = audio_processed / peak
            logger.debug(f"Clipped peak from {peak:.4f} to 1.0")
        
        return audio_processed
    
    def detect_speech_segments(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """
        Detect speech segments using energy-based VAD.
        
        Args:
            audio: Input audio array (mono, 16kHz)
            sr: Sample rate in Hz
            
        Returns:
            List of (start_sec, end_sec) tuples for detected speech segments
        """
        logger.info("Running VAD to detect speech segments")
        
        frame_length = int(0.03 * sr)
        hop_length = int(0.015 * sr)
        
        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        max_energy = rms.max()
        if max_energy < 1e-6:
            logger.warning("Audio has near-zero energy, no speech detected")
            return []
        
        threshold = self.vad_energy_threshold * max_energy
        logger.debug(f"VAD threshold: {threshold:.6f} (max_energy: {max_energy:.6f})")
        
        is_speech = rms > threshold
        
        segments = []
        in_segment = False
        segment_start = 0
        
        for i, speech in enumerate(is_speech):
            time = i * hop_length / sr
            
            if speech and not in_segment:
                segment_start = time
                in_segment = True
            elif not speech and in_segment:
                segments.append((segment_start, time))
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
                logger.warning(f"Segment {start:.2f}s-{end:.2f}s ({segment_duration:.2f}s) exceeds max length, splitting...")
                num_splits = int(np.ceil(segment_duration / self.max_segment_length))
                split_duration = segment_duration / num_splits
                for i in range(num_splits):
                    split_start = start + i * split_duration
                    split_end = min(start + (i + 1) * split_duration, end)
                    final_segments.append((split_start, split_end))
                    logger.debug(f"  Split {i+1}/{num_splits}: {split_start:.2f}s - {split_end:.2f}s ({split_end-split_start:.2f}s)")
            else:
                final_segments.append((start, end))
        
        total_speech_duration = sum(e - s for s, e in final_segments)
        logger.info(f"Detected {len(final_segments)} speech segments, "
                   f"total duration: {total_speech_duration:.2f}s")
        
        for i, (start, end) in enumerate(final_segments[:5]):
            logger.debug(f"  Segment {i+1}: {start:.2f}s - {end:.2f}s ({end-start:.2f}s)")
        if len(final_segments) > 5:
            logger.debug(f"  ... and {len(final_segments) - 5} more segments")
        
        return final_segments
    
    def _is_repetitive(self, text: str, char_threshold: float = 0.6, word_threshold: float = 0.7) -> bool:
        """
        Detect if text contains excessive repetition.
        
        Args:
            text: Text to check
            char_threshold: Ratio of repeated characters to total length
            word_threshold: Ratio of repeated words to total words
            
        Returns:
            True if text is repetitive, False otherwise
        """
        if len(text) < 10:
            return False
        
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        max_char_count = max(char_counts.values())
        repetition_ratio = max_char_count / len(text)
        
        if repetition_ratio > char_threshold:
            logger.warning(f"High char repetition: {repetition_ratio:.2%} (char: '{max(char_counts, key=char_counts.get)}')")
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
            word_repetition_ratio = max_word_count / len(words)
            
            if word_repetition_ratio > word_threshold and max_word_count >= 3:
                most_repeated = max(word_counts, key=word_counts.get)
                logger.warning(f"High word repetition: {word_repetition_ratio:.2%} (word: '{most_repeated}' × {max_word_count})")
                return True
        
        return False
    
    def transcribe(self, file_path: str) -> Dict[str, Any]:
        """
        Transcribe an audio file and return text with timestamps.
        
        Args:
            file_path: Path to the audio file to transcribe
            
        Returns:
            Dictionary containing:
                - text: Full transcription text
                - timestamps: List of timestamp dictionaries with word/chunk timing
                - duration_seconds: Audio duration in seconds
                - error: Error message if transcription failed, None otherwise
        """
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
            
            speech_segments = self.detect_speech_segments(audio_preprocessed, sr=16000)
            
            if not speech_segments:
                logger.warning("No speech segments detected by VAD, falling back to full audio transcription")
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
            
            all_text_parts = []
            all_timestamps = []
            
            for seg_idx, (seg_start, seg_end) in enumerate(speech_segments):
                start_sample = int(seg_start * 16000)
                end_sample = int(seg_end * 16000)
                segment_audio = audio_preprocessed[start_sample:end_sample]
                
                logger.info(f"Transcribing segment {seg_idx+1}/{len(speech_segments)}: "
                           f"{seg_start:.2f}s - {seg_end:.2f}s")
                
                with torch.no_grad():
                    output = self.pipeline(
                        segment_audio,
                        return_timestamps=self.return_timestamps,
                        generate_kwargs=generate_kwargs
                    )
                
                segment_text = output.get("text", "")
                
                if self._is_repetitive(segment_text):
                    logger.warning(f"Detected repetitive output in segment {seg_idx+1}, skipping")
                    continue
                
                all_text_parts.append(segment_text)
                
                if "chunks" in output:
                    for chunk_idx, chunk in enumerate(output["chunks"]):
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
                            logger.debug(f"Chunk {chunk_idx+1} missing end timestamp, estimated: {global_start:.2f}s -> {global_end:.2f}s")
                        
                        all_timestamps.append({
                            "text": chunk_text,
                            "timestamp": [global_start, global_end]
                        })
            
            result["text"] = " ".join(all_text_parts)
            result["timestamps"] = all_timestamps
            
            logger.info(f"Transcription completed successfully. Text length: {len(result['text'])} chars, "
                       f"{len(all_timestamps)} timestamp chunks, {len(all_text_parts)} segments processed")
            
        except FileNotFoundError as e:
            result["error"] = str(e)
            logger.error(f"File not found error: {e}")
        except ValueError as e:
            result["error"] = str(e)
            logger.error(f"Audio processing error: {e}")
        except Exception as e:
            result["error"] = f"Unexpected error during transcription: {str(e)}"
            logger.exception(f"Unexpected error: {e}")
        
        return result


def main() -> None:
    """
    CLI entry point for the transcription service.
    
    Parses command-line arguments and runs transcription on the specified file.
    """
    parser = argparse.ArgumentParser(
        description="Transcribe Cantonese audio files using Whisper Large V3 Turbo"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the input audio file (MP3 or WAV)"
    )
    
    args = parser.parse_args()
    
    try:
        service = TranscriptionService()
        result = service.transcribe(args.input_file)
        
        if result["error"]:
            logger.error(f"Transcription failed: {result['error']}")
            print(f"ERROR: {result['error']}", file=__import__('sys').stderr)
            exit(1)
        
        print("\n" + "="*80)
        print("TRANSCRIPTION SUMMARY")
        print("="*80)
        print(f"File: {args.input_file}")
        print(f"Duration: {result['duration_seconds']:.2f} seconds")
        print(f"Transcript length: {len(result['text'])} characters")
        print(f"Timestamps: {len(result['timestamps'])} chunks")
        print("\nFirst 100 characters:")
        print(result['text'][:100] + ("..." if len(result['text']) > 100 else ""))
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        logger.info("Transcription interrupted by user")
        exit(130)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        print(f"FATAL ERROR: {e}", file=__import__('sys').stderr)
        exit(1)


if __name__ == "__main__":
    main()
