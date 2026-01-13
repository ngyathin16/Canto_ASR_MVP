"""Test script for khleeloo/whisper-large-v3-cantonese model.

This script tests the khleeloo Whisper model using the same parameters
and structure as the existing production test modules.
"""
from pathlib import Path
import librosa
import soundfile as sf

from transcription_service import TranscriptionService
from output_formatter import write_transcript

print("="*80)
print("TESTING khleeloo/whisper-large-v3-cantonese MODEL")
print("="*80)

# Create a 30-second test audio file
print("\n[SETUP] Creating 30-second test audio...")
print("-"*80)
audio_path = "samples/test_1.mp3"
audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
full_duration = len(audio_array) / sr
print(f"Full audio: {full_duration:.2f} seconds")

test_audio_path = Path("test_output/test_30sec_khleeloo.wav")
test_audio_path.parent.mkdir(exist_ok=True)
max_samples = 30 * 16000
sf.write(test_audio_path, audio_array[:max_samples], 16000)
print(f"✓ Created 30-second test file: {test_audio_path}")

# Test 1: TranscriptionService with khleeloo model
print("\n[TEST 1] Testing TranscriptionService with khleeloo/whisper-large-v3-cantonese...")
print("-"*80)

service = TranscriptionService(
    model_name="khleeloo/whisper-large-v3-cantonese",
    temperature=0.0,
    compression_ratio_threshold=2.4,
    logprob_threshold=-1.0,
    no_speech_threshold=0.6,
    vad_energy_threshold=0.015,
    vad_min_segment_duration=0.3,
    vad_merge_gap=0.3,
    max_segment_length=30.0,
    return_timestamps="chunk"
)
print("✓ TranscriptionService initialized with khleeloo model")

print("\nCalling service.transcribe()...")
result = service.transcribe(str(test_audio_path))

print("\nTranscription result:")
print(f"  - Error: {result['error']}")
print(f"  - Duration: {result['duration_seconds']:.2f} seconds")
print(f"  - Text length: {len(result['text'])} characters")
print(f"  - Timestamps: {len(result['timestamps'])} chunks")
if result['text']:
    print(f"  - First 100 chars: {result['text'][:100]}...")

if result['error']:
    print("✗ TranscriptionService FAILED")
    exit(1)
else:
    print("✓ TranscriptionService PASSED")

# Test 2: output_formatter
print("\n[TEST 2] Testing output_formatter.write_transcript()...")
print("-"*80)

output_dir = Path("test_output/khleeloo")
original_path = Path("samples/test_1.mp3")

# Test SRT
srt_path = write_transcript(
    transcription=result,
    original_audio_path=original_path,
    output_dir=output_dir,
    preferred_format="srt"
)
print(f"✓ SRT created: {srt_path}")

# Test TXT
txt_path = write_transcript(
    transcription=result,
    original_audio_path=original_path,
    output_dir=output_dir,
    preferred_format="txt"
)
print(f"✓ TXT created: {txt_path}")

print("\n" + "="*80)
print("khleeloo MODEL TESTS COMPLETED ✓")
print("="*80)
print("\nModel tested: khleeloo/whisper-large-v3-cantonese")
print("Parameters used (same as production):")
print("  - temperature: 0.0")
print("  - compression_ratio_threshold: 2.4")
print("  - logprob_threshold: -1.0")
print("  - no_speech_threshold: 0.6")
print("  - vad_energy_threshold: 0.015")
print("  - vad_min_segment_duration: 0.3")
print("  - vad_merge_gap: 0.3")
print("  - max_segment_length: 30.0")
print("  - return_timestamps: chunk")
print("="*80)
