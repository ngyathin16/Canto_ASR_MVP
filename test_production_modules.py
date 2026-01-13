"""Test script that uses the actual production modules."""
from pathlib import Path
import librosa
import soundfile as sf

from transcription_service import TranscriptionService
from output_formatter import write_transcript

print("="*80)
print("TESTING PRODUCTION MODULES")
print("="*80)

# Create a 30-second test audio file
print("\n[SETUP] Creating 30-second test audio...")
print("-"*80)
audio_path = "samples/test_1.mp3"
audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
full_duration = len(audio_array) / sr
print(f"Full audio: {full_duration:.2f} seconds")

test_audio_path = Path("test_output/test_30sec.wav")
test_audio_path.parent.mkdir(exist_ok=True)
max_samples = 30 * 16000
sf.write(test_audio_path, audio_array[:max_samples], 16000)
print(f"✓ Created 30-second test file: {test_audio_path}")

# Test 1: TranscriptionService
print("\n[TEST 1] Testing TranscriptionService class...")
print("-"*80)

service = TranscriptionService()
print("✓ TranscriptionService initialized")

print("\nCalling service.transcribe()...")
result = service.transcribe(str(test_audio_path))

print("\nTranscription result:")
print(f"  - Error: {result['error']}")
print(f"  - Duration: {result['duration_seconds']:.2f} seconds")
print(f"  - Text length: {len(result['text'])} characters")
print(f"  - Timestamps: {len(result['timestamps'])} chunks")
print(f"  - First 100 chars: {result['text'][:100]}...")

if result['error']:
    print("✗ TranscriptionService FAILED")
    exit(1)
else:
    print("✓ TranscriptionService PASSED")

# Test 2: output_formatter
print("\n[TEST 2] Testing output_formatter.write_transcript()...")
print("-"*80)

output_dir = Path("test_output")
original_path = Path("samples/test_1.mp3")

# Test SRT
srt_path = write_transcript(
    transcription=result,
    original_audio_path=original_path,
    output_dir=output_dir,
    preferred_format="srt"
)
print(f"✓ SRT created: {srt_path}")

# Verify SRT content
with open(srt_path, 'r', encoding='utf-8') as f:
    srt_content = f.read()
    # Check for valid SRT structure and that some transcribed text is present
    if "00:00:00" in srt_content and "身為一個間諜" in srt_content:
        print("✓ SRT content verified")
    else:
        print("✗ SRT content invalid")
        exit(1)

# Test TXT
txt_path = write_transcript(
    transcription=result,
    original_audio_path=original_path,
    output_dir=output_dir,
    preferred_format="txt"
)
print(f"✓ TXT created: {txt_path}")

# Verify TXT content
with open(txt_path, 'r', encoding='utf-8') as f:
    txt_content = f.read()
    if "TRANSCRIPTION" in txt_content and result['text'] in txt_content:
        print("✓ TXT content verified")
    else:
        print("✗ TXT content invalid")
        exit(1)

print("\n" + "="*80)
print("ALL PRODUCTION MODULE TESTS PASSED ✓")
print("="*80)
print("\nProduction modules verified:")
print("  ✓ transcription_service.TranscriptionService")
print("  ✓ output_formatter.write_transcript (SRT)")
print("  ✓ output_formatter.write_transcript (TXT)")
print("="*80)
