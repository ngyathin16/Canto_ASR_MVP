"""
Generate audio clips from training sentences using Azure TTS.

This script reads sentences from generated_sentences.json and creates
audio files for fine-tuning the ASR model.
"""

import json
import os
import sys
import time
from pathlib import Path

import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

load_dotenv()

# Azure Speech configuration
SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "eastus2")

# Cantonese voice - Azure offers several HK Cantonese voices
# Options: zh-HK-HiuMaanNeural (female), zh-HK-HiuGaaiNeural (female), zh-HK-WanLungNeural (male)
VOICE_NAME = "zh-HK-HiuMaanNeural"

# Paths
SENTENCES_FILE = Path(__file__).parent.parent / "sentence_generation" / "generated_sentences.json"
OUTPUT_DIR = Path(__file__).parent / "audio_clips"


def create_speech_config():
    """Create and configure Azure Speech SDK config."""
    if not SPEECH_KEY:
        print("Error: AZURE_SPEECH_KEY not found in environment variables.")
        print("Please add AZURE_SPEECH_KEY and AZURE_SPEECH_REGION to your .env file.")
        sys.exit(1)
    
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_synthesis_voice_name = VOICE_NAME
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
    )
    return speech_config


def generate_audio(speech_config, text: str, output_path: Path) -> bool:
    """Generate audio file from text using Azure TTS."""
    audio_config = speechsdk.audio.AudioOutputConfig(filename=str(output_path))
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, 
        audio_config=audio_config
    )
    
    result = synthesizer.speak_text_async(text).get()
    
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return True
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"  Speech synthesis canceled: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"  Error details: {cancellation_details.error_details}")
        return False
    return False


def load_sentences() -> list:
    """Load sentences from JSON file."""
    if not SENTENCES_FILE.exists():
        print(f"Error: Sentences file not found: {SENTENCES_FILE}")
        sys.exit(1)
    
    with open(SENTENCES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_manifest(sentences: list, output_dir: Path):
    """Generate a manifest file mapping audio files to transcripts."""
    manifest = []
    for i, item in enumerate(sentences):
        audio_file = f"clip_{i:04d}.wav"
        if (output_dir / audio_file).exists():
            manifest.append({
                "audio_file": audio_file,
                "transcript": item["sentence"],
                "term": item["term"],
                "type": item.get("type", "")
            })
    
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nManifest saved to: {manifest_path}")
    return manifest


def main():
    print("=== Azure TTS Audio Generation ===\n")
    
    # Load sentences
    sentences = load_sentences()
    print(f"Loaded {len(sentences)} sentences from {SENTENCES_FILE.name}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Initialize Azure Speech
    speech_config = create_speech_config()
    print(f"Using voice: {VOICE_NAME}")
    print(f"Region: {SPEECH_REGION}\n")
    
    # Generate audio clips
    success_count = 0
    fail_count = 0
    
    for i, item in enumerate(sentences):
        sentence = item["sentence"]
        term = item["term"]
        output_path = OUTPUT_DIR / f"clip_{i:04d}.wav"
        
        # Skip if already exists
        if output_path.exists():
            print(f"[{i+1}/{len(sentences)}] Skipping (exists): {term}")
            success_count += 1
            continue
        
        print(f"[{i+1}/{len(sentences)}] Generating: {term} - {sentence[:30]}...")
        
        if generate_audio(speech_config, sentence, output_path):
            success_count += 1
        else:
            fail_count += 1
        
        # Small delay to avoid rate limiting
        time.sleep(0.1)
    
    print(f"\n=== Generation Complete ===")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    
    # Generate manifest
    generate_manifest(sentences, OUTPUT_DIR)


if __name__ == "__main__":
    main()
