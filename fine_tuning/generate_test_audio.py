"""
Generate a 1-minute test audio clip with construction domain terms for evaluation.
"""

import json
import os
import time
from pathlib import Path

import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

load_dotenv()

SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "eastus2")
VOICE_NAME = "zh-HK-WanLungNeural"  # Male voice

OUTPUT_DIR = Path(__file__).parent / "test_evaluation"

# Test sentences using various construction terms
# These are NEW sentences not in the training set
TEST_SENTENCES = [
    "今日要用天秤吊批鋼筋上去十五樓。",
    "扎鐵師傅話要加多啲花鐵先夠穩陣。",
    "石屎車遲咗，我哋要等多半個鐘。",
    "嗰邊有個吼要補返，記住用牛奶水。",
    "雞頭司機話部機有啲問題，要搵大偈睇下。",
    "呢度啲瓦仔貼得唔靚，要剷走重新貼過。",
    "安全主任話要戴齊地盤三寶先可以入場。",
    "聽日有皇家佬嚟巡查，全部位執好啲。",
    "個震船壞咗，要叫人嚟整返先繼續壓實。",
    "呢幅牆有啲耳鵝，要執返正先油漆。",
    "師傅，幫手攞支士巴拿過嚟擰實啲螺絲。",
    "落石屎之前記住檢查晒啲飛碟有冇放好。",
]


def create_speech_config():
    """Create Azure Speech SDK config."""
    if not SPEECH_KEY:
        print("Error: AZURE_SPEECH_KEY not found")
        return None
    
    config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    config.speech_synthesis_voice_name = VOICE_NAME
    config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
    )
    return config


def generate_audio(speech_config, text: str, output_path: Path) -> bool:
    """Generate audio file from text."""
    audio_config = speechsdk.audio.AudioOutputConfig(filename=str(output_path))
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=audio_config
    )
    
    result = synthesizer.speak_text_async(text).get()
    
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return True
    else:
        print(f"TTS failed: {result.reason}")
        return False


def concatenate_audio_files(audio_files: list, output_path: Path):
    """Concatenate multiple WAV files into one."""
    import wave
    
    with wave.open(str(output_path), 'wb') as outfile:
        for i, audio_file in enumerate(audio_files):
            with wave.open(str(audio_file), 'rb') as infile:
                if i == 0:
                    outfile.setparams(infile.getparams())
                outfile.writeframes(infile.readframes(infile.getnframes()))


def main():
    print("=== Generating Test Audio for Evaluation ===\n")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    speech_config = create_speech_config()
    if not speech_config:
        return
    
    # Generate individual clips
    audio_files = []
    for i, sentence in enumerate(TEST_SENTENCES):
        clip_path = OUTPUT_DIR / f"test_clip_{i:02d}.wav"
        print(f"[{i+1}/{len(TEST_SENTENCES)}] Generating: {sentence[:30]}...")
        
        if generate_audio(speech_config, sentence, clip_path):
            audio_files.append(clip_path)
        time.sleep(0.1)
    
    # Concatenate into single file
    combined_path = OUTPUT_DIR / "test_audio_combined.wav"
    print(f"\nConcatenating {len(audio_files)} clips...")
    concatenate_audio_files(audio_files, combined_path)
    
    # Save ground truth transcript
    ground_truth = "\n".join(TEST_SENTENCES)
    ground_truth_path = OUTPUT_DIR / "ground_truth.txt"
    with open(ground_truth_path, "w", encoding="utf-8") as f:
        f.write(ground_truth)
    
    # Save test info
    test_info = {
        "sentences": TEST_SENTENCES,
        "num_sentences": len(TEST_SENTENCES),
        "voice": VOICE_NAME,
        "combined_audio": str(combined_path),
        "ground_truth": str(ground_truth_path),
    }
    with open(OUTPUT_DIR / "test_info.json", "w", encoding="utf-8") as f:
        json.dump(test_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== Test Audio Generated ===")
    print(f"Combined audio: {combined_path}")
    print(f"Ground truth: {ground_truth_path}")
    print(f"Total sentences: {len(TEST_SENTENCES)}")


if __name__ == "__main__":
    main()
