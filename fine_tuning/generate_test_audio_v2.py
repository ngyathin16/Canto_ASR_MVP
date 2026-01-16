"""
Generate a NEW test audio clip with DIFFERENT construction domain terms for evaluation.
Uses terms NOT in the original test set to validate model generalization.
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

OUTPUT_DIR = Path(__file__).parent / "test_evaluation_v2"

# NEW test sentences using DIFFERENT construction terms
# These use terms NOT in the original test set
TEST_SENTENCES_V2 = [
    "判頭話今日要起貨，全部人加快手腳。",
    "叻架佬將啲鋼筋吊上去，小心啲唔好撞到人。",
    "呢度要裝個嘩佬，控制水流大細。",
    "蛇佬話啲水線要重新拉過，有啲位唔啱。",
    "電王檢查完話個九線掣有問題，要換過。",
    "用風煤切開嗰條工字鐵，記住戴好眼罩。",
    "個沙井蓋爛咗，要搵人嚟換返個新嘅。",
    "墨王已經彈好線，可以開始砌磚喇。",
    "呢度啲黃蜂竇要執返靚，唔好俾人睇到。",
    "阿SIR話要檢查埋啲過山螺絲有冇擰實。",
    "今日有三判嘅人嚟做嘢，記住分配好位置。",
    "架步入面太熱，開返部寶路華涼下先。",
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
    print("=== Generating NEW Test Audio (V2) for Evaluation ===\n")
    print("Using DIFFERENT terms from original test set\n")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    speech_config = create_speech_config()
    if not speech_config:
        return
    
    # Generate individual clips
    audio_files = []
    for i, sentence in enumerate(TEST_SENTENCES_V2):
        clip_path = OUTPUT_DIR / f"test_clip_{i:02d}.wav"
        print(f"[{i+1}/{len(TEST_SENTENCES_V2)}] Generating: {sentence[:30]}...")
        
        if generate_audio(speech_config, sentence, clip_path):
            audio_files.append(clip_path)
        time.sleep(0.1)
    
    # Concatenate into single file
    combined_path = OUTPUT_DIR / "test_audio_combined_v2.wav"
    print(f"\nConcatenating {len(audio_files)} clips...")
    concatenate_audio_files(audio_files, combined_path)
    
    # Save ground truth transcript
    ground_truth = "\n".join(TEST_SENTENCES_V2)
    ground_truth_path = OUTPUT_DIR / "ground_truth_v2.txt"
    with open(ground_truth_path, "w", encoding="utf-8") as f:
        f.write(ground_truth)
    
    # Save test info
    test_info = {
        "sentences": TEST_SENTENCES_V2,
        "num_sentences": len(TEST_SENTENCES_V2),
        "voice": VOICE_NAME,
        "combined_audio": str(combined_path),
        "ground_truth": str(ground_truth_path),
        "terms_used": [
            "判頭", "起貨", "叻架佬", "嘩佬", "蛇佬", "水線",
            "電王", "九線掣", "風煤", "工字", "沙井", "墨王",
            "黃蜂竇", "阿SIR", "過山螺絲", "三判", "架步", "寶路華"
        ],
        "description": "New test set with different terms to validate model generalization"
    }
    with open(OUTPUT_DIR / "test_info_v2.json", "w", encoding="utf-8") as f:
        json.dump(test_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== Test Audio V2 Generated ===")
    print(f"Combined audio: {combined_path}")
    print(f"Ground truth: {ground_truth_path}")
    print(f"Total sentences: {len(TEST_SENTENCES_V2)}")
    print(f"\nTerms used: 判頭, 起貨, 叻架佬, 嘩佬, 蛇佬, 水線, 電王, 九線掣, 風煤, 工字, 沙井, 墨王, 黃蜂竇, 阿SIR, 過山螺絲, 三判, 架步, 寶路華")


if __name__ == "__main__":
    main()
