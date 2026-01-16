"""
Generate test audio V4 with even more unused construction terms.
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
VOICE_NAME = "zh-HK-WanLungNeural"

OUTPUT_DIR = Path(__file__).parent / "test_evaluation_v4"

# Test sentences with MORE unused terms (positions, slang, etc.)
TEST_SENTENCES_V4 = [
    "大判話要加人手，呢個項目趕住起貨。",
    "蛇頭今日搵咗十個人返工，夠唔夠用？",
    "英展睇完圖紙話要改設計，麻煩晒。",
    "安督巡完場話有幾個位唔合格，要執返。",
    "銀卡課程下星期開班，你報咗名未？",
    "貓仔開過嚟剷走啲泥頭，快啲搞掂佢。",
    "佛沙板今日送到，準備好位置放。",
    "威也要檢查下，睇下有冇斷絲。",
    "吊船升上去之前，記住扣好安全帶。",
    "圍街板要加高啲，唔好俾人睇到入面。",
    "水廁嗰邊太污糟，要搵人清潔下。",
    "牛仔話今日要加班，大家辛苦啲。",
]


def create_speech_config():
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
    audio_config = speechsdk.audio.AudioOutputConfig(filename=str(output_path))
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=audio_config
    )
    result = synthesizer.speak_text_async(text).get()
    return result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted


def concatenate_audio_files(audio_files: list, output_path: Path):
    import wave
    with wave.open(str(output_path), 'wb') as outfile:
        for i, audio_file in enumerate(audio_files):
            with wave.open(str(audio_file), 'rb') as infile:
                if i == 0:
                    outfile.setparams(infile.getparams())
                outfile.writeframes(infile.readframes(infile.getnframes()))


def main():
    print("=== Generating Test Audio V4 ===\n")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    speech_config = create_speech_config()
    if not speech_config:
        return
    
    audio_files = []
    for i, sentence in enumerate(TEST_SENTENCES_V4):
        clip_path = OUTPUT_DIR / f"test_clip_{i:02d}.wav"
        print(f"[{i+1}/{len(TEST_SENTENCES_V4)}] {sentence[:30]}...")
        if generate_audio(speech_config, sentence, clip_path):
            audio_files.append(clip_path)
        time.sleep(0.1)
    
    combined_path = OUTPUT_DIR / "test_audio_combined.wav"
    concatenate_audio_files(audio_files, combined_path)
    
    ground_truth = "\n".join(TEST_SENTENCES_V4)
    with open(OUTPUT_DIR / "ground_truth.txt", "w", encoding="utf-8") as f:
        f.write(ground_truth)
    
    print(f"\n=== V4 Generated: {len(TEST_SENTENCES_V4)} sentences ===")
    print(f"Terms: 大判, 蛇頭, 英展, 安督, 銀卡, 貓仔, 佛沙, 威也, 吊船, 圍街板, 水廁, 牛仔")


if __name__ == "__main__":
    main()
