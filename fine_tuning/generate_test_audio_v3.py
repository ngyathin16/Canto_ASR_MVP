"""
Generate test audio V3 with more unused construction terms.
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

OUTPUT_DIR = Path(__file__).parent / "test_evaluation_v3"

# Test sentences with MORE unused terms
TEST_SENTENCES_V3 = [
    "個狗臂架要裝喺嗰邊，支撐住啲重嘢。",
    "趷雞陣嗰度要加多條鐵，唔夠穩陣。",
    "街吊今日會嚟，準備好啲嘢等佢吊。",
    "門頭陣上面要加多啲支撐，唔好塌落嚟。",
    "靚仔面做好咗，唔好再行上去整花佢。",
    "膠凳仔放好未？鋼筋要有足夠保護層。",
    "鹽水喉要換過，舊嗰條已經生晒鏽。",
    "埋碼嗰個師傅好叻，吊嘢好穩陣。",
    "震筆開咗成日，啲石屎應該夠實淨。",
    "開士機壞咗，啲材料要行樓梯搬上去。",
    "鉸剪台要有牌先可以用，你有冇牌？",
    "花籃入面啲鐵紮好未？等陣要落石屎。",
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
    print("=== Generating Test Audio V3 ===\n")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    speech_config = create_speech_config()
    if not speech_config:
        return
    
    audio_files = []
    for i, sentence in enumerate(TEST_SENTENCES_V3):
        clip_path = OUTPUT_DIR / f"test_clip_{i:02d}.wav"
        print(f"[{i+1}/{len(TEST_SENTENCES_V3)}] {sentence[:30]}...")
        if generate_audio(speech_config, sentence, clip_path):
            audio_files.append(clip_path)
        time.sleep(0.1)
    
    combined_path = OUTPUT_DIR / "test_audio_combined.wav"
    concatenate_audio_files(audio_files, combined_path)
    
    ground_truth = "\n".join(TEST_SENTENCES_V3)
    with open(OUTPUT_DIR / "ground_truth.txt", "w", encoding="utf-8") as f:
        f.write(ground_truth)
    
    print(f"\n=== V3 Generated: {len(TEST_SENTENCES_V3)} sentences ===")
    print(f"Terms: 狗臂架, 趷雞陣, 街吊, 門頭陣, 靚仔面, 膠凳仔, 鹽水喉, 埋碼, 震筆, 開士機, 鉸剪台, 花籃")


if __name__ == "__main__":
    main()
