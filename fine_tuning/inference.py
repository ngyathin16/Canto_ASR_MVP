"""
Inference script for the fine-tuned Whisper model.

This script loads the fine-tuned LoRA adapter and runs inference on audio files.
"""

import argparse
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import librosa

# Configuration
BASE_MODEL = "khleeloo/whisper-large-v3-cantonese"
FINE_TUNED_DIR = Path(__file__).parent / "model"


def load_model(use_fine_tuned: bool = True):
    """
    Load Whisper model, optionally with fine-tuned LoRA adapter.
    
    Args:
        use_fine_tuned: If True, load LoRA adapter on top of base model
    
    Returns:
        Tuple of (model, processor, device)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Loading base model: {BASE_MODEL}")
    print(f"Device: {device}")
    
    # Load processor
    if use_fine_tuned and FINE_TUNED_DIR.exists():
        processor = WhisperProcessor.from_pretrained(str(FINE_TUNED_DIR))
        print(f"Loaded processor from fine-tuned model")
    else:
        processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    
    # Load base model
    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch_dtype,
    )
    
    # Load LoRA adapter if available
    if use_fine_tuned and FINE_TUNED_DIR.exists():
        print(f"Loading LoRA adapter from: {FINE_TUNED_DIR}")
        model = PeftModel.from_pretrained(model, str(FINE_TUNED_DIR))
        model = model.merge_and_unload()  # Merge for faster inference
        print("LoRA adapter merged successfully")
    else:
        print("Using base model without fine-tuning")
    
    model.to(device)
    model.eval()
    
    return model, processor, device


def transcribe(audio_path: str, model, processor, device) -> str:
    """
    Transcribe an audio file with proper long-form audio support.
    
    Args:
        audio_path: Path to audio file
        model: Whisper model
        processor: Whisper processor
        device: Device to run inference on
    
    Returns:
        Transcribed text
    """
    # Load audio
    print(f"Loading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(audio) / sr
    print(f"Audio duration: {duration:.2f}s")
    
    # For long audio (>30s), process in chunks
    CHUNK_LENGTH = 30 * sr  # 30 seconds in samples
    
    if len(audio) <= CHUNK_LENGTH:
        # Short audio - process directly
        chunks = [audio]
    else:
        # Long audio - split into overlapping chunks
        chunks = []
        stride = 25 * sr  # 25 second stride (5 second overlap)
        for start in range(0, len(audio), stride):
            end = min(start + CHUNK_LENGTH, len(audio))
            chunks.append(audio[start:end])
            if end >= len(audio):
                break
    
    print(f"Processing {len(chunks)} chunk(s)...")
    
    transcriptions = []
    for i, chunk in enumerate(chunks):
        # Extract features and cast to model dtype
        input_features = processor.feature_extractor(
            chunk, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(device, dtype=torch.float16 if device == "cuda" else torch.float32)
        
        # Generate
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                language="yue",
                task="transcribe",
            )
        
        # Decode
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        transcriptions.append(text.strip())
    
    # Join chunks (simple concatenation - could be improved with deduplication)
    return " ".join(transcriptions)


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with fine-tuned Whisper")
    parser.add_argument("audio_file", type=str, help="Path to audio file")
    parser.add_argument("--base-only", action="store_true", help="Use base model without fine-tuning")
    args = parser.parse_args()
    
    # Validate input
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1
    
    # Load model
    use_fine_tuned = not args.base_only
    model, processor, device = load_model(use_fine_tuned=use_fine_tuned)
    
    # Transcribe
    print("\nTranscribing...")
    result = transcribe(str(audio_path), model, processor, device)
    
    print("\n" + "=" * 60)
    print("Transcription:")
    print("=" * 60)
    print(result)
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
