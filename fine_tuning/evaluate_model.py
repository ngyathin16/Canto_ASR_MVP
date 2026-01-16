"""
Evaluate fine-tuned Whisper model against base model.

Compares transcription accuracy using CER (Character Error Rate).
"""

import json
from pathlib import Path

import torch
import librosa
import jiwer
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

# Configuration
BASE_MODEL = "khleeloo/whisper-large-v3-cantonese"
FINE_TUNED_DIR = Path(__file__).parent / "model"
TEST_DIR = Path(__file__).parent / "test_evaluation"

# Initial prompt with top construction jargon terms to prime the model
# This significantly lowers CER for rare domain-specific vocabulary
INITIAL_PROMPT = (
    "天秤, 扎鐵, 石屎, 狗臂架, 趷雞陣, 士巴拿, 工字, 街吊, 門頭陣, "
    "靚仔面, 瓦仔, 膠凳仔, 鹽水喉, 埋碼, 震筆, 開士機, 鉸剪台, "
    "花籃, 風煤, 判頭, 大判, 蛇頭, 英展, 安督, 地盤三寶, 銀卡"
)


def load_base_model():
    """Load base Whisper model without fine-tuning."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Loading base model: {BASE_MODEL}")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=dtype
    )
    model.to(device)
    model.eval()
    
    return model, processor, device


def load_fine_tuned_model():
    """Load fine-tuned model with LoRA adapter."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Loading fine-tuned model from: {FINE_TUNED_DIR}")
    processor = WhisperProcessor.from_pretrained(str(FINE_TUNED_DIR))
    
    # Load base model
    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=dtype
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, str(FINE_TUNED_DIR))
    model = model.merge_and_unload()  # Merge for faster inference
    
    model.to(device)
    model.eval()
    
    return model, processor, device


def transcribe_audio(audio_path: str, model, processor, device) -> str:
    """Transcribe audio file with chunked processing for long audio."""
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # For long audio (>30s), process in chunks
    CHUNK_LENGTH = 30 * sr  # 30 seconds in samples
    
    if len(audio) <= CHUNK_LENGTH:
        chunks = [audio]
    else:
        # Split into overlapping chunks
        chunks = []
        stride = 25 * sr  # 25 second stride (5 second overlap)
        for start in range(0, len(audio), stride):
            end = min(start + CHUNK_LENGTH, len(audio))
            chunks.append(audio[start:end])
            if end >= len(audio):
                break
    
    transcriptions = []
    for chunk in chunks:
        # Extract features
        input_features = processor.feature_extractor(
            chunk, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)
        
        # Ensure correct dtype
        if next(model.parameters()).dtype == torch.float16:
            input_features = input_features.half()
        
        # Generate with optimized settings
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                language="yue",
                task="transcribe",
                # Inference optimizations for better accuracy
                num_beams=5,  # Beam search for better accuracy (trades speed)
                condition_on_prev_tokens=False,  # Prevent hallucination loops
                prompt_ids=processor.get_prompt_ids(INITIAL_PROMPT),  # Prime with jargon
                # Anti-hallucination settings
                no_repeat_ngram_size=3,  # Prevent repeating 3-grams
                repetition_penalty=1.2,  # Penalize repeated tokens
            )
        
        # Decode
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        transcriptions.append(text.strip())
    
    return " ".join(transcriptions)


def normalize_text(text: str) -> str:
    """Remove punctuation and normalize text for fair comparison."""
    import re
    # Remove Chinese and English punctuation
    punctuation = r'[，。！？、；：""''（）【】《》\s,\.!?\-\(\)\[\]\'\""]'
    text = re.sub(punctuation, '', text)
    return text


def compute_metrics(prediction: str, reference: str, normalize: bool = True) -> dict:
    """Compute CER and other metrics."""
    if normalize:
        prediction = normalize_text(prediction)
        reference = normalize_text(reference)
    
    # Character Error Rate (better for Chinese)
    cer = jiwer.cer(reference, prediction)
    
    # Word Error Rate (for comparison)
    wer = jiwer.wer(reference, prediction)
    
    # Character-level accuracy
    char_accuracy = 1 - cer
    
    return {
        "cer": cer,
        "wer": wer,
        "char_accuracy": char_accuracy,
        "reference_length": len(reference),
        "prediction_length": len(prediction),
    }


def main():
    print("=" * 60)
    print("Model Evaluation: Fine-tuned vs Base Model")
    print("=" * 60)
    
    # Load test data
    test_audio = TEST_DIR / "test_audio_combined.wav"
    ground_truth_path = TEST_DIR / "ground_truth.txt"
    
    if not test_audio.exists():
        print(f"Error: Test audio not found at {test_audio}")
        return
    
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth = f.read().replace("\n", "")  # Join all sentences
    
    print(f"\nTest audio: {test_audio}")
    print(f"Ground truth length: {len(ground_truth)} characters")
    
    # Evaluate base model
    print("\n" + "=" * 60)
    print("Evaluating BASE model...")
    print("=" * 60)
    
    base_model, base_processor, device = load_base_model()
    base_transcription = transcribe_audio(str(test_audio), base_model, base_processor, device)
    base_metrics = compute_metrics(base_transcription, ground_truth)
    
    print(f"\nBase Model Transcription:")
    print(f"{base_transcription[:200]}...")
    print(f"\nBase Model Metrics:")
    print(f"  CER: {base_metrics['cer']:.4f} ({base_metrics['cer']*100:.2f}%)")
    print(f"  Character Accuracy: {base_metrics['char_accuracy']*100:.2f}%")
    
    # Free memory
    del base_model
    torch.cuda.empty_cache()
    
    # Evaluate fine-tuned model
    print("\n" + "=" * 60)
    print("Evaluating FINE-TUNED model...")
    print("=" * 60)
    
    ft_model, ft_processor, device = load_fine_tuned_model()
    ft_transcription = transcribe_audio(str(test_audio), ft_model, ft_processor, device)
    ft_metrics = compute_metrics(ft_transcription, ground_truth)
    
    print(f"\nFine-tuned Model Transcription:")
    print(f"{ft_transcription[:200]}...")
    print(f"\nFine-tuned Model Metrics:")
    print(f"  CER: {ft_metrics['cer']:.4f} ({ft_metrics['cer']*100:.2f}%)")
    print(f"  Character Accuracy: {ft_metrics['char_accuracy']*100:.2f}%")
    
    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    cer_improvement = base_metrics['cer'] - ft_metrics['cer']
    cer_improvement_pct = (cer_improvement / base_metrics['cer']) * 100 if base_metrics['cer'] > 0 else 0
    
    print(f"\n{'Metric':<25} {'Base Model':<15} {'Fine-tuned':<15} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'CER':<25} {base_metrics['cer']*100:>12.2f}% {ft_metrics['cer']*100:>12.2f}% {cer_improvement*100:>+12.2f}%")
    print(f"{'Character Accuracy':<25} {base_metrics['char_accuracy']*100:>12.2f}% {ft_metrics['char_accuracy']*100:>12.2f}% {cer_improvement*100:>+12.2f}%")
    
    if cer_improvement > 0:
        print(f"\n✓ Fine-tuning IMPROVED CER by {cer_improvement_pct:.1f}% relative")
    else:
        print(f"\n✗ Fine-tuning did not improve CER (may need more training data)")
    
    # Save results
    results = {
        "ground_truth": ground_truth,
        "base_model": {
            "transcription": base_transcription,
            "metrics": base_metrics,
        },
        "fine_tuned_model": {
            "transcription": ft_transcription,
            "metrics": ft_metrics,
        },
        "improvement": {
            "cer_absolute": cer_improvement,
            "cer_relative_pct": cer_improvement_pct,
        }
    }
    
    results_path = TEST_DIR / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
