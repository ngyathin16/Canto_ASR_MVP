"""
Evaluate and compare both fine-tuned Whisper models.

Compares:
1. khleeloo/whisper-large-v3-cantonese (existing fine-tuned model)
2. JackyHoCL/whisper-large-v3-turbo-cantonese-yue-english (new fine-tuned model)

Uses optimized inference settings: beam search, initial_prompt, no condition_on_prev_tokens
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import librosa
import jiwer
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

# Configuration
KHLEELOO_BASE = "khleeloo/whisper-large-v3-cantonese"
JACKY_BASE = "JackyHoCL/whisper-large-v3-turbo-cantonese-yue-english"

KHLEELOO_FINE_TUNED_DIR = Path(__file__).parent / "model"
KHLEELOO_V2_FINE_TUNED_DIR = Path(__file__).parent / "model_v2"  # Retrained with augmented data
JACKY_FINE_TUNED_DIR = Path(__file__).parent / "model_jacky"
DEFAULT_TEST_DIR = Path(__file__).parent / "test_evaluation"

# Initial prompt with top construction jargon terms to prime the model
INITIAL_PROMPT = (
    "天秤, 扎鐵, 石屎, 狗臂架, 趷雞陣, 士巴拿, 工字, 街吊, 門頭陣, "
    "靚仔面, 瓦仔, 膠凳仔, 鹽水喉, 埋碼, 震筆, 開士機, 鉸剪台, "
    "花籃, 風煤, 判頭, 大判, 蛇頭, 英展, 安督, 地盤三寶, 銀卡"
)


def load_fine_tuned_model(base_model: str, adapter_dir: Path):
    """Load fine-tuned model with LoRA adapter."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Loading fine-tuned model from: {adapter_dir}")
    print(f"Base model: {base_model}")
    
    processor = WhisperProcessor.from_pretrained(str(adapter_dir))
    
    # Load base model
    model = WhisperForConditionalGeneration.from_pretrained(
        base_model, torch_dtype=dtype
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()  # Merge for faster inference
    
    model.to(device)
    model.eval()
    
    return model, processor, device


def transcribe_audio(audio_path: str, model, processor, device) -> str:
    """Transcribe audio file with optimized settings."""
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # For long audio (>30s), process in chunks
    CHUNK_LENGTH = 30 * sr  # 30 seconds in samples
    
    if len(audio) <= CHUNK_LENGTH:
        chunks = [audio]
    else:
        chunks = []
        stride = 25 * sr  # 25 second stride (5 second overlap)
        for start in range(0, len(audio), stride):
            end = min(start + CHUNK_LENGTH, len(audio))
            chunks.append(audio[start:end])
            if end >= len(audio):
                break
    
    transcriptions = []
    for chunk in chunks:
        input_features = processor.feature_extractor(
            chunk, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)
        
        if next(model.parameters()).dtype == torch.float16:
            input_features = input_features.half()
        
        # Generate with optimized settings
        with torch.no_grad():
            # Get prompt IDs and convert to tensor
            prompt_ids = processor.get_prompt_ids(INITIAL_PROMPT)
            if isinstance(prompt_ids, np.ndarray):
                prompt_ids = torch.from_numpy(prompt_ids).to(device)
            
            predicted_ids = model.generate(
                input_features,
                language="yue",
                task="transcribe",
                num_beams=5,
                condition_on_prev_tokens=False,
                prompt_ids=prompt_ids,
            )
        
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        transcriptions.append(text.strip())
    
    return " ".join(transcriptions)


def remove_repeated_sentences(text: str) -> str:
    """Remove repeated sentences/phrases from transcription (hallucination fix)."""
    import re
    # Split by common sentence delimiters (including space for Whisper output)
    sentences = re.split(r'[。！？\n\s]+', text)
    seen = set()
    unique_sentences = []
    for s in sentences:
        s_clean = s.strip()
        # Skip empty or very short fragments
        if len(s_clean) < 3:
            continue
        # Check if this sentence (or very similar) already exists
        is_duplicate = False
        for existing in seen:
            # Check for exact match or substring match (for partial repetitions)
            if s_clean == existing or s_clean in existing or existing in s_clean:
                is_duplicate = True
                break
        if not is_duplicate:
            seen.add(s_clean)
            unique_sentences.append(s_clean)
    return ' '.join(unique_sentences)


def normalize_text(text: str) -> str:
    """Remove punctuation and normalize text for fair comparison."""
    import re
    punctuation = r'[，。！？、；：""''（）【】《》\s,\.!?\-\(\)\[\]\'\""]'
    text = re.sub(punctuation, '', text)
    return text


def compute_metrics(prediction: str, reference: str) -> dict:
    """Compute CER and other metrics."""
    prediction = normalize_text(prediction)
    reference = normalize_text(reference)
    
    cer = jiwer.cer(reference, prediction)
    wer = jiwer.wer(reference, prediction)
    
    return {
        "cer": cer,
        "wer": wer,
        "char_accuracy": 1 - cer,
        "reference_length": len(reference),
        "prediction_length": len(prediction),
    }


def main(test_dir: Path = None):
    if test_dir is None:
        test_dir = DEFAULT_TEST_DIR
    
    print("=" * 70)
    print("Model Comparison: khleeloo vs JackyHoCL Fine-tuned Models")
    print(f"Test directory: {test_dir}")
    print("=" * 70)
    
    # Load test data - find audio and ground truth files
    # Support both naming conventions
    test_audio = None
    for name in ["test_audio_combined.wav", "test_audio_combined_v2.wav"]:
        if (test_dir / name).exists():
            test_audio = test_dir / name
            break
    
    ground_truth_path = None
    for name in ["ground_truth.txt", "ground_truth_v2.txt"]:
        if (test_dir / name).exists():
            ground_truth_path = test_dir / name
            break
    
    if not test_audio or not test_audio.exists():
        print(f"Error: Test audio not found in {test_dir}")
        return
    
    if not ground_truth_path or not ground_truth_path.exists():
        print(f"Error: Ground truth not found in {test_dir}")
        return
    
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth = f.read().replace("\n", "")
    
    print(f"\nTest audio: {test_audio}")
    print(f"Ground truth length: {len(ground_truth)} characters")
    
    results = {}
    
    # Evaluate khleeloo fine-tuned model (original)
    print("\n" + "=" * 70)
    print("Evaluating KHLEELOO fine-tuned model (original)...")
    print("=" * 70)
    
    if KHLEELOO_FINE_TUNED_DIR.exists():
        khleeloo_model, khleeloo_processor, device = load_fine_tuned_model(
            KHLEELOO_BASE, KHLEELOO_FINE_TUNED_DIR
        )
        khleeloo_transcription = transcribe_audio(
            str(test_audio), khleeloo_model, khleeloo_processor, device
        )
        # Post-process to remove repeated sentences (hallucination fix)
        khleeloo_transcription = remove_repeated_sentences(khleeloo_transcription)
        khleeloo_metrics = compute_metrics(khleeloo_transcription, ground_truth)
        
        print(f"\nKhleeloo Transcription (first 200 chars):")
        print(f"{khleeloo_transcription[:200]}...")
        print(f"\nKhleeloo Metrics:")
        print(f"  CER: {khleeloo_metrics['cer']:.4f} ({khleeloo_metrics['cer']*100:.2f}%)")
        print(f"  Character Accuracy: {khleeloo_metrics['char_accuracy']*100:.2f}%")
        
        results["khleeloo"] = {
            "base_model": KHLEELOO_BASE,
            "transcription": khleeloo_transcription,
            "metrics": khleeloo_metrics,
        }
        
        del khleeloo_model
        torch.cuda.empty_cache()
    else:
        print(f"Warning: Khleeloo fine-tuned model not found at {KHLEELOO_FINE_TUNED_DIR}")
    
    # Evaluate khleeloo V2 fine-tuned model (retrained with augmented data)
    print("\n" + "=" * 70)
    print("Evaluating KHLEELOO V2 fine-tuned model (augmented data + early stopping)...")
    print("=" * 70)
    
    if KHLEELOO_V2_FINE_TUNED_DIR.exists():
        khleeloo_v2_model, khleeloo_v2_processor, device = load_fine_tuned_model(
            KHLEELOO_BASE, KHLEELOO_V2_FINE_TUNED_DIR
        )
        khleeloo_v2_transcription = transcribe_audio(
            str(test_audio), khleeloo_v2_model, khleeloo_v2_processor, device
        )
        # Post-process to remove repeated sentences (hallucination fix)
        khleeloo_v2_transcription = remove_repeated_sentences(khleeloo_v2_transcription)
        khleeloo_v2_metrics = compute_metrics(khleeloo_v2_transcription, ground_truth)
        
        print(f"\nKhleeloo V2 Transcription (first 200 chars):")
        print(f"{khleeloo_v2_transcription[:200]}...")
        print(f"\nKhleeloo V2 Metrics:")
        print(f"  CER: {khleeloo_v2_metrics['cer']:.4f} ({khleeloo_v2_metrics['cer']*100:.2f}%)")
        print(f"  Character Accuracy: {khleeloo_v2_metrics['char_accuracy']*100:.2f}%")
        
        results["khleeloo_v2"] = {
            "base_model": KHLEELOO_BASE,
            "description": "Retrained with augmented data (silence trimming + noise) + early stopping",
            "transcription": khleeloo_v2_transcription,
            "metrics": khleeloo_v2_metrics,
        }
        
        del khleeloo_v2_model
        torch.cuda.empty_cache()
    else:
        print(f"Warning: Khleeloo V2 fine-tuned model not found at {KHLEELOO_V2_FINE_TUNED_DIR}")
    
    # Evaluate JackyHoCL fine-tuned model
    print("\n" + "=" * 70)
    print("Evaluating JACKYHOCL fine-tuned model...")
    print("=" * 70)
    
    if JACKY_FINE_TUNED_DIR.exists():
        jacky_model, jacky_processor, device = load_fine_tuned_model(
            JACKY_BASE, JACKY_FINE_TUNED_DIR
        )
        jacky_transcription = transcribe_audio(
            str(test_audio), jacky_model, jacky_processor, device
        )
        # Post-process to remove repeated sentences (hallucination fix)
        jacky_transcription = remove_repeated_sentences(jacky_transcription)
        jacky_metrics = compute_metrics(jacky_transcription, ground_truth)
        
        print(f"\nJackyHoCL Transcription (first 200 chars):")
        print(f"{jacky_transcription[:200]}...")
        print(f"\nJackyHoCL Metrics:")
        print(f"  CER: {jacky_metrics['cer']:.4f} ({jacky_metrics['cer']*100:.2f}%)")
        print(f"  Character Accuracy: {jacky_metrics['char_accuracy']*100:.2f}%")
        
        results["jackyhocl"] = {
            "base_model": JACKY_BASE,
            "transcription": jacky_transcription,
            "metrics": jacky_metrics,
        }
        
        del jacky_model
        torch.cuda.empty_cache()
    else:
        print(f"Warning: JackyHoCL fine-tuned model not found at {JACKY_FINE_TUNED_DIR}")
    
    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    # Collect all available models
    model_results = []
    if "khleeloo" in results:
        model_results.append(("khleeloo (original)", results["khleeloo"]["metrics"]["cer"], "model"))
    if "khleeloo_v2" in results:
        model_results.append(("khleeloo V2 (augmented)", results["khleeloo_v2"]["metrics"]["cer"], "model_v2"))
    if "jackyhocl" in results:
        model_results.append(("JackyHoCL", results["jackyhocl"]["metrics"]["cer"], "model_jacky"))
    
    if model_results:
        print(f"\n{'Model':<35} {'CER':<15} {'Accuracy':<15}")
        print("-" * 70)
        
        for name, cer, _ in model_results:
            print(f"{name:<35} {cer*100:>12.2f}% {(1-cer)*100:>12.2f}%")
        
        # Find best model
        best_model = min(model_results, key=lambda x: x[1])
        print(f"\n{'='*70}")
        print(f"🏆 BEST MODEL: {best_model[0]} with CER {best_model[1]*100:.2f}%")
        print(f"   RECOMMENDATION: Use {best_model[2]} for production")
        print(f"{'='*70}")
    
    # Save results
    results["ground_truth"] = ground_truth
    results_path = test_dir / "comparison_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Whisper models")
    parser.add_argument(
        "--test-dir", "-t",
        type=Path,
        default=DEFAULT_TEST_DIR,
        help="Directory containing test audio and ground truth"
    )
    args = parser.parse_args()
    main(args.test_dir)
