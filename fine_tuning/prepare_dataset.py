"""
Prepare audio clips and transcripts for Whisper fine-tuning.

This script combines audio clips from multiple folders (male/female voices)
and creates a HuggingFace Dataset suitable for training.
"""

import json
import os
from pathlib import Path
import random

import librosa
import numpy as np
import soundfile as sf
from datasets import Dataset, DatasetDict

# Paths
TTS_DIR = Path(__file__).parent.parent / "tts_generation"
AUDIO_DIRS = [
    TTS_DIR / "audio_clips",        # Female voice
    TTS_DIR / "audio_clips_male",   # Male voice
]
OUTPUT_DIR = Path(__file__).parent / "dataset"


def load_manifest(audio_dir: Path) -> list[dict]:
    """Load manifest.json from an audio directory."""
    manifest_path = audio_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Warning: No manifest found at {manifest_path}")
        return []
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    # Add full audio path to each entry
    for entry in manifest:
        entry["audio_path"] = str(audio_dir / entry["audio_file"])
    
    return manifest


def trim_silence(audio_array: np.ndarray, top_db: int = 20) -> np.ndarray:
    """
    Trim leading and trailing silence from audio.
    
    TTS engines often leave 1-2 seconds of absolute silence at start/end.
    Whisper treats long silence as a trigger to hallucinate (repeat previous text).
    
    Args:
        audio_array: Audio samples as numpy array
        top_db: Threshold in dB below reference to consider as silence
    
    Returns:
        Trimmed audio array
    """
    trimmed, _ = librosa.effects.trim(audio_array, top_db=top_db)
    return trimmed


def add_noise_augmentation(audio_array: np.ndarray, snr_db: float = None) -> np.ndarray:
    """
    Add Gaussian noise to audio at specified SNR.
    
    TTS audio is "too clean" - Whisper overfits to perfect digital silence
    and robotic prosody. Adding noise forces the model to focus on phonemes.
    
    Args:
        audio_array: Audio samples as numpy array
        snr_db: Signal-to-noise ratio in dB. If None, randomly choose 20-30dB.
    
    Returns:
        Audio array with added noise
    """
    if snr_db is None:
        snr_db = random.uniform(20, 30)  # Random SNR between 20-30dB
    
    # Calculate signal power
    signal_power = np.mean(audio_array ** 2)
    
    # Calculate noise power based on SNR
    # SNR = 10 * log10(signal_power / noise_power)
    # noise_power = signal_power / (10 ** (snr_db / 10))
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    # Generate Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio_array))
    
    # Add noise to signal
    noisy_audio = audio_array + noise.astype(audio_array.dtype)
    
    return noisy_audio


def load_audio(audio_path: str, apply_augmentation: bool = True) -> dict:
    """
    Load audio file with optional augmentation.
    
    Applies:
    1. Silence trimming (always) - prevents hallucination on long silence
    2. Noise injection (optional) - combats TTS "too clean" problem
    
    Args:
        audio_path: Path to audio file
        apply_augmentation: Whether to apply noise augmentation
    
    Returns:
        Dict with 'array' and 'sampling_rate' keys
    """
    audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # Always trim silence (critical for preventing hallucination)
    audio_array = trim_silence(audio_array)
    
    # Apply noise augmentation for training robustness
    if apply_augmentation:
        audio_array = add_noise_augmentation(audio_array)
    
    return {"array": audio_array, "sampling_rate": sr}


def create_dataset() -> Dataset:
    """
    Create a HuggingFace Dataset from all audio clips.
    
    Returns:
        Dataset with columns: audio, transcript, term, type, source
    """
    all_entries = []
    
    for audio_dir in AUDIO_DIRS:
        if not audio_dir.exists():
            print(f"Warning: Directory not found: {audio_dir}")
            continue
        
        source = audio_dir.name  # e.g., "audio_clips" or "audio_clips_male"
        manifest = load_manifest(audio_dir)
        
        for entry in manifest:
            audio_path = entry["audio_path"]
            if not Path(audio_path).exists():
                print(f"Warning: Audio file not found: {audio_path}")
                continue
            
            # Load audio directly
            audio_data = load_audio(audio_path)
            
            all_entries.append({
                "audio": audio_data,
                "transcript": entry["transcript"],
                "term": entry["term"],
                "type": entry.get("type", ""),
                "source": source,
            })
    
    print(f"Total samples collected: {len(all_entries)}")
    
    # Create Dataset
    dataset = Dataset.from_dict({
        "audio": [e["audio"] for e in all_entries],
        "transcript": [e["transcript"] for e in all_entries],
        "term": [e["term"] for e in all_entries],
        "type": [e["type"] for e in all_entries],
        "source": [e["source"] for e in all_entries],
    })
    
    return dataset


def split_dataset(dataset: Dataset, test_size: float = 0.1, seed: int = 42) -> DatasetDict:
    """
    Split dataset into train and validation sets.
    
    Args:
        dataset: Full dataset
        test_size: Fraction for validation (default 10%)
        seed: Random seed for reproducibility
    
    Returns:
        DatasetDict with 'train' and 'validation' splits
    """
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    return DatasetDict({
        "train": split["train"],
        "validation": split["test"],
    })


def main():
    print("=== Preparing Fine-tuning Dataset ===\n")
    
    # Create dataset
    dataset = create_dataset()
    print(f"\nDataset created with {len(dataset)} samples")
    print(f"Columns: {dataset.column_names}")
    
    # Show sample
    print("\nSample entry:")
    sample = dataset[0]
    print(f"  Transcript: {sample['transcript']}")
    print(f"  Term: {sample['term']}")
    print(f"  Type: {sample['type']}")
    print(f"  Source: {sample['source']}")
    print(f"  Audio length: {len(sample['audio']['array'])} samples")
    print(f"  Audio sample rate: {sample['audio']['sampling_rate']}")
    
    # Split into train/validation
    dataset_dict = split_dataset(dataset)
    print(f"\nTrain samples: {len(dataset_dict['train'])}")
    print(f"Validation samples: {len(dataset_dict['validation'])}")
    
    # Save to disk
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(OUTPUT_DIR))
    print(f"\nDataset saved to: {OUTPUT_DIR}")
    
    # Also save as individual files for inspection
    train_info = {
        "num_train": len(dataset_dict["train"]),
        "num_validation": len(dataset_dict["validation"]),
        "columns": dataset_dict["train"].column_names,
        "audio_dirs": [str(d) for d in AUDIO_DIRS],
    }
    with open(OUTPUT_DIR / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(train_info, f, indent=2)
    
    print("\n=== Dataset Preparation Complete ===")


if __name__ == "__main__":
    main()
