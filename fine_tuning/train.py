"""
Fine-tune Whisper for Cantonese construction domain ASR.

Base model: khleeloo/whisper-large-v3-cantonese (already fine-tuned for Cantonese)

This script further fine-tunes the model on domain-specific audio data using
LoRA (Low-Rank Adaptation) for efficient training with reduced memory usage.

Key design decisions:
1. LoRA fine-tuning: Only trains ~1% of parameters, prevents catastrophic forgetting
2. Mixed precision (fp16): Faster training, lower memory on GPU
3. Gradient checkpointing: Trades compute for memory efficiency
4. Seq2SeqTrainer: Optimized for encoder-decoder models like Whisper

Why khleeloo/whisper-large-v3-cantonese:
- Already fine-tuned specifically for Cantonese
- Better baseline for Cantonese ASR than generic Whisper
- Domain adaptation builds on existing Cantonese capabilities
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
try:
    import nvtx
    HAS_NVTX = True
except ImportError:
    HAS_NVTX = False
    print("Warning: nvtx not installed. NVTX markers will be disabled.")
from datasets import DatasetDict, load_from_disk
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)
from peft import LoraConfig, get_peft_model


class ProfiledTrainer(Trainer):
    """
    Custom Trainer with NVTX markers and Torch Profiler integration.
    
    Instruments:
    - Data loading step
    - Forward pass
    - Backward pass
    """
    
    def __init__(self, *args, enable_profiling=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_profiling = enable_profiling
        self.profiler = None
        self.step_count = 0
        
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training_step to add NVTX markers."""
        self.step_count += 1
        
        # NVTX marker for forward pass
        if HAS_NVTX:
            nvtx.push_range("Forward")
        
        with record_function("forward_pass"):
            loss = super().training_step(model, inputs, num_items_in_batch)
        
        if HAS_NVTX:
            nvtx.pop_range()
        
        # Advance profiler schedule
        if self.profiler is not None:
            self.profiler.step()
        
        return loss
    
    def get_train_dataloader(self):
        """Override to add NVTX markers around data loading."""
        dataloader = super().get_train_dataloader()
        return InstrumentedDataLoader(dataloader)


class InstrumentedDataLoader:
    """
    Wrapper around DataLoader to add NVTX markers for data loading.
    """
    def __init__(self, dataloader):
        self.dataloader = dataloader
        
    def __iter__(self):
        for batch in self.dataloader:
            if HAS_NVTX:
                nvtx.push_range("DataLoading")
            yield batch
            if HAS_NVTX:
                nvtx.pop_range()
    
    def __len__(self):
        return len(self.dataloader)
    
    @property
    def batch_size(self):
        return self.dataloader.batch_size
    
    @property
    def dataset(self):
        return self.dataloader.dataset


# Configuration
# Using khleeloo model - best performer on construction domain
BASE_MODEL = "khleeloo/whisper-large-v3-cantonese"
LANGUAGE = "yue"  # Cantonese language code
TASK = "transcribe"

# Paths - output to model_v2 to preserve original khleeloo fine-tuned model
DATASET_DIR = Path(__file__).parent / "dataset"
OUTPUT_DIR = Path(__file__).parent / "checkpoints_v2"
FINAL_MODEL_DIR = Path(__file__).parent / "model_v2"


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator for Whisper fine-tuning.
    
    Handles:
    - Padding input features (mel spectrograms) to max length in batch
    - Padding labels (token IDs) and replacing padding with -100 for loss masking
    """
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        
        # Pad input features
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels using tokenizer's __call__ method (more efficient for fast tokenizer)
        label_ids = [feature["labels"] for feature in features]
        max_label_length = max(len(l) for l in label_ids)
        
        # Pad labels manually to avoid the warning
        padded_labels = []
        for label in label_ids:
            padding_length = max_label_length - len(label)
            padded_label = label + [self.processor.tokenizer.pad_token_id] * padding_length
            padded_labels.append(padded_label)
        
        labels = torch.tensor(padded_labels)
        
        # Replace padding token id with -100 so it's ignored in loss computation
        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)

        # Remove BOS token if present (Whisper doesn't need it)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        # Only return input_features and labels
        return {
            "input_features": batch["input_features"],
            "labels": labels,
        }


def load_processor(model_name: str) -> WhisperProcessor:
    """Load and configure Whisper processor."""
    processor = WhisperProcessor.from_pretrained(model_name, language=LANGUAGE, task=TASK)
    return processor


def prepare_dataset(dataset: DatasetDict, processor: WhisperProcessor) -> DatasetDict:
    """
    Prepare dataset for training by extracting features and tokenizing transcripts.
    
    Args:
        dataset: Raw dataset with audio and transcript columns
        processor: Whisper processor for feature extraction and tokenization
    
    Returns:
        Processed dataset ready for training
    """
    def prepare_example(batch):
        # Extract mel spectrogram features from audio
        # Audio is stored as dict with 'array' and 'sampling_rate' keys
        audio = batch["audio"]
        audio_array = audio["array"]
        sampling_rate = audio["sampling_rate"]
        
        # Convert list to numpy array if needed
        if isinstance(audio_array, list):
            audio_array = np.array(audio_array, dtype=np.float32)
        
        batch["input_features"] = processor.feature_extractor(
            audio_array, 
            sampling_rate=sampling_rate
        ).input_features[0]
        
        # Tokenize transcript
        batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids
        return batch
    
    # Process dataset
    processed = dataset.map(
        prepare_example,
        remove_columns=dataset["train"].column_names,
        num_proc=1,  # Audio processing doesn't parallelize well
    )
    
    return processed


class WhisperLoRAWrapper(torch.nn.Module):
    """
    Wrapper for PEFT Whisper model that correctly routes input_features.
    
    PEFT's default forward signature expects input_ids, but Whisper uses
    input_features. This wrapper ensures correct argument passing.
    """
    def __init__(self, peft_model):
        super().__init__()
        self.peft_model = peft_model
        self.config = peft_model.config
        self._keys_to_ignore_on_save = None
    
    def forward(self, input_features=None, labels=None, **kwargs):
        # Get the base model with LoRA adapters applied
        base_model = self.peft_model.get_base_model()
        # Filter out kwargs that Whisper doesn't accept
        valid_kwargs = {k: v for k, v in kwargs.items() 
                       if k in ['attention_mask', 'decoder_input_ids', 'decoder_attention_mask',
                               'head_mask', 'decoder_head_mask', 'cross_attn_head_mask',
                               'encoder_outputs', 'past_key_values', 'decoder_inputs_embeds',
                               'use_cache', 'output_attentions', 'output_hidden_states', 'return_dict']}
        return base_model(input_features=input_features, labels=labels, **valid_kwargs)
    
    def save_pretrained(self, *args, **kwargs):
        # Use safe_serialization=False to handle shared tensors
        kwargs['safe_serialization'] = False
        return self.peft_model.save_pretrained(*args, **kwargs)
    
    def state_dict(self):
        return self.peft_model.state_dict()
    
    def gradient_checkpointing_enable(self):
        return self.peft_model.gradient_checkpointing_enable()
    
    def parameters(self):
        return self.peft_model.parameters()
    
    def named_parameters(self):
        return self.peft_model.named_parameters()


def load_model_with_lora(model_name: str) -> WhisperLoRAWrapper:
    """
    Load Whisper model with LoRA adapters for efficient fine-tuning.
    
    LoRA (Low-Rank Adaptation) benefits:
    - Trains only ~1% of parameters
    - Prevents catastrophic forgetting of Cantonese ASR capabilities
    - Reduces memory requirements significantly
    - Produces small adapter files (~10-50MB vs 3GB full model)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Load base model
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch_dtype,
        use_cache=False,  # Required for gradient checkpointing
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Configure LoRA - reduced capacity for better generalization
    lora_config = LoraConfig(
        r=32,  # Reduced rank (was 64) - less capacity = less overfitting
        lora_alpha=64,  # Scaling factor (2 * r)
        # Target attention layers only - removing FFN reduces overfitting
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention projections only
        ],
        lora_dropout=0.15,  # Higher dropout for regularization (was 0.1)
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    
    # Apply LoRA
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    # Wrap to fix forward signature for Whisper
    return WhisperLoRAWrapper(peft_model)


def compute_metrics(pred, processor: WhisperProcessor, metric):
    """Compute Character Error Rate (CER) for evaluation."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Replace -100 with pad token for decoding
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    # Decode predictions and references
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    # Compute CER (more appropriate for Chinese than WER)
    cer = metric.compute(predictions=pred_str, references=label_str)
    
    return {"cer": cer}


def main():
    print("=" * 60)
    print("Whisper Fine-tuning for Cantonese Construction ASR")
    print(f"Base model: {BASE_MODEL}")
    print("=" * 60)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Enable cuDNN autotuner for faster convolutions
        torch.backends.cudnn.benchmark = True
        print("Enabled: cudnn.benchmark=True")
    
    # Load dataset
    print(f"\nLoading dataset from: {DATASET_DIR}")
    if not DATASET_DIR.exists():
        print("Error: Dataset not found. Run prepare_dataset.py first.")
        return
    
    dataset = load_from_disk(str(DATASET_DIR))
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")
    
    # Load processor
    print(f"\nLoading processor from: {BASE_MODEL}")
    processor = load_processor(BASE_MODEL)
    
    # Prepare dataset
    print("\nPreparing dataset (extracting features, tokenizing)...")
    processed_dataset = prepare_dataset(dataset, processor)
    print(f"Processed train samples: {len(processed_dataset['train'])}")
    
    # Load model with LoRA
    print(f"\nLoading model with LoRA: {BASE_MODEL}")
    model = load_model_with_lora(BASE_MODEL)
    
    # Note: torch.compile() is incompatible with custom wrapper classes
    # The main optimizations are: larger batch size + no gradient accumulation
    
    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    # Metric
    cer_metric = evaluate.load("cer")
    
    # Calculate optimal num_workers
    num_workers = min(os.cpu_count() // 2, 4) if os.cpu_count() else 2
    print(f"\nDataLoader optimization: num_workers={num_workers}, pin_memory=True")
    
    # Training arguments with optimizations
    # PROFILER FINDINGS:
    # - GPU Occupancy was 36.53% (low) -> increase batch size
    # - CPU Exec was 24.46% -> reduce gradient accumulation
    # - Kernel time 60.95% -> enable torch.compile for fusion
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        
        # Training hyperparameters - OPTIMIZED for generalization (anti-overfitting)
        per_device_train_batch_size=16,  # Doubled for better GPU occupancy (36%->70%+)
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,  # Removed to reduce CPU overhead (24%->~12%)
        learning_rate=3e-5,  # Lower LR for better generalization
        warmup_steps=150,  # Longer warmup for stability
        num_train_epochs=30,  # Fewer epochs to prevent overfitting
        weight_decay=0.05,  # Stronger regularization (was 0.01)
        
        # Optimization - use 8-bit AdamW for memory efficiency
        fp16=device == "cuda",
        optim="adamw_bnb_8bit" if device == "cuda" else "adamw_torch",
        
        # DataLoader optimization (CRITICAL for throughput)
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True if device == "cuda" else False,
        dataloader_persistent_workers=True if num_workers > 0 else False,
        dataloader_prefetch_factor=2 if num_workers > 0 else None,
        
        # Evaluation & saving with early stopping
        eval_strategy="steps",
        eval_steps=25,  # More frequent evaluation for early stopping
        save_strategy="steps",
        save_steps=25,
        save_total_limit=5,  # Keep more checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Logging
        logging_steps=10,
        report_to="none",  # Disable wandb/tensorboard
        
        # Disable label_names to prevent Trainer from looking for input_ids
        label_names=["labels"],
        
        # Disable safetensors to handle shared tensor weights in Whisper
        save_safetensors=False,
    )
    
    # Early stopping callback to prevent overfitting
    # Stop earlier to prevent overfitting (3 evaluations = 75 steps)
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,  # Reduced from 5 - stop earlier
        early_stopping_threshold=0.005,  # Higher threshold - need more improvement to continue
    )
    
    # Trainer with profiling instrumentation and early stopping
    trainer = ProfiledTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        data_collator=data_collator,
        callbacks=[early_stopping],
        enable_profiling=True,
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    # Set up Torch Profiler for first 10 steps
    PROFILE_DIR = Path(__file__).parent / "profiler_logs"
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    
    profiler_schedule = schedule(
        wait=1,      # Skip first step (warmup)
        warmup=1,    # Warmup for 1 step
        active=8,    # Profile 8 steps
        repeat=1     # Only run once
    )
    
    print(f"\nProfiler logs will be saved to: {PROFILE_DIR}")
    print("Run 'tensorboard --logdir=profiler_logs' to visualize traces")
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=profiler_schedule,
        on_trace_ready=tensorboard_trace_handler(str(PROFILE_DIR)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        # Custom training loop with profiler step callback
        trainer.profiler = prof
        trainer.train()
    
    # Save final model
    print(f"\nSaving final model to: {FINAL_MODEL_DIR}")
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save LoRA adapter
    model.save_pretrained(str(FINAL_MODEL_DIR))
    processor.save_pretrained(str(FINAL_MODEL_DIR))
    
    # Save training info
    training_info = {
        "base_model": BASE_MODEL,
        "language": LANGUAGE,
        "task": TASK,
        "train_samples": len(dataset["train"]),
        "validation_samples": len(dataset["validation"]),
        "lora_config": {
            "r": 64,
            "lora_alpha": 128,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"],
        },
    }
    with open(FINAL_MODEL_DIR / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {FINAL_MODEL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
