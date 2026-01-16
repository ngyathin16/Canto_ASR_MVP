# Canto ASR MVP

Cantonese Automatic Speech Recognition (ASR) pipeline for transcribing audio recordings, optimized for noisy real-world environments such as construction sites. Includes fine-tuning pipeline with LoRA for domain-specific vocabulary.

## Model Performance

Fine-tuned Whisper model achieves **4.00% average CER** across 4 test sets with 48 unique sentences.

| Model | Avg CER | Description |
|-------|---------|-------------|
| **khleeloo V2** | **4.00%** | Fine-tuned with anti-overfitting (recommended) |
| khleeloo original | 6.29% | First fine-tuned version |
| JackyHoCL | 9.00% | Alternative base model |

### Per-Test-Set Results (khleeloo V2)

| Test Set | CER | Terms Tested |
|----------|-----|--------------|
| V1 | 2.16% | 天秤, 扎鐵, 石屎, 瓦仔, 震船, 地盤三寶 |
| V2 | 7.41% | 判頭, 叻架佬, 嘩佬, 九線掣, 黃蜂竇 |
| V3 | 2.30% | 狗臂架, 趷雞陣, 街吊, 膠凳仔, 鉸剪台 |
| V4 | 4.14% | 大判, 蛇頭, 英展, 安督, 佛沙, 威也 |

## Features

- **Fine-tuning**: LoRA-based fine-tuning on domain-specific Cantonese construction terminology
- **Transcription**: Uses fine-tuned [khleeloo/whisper-large-v3-cantonese](https://huggingface.co/khleeloo/whisper-large-v3-cantonese) model
- **Post-processing**: Hallucination removal and Azure OpenAI-powered transcript cleaning
- **TTS Data Generation**: Azure TTS for synthetic training data with noise augmentation
- **Output formats**: SRT (subtitles) and plain text

## Setup

1. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   Create a `.env` file with:
   ```
   # For TTS data generation
   AZURE_SPEECH_KEY=your_speech_key
   AZURE_SPEECH_REGION=eastus2

   # For transcript cleaning (optional)
   AZURE_OPENAI_API_KEY=your_api_key
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
   ```

## Usage

### Fine-tuning

```bash
# Generate training sentences
python sentence_generation/generate_training_sentences.py

# Generate TTS audio clips
python tts_generation/generate_audio_clips.py

# Prepare dataset
python fine_tuning/prepare_dataset.py

# Train model (outputs to fine_tuning/model_v2/)
python fine_tuning/train.py
```

### Evaluation

```bash
# Evaluate on default test set
python fine_tuning/evaluate_both_models.py

# Evaluate on specific test set
python fine_tuning/evaluate_both_models.py --test-dir fine_tuning/test_evaluation_v2
```

### Transcribe Audio

```bash
python transcription/transcribe_audio_khleeloo.py <audio_file>

# Examples:
python transcription/transcribe_audio_khleeloo.py site_01.mp3
python transcription/transcribe_audio_khleeloo.py site_01.mp3 --output-dir ./transcripts
```

### Clean Transcript (Post-processing)

```bash
python transcription/clean_transcript.py <input_file.srt>
```

## Project Structure

```
├── fine_tuning/
│   ├── train.py                 # LoRA fine-tuning script
│   ├── evaluate_both_models.py  # Model evaluation with --test-dir support
│   ├── prepare_dataset.py       # Dataset preparation with augmentation
│   ├── generate_test_audio*.py  # Test audio generation scripts
│   └── test_evaluation*/        # Test sets (V1-V4)
├── sentence_generation/
│   ├── generate_training_sentences.py
│   └── normalized_terms_construction.json
├── tts_generation/
│   ├── generate_audio_clips.py  # Azure TTS generation
│   └── audio_clips*/            # Generated audio clips
├── transcription/
│   ├── transcribe_audio_*.py    # Transcription scripts
│   └── clean_transcript.py      # Post-processing
├── ACCURACY_OPTIMIZATION_PROTOCOL.md  # Optimization guide
└── requirements.txt
```

## Training Configuration (Anti-Overfitting)

The V2 model uses optimized hyperparameters to prevent overfitting:

| Parameter | Value |
|-----------|-------|
| LoRA rank | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.15 |
| Learning rate | 3e-5 |
| Weight decay | 0.05 |
| Early stopping patience | 3 |

## License

MIT
