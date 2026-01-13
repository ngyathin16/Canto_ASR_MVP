# Canto ASR MVP

Cantonese Automatic Speech Recognition (ASR) pipeline for transcribing audio recordings, optimized for noisy real-world environments such as construction sites.

## Features

- **Transcription**: Uses the [khleeloo/whisper-large-v3-cantonese](https://huggingface.co/khleeloo/whisper-large-v3-cantonese) model for accurate Cantonese speech recognition
- **Post-processing**: Azure OpenAI-powered transcript cleaning to fix ASR errors, homophones, and domain-specific jargon
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

3. **Configure environment variables** (for transcript cleaning):
   Create a `.env` file with:
   ```
   AZURE_OPENAI_API_KEY=your_api_key
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
   ```

## Usage

### Transcribe Audio

```bash
python transcribe_audio_khleeloo.py <audio_file>

# Examples:
python transcribe_audio_khleeloo.py site_01.mp3
python transcribe_audio_khleeloo.py site_01.mp3 --output-dir ./transcripts --verbose
python transcribe_audio_khleeloo.py site_01.mp3 --output-format txt
```

### Clean Transcript (Post-processing)

```bash
python clean_transcript.py <input_file.srt>

# Example:
python clean_transcript.py transcripts_khleeloo/test_1.srt
```

This uses Azure OpenAI to correct phonetic errors, homophones, and domain jargon in the transcript. Output is saved as `<filename>_cleaned.srt`.

## Project Structure

- `transcribe_audio_khleeloo.py` - Main transcription script
- `transcription_service.py` - Core transcription service module
- `output_formatter.py` - Output formatting utilities
- `clean_transcript.py` - Azure OpenAI-based transcript cleaning
- `terms.json` - Domain terminology for transcript cleaning context
- `requirements.txt` - Python dependencies

## License

MIT
