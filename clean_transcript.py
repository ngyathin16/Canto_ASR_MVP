"""
Clean transcript using Azure OpenAI to fix Whisper V3 transcription errors.
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI


TERMS_FILE = Path(__file__).parent / "terms.json"


def load_config():
    """Load Azure OpenAI configuration from environment variables."""
    load_dotenv()
    
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    
    if not all([api_key, endpoint, deployment_name]):
        raise ValueError(
            "Missing required environment variables. "
            "Please set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, "
            "and AZURE_OPENAI_DEPLOYMENT_NAME in your .env file."
        )
    
    return {
        "api_key": api_key,
        "endpoint": endpoint,
        "deployment_name": deployment_name,
        "api_version": api_version,
    }


def create_client(config: dict) -> AzureOpenAI:
    """Initialize the Azure OpenAI client."""
    return AzureOpenAI(
        api_key=config["api_key"],
        api_version=config["api_version"],
        azure_endpoint=config["endpoint"],
    )


def load_term_context() -> str:
    """Load terminology context from terms.json if available."""
    if not TERMS_FILE.exists():
        return "No specialized terminology provided."
    
    try:
        data = json.loads(TERMS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "No specialized terminology provided."
    
    context_desc = data.get("context_description", "")
    terms = data.get("terms", {})
    
    if not terms:
        return "No specialized terminology provided."
    
    lines = []
    if context_desc:
        lines.append(f"Context Description: {context_desc}")
    
    lines.append("Terminology Reference:")
    for category, items in terms.items():
        category_display = category.replace("_", " ")
        items_str = ", ".join(items) if isinstance(items, list) else str(items)
        lines.append(f"- {category_display}: {items_str}")
    
    return "\n".join(lines)


def load_json_context() -> str:
    """Load the raw JSON context for embedding in the prompt."""
    if not TERMS_FILE.exists():
        return "{}"
    
    try:
        return TERMS_FILE.read_text(encoding="utf-8")
    except OSError:
        return "{}"


def split_srt_into_chunks(content: str, max_lines: int = 50) -> list:
    """
    Split SRT content into chunks of approximately max_lines subtitle entries.
    Each SRT entry consists of: index, timestamp, text, blank line.
    """
    lines = content.strip().split('\n')
    
    if len(lines) <= max_lines * 4:
        return [content]
    
    chunks = []
    current_chunk = []
    line_count = 0
    
    i = 0
    while i < len(lines):
        if lines[i].strip().isdigit():
            entry_lines = []
            while i < len(lines) and lines[i].strip():
                entry_lines.append(lines[i])
                i += 1
            entry_lines.append('')
            current_chunk.extend(entry_lines)
            line_count += 1
            
            if line_count >= max_lines:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                line_count = 0
        else:
            i += 1
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks if chunks else [content]


def clean_transcript(client: AzureOpenAI, deployment_name: str, content: str, json_context: str) -> str:
    """
    Send transcript to Azure OpenAI for cleaning.
    Processes entire SRT at once, or in chunks of 50 entries for very long files.
    """
    system_prompt = f"""You are an expert Cantonese transcription editor specializing in correcting ASR (Automatic Speech Recognition) transcription errors in noisy, real-world recordings (e.g., construction sites, tool operation, and safety briefings).

## Task
Read the raw SRT transcript below. Identify and correct phonetic errors, homophones, and hallucinations based on the surrounding dialogue and the provided terminology context.

## Reasoning Process
Before making corrections, consider:
1. What is the work situation / topic based on surrounding dialogue?
2. Does this phrase make sense in context, or is it a phonetic mishearing?
3. What Cantonese word/phrase or HK English term sounds similar and fits the context?

## Context
{json_context}

## Rules

1. **Phonetic Priority**: Identify Cantonese homophones. If a phrase sounds like a term in the Context but looks wrong, replace it.

2. **Fix Grammatical Disjoints**: If you see meaningless character combinations, infer the intended phrase from context. Prefer small edits and keep the speaker's intent.

3. **Domain Jargon Matching**: Preserve and correct technical terms common in construction audio (e.g., tools, materials, safety terms, measurements/units, model numbers, and acronyms). If nonsense characters sound like a technical term from the Context, switch it to the correct term.

4. **HK English Code-Switching**: Hong Kong speakers mix English technical terms into Cantonese. If a Cantonese phrase sounds like an English word from the Context, switch it.

5. **Preserve SRT Format**: Keep all timestamp lines exactly as they are. Only modify the subtitle text lines.

6. **Preserve Language**: Do not translate. Keep Cantonese in Traditional Chinese and English in English.

7. **Output Format**: Output valid SRT format ONLY. No explanations, no markdown, no intro/outro."""

    chunks = split_srt_into_chunks(content)
    
    if len(chunks) == 1:
        print(f"Processing entire file ({len(content)} characters)...")
    else:
        print(f"Processing {len(chunks)} chunks...")
    
    cleaned_chunks = []
    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            print(f"  Processing chunk {i + 1}/{len(chunks)}...")
        
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk},
            ],
        )
        cleaned_chunks.append(response.choices[0].message.content)
    
    return '\n\n'.join(cleaned_chunks)


def main():
    if len(sys.argv) != 2:
        print("Usage: python clean_transcript.py <input_file.srt>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    if not input_path.exists():
        print(f"Error: File '{input_path}' not found.")
        sys.exit(1)
    
    # Build output filename: [original_filename]_cleaned.srt
    output_path = input_path.with_stem(f"{input_path.stem}_cleaned")
    
    print(f"Loading configuration...")
    config = load_config()
    
    print(f"Reading '{input_path}'...")
    content = input_path.read_text(encoding="utf-8")
    
    print("Loading JSON context...")
    json_context = load_json_context()
    
    print(f"Sending to Azure OpenAI ({config['deployment_name']})...")
    client = create_client(config)
    cleaned_content = clean_transcript(client, config["deployment_name"], content, json_context)
    
    print(f"Writing cleaned transcript to '{output_path}'...")
    output_path.write_text(cleaned_content, encoding="utf-8")
    
    print("Done!")


if __name__ == "__main__":
    main()
