"""
Generate training sentences for Cantonese ASR fine-tuning using Azure OpenAI.

This script reads construction domain terms from normalized_terms_construction.json
and generates naturalistic Cantonese sentences containing those terms.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import AzureOpenAI


TERMS_FILE = Path(__file__).parent / "normalized_terms_construction.json"
OUTPUT_FILE = Path(__file__).parent / "generated_sentences.json"

SYSTEM_PROMPT = """You are a Linguistic Data Specialist for a Hong Kong Construction ASR project. Your task is to generate naturalistic Cantonese training sentences containing specific domain jargon.

**Objective:**
Create a dataset of sentences where specific "Target Terms" are used in realistic, grammatically correct contexts that reflect the term's actual meaning.

**Guidelines for Sentence Generation:**

1. **Use the Definition:** Each term comes with a definition. Your sentence MUST reflect how the term is actually used based on its meaning. The sentence should make sense only if the term means what the definition says.
   - *Example:* If "天秤" means "建築工地塔式起重機" (tower crane), the sentence should involve lifting, moving materials, crane operation, etc.
   - *Bad:* Generic sentence that could apply to anything
   - *Good:* Sentence that only makes sense for a tower crane

2. **Language Style:** Use "Hong Kong Site Cantonese" (Code-mixed/Chinglish).
   - *Bad:* "請檢查混凝土的質量" (Too formal/written)
   - *Bad:* "The Slab is a concrete floor." (Too explanatory/dictionary-like)
   - *Good:* "Check 下個 Concrete 乾咗未" (Natural spoken)
   - *Good:* "個天秤吊緊嘢，你哋退後啲。" (Natural usage reflecting crane function)

3. **Sentence Length:** Keep sentences between 5-15 words to match typical site communication.

4. **Speaker Context:** Sentences should sound like they come from: foremen, workers, engineers, or safety officers on a construction site.

5. **Context Variety:** For each term, generate sentences in varied modes:
   - *Imperative:* Giving an order (e.g., "Move the [Term]...")
   - *Interrogative:* Asking a question (e.g., "Where is the [Term]?")
   - *Declarative:* Stating a fact (e.g., "The [Term] is broken.")

6. **Code-Switching Rule:**
   - If the term is English (e.g., "Excavator"), keep it English in the sentence.
   - If the term is Chinese (e.g., "狗臂架"), keep it Chinese.
   - Do *not* translate the target term itself.

**Input Data:**
I will provide a JSON list of terms with "definition" and "priority" fields.
- Use the definition to understand what the term means and generate contextually accurate sentences.
- If priority is "High", generate **3 unique sentences** (one of each type: Imperative, Interrogative, Declarative).
- If priority is "Low", generate **1 unique sentence** (vary the type across terms).

**Output Format:**
Strict JSON array format only. No markdown, no explanation, just the JSON:
[
  {
    "term": "Target Term",
    "sentence": "The generated sentence containing the term.",
    "type": "Imperative/Interrogative/Declarative"
  }
]

**Example Input:**
[{"term": "天秤", "definition": "建築工地塔式起重機", "priority": "High"}]

**Example Output:**
[
  {"term": "天秤", "sentence": "叫天秤師傅吊批鐵上十二樓。", "type": "Imperative"},
  {"term": "天秤", "sentence": "個天秤今日有冇 Book 咗時間吊嘢？", "type": "Interrogative"},
  {"term": "天秤", "sentence": "天秤轉緊個陣唔好企喺下面。", "type": "Declarative"}
]"""


def load_config() -> dict:
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


def load_terms(exclude_hidden: bool = True) -> list[dict]:
    """
    Load terms from normalized_terms_construction.json.
    
    Returns a list of dicts with 'term', 'definition', 'category', and 'priority'.
    """
    if not TERMS_FILE.exists():
        raise FileNotFoundError(f"Terms file not found: {TERMS_FILE}")
    
    data = json.loads(TERMS_FILE.read_text(encoding="utf-8"))
    
    terms = []
    for category, category_terms in data.items():
        if exclude_hidden and category.startswith("_"):
            continue
        
        if not isinstance(category_terms, dict):
            continue
        
        for term, definition in category_terms.items():
            terms.append({
                "term": term,
                "definition": definition,
                "category": category,
                "priority": "Low",  # Default to Low priority
            })
    
    return terms


def batch_terms(terms: list[dict], batch_size: int = 20) -> list[list[dict]]:
    """Split terms into batches for API calls."""
    return [terms[i:i + batch_size] for i in range(0, len(terms), batch_size)]


def generate_sentences_for_batch(
    client: AzureOpenAI,
    deployment_name: str,
    terms_batch: list[dict],
    max_retries: int = 3,
) -> list[dict]:
    """
    Generate sentences for a batch of terms using Azure OpenAI.
    
    Returns a list of generated sentence objects.
    """
    # Prepare input for the LLM
    input_data = [{"term": t["term"], "definition": t["definition"], "priority": t["priority"]} for t in terms_batch]
    user_content = json.dumps(input_data, ensure_ascii=False)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.7,  # Some creativity for varied sentences
                max_completion_tokens=4000,
            )
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from API")
            
            # Clean up response - remove markdown code blocks if present
            content = content.strip()
            if content.startswith("```"):
                # Remove markdown code block wrapper
                lines = content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)
            
            # Parse JSON response
            sentences = json.loads(content)
            return sentences
            
        except json.JSONDecodeError as e:
            print(f"  Warning: JSON parse error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            continue
        except Exception as e:
            print(f"  Warning: API error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            continue
    
    print(f"  Failed to generate sentences for batch after {max_retries} attempts")
    return []


def generate_all_sentences(
    client: AzureOpenAI,
    deployment_name: str,
    terms: list[dict],
    batch_size: int = 20,
    output_file: Optional[Path] = None,
) -> list[dict]:
    """
    Generate sentences for all terms in batches.
    
    Saves progress incrementally to output_file if provided.
    """
    batches = batch_terms(terms, batch_size)
    all_sentences = []
    
    print(f"Generating sentences for {len(terms)} terms in {len(batches)} batches...")
    
    for i, batch in enumerate(batches):
        print(f"Processing batch {i + 1}/{len(batches)} ({len(batch)} terms)...")
        
        sentences = generate_sentences_for_batch(client, deployment_name, batch)
        all_sentences.extend(sentences)
        
        print(f"  Generated {len(sentences)} sentences")
        
        # Save progress incrementally
        if output_file:
            output_file.write_text(
                json.dumps(all_sentences, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        
        # Rate limiting - avoid hitting API limits
        if i < len(batches) - 1:
            time.sleep(1)
    
    return all_sentences


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate training sentences for Cantonese ASR fine-tuning"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of terms to process per API call (default: 20)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help=f"Output file path (default: {OUTPUT_FILE})"
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include terms from _hidden category"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load terms and show stats without generating sentences"
    )
    
    args = parser.parse_args()
    
    print("Loading terms...")
    terms = load_terms(exclude_hidden=not args.include_hidden)
    print(f"Loaded {len(terms)} terms")
    
    # Show category breakdown
    categories = {}
    for t in terms:
        cat = t["category"]
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nTerms by category:")
    for cat, count in categories.items():
        print(f"  - {cat}: {count} terms")
    
    if args.dry_run:
        print("\nDry run complete. No sentences generated.")
        return
    
    print("\nLoading Azure OpenAI configuration...")
    config = load_config()
    client = create_client(config)
    
    print(f"Using deployment: {config['deployment_name']}")
    print(f"Output file: {args.output}")
    
    sentences = generate_all_sentences(
        client,
        config["deployment_name"],
        terms,
        batch_size=args.batch_size,
        output_file=args.output,
    )
    
    print(f"\nGeneration complete!")
    print(f"Total sentences generated: {len(sentences)}")
    print(f"Output saved to: {args.output}")
    
    # Show type breakdown
    types = {}
    for s in sentences:
        t = s.get("type", "Unknown")
        types[t] = types.get(t, 0) + 1
    
    print("\nSentences by type:")
    for t, count in types.items():
        print(f"  - {t}: {count}")


if __name__ == "__main__":
    main()
