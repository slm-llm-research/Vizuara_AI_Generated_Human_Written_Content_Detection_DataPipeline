#!/usr/bin/env python3
"""
Script to download the Poetry Foundation Poems dataset from HuggingFace
and construct a human-written poetry corpus using RAID constraints.

Filters poems to 700-1600 characters and outputs a clean CSV.
"""

import re
import uuid
from pathlib import Path
from datasets import load_dataset
import pandas as pd

def clean_text(text):
    """
    Normalize text by:
    - Stripping leading/trailing whitespace
    - Replacing multiple newlines with single newline
    - NOT removing punctuation
    - NOT lowercasing
    """
    if not text:
        return ""
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Replace multiple newlines with single newline
    text = re.sub(r'\n+', '\n', text)
    
    return text

def build_poetry_foundation_human():
    """
    Download and process the Poetry Foundation Poems dataset.
    """
    print("Loading Poetry Foundation Poems dataset from HuggingFace...")
    
    # Load the dataset
    dataset = load_dataset("suayptalha/Poetry-Foundation-Poems")
    
    # Assume train split
    poems_data = dataset['train']
    
    print(f"Total poems loaded: {len(poems_data)}")
    
    # Debug: Print first record to see available fields
    if len(poems_data) > 0:
        print(f"\nAvailable fields in dataset: {list(poems_data[0].keys())}")
        print(f"\nSample record:")
        for key, value in poems_data[0].items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}...")
            else:
                print(f"  {key}: {value}")
    
    # Process poems
    processed_poems = []
    
    for idx, record in enumerate(poems_data):
        # Extract required fields
        poem_text = record.get('poem', record.get('Poem', None))
        title = record.get('title', record.get('Title', None))
        author = record.get('author', record.get('Poet', None))
        tags = record.get('tags', record.get('Tags', None))
        
        # Skip if any required field is missing
        if not poem_text or not title or not author:
            continue
        
        # Clean the poem text
        cleaned_text = clean_text(poem_text)
        
        # Calculate character length
        char_length = len(cleaned_text)
        
        # Apply RAID constraints: 700-1600 characters
        if not (700 <= char_length <= 1600):
            continue
        
        # Generate deterministic ID (using UUID v5 with namespace)
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
        poem_id = str(uuid.uuid5(namespace, cleaned_text))
        
        # Build record matching RAID schema
        poem_record = {
            'id': poem_id,
            'domain': 'poetry',
            'model': None,  # Human-written, no model
            'decoding': None,  # Human-written, no decoding
            'attack': None,  # No adversarial attack
            'text': cleaned_text,
            'title': title,
            'author': author,
            'tags': tags if tags else "",
            'label_generated': 'Human_Written',
            'source': 'PoetryFoundation',
            'char_length': char_length
        }
        
        processed_poems.append(poem_record)
    
    # Convert to DataFrame (preserves order, no shuffling)
    df = pd.DataFrame(processed_poems)
    
    # Calculate statistics
    total_loaded = len(poems_data)
    total_retained = len(df)
    
    # Print sanity checks
    print(f"\nTotal poems loaded: {total_loaded}")
    print(f"Poems after filtering (700–1600 chars): {total_retained}")
    
    # Check if we have any data
    if total_retained == 0:
        print("\n⚠️  WARNING: No poems passed the filtering criteria!")
        print("Check that the dataset has poems between 700-1600 characters.")
        return None
    
    # Calculate length statistics
    min_length = df['char_length'].min()
    max_length = df['char_length'].max()
    mean_length = df['char_length'].mean()
    print(f"Char length stats: min={min_length}, max={max_length}, mean={mean_length:.1f}")
    
    # Create output directory
    output_dir = Path("/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/poetry_foundation_human_700_1600")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    output_path = output_dir / "poetry_foundation_human_700_1600.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Dataset saved to: {output_path}")
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df

if __name__ == "__main__":
    build_poetry_foundation_human()

