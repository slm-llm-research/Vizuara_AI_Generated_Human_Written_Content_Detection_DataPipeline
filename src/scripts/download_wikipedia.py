#!/usr/bin/env python3
"""
Memory-efficient script to download and sample wikimedia/wikipedia dataset 
based on RAID wiki dataset characteristics.

This script:
1. Downloads Wikipedia (English, streaming mode)
2. Combines title + text into single column
3. Analyzes RAID wiki dataset (title + generation)
4. Samples 100K articles based on RAID characteristics
5. Ensures NO overlap with RAID human Wikipedia articles
6. Saves sampled dataset
"""

import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from pathlib import Path
import argparse
from typing import Dict, Set
import warnings
import hashlib
warnings.filterwarnings('ignore')


def analyze_raid_characteristics(raid_csv_path: str, sample_size: int = 50000) -> Dict:
    """
    Analyze RAID wiki dataset to extract text characteristics.
    
    Args:
        raid_csv_path: Path to RAID wiki.csv file
        sample_size: Number of samples to analyze
    
    Returns:
        Dictionary with text characteristics
    """
    print("=" * 80)
    print("STEP 1: Analyzing RAID Wiki Dataset Characteristics")
    print("=" * 80)
    
    print(f"\nReading sample of {sample_size:,} rows from RAID wiki dataset...")
    raid_df = pd.read_csv(raid_csv_path, nrows=sample_size)
    
    print(f"✓ Loaded {len(raid_df):,} samples from RAID dataset")
    
    # Combine title and generation
    print("\nCombining 'title' + 'generation' columns...")
    raid_df['text'] = raid_df['title'].fillna('') + ' ' + raid_df['generation'].fillna('')
    raid_df['text_length'] = raid_df['text'].str.len()
    raid_df['word_count'] = raid_df['text'].str.split().str.len()
    
    # Remove zero-length texts
    raid_df = raid_df[raid_df['text_length'] > 0]
    
    # Calculate statistics
    stats = {
        'char_length': {
            'min': raid_df['text_length'].min(),
            'max': raid_df['text_length'].max(),
            'mean': raid_df['text_length'].mean(),
            'median': raid_df['text_length'].median(),
            'q25': raid_df['text_length'].quantile(0.25),
            'q75': raid_df['text_length'].quantile(0.75),
            'std': raid_df['text_length'].std()
        },
        'word_count': {
            'min': raid_df['word_count'].min(),
            'max': raid_df['word_count'].max(),
            'mean': raid_df['word_count'].mean(),
            'median': raid_df['word_count'].median(),
            'q25': raid_df['word_count'].quantile(0.25),
            'q75': raid_df['word_count'].quantile(0.75),
            'std': raid_df['word_count'].std()
        }
    }
    
    print("\n" + "=" * 80)
    print("RAID Wiki Dataset Text Characteristics (title + generation):")
    print("=" * 80)
    print(f"\n📊 CHARACTER LENGTH STATISTICS:")
    print(f"   Min:    {stats['char_length']['min']:>8.0f} characters")
    print(f"   Q25:    {stats['char_length']['q25']:>8.0f} characters")
    print(f"   Median: {stats['char_length']['median']:>8.0f} characters")
    print(f"   Mean:   {stats['char_length']['mean']:>8.0f} characters")
    print(f"   Q75:    {stats['char_length']['q75']:>8.0f} characters")
    print(f"   Max:    {stats['char_length']['max']:>8.0f} characters")
    print(f"   Std:    {stats['char_length']['std']:>8.0f} characters")
    
    print(f"\n📊 WORD COUNT STATISTICS:")
    print(f"   Min:    {stats['word_count']['min']:>8.0f} words")
    print(f"   Q25:    {stats['word_count']['q25']:>8.0f} words")
    print(f"   Median: {stats['word_count']['median']:>8.0f} words")
    print(f"   Mean:   {stats['word_count']['mean']:>8.0f} words")
    print(f"   Q75:    {stats['word_count']['q75']:>8.0f} words")
    print(f"   Max:    {stats['word_count']['max']:>8.0f} words")
    print(f"   Std:    {stats['word_count']['std']:>8.0f} words")
    
    return stats


def get_raid_human_titles(raid_csv_path: str) -> Set[str]:
    """
    Extract titles of human Wikipedia articles from RAID dataset to avoid overlap.
    
    Args:
        raid_csv_path: Path to RAID wiki.csv file
    
    Returns:
        Set of human Wikipedia article titles (normalized)
    """
    print("\n" + "=" * 80)
    print("STEP 2: Extracting RAID Human Wikipedia Titles (for overlap prevention)")
    print("=" * 80)
    
    print(f"\nReading RAID wiki dataset to find human articles...")
    
    # Read in chunks to handle large file
    human_titles = set()
    chunk_size = 50000
    chunk_num = 0
    
    for chunk in pd.read_csv(raid_csv_path, chunksize=chunk_size):
        chunk_num += 1
        # Filter for human-written articles
        if 'domain' in chunk.columns:
            human_chunk = chunk[chunk['domain'] == 'human']
        elif 'model' in chunk.columns:
            # If no domain column, look for model column (human might be indicated differently)
            human_chunk = chunk[chunk['model'].isna() | (chunk['model'] == 'human')]
        else:
            print(f"⚠️  Warning: Cannot identify human articles (no domain/model column)")
            human_chunk = pd.DataFrame()
        
        if len(human_chunk) > 0:
            # Extract titles and normalize (lowercase, strip whitespace)
            titles = human_chunk['title'].fillna('').astype(str).str.lower().str.strip()
            human_titles.update(titles.tolist())
        
        if chunk_num % 10 == 0:
            print(f"   Processed {chunk_num * chunk_size:,} rows | Human titles found: {len(human_titles):,}")
    
    print(f"\n✓ Total RAID human Wikipedia titles to exclude: {len(human_titles):,}")
    
    return human_titles


def download_and_sample_wikipedia_streaming(raid_stats: Dict, excluded_titles: Set[str],
                                            target_samples: int, chunk_size: int = 5000):
    """
    Download and sample Wikipedia using STREAMING to avoid OOM.
    Ensures no overlap with RAID human articles.
    
    Args:
        raid_stats: Statistics from RAID dataset
        excluded_titles: Set of RAID human article titles to exclude
        target_samples: Number of samples to collect
        chunk_size: Size of chunks to process
    
    Returns:
        Sampled DataFrame
    """
    print("\n" + "=" * 80)
    print("STEP 3: Downloading & Sampling Wikipedia (STREAMING MODE)")
    print("=" * 80)
    
    print("\nSource: https://huggingface.co/datasets/wikimedia/wikipedia")
    print("Version: 20231101.en (English Wikipedia, November 2023)")
    print("⚠️  This is a LARGE dataset - using STREAMING mode")
    
    # Calculate filtering criteria
    char_min = max(0, raid_stats['char_length']['mean'] - 2 * raid_stats['char_length']['std'])
    char_max = raid_stats['char_length']['mean'] + 2 * raid_stats['char_length']['std']
    word_min = max(0, raid_stats['word_count']['mean'] - 2 * raid_stats['word_count']['std'])
    word_max = raid_stats['word_count']['mean'] + 2 * raid_stats['word_count']['std']
    
    print(f"\n📏 Sampling Criteria (based on RAID mean ± 2σ):")
    print(f"   Character length: {char_min:.0f} - {char_max:.0f} characters")
    print(f"   Word count:       {word_min:.0f} - {word_max:.0f} words")
    print(f"   Target samples:   {target_samples:,}")
    print(f"   Excluded titles:  {len(excluded_titles):,}")
    print(f"   Chunk size:       {chunk_size:,}")
    
    # Load dataset in streaming mode
    print("\nLoading Wikipedia in streaming mode...")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    
    # Examine first example
    first_example = next(iter(dataset))
    print(f"\n📋 Dataset fields: {list(first_example.keys())}")
    
    # Identify text fields (should be 'title' and 'text')
    if 'title' in first_example and 'text' in first_example:
        title_field, text_field = 'title', 'text'
        print(f"✓ Using 'title' + 'text' fields")
    else:
        raise ValueError(f"Expected 'title' and 'text' fields, found: {list(first_example.keys())}")
    
    # Show sample
    sample_title = first_example[title_field]
    sample_text = first_example[text_field]
    combined = sample_title + ' ' + sample_text
    print(f"\n📝 Sample Wikipedia article:")
    print(f"   Title: {sample_title[:100]}...")
    print(f"   Text length: {len(sample_text)} chars")
    print(f"   Combined: {len(combined)} chars, {len(combined.split())} words")
    
    # Process in chunks using streaming
    print(f"\n🔄 Processing Wikipedia in streaming mode...")
    
    sampled_data = []
    total_processed = 0
    total_matched = 0
    total_excluded = 0
    chunk_buffer = []
    
    for example in dataset:
        chunk_buffer.append(example)
        
        # Process when chunk is full
        if len(chunk_buffer) >= chunk_size:
            # Create DataFrame from chunk
            chunk_df = pd.DataFrame(chunk_buffer)
            
            # Combine title and text
            chunk_df['combined_text'] = (
                chunk_df[title_field].fillna('').astype(str) + ' ' + 
                chunk_df[text_field].fillna('').astype(str)
            )
            chunk_df['text_length'] = chunk_df['combined_text'].str.len()
            chunk_df['word_count'] = chunk_df['combined_text'].str.split().str.len()
            
            # Filter based on RAID criteria
            filtered_chunk = chunk_df[
                (chunk_df['text_length'] >= char_min) &
                (chunk_df['text_length'] <= char_max) &
                (chunk_df['word_count'] >= word_min) &
                (chunk_df['word_count'] <= word_max)
            ]
            
            # Exclude RAID human articles (check by normalized title)
            if len(excluded_titles) > 0:
                filtered_chunk['title_normalized'] = filtered_chunk[title_field].str.lower().str.strip()
                before_exclusion = len(filtered_chunk)
                filtered_chunk = filtered_chunk[~filtered_chunk['title_normalized'].isin(excluded_titles)]
                excluded_count = before_exclusion - len(filtered_chunk)
                total_excluded += excluded_count
                filtered_chunk = filtered_chunk.drop('title_normalized', axis=1)
            
            total_processed += len(chunk_buffer)
            total_matched += len(filtered_chunk)
            
            # Add to sampled data
            if len(filtered_chunk) > 0:
                sampled_data.append(filtered_chunk)
            
            # Progress update
            if total_processed % 50000 == 0:
                current_total = sum(len(df) for df in sampled_data)
                print(f"   Processed: {total_processed:,} | Matched: {total_matched:,} "
                      f"({total_matched/total_processed*100:.2f}%) | Excluded: {total_excluded:,} | Collected: {current_total:,}")
            
            # Check if we have enough samples
            current_total = sum(len(df) for df in sampled_data)
            if current_total >= target_samples:
                print(f"\n✓ Collected {current_total:,} samples (target: {target_samples:,})")
                break
            
            # Clear chunk buffer
            chunk_buffer = []
    
    # Process remaining buffer
    if chunk_buffer and sum(len(df) for df in sampled_data) < target_samples:
        chunk_df = pd.DataFrame(chunk_buffer)
        chunk_df['combined_text'] = (
            chunk_df[title_field].fillna('').astype(str) + ' ' + 
            chunk_df[text_field].fillna('').astype(str)
        )
        chunk_df['text_length'] = chunk_df['combined_text'].str.len()
        chunk_df['word_count'] = chunk_df['combined_text'].str.split().str.len()
        
        filtered_chunk = chunk_df[
            (chunk_df['text_length'] >= char_min) &
            (chunk_df['text_length'] <= char_max) &
            (chunk_df['word_count'] >= word_min) &
            (chunk_df['word_count'] <= word_max)
        ]
        
        if len(excluded_titles) > 0:
            chunk_df['title_normalized'] = chunk_df[title_field].str.lower().str.strip()
            before_exclusion = len(filtered_chunk)
            filtered_chunk = filtered_chunk[~filtered_chunk['title_normalized'].isin(excluded_titles)]
            total_excluded += before_exclusion - len(filtered_chunk)
            filtered_chunk = filtered_chunk.drop('title_normalized', axis=1)
        
        total_processed += len(chunk_buffer)
        total_matched += len(filtered_chunk)
        
        if len(filtered_chunk) > 0:
            sampled_data.append(filtered_chunk)
    
    # Summary
    print(f"\n📊 Processing Summary:")
    print(f"   Total processed:     {total_processed:,}")
    print(f"   Total matched:       {total_matched:,}")
    print(f"   Total excluded:      {total_excluded:,} (RAID human duplicates)")
    print(f"   Match rate:          {total_matched/total_processed*100:.2f}%")
    print(f"   Exclusion rate:      {total_excluded/total_matched*100:.2f}% of matches")
    
    if not sampled_data:
        print("\n❌ No samples matched the RAID criteria!")
        return None
    
    # Combine all sampled data
    print(f"\nCombining {len(sampled_data)} chunks...")
    final_df = pd.concat(sampled_data, ignore_index=True)
    
    # Limit to target samples if we have more
    if len(final_df) > target_samples:
        print(f"Sampling {target_samples:,} from {len(final_df):,} matches...")
        final_df = final_df.sample(n=target_samples, random_state=42).reset_index(drop=True)
    
    # Rename combined_text to text for consistency
    final_df['text'] = final_df['combined_text']
    
    # Add metadata
    final_df['domain'] = 'wikipedia'
    final_df['source'] = 'wikimedia/wikipedia'
    final_df['raid_aligned'] = True
    
    print(f"\n📊 Final Sampled Dataset Statistics:")
    print(f"   Total samples: {len(final_df):,}")
    print(f"   Char length - Min: {final_df['text_length'].min()}, "
          f"Mean: {final_df['text_length'].mean():.0f}, "
          f"Max: {final_df['text_length'].max()}")
    print(f"   Word count  - Min: {final_df['word_count'].min()}, "
          f"Mean: {final_df['word_count'].mean():.0f}, "
          f"Max: {final_df['word_count'].max()}")
    
    return final_df


def save_dataset(df: pd.DataFrame, output_dir: str):
    """
    Save dataset in multiple formats.
    
    Args:
        df: DataFrame to save
        output_dir: Output directory path
    """
    print("\n" + "=" * 80)
    print("STEP 4: Saving Dataset")
    print("=" * 80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    csv_path = output_path / 'wikipedia_sampled.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved CSV: {csv_path}")
    print(f"  Size: {csv_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Save as JSON
    json_path = output_path / 'wikipedia_sampled.json'
    df.to_json(json_path, orient='records', lines=True)
    print(f"✓ Saved JSON: {json_path}")
    print(f"  Size: {json_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Create summary file
    summary_path = output_path / 'SAMPLING_SUMMARY.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("WIKIPEDIA DATASET - RAID ALIGNED SAMPLING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Source: wikimedia/wikipedia (20231101.en)\n")
        f.write(f"Total Samples: {len(df):,}\n")
        f.write(f"Character Length: {df['text_length'].min()} - {df['text_length'].max()} "
                f"(mean: {df['text_length'].mean():.0f})\n")
        f.write(f"Word Count: {df['word_count'].min()} - {df['word_count'].max()} "
                f"(mean: {df['word_count'].mean():.0f})\n\n")
        f.write("Files:\n")
        f.write("  - wikipedia_sampled.csv\n")
        f.write("  - wikipedia_sampled.json\n\n")
        f.write("Sampling Strategy:\n")
        f.write("  Combined: title + text\n")
        f.write("  Filter: RAID mean ± 2σ\n")
        f.write("  Overlap prevention: Excluded RAID human articles\n")
        f.write("  Random seed: 42\n")
        f.write("  Processing: Streaming mode (memory-efficient)\n")
    
    print(f"✓ Created summary: {summary_path}")
    
    print(f"\n✅ Dataset saved successfully to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Download and sample Wikipedia based on RAID characteristics (streaming mode)'
    )
    parser.add_argument(
        '--raid-csv',
        type=str,
        default='/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/raid/wiki.csv',
        help='Path to RAID wiki.csv file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/wiki',
        help='Output directory for sampled dataset'
    )
    parser.add_argument(
        '--raid-sample-size',
        type=int,
        default=50000,
        help='Number of RAID samples to analyze (default: 50000)'
    )
    parser.add_argument(
        '--target-samples',
        type=int,
        default=100000,
        help='Target number of Wikipedia samples to extract (default: 100000)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=5000,
        help='Chunk size for streaming processing (default: 5000)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("WIKIPEDIA DATASET DOWNLOAD AND SAMPLING SCRIPT")
    print("(Memory-Efficient Streaming Mode with Overlap Prevention)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  RAID CSV:        {args.raid_csv}")
    print(f"  Output Dir:      {args.output_dir}")
    print(f"  RAID samples:    {args.raid_sample_size:,}")
    print(f"  Target samples:  {args.target_samples:,}")
    print(f"  Chunk size:      {args.chunk_size:,}")
    
    try:
        # Step 1: Analyze RAID characteristics
        raid_stats = analyze_raid_characteristics(args.raid_csv, args.raid_sample_size)
        
        # Step 2: Get RAID human titles to exclude
        excluded_titles = get_raid_human_titles(args.raid_csv)
        
        # Step 3: Download and sample Wikipedia (streaming, with exclusion)
        sampled_df = download_and_sample_wikipedia_streaming(
            raid_stats,
            excluded_titles,
            args.target_samples,
            args.chunk_size
        )
        
        if sampled_df is not None:
            # Step 4: Save dataset
            save_dataset(sampled_df, args.output_dir)
            
            print("\n" + "=" * 80)
            print("✅ SCRIPT COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"\nDataset location: {args.output_dir}")
            print(f"Total samples: {len(sampled_df):,}")
            print(f"RAID human articles excluded: {len(excluded_titles):,}")
            print("\nFiles created:")
            print(f"  - wikipedia_sampled.csv")
            print(f"  - wikipedia_sampled.json")
            print(f"  - SAMPLING_SUMMARY.txt")
        else:
            print("\n⚠️  Script completed but no matching samples were found.")
            return 1
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

