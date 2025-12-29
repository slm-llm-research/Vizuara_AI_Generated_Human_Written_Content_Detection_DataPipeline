#!/usr/bin/env python3
"""
Memory-efficient script to download and sample sentence-transformers/reddit-title-body 
dataset based on RAID reddit dataset characteristics.

This script uses STREAMING mode to handle the massive Reddit dataset (60M+ examples)
without running out of memory.
"""

import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from pathlib import Path
import argparse
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


def analyze_raid_characteristics(raid_csv_path: str, sample_size: int = 50000) -> Dict:
    """
    Analyze RAID reddit dataset to extract text characteristics.
    
    Args:
        raid_csv_path: Path to RAID reddit.csv file
        sample_size: Number of samples to analyze
    
    Returns:
        Dictionary with text characteristics
    """
    print("=" * 80)
    print("STEP 1: Analyzing RAID Reddit Dataset Characteristics")
    print("=" * 80)
    
    print(f"\nReading sample of {sample_size:,} rows from RAID reddit dataset...")
    raid_df = pd.read_csv(raid_csv_path, nrows=sample_size)
    
    print(f"✓ Loaded {len(raid_df):,} samples from RAID dataset")
    
    # Combine title and generation
    print("\nCombining 'title' + 'generation' columns...")
    raid_df['combined_text'] = raid_df['title'].fillna('') + ' ' + raid_df['generation'].fillna('')
    raid_df['text_length'] = raid_df['combined_text'].str.len()
    raid_df['word_count'] = raid_df['combined_text'].str.split().str.len()
    
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
    print("RAID Reddit Dataset Text Characteristics (title + generation):")
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


def download_and_sample_reddit_streaming(raid_stats: Dict, target_samples: int, 
                                         output_dir: str, chunk_size: int = 10000):
    """
    Download and sample Reddit dataset using STREAMING to avoid OOM.
    
    Args:
        raid_stats: Statistics from RAID dataset
        target_samples: Number of samples to collect
        output_dir: Output directory path
        chunk_size: Size of chunks to process at a time
    """
    print("\n" + "=" * 80)
    print("STEP 2: Downloading & Sampling Reddit Dataset (STREAMING MODE)")
    print("=" * 80)
    
    print("\nSource: https://huggingface.co/datasets/sentence-transformers/reddit-title-body")
    print("⚠️  This is a MASSIVE dataset (60M+ examples) - using STREAMING mode")
    
    # Calculate filtering criteria
    char_min = max(0, raid_stats['char_length']['mean'] - 2 * raid_stats['char_length']['std'])
    char_max = raid_stats['char_length']['mean'] + 2 * raid_stats['char_length']['std']
    word_min = max(0, raid_stats['word_count']['mean'] - 2 * raid_stats['word_count']['std'])
    word_max = raid_stats['word_count']['mean'] + 2 * raid_stats['word_count']['std']
    
    print(f"\n📏 Sampling Criteria (based on RAID mean ± 2σ):")
    print(f"   Character length: {char_min:.0f} - {char_max:.0f} characters")
    print(f"   Word count:       {word_min:.0f} - {word_max:.0f} words")
    print(f"   Target samples:   {target_samples:,}")
    print(f"   Chunk size:       {chunk_size:,}")
    
    # Load dataset in streaming mode
    print("\nLoading dataset in STREAMING mode...")
    dataset = load_dataset("sentence-transformers/reddit-title-body", split="train", streaming=True)
    
    # Examine first example to understand structure
    first_example = next(iter(dataset))
    print(f"\n📋 Dataset fields: {list(first_example.keys())}")
    
    # Identify text fields
    if 'title' in first_example and 'body' in first_example:
        title_field, body_field = 'title', 'body'
        print(f"✓ Using 'title' + 'body' fields")
    elif 'title' in first_example and 'selftext' in first_example:
        title_field, body_field = 'title', 'selftext'
        print(f"✓ Using 'title' + 'selftext' fields")
    else:
        # Find appropriate fields
        title_field = [k for k in first_example.keys() if 'title' in k.lower()][0]
        body_field = [k for k in first_example.keys() if 'body' in k.lower() or 'text' in k.lower()][0]
        print(f"✓ Using '{title_field}' + '{body_field}' fields")
    
    # Show sample
    sample_text = str(first_example.get(title_field, '')) + ' ' + str(first_example.get(body_field, ''))
    print(f"\n📝 Sample Reddit post (first 250 chars):")
    print(f"   Title: {first_example.get(title_field, '')[:100]}...")
    print(f"   Body: {first_example.get(body_field, '')[:100]}...")
    print(f"   Combined length: {len(sample_text)} chars, {len(sample_text.split())} words")
    
    # Process in chunks using streaming
    print(f"\n🔄 Processing dataset in streaming mode (chunk size: {chunk_size:,})...")
    
    sampled_data = []
    total_processed = 0
    total_matched = 0
    chunk_buffer = []
    
    for example in dataset:
        chunk_buffer.append(example)
        
        # Process when chunk is full
        if len(chunk_buffer) >= chunk_size:
            # Create DataFrame from chunk
            chunk_df = pd.DataFrame(chunk_buffer)
            chunk_df['combined_text'] = (
                chunk_df[title_field].fillna('').astype(str) + ' ' + 
                chunk_df[body_field].fillna('').astype(str)
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
            
            total_processed += len(chunk_buffer)
            total_matched += len(filtered_chunk)
            
            # Add to sampled data
            if len(filtered_chunk) > 0:
                sampled_data.append(filtered_chunk)
            
            # Progress update every 100K
            if total_processed % 100000 == 0:
                current_total = sum(len(df) for df in sampled_data)
                print(f"   Processed: {total_processed:,} | Matched: {total_matched:,} "
                      f"({total_matched/total_processed*100:.2f}%) | Collected: {current_total:,}")
            
            # Check if we have enough samples
            current_total = sum(len(df) for df in sampled_data)
            if current_total >= target_samples:
                print(f"\n✓ Collected {current_total:,} samples (target: {target_samples:,})")
                break
            
            # Clear chunk buffer to free memory
            chunk_buffer = []
    
    # Process remaining buffer if needed
    if chunk_buffer and sum(len(df) for df in sampled_data) < target_samples:
        chunk_df = pd.DataFrame(chunk_buffer)
        chunk_df['combined_text'] = (
            chunk_df[title_field].fillna('').astype(str) + ' ' + 
            chunk_df[body_field].fillna('').astype(str)
        )
        chunk_df['text_length'] = chunk_df['combined_text'].str.len()
        chunk_df['word_count'] = chunk_df['combined_text'].str.split().str.len()
        
        filtered_chunk = chunk_df[
            (chunk_df['text_length'] >= char_min) &
            (chunk_df['text_length'] <= char_max) &
            (chunk_df['word_count'] >= word_min) &
            (chunk_df['word_count'] <= word_max)
        ]
        
        total_processed += len(chunk_buffer)
        total_matched += len(filtered_chunk)
        
        if len(filtered_chunk) > 0:
            sampled_data.append(filtered_chunk)
    
    # Summary
    print(f"\n📊 Processing Summary:")
    print(f"   Total processed: {total_processed:,}")
    print(f"   Total matched:   {total_matched:,}")
    print(f"   Match rate:      {total_matched/total_processed*100:.2f}%")
    
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
    
    # Add metadata
    final_df['domain'] = 'reddit'
    final_df['source'] = 'sentence-transformers/reddit-title-body'
    final_df['raid_aligned'] = True
    
    # Rename combined_text to text for consistency
    final_df['text'] = final_df['combined_text']
    
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
    Save dataset in multiple formats (CSV, JSON).
    
    Args:
        df: DataFrame to save
        output_dir: Output directory path
    """
    print("\n" + "=" * 80)
    print("STEP 3: Saving Dataset")
    print("=" * 80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    csv_path = output_path / 'reddit_sampled.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved CSV: {csv_path}")
    print(f"  Size: {csv_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Save as JSON
    json_path = output_path / 'reddit_sampled.json'
    df.to_json(json_path, orient='records', lines=True)
    print(f"✓ Saved JSON: {json_path}")
    print(f"  Size: {json_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Create summary file
    summary_path = output_path / 'SAMPLING_SUMMARY.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("REDDIT DATASET - RAID ALIGNED SAMPLING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Source: sentence-transformers/reddit-title-body\n")
        f.write(f"Total Samples: {len(df):,}\n")
        f.write(f"Character Length: {df['text_length'].min()} - {df['text_length'].max()} "
                f"(mean: {df['text_length'].mean():.0f})\n")
        f.write(f"Word Count: {df['word_count'].min()} - {df['word_count'].max()} "
                f"(mean: {df['word_count'].mean():.0f})\n\n")
        f.write("Files:\n")
        f.write("  - reddit_sampled.csv\n")
        f.write("  - reddit_sampled.json\n\n")
        f.write("Sampling Strategy:\n")
        f.write("  Combined: title + body\n")
        f.write("  Filter: RAID mean ± 2σ\n")
        f.write("  Random seed: 42\n")
        f.write("  Processing: Streaming mode (memory-efficient)\n")
    
    print(f"✓ Created summary: {summary_path}")
    
    print(f"\n✅ Dataset saved successfully to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Download and sample Reddit dataset based on RAID characteristics (streaming mode)'
    )
    parser.add_argument(
        '--raid-csv',
        type=str,
        default='/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/raid/reddit.csv',
        help='Path to RAID reddit.csv file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/reddit',
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
        help='Target number of Reddit samples to extract (default: 100000)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=10000,
        help='Chunk size for streaming processing (default: 10000)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("REDDIT DATASET DOWNLOAD AND SAMPLING SCRIPT")
    print("(Memory-Efficient Streaming Mode)")
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
        
        # Step 2: Download and sample Reddit dataset (streaming)
        sampled_df = download_and_sample_reddit_streaming(
            raid_stats, 
            args.target_samples, 
            args.output_dir,
            args.chunk_size
        )
        
        if sampled_df is not None:
            # Step 3: Save dataset
            save_dataset(sampled_df, args.output_dir)
            
            print("\n" + "=" * 80)
            print("✅ SCRIPT COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"\nDataset location: {args.output_dir}")
            print(f"Total samples: {len(sampled_df):,}")
            print("\nFiles created:")
            print(f"  - reddit_sampled.csv")
            print(f"  - reddit_sampled.json")
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

