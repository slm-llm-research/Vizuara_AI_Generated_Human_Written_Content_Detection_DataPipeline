#!/usr/bin/env python3
"""
Memory-efficient script to download and sample dell-research-harvard/newswire dataset 
based on RAID news dataset characteristics.

This script uses chunked processing to handle the large newswire dataset (2.7M+ examples)
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
    Analyze RAID news dataset to extract text characteristics.
    
    Args:
        raid_csv_path: Path to RAID news.csv file
        sample_size: Number of samples to analyze (due to large file size)
    
    Returns:
        Dictionary with text characteristics
    """
    print("=" * 80)
    print("STEP 1: Analyzing RAID News Dataset Characteristics")
    print("=" * 80)
    
    # Read a sample of the RAID dataset (it's very large)
    print(f"\nReading sample of {sample_size:,} rows from RAID news dataset...")
    raid_df = pd.read_csv(raid_csv_path, nrows=sample_size)
    
    print(f"✓ Loaded {len(raid_df):,} samples from RAID dataset")
    
    # Combine title and generation
    print("\nCombining 'title' and 'generation' columns...")
    raid_df['combined_text'] = raid_df['title'].fillna('') + ' ' + raid_df['generation'].fillna('')
    raid_df['text_length'] = raid_df['combined_text'].str.len()
    raid_df['word_count'] = raid_df['combined_text'].str.split().str.len()
    
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
    print("RAID Dataset Text Characteristics (title + generation):")
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


def download_and_sample_newswire_chunked(raid_stats: Dict, target_samples: int, 
                                         output_dir: str, chunk_size: int = 10000):
    """
    Download and sample newswire dataset using chunked processing to avoid OOM.
    
    Args:
        raid_stats: Statistics from RAID dataset
        target_samples: Number of samples to collect
        output_dir: Output directory path
        chunk_size: Size of chunks to process at a time
    """
    print("\n" + "=" * 80)
    print("STEP 2: Downloading & Sampling Newswire Dataset (Chunked Processing)")
    print("=" * 80)
    
    print("\nSource: https://huggingface.co/datasets/dell-research-harvard/newswire")
    print("⚠️  This is a LARGE dataset (2.7M+ examples) - using chunked processing")
    
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
    print("\nLoading dataset in streaming mode...")
    dataset = load_dataset("dell-research-harvard/newswire", split="train", streaming=True)
    
    # Determine text column
    first_example = next(iter(dataset))
    print(f"\n📋 Dataset columns: {list(first_example.keys())}")
    
    # Find text column
    text_columns = ['text', 'article', 'content', 'body', 'headline', 'description']
    text_col = None
    for col in text_columns:
        if col in first_example:
            text_col = col
            break
    
    if text_col is None:
        # Use first string column
        for col, value in first_example.items():
            if isinstance(value, str) and len(value) > 50:
                text_col = col
                break
    
    print(f"✓ Using '{text_col}' as text column")
    
    # Sample preview
    sample_text = first_example[text_col]
    print(f"\n📝 Sample text (first 200 chars):")
    print(f"   {sample_text[:200]}...")
    print(f"   Length: {len(sample_text)} chars, {len(sample_text.split())} words")
    
    # Process in chunks and collect matching samples
    print(f"\n🔄 Processing dataset in chunks of {chunk_size:,}...")
    
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
            chunk_df['combined_text'] = chunk_df[text_col].astype(str)
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
            
            # Progress update
            if total_processed % 50000 == 0:
                print(f"   Processed: {total_processed:,} | Matched: {total_matched:,} "
                      f"({total_matched/total_processed*100:.2f}%)")
            
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
        chunk_df['combined_text'] = chunk_df[text_col].astype(str)
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
    
    # Combine all sampled data
    print(f"\n📊 Processing Summary:")
    print(f"   Total processed: {total_processed:,}")
    print(f"   Total matched:   {total_matched:,}")
    print(f"   Match rate:      {total_matched/total_processed*100:.2f}%")
    
    if not sampled_data:
        print("\n❌ No samples matched the RAID criteria!")
        return None
    
    print(f"\nCombining {len(sampled_data)} chunks...")
    final_df = pd.concat(sampled_data, ignore_index=True)
    
    # Limit to target samples if we have more
    if len(final_df) > target_samples:
        print(f"Sampling {target_samples:,} from {len(final_df):,} matches...")
        final_df = final_df.sample(n=target_samples, random_state=42).reset_index(drop=True)
    
    # Add metadata
    final_df['domain'] = 'news'
    final_df['source'] = 'newswire'
    final_df['raid_aligned'] = True
    
    print(f"\n📊 Final Sampled Dataset Statistics:")
    print(f"   Total samples: {len(final_df):,}")
    print(f"   Char length - Min: {final_df['text_length'].min()}, "
          f"Mean: {final_df['text_length'].mean():.0f}, "
          f"Max: {final_df['text_length'].max()}")
    print(f"   Word count  - Min: {final_df['word_count'].min()}, "
          f"Mean: {final_df['word_count'].mean():.0f}, "
          f"Max: {final_df['word_count'].max()}")
    
    # Save dataset
    print("\n" + "=" * 80)
    print("STEP 3: Saving Dataset")
    print("=" * 80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    csv_path = output_path / 'newswire_sampled.csv'
    final_df.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV: {csv_path}")
    print(f"  Size: {csv_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Save as JSON
    json_path = output_path / 'newswire_sampled.json'
    final_df.to_json(json_path, orient='records', lines=True)
    print(f"✓ Saved JSON: {json_path}")
    print(f"  Size: {json_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    return final_df


def main():
    parser = argparse.ArgumentParser(
        description='Download and sample dell-research-harvard/newswire dataset based on RAID characteristics'
    )
    parser.add_argument(
        '--raid-csv',
        type=str,
        default='/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/raid/news.csv',
        help='Path to RAID news.csv file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/news',
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
        help='Target number of samples to extract (default: 100000)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=10000,
        help='Chunk size for processing (default: 10000)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("NEWSWIRE DATASET DOWNLOAD AND SAMPLING SCRIPT")
    print("(Memory-Efficient Chunked Processing)")
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
        
        # Step 2-3: Download, sample, and save newswire dataset
        result_df = download_and_sample_newswire_chunked(
            raid_stats, 
            args.target_samples, 
            args.output_dir,
            args.chunk_size
        )
        
        if result_df is not None:
            print("\n" + "=" * 80)
            print("✅ SCRIPT COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"\nDataset location: {args.output_dir}")
            print(f"Total samples: {len(result_df):,}")
            print("\nFiles created:")
            print(f"  - newswire_sampled.csv")
            print(f"  - newswire_sampled.json")
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

