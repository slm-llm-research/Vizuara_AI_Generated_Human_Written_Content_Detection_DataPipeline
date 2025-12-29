#!/usr/bin/env python3
"""
Script to download and sample sentence-transformers/reddit-title-body dataset 
based on RAID reddit dataset characteristics.

This script:
1. Analyzes RAID reddit dataset to understand text characteristics (title + generation)
2. Downloads sentence-transformers/reddit-title-body dataset from HuggingFace
3. Samples based on RAID characteristics
4. Saves the sampled dataset
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
    print(f"✓ Columns: {list(raid_df.columns)}")
    
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


def download_reddit_dataset() -> pd.DataFrame:
    """
    Download sentence-transformers/reddit-title-body dataset from HuggingFace.
    
    Returns:
        Combined DataFrame with all samples
    """
    print("\n" + "=" * 80)
    print("STEP 2: Downloading Reddit Dataset from HuggingFace")
    print("=" * 80)
    
    print("\nSource: https://huggingface.co/datasets/sentence-transformers/reddit-title-body")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset("sentence-transformers/reddit-title-body")
    
    print(f"✓ Available splits: {list(dataset.keys())}")
    
    # Combine all splits
    all_dfs = []
    total_samples = 0
    for split_name in dataset.keys():
        split_df = pd.DataFrame(dataset[split_name])
        all_dfs.append(split_df)
        total_samples += len(split_df)
        print(f"  {split_name}: {len(split_df):,} samples")
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    print(f"\n✓ Total samples: {len(combined_df):,}")
    print(f"\n📋 Dataset columns: {list(combined_df.columns)}")
    
    # Identify text columns
    # Reddit dataset typically has 'title' and 'body' or similar
    print("\nSample of first row to understand structure:")
    print(combined_df.head(1).to_dict('records')[0])
    
    # Combine title and body (or equivalent fields)
    if 'title' in combined_df.columns and 'body' in combined_df.columns:
        text_cols = ('title', 'body')
        print(f"\n✓ Using 'title' + 'body' columns")
    elif 'title' in combined_df.columns and 'selftext' in combined_df.columns:
        text_cols = ('title', 'selftext')
        print(f"\n✓ Using 'title' + 'selftext' columns")
    else:
        # Find text columns
        text_cols = [col for col in combined_df.columns if 'text' in col.lower() or 'title' in col.lower()][:2]
        print(f"\n✓ Using columns: {text_cols}")
    
    # Combine text
    if len(text_cols) == 2:
        combined_df['combined_text'] = (
            combined_df[text_cols[0]].fillna('').astype(str) + ' ' + 
            combined_df[text_cols[1]].fillna('').astype(str)
        )
    else:
        combined_df['combined_text'] = combined_df[text_cols[0]].fillna('').astype(str)
    
    combined_df['text_length'] = combined_df['combined_text'].str.len()
    combined_df['word_count'] = combined_df['combined_text'].str.split().str.len()
    
    # Remove zero-length texts
    combined_df = combined_df[combined_df['text_length'] > 0]
    
    print(f"\n📊 Reddit Dataset Text Statistics:")
    print(f"   Char length - Min: {combined_df['text_length'].min()}, "
          f"Mean: {combined_df['text_length'].mean():.0f}, "
          f"Max: {combined_df['text_length'].max()}")
    print(f"   Word count  - Min: {combined_df['word_count'].min()}, "
          f"Mean: {combined_df['word_count'].mean():.0f}, "
          f"Max: {combined_df['word_count'].max()}")
    
    # Show sample
    print(f"\n📝 Sample Reddit post (first 300 chars):")
    sample_text = combined_df['combined_text'].iloc[0]
    print(f"   {sample_text[:300]}...")
    
    return combined_df


def sample_based_on_raid(reddit_df: pd.DataFrame, raid_stats: Dict, 
                         target_samples: int = 100000) -> pd.DataFrame:
    """
    Sample Reddit posts based on RAID characteristics.
    
    Args:
        reddit_df: Reddit DataFrame
        raid_stats: Statistics from RAID dataset
        target_samples: Number of samples to extract
    
    Returns:
        Sampled DataFrame
    """
    print("\n" + "=" * 80)
    print("STEP 3: Sampling Reddit Posts Based on RAID Characteristics")
    print("=" * 80)
    
    # Define filtering criteria based on RAID statistics
    # Using mean ± 2*std to capture most of the distribution
    char_min = max(0, raid_stats['char_length']['mean'] - 2 * raid_stats['char_length']['std'])
    char_max = raid_stats['char_length']['mean'] + 2 * raid_stats['char_length']['std']
    
    word_min = max(0, raid_stats['word_count']['mean'] - 2 * raid_stats['word_count']['std'])
    word_max = raid_stats['word_count']['mean'] + 2 * raid_stats['word_count']['std']
    
    print(f"\n📏 Sampling Criteria (based on RAID mean ± 2σ):")
    print(f"   Character length: {char_min:.0f} - {char_max:.0f} characters")
    print(f"   Word count:       {word_min:.0f} - {word_max:.0f} words")
    print(f"   Target samples:   {target_samples:,}")
    
    # Filter based on criteria
    print(f"\nFiltering Reddit dataset...")
    filtered_df = reddit_df[
        (reddit_df['text_length'] >= char_min) &
        (reddit_df['text_length'] <= char_max) &
        (reddit_df['word_count'] >= word_min) &
        (reddit_df['word_count'] <= word_max)
    ].copy()
    
    print(f"✓ Posts matching criteria: {len(filtered_df):,} / {len(reddit_df):,} "
          f"({len(filtered_df)/len(reddit_df)*100:.1f}%)")
    
    # Sample target number of samples (or all if less than target)
    if len(filtered_df) > target_samples:
        print(f"\nSampling {target_samples:,} posts randomly...")
        sampled_df = filtered_df.sample(n=target_samples, random_state=42).reset_index(drop=True)
    else:
        print(f"\nUsing all {len(filtered_df):,} matching posts...")
        sampled_df = filtered_df.copy()
    
    print(f"✓ Final sample size: {len(sampled_df):,}")
    
    # Add metadata
    sampled_df['domain'] = 'reddit'
    sampled_df['source'] = 'sentence-transformers/reddit-title-body'
    sampled_df['raid_aligned'] = True
    
    # Display final statistics
    print(f"\n📊 Sampled Dataset Statistics:")
    print(f"   Total samples: {len(sampled_df):,}")
    print(f"   Char length - Min: {sampled_df['text_length'].min()}, "
          f"Mean: {sampled_df['text_length'].mean():.0f}, "
          f"Max: {sampled_df['text_length'].max()}")
    print(f"   Word count  - Min: {sampled_df['word_count'].min()}, "
          f"Mean: {sampled_df['word_count'].mean():.0f}, "
          f"Max: {sampled_df['word_count'].max()}")
    
    return sampled_df


def save_dataset(df: pd.DataFrame, output_dir: str):
    """
    Save dataset in multiple formats (CSV, JSON).
    
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
    
    print(f"✓ Created summary: {summary_path}")
    
    print(f"\n✅ Dataset saved successfully to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Download and sample Reddit dataset based on RAID characteristics'
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
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("REDDIT DATASET DOWNLOAD AND SAMPLING SCRIPT")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  RAID CSV:        {args.raid_csv}")
    print(f"  Output Dir:      {args.output_dir}")
    print(f"  RAID samples:    {args.raid_sample_size:,}")
    print(f"  Target samples:  {args.target_samples:,}")
    
    try:
        # Step 1: Analyze RAID characteristics
        raid_stats = analyze_raid_characteristics(args.raid_csv, args.raid_sample_size)
        
        # Step 2: Download Reddit dataset
        reddit_df = download_reddit_dataset()
        
        # Step 3: Sample based on RAID
        sampled_df = sample_based_on_raid(reddit_df, raid_stats, args.target_samples)
        
        # Step 4: Save dataset
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
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

