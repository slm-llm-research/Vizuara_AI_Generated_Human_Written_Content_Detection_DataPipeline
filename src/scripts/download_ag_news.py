#!/usr/bin/env python3
"""
Script to download AG News dataset and sample based on RAID news dataset characteristics.

This script:
1. Analyzes RAID news dataset to understand text characteristics (title + generation)
2. Downloads AG News dataset from HuggingFace
3. Samples AG News based on RAID characteristics
4. Saves the sampled dataset in multiple formats
"""

import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from pathlib import Path
import argparse
from typing import Dict, Tuple
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
    print(f"✓ Columns: {list(raid_df.columns)}")
    
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
        },
        'sample_size': len(raid_df),
        'raid_samples': raid_df['combined_text'].head(10).tolist()
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
    
    print(f"\n📝 Sample RAID texts (first 3):")
    for i, text in enumerate(stats['raid_samples'][:3], 1):
        preview = text[:200] + "..." if len(text) > 200 else text
        print(f"\n   Sample {i} ({len(text)} chars):")
        print(f"   {preview}")
    
    return stats


def download_ag_news() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Download AG News dataset from HuggingFace.
    
    Returns:
        Tuple of (train_df, test_df, combined_df)
    """
    print("\n" + "=" * 80)
    print("STEP 2: Downloading AG News Dataset from HuggingFace")
    print("=" * 80)
    
    print("\nDownloading ag_news dataset...")
    print("Source: https://huggingface.co/datasets/ag_news")
    
    # Load dataset
    dataset = load_dataset("ag_news")
    
    # Convert to DataFrames
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    
    print(f"\n✓ Downloaded successfully!")
    print(f"  Train samples: {len(train_df):,}")
    print(f"  Test samples:  {len(test_df):,}")
    print(f"  Total samples: {len(train_df) + len(test_df):,}")
    
    # Combine train and test for sampling
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Display class distribution
    print(f"\n📊 AG News Class Distribution:")
    class_names = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
    for label, name in class_names.items():
        count = (combined_df['label'] == label).sum()
        pct = count / len(combined_df) * 100
        print(f"   {label} - {name:10s}: {count:>7,} samples ({pct:>5.2f}%)")
    
    # Combine text and label for AG News
    combined_df['combined_text'] = combined_df['text']
    combined_df['text_length'] = combined_df['combined_text'].str.len()
    combined_df['word_count'] = combined_df['combined_text'].str.split().str.len()
    
    print(f"\n📊 AG News Original Text Statistics:")
    print(f"   Char length - Min: {combined_df['text_length'].min()}, "
          f"Mean: {combined_df['text_length'].mean():.0f}, "
          f"Max: {combined_df['text_length'].max()}")
    print(f"   Word count  - Min: {combined_df['word_count'].min()}, "
          f"Mean: {combined_df['word_count'].mean():.0f}, "
          f"Max: {combined_df['word_count'].max()}")
    
    return train_df, test_df, combined_df


def sample_based_on_raid(ag_news_df: pd.DataFrame, raid_stats: Dict, 
                         target_samples: int = 100000) -> pd.DataFrame:
    """
    Sample AG News based on RAID characteristics.
    
    Args:
        ag_news_df: AG News DataFrame
        raid_stats: Statistics from RAID dataset
        target_samples: Number of samples to extract
    
    Returns:
        Sampled DataFrame
    """
    print("\n" + "=" * 80)
    print("STEP 3: Sampling AG News Based on RAID Characteristics")
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
    
    # Filter based on criteria
    print(f"\nFiltering AG News dataset...")
    filtered_df = ag_news_df[
        (ag_news_df['text_length'] >= char_min) &
        (ag_news_df['text_length'] <= char_max) &
        (ag_news_df['word_count'] >= word_min) &
        (ag_news_df['word_count'] <= word_max)
    ].copy()
    
    print(f"✓ Samples matching criteria: {len(filtered_df):,} / {len(ag_news_df):,} "
          f"({len(filtered_df)/len(ag_news_df)*100:.1f}%)")
    
    # Sample target number of samples (or all if less than target)
    if len(filtered_df) > target_samples:
        print(f"\nSampling {target_samples:,} samples (stratified by class)...")
        # Stratified sampling to maintain class balance
        sampled_df = filtered_df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), target_samples // 4), random_state=42)
        ).reset_index(drop=True)
    else:
        print(f"\nUsing all {len(filtered_df):,} matching samples...")
        sampled_df = filtered_df.copy()
    
    print(f"✓ Final sample size: {len(sampled_df):,}")
    
    # Add metadata
    sampled_df['domain'] = 'news'
    sampled_df['source'] = 'ag_news'
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
    
    # Class distribution
    print(f"\n📊 Class Distribution in Sampled Data:")
    class_names = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
    for label, name in class_names.items():
        count = (sampled_df['label'] == label).sum()
        pct = count / len(sampled_df) * 100
        print(f"   {label} - {name:10s}: {count:>7,} samples ({pct:>5.2f}%)")
    
    return sampled_df


def save_dataset(df: pd.DataFrame, output_dir: str):
    """
    Save dataset in multiple formats (CSV, JSON, Arrow).
    
    Args:
        df: DataFrame to save
        output_dir: Output directory path
    """
    print("\n" + "=" * 80)
    print("STEP 4: Saving Dataset")
    print("=" * 80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    csv_dir = output_path / 'csv'
    json_dir = output_path / 'json'
    arrow_dir = output_path / 'arrow' / 'train'
    
    csv_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    arrow_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to: {output_path}")
    
    # Save as CSV
    csv_path = csv_dir / 'train.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV: {csv_path}")
    print(f"  Size: {csv_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Save as JSON
    json_path = json_dir / 'train.json'
    df.to_json(json_path, orient='records', lines=True)
    print(f"✓ Saved JSON: {json_path}")
    print(f"  Size: {json_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Save as Arrow (HuggingFace format)
    from datasets import Dataset
    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(str(arrow_dir.parent))
    print(f"✓ Saved Arrow format: {arrow_dir.parent}")
    
    # Create a summary file
    summary_path = output_path / 'README.md'
    with open(summary_path, 'w') as f:
        f.write(f"# AG News Dataset - RAID Aligned\n\n")
        f.write(f"**Source**: https://huggingface.co/datasets/ag_news\n")
        f.write(f"**Sampling Strategy**: Based on RAID news dataset characteristics\n\n")
        f.write(f"## Statistics\n\n")
        f.write(f"- Total samples: {len(df):,}\n")
        f.write(f"- Character length: {df['text_length'].min()} - {df['text_length'].max()} "
                f"(mean: {df['text_length'].mean():.0f})\n")
        f.write(f"- Word count: {df['word_count'].min()} - {df['word_count'].max()} "
                f"(mean: {df['word_count'].mean():.0f})\n\n")
        f.write(f"## Class Distribution\n\n")
        class_names = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
        for label, name in class_names.items():
            count = (df['label'] == label).sum()
            pct = count / len(df) * 100
            f.write(f"- {name}: {count:,} samples ({pct:.2f}%)\n")
    
    print(f"✓ Created README: {summary_path}")
    
    print(f"\n✅ Dataset saved successfully to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Download and sample AG News dataset based on RAID characteristics'
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
        default='/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/ag_news',
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
        help='Target number of AG News samples to extract (default: 100000)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("AG NEWS DATASET DOWNLOAD AND SAMPLING SCRIPT")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  RAID CSV:        {args.raid_csv}")
    print(f"  Output Dir:      {args.output_dir}")
    print(f"  RAID samples:    {args.raid_sample_size:,}")
    print(f"  Target samples:  {args.target_samples:,}")
    
    try:
        # Step 1: Analyze RAID characteristics
        raid_stats = analyze_raid_characteristics(args.raid_csv, args.raid_sample_size)
        
        # Step 2: Download AG News
        train_df, test_df, combined_df = download_ag_news()
        
        # Step 3: Sample based on RAID
        sampled_df = sample_based_on_raid(combined_df, raid_stats, args.target_samples)
        
        # Step 4: Save dataset
        save_dataset(sampled_df, args.output_dir)
        
        print("\n" + "=" * 80)
        print("✅ SCRIPT COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nDataset location: {args.output_dir}")
        print(f"Total samples: {len(sampled_df):,}")
        print("\nNext steps:")
        print("  1. Review the dataset in the output directory")
        print("  2. Check the README.md for dataset statistics")
        print("  3. Update DATASETS.md with information about this dataset")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

