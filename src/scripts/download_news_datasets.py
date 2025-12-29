#!/usr/bin/env python3
"""
Script to download multiple news datasets and sample based on RAID news dataset characteristics.

This script:
1. Analyzes RAID news dataset to understand text characteristics (title + generation)
2. Downloads news datasets from HuggingFace (AG News, Newswire, etc.)
3. Samples datasets based on RAID characteristics
4. Saves the sampled datasets in multiple formats
"""

import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from pathlib import Path
import argparse
from typing import Dict, Tuple, List
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


def download_ag_news() -> pd.DataFrame:
    """
    Download AG News dataset from HuggingFace.
    
    Returns:
        Combined DataFrame with all samples
    """
    print("\n" + "=" * 80)
    print("Downloading AG News Dataset from HuggingFace")
    print("=" * 80)
    
    print("\nSource: https://huggingface.co/datasets/ag_news")
    
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
    
    # Combine text and calculate stats
    combined_df['combined_text'] = combined_df['text']
    combined_df['text_length'] = combined_df['combined_text'].str.len()
    combined_df['word_count'] = combined_df['combined_text'].str.split().str.len()
    combined_df['dataset_source'] = 'ag_news'
    
    print(f"\n📊 AG News Text Statistics:")
    print(f"   Char length - Min: {combined_df['text_length'].min()}, "
          f"Mean: {combined_df['text_length'].mean():.0f}, "
          f"Max: {combined_df['text_length'].max()}")
    print(f"   Word count  - Min: {combined_df['word_count'].min()}, "
          f"Mean: {combined_df['word_count'].mean():.0f}, "
          f"Max: {combined_df['word_count'].max()}")
    
    return combined_df


def download_newswire() -> pd.DataFrame:
    """
    Download Dell Research Harvard Newswire dataset from HuggingFace.
    
    Returns:
        DataFrame with all samples
    """
    print("\n" + "=" * 80)
    print("Downloading Dell Research Harvard Newswire Dataset from HuggingFace")
    print("=" * 80)
    
    print("\nSource: https://huggingface.co/datasets/dell-research-harvard/newswire")
    
    try:
        # Load dataset
        dataset = load_dataset("dell-research-harvard/newswire")
        
        # Check available splits
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
        
        print(f"\n✓ Downloaded successfully!")
        print(f"  Total samples: {len(combined_df):,}")
        
        # Display columns
        print(f"\n📋 Available columns: {list(combined_df.columns)}")
        
        # Find the text column (could be 'text', 'article', 'content', etc.)
        text_columns = ['text', 'article', 'content', 'body', 'headline']
        text_col = None
        for col in text_columns:
            if col in combined_df.columns:
                text_col = col
                break
        
        if text_col is None:
            # If no standard column, show first few rows to understand structure
            print("\n⚠️  Standard text column not found. First row sample:")
            print(combined_df.head(1).to_dict('records'))
            # Use the first string column as text
            for col in combined_df.columns:
                if combined_df[col].dtype == 'object':
                    text_col = col
                    print(f"\n✓ Using '{col}' as text column")
                    break
        else:
            print(f"\n✓ Using '{text_col}' as text column")
        
        # Combine text and calculate stats
        combined_df['combined_text'] = combined_df[text_col].astype(str)
        combined_df['text_length'] = combined_df['combined_text'].str.len()
        combined_df['word_count'] = combined_df['combined_text'].str.split().str.len()
        combined_df['dataset_source'] = 'newswire'
        
        print(f"\n📊 Newswire Text Statistics:")
        print(f"   Char length - Min: {combined_df['text_length'].min()}, "
              f"Mean: {combined_df['text_length'].mean():.0f}, "
              f"Max: {combined_df['text_length'].max()}")
        print(f"   Word count  - Min: {combined_df['word_count'].min()}, "
              f"Mean: {combined_df['word_count'].mean():.0f}, "
              f"Max: {combined_df['word_count'].max()}")
        
        # Show sample
        print(f"\n📝 Sample text (first 300 chars):")
        sample_text = combined_df['combined_text'].iloc[0]
        print(f"   {sample_text[:300]}...")
        
        return combined_df
        
    except Exception as e:
        print(f"\n❌ Error downloading newswire dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def sample_based_on_raid(df: pd.DataFrame, dataset_name: str, raid_stats: Dict, 
                         target_samples: int = 100000) -> pd.DataFrame:
    """
    Sample dataset based on RAID characteristics.
    
    Args:
        df: Dataset DataFrame
        dataset_name: Name of the dataset
        raid_stats: Statistics from RAID dataset
        target_samples: Number of samples to extract
    
    Returns:
        Sampled DataFrame
    """
    print("\n" + "=" * 80)
    print(f"Sampling {dataset_name} Based on RAID Characteristics")
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
    print(f"\nFiltering {dataset_name} dataset...")
    filtered_df = df[
        (df['text_length'] >= char_min) &
        (df['text_length'] <= char_max) &
        (df['word_count'] >= word_min) &
        (df['word_count'] <= word_max)
    ].copy()
    
    print(f"✓ Samples matching criteria: {len(filtered_df):,} / {len(df):,} "
          f"({len(filtered_df)/len(df)*100:.1f}%)")
    
    # Sample target number of samples (or all if less than target)
    if len(filtered_df) > target_samples:
        print(f"\nSampling {target_samples:,} samples randomly...")
        sampled_df = filtered_df.sample(n=target_samples, random_state=42).reset_index(drop=True)
    else:
        print(f"\nUsing all {len(filtered_df):,} matching samples...")
        sampled_df = filtered_df.copy()
    
    print(f"✓ Final sample size: {len(sampled_df):,}")
    
    # Add metadata
    sampled_df['domain'] = 'news'
    sampled_df['source'] = sampled_df['dataset_source']
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


def save_dataset(df: pd.DataFrame, output_dir: str, dataset_name: str):
    """
    Save dataset in multiple formats (CSV, JSON, Arrow).
    
    Args:
        df: DataFrame to save
        output_dir: Output directory path
        dataset_name: Name of the dataset for file naming
    """
    print("\n" + "=" * 80)
    print(f"Saving {dataset_name} Dataset")
    print("=" * 80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    csv_path = output_path / f'{dataset_name}_sampled.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV: {csv_path}")
    print(f"  Size: {csv_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Save as JSON
    json_path = output_path / f'{dataset_name}_sampled.json'
    df.to_json(json_path, orient='records', lines=True)
    print(f"✓ Saved JSON: {json_path}")
    print(f"  Size: {json_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    print(f"\n✅ {dataset_name} dataset saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Download and sample news datasets based on RAID characteristics'
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
        help='Output directory for sampled datasets'
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
        help='Target number of samples to extract per dataset (default: 100000)'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=['ag_news', 'newswire'],
        choices=['ag_news', 'newswire', 'both'],
        help='Which datasets to download (default: both)'
    )
    
    args = parser.parse_args()
    
    # Handle 'both' option
    if 'both' in args.datasets:
        datasets_to_download = ['ag_news', 'newswire']
    else:
        datasets_to_download = args.datasets
    
    print("\n" + "=" * 80)
    print("NEWS DATASETS DOWNLOAD AND SAMPLING SCRIPT")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  RAID CSV:        {args.raid_csv}")
    print(f"  Output Dir:      {args.output_dir}")
    print(f"  RAID samples:    {args.raid_sample_size:,}")
    print(f"  Target samples:  {args.target_samples:,}")
    print(f"  Datasets:        {', '.join(datasets_to_download)}")
    
    try:
        # Step 1: Analyze RAID characteristics (once for all datasets)
        raid_stats = analyze_raid_characteristics(args.raid_csv, args.raid_sample_size)
        
        # Step 2-4: Download, sample, and save each dataset
        for dataset_name in datasets_to_download:
            print("\n" + "=" * 80)
            print(f"PROCESSING: {dataset_name.upper()}")
            print("=" * 80)
            
            # Download dataset
            if dataset_name == 'ag_news':
                df = download_ag_news()
            elif dataset_name == 'newswire':
                df = download_newswire()
                if df is None:
                    print(f"\n⚠️  Skipping {dataset_name} due to download error")
                    continue
            
            # Sample based on RAID
            sampled_df = sample_based_on_raid(df, dataset_name, raid_stats, args.target_samples)
            
            # Save dataset
            save_dataset(sampled_df, args.output_dir, dataset_name)
        
        print("\n" + "=" * 80)
        print("✅ ALL DATASETS PROCESSED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nDataset location: {args.output_dir}")
        print("\nNext steps:")
        print("  1. Review the datasets in the output directory")
        print("  2. Update DATASETS.md with information about these datasets")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

