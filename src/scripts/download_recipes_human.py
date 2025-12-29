#!/usr/bin/env python3
"""
Script to process and sample recipes dataset based on RAID recipes characteristics.

This script:
1. Loads the human-written recipes dataset
2. Combines title + ingredients + directions into single 'text' column
3. Analyzes RAID recipes dataset (title + generation)
4. Samples human recipes based on RAID characteristics
5. Saves sampled dataset
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


def analyze_raid_characteristics(raid_csv_path: str, sample_size: int = 50000) -> Dict:
    """
    Analyze RAID recipes dataset to extract text characteristics.
    
    Args:
        raid_csv_path: Path to RAID recipes.csv file
        sample_size: Number of samples to analyze
    
    Returns:
        Dictionary with text characteristics
    """
    print("=" * 80)
    print("STEP 1: Analyzing RAID Recipes Dataset Characteristics")
    print("=" * 80)
    
    print(f"\nReading {sample_size:,} samples from RAID recipes dataset...")
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
    print("RAID Recipes Dataset Text Characteristics:")
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


def process_human_recipes(recipes_csv_path: str) -> pd.DataFrame:
    """
    Load and process human-written recipes dataset.
    Combines title + ingredients + directions into single text column.
    
    Args:
        recipes_csv_path: Path to recipes dataset CSV
    
    Returns:
        Processed DataFrame with 'text' column
    """
    print("\n" + "=" * 80)
    print("STEP 2: Processing Human-Written Recipes Dataset")
    print("=" * 80)
    
    print(f"\nLoading recipes from: {recipes_csv_path}")
    df = pd.read_csv(recipes_csv_path)
    
    print(f"✓ Loaded {len(df):,} recipes")
    print(f"✓ Columns: {list(df.columns)}")
    
    # Check for required columns
    required_cols = ['title', 'ingredients', 'directions']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"\n⚠️  Warning: Missing columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        # Try to find alternative column names
        for col in df.columns:
            print(f"  - {col}: {df[col].dtype}")
    
    # Combine title, ingredients, and directions
    print("\nCombining 'title' + 'ingredients' + 'directions' into 'text' column...")
    
    df['title'] = df['title'].fillna('').astype(str)
    df['ingredients'] = df['ingredients'].fillna('').astype(str)
    df['directions'] = df['directions'].fillna('').astype(str)
    
    # Combine with formatting
    df['text'] = (
        df['title'] + '\n\n' +
        'Ingredients:\n' + df['ingredients'] + '\n\n' +
        'Directions:\n' + df['directions']
    )
    
    # Calculate text statistics
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    
    # Remove any zero-length texts
    original_len = len(df)
    df = df[df['text_length'] > 0]
    if len(df) < original_len:
        print(f"⚠️  Removed {original_len - len(df):,} recipes with no text")
    
    print(f"\n✓ Processed {len(df):,} recipes")
    print(f"\n📊 Human Recipes Text Statistics:")
    print(f"   Char length - Min: {df['text_length'].min()}, "
          f"Mean: {df['text_length'].mean():.0f}, "
          f"Max: {df['text_length'].max()}")
    print(f"   Word count  - Min: {df['word_count'].min()}, "
          f"Mean: {df['word_count'].mean():.0f}, "
          f"Max: {df['word_count'].max()}")
    
    # Show sample
    print(f"\n📝 Sample recipe (first 300 chars):")
    sample = df.iloc[0]['text']
    print(f"   {sample[:300]}...")
    
    return df


def sample_based_on_raid(recipes_df: pd.DataFrame, raid_stats: Dict, 
                         target_samples: int = 50000) -> pd.DataFrame:
    """
    Sample recipes based on RAID characteristics.
    
    Args:
        recipes_df: Recipes DataFrame with 'text' column
        raid_stats: Statistics from RAID dataset
        target_samples: Number of samples to extract
    
    Returns:
        Sampled DataFrame
    """
    print("\n" + "=" * 80)
    print("STEP 3: Sampling Recipes Based on RAID Characteristics")
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
    print(f"\nFiltering recipes dataset...")
    filtered_df = recipes_df[
        (recipes_df['text_length'] >= char_min) &
        (recipes_df['text_length'] <= char_max) &
        (recipes_df['word_count'] >= word_min) &
        (recipes_df['word_count'] <= word_max)
    ].copy()
    
    print(f"✓ Recipes matching criteria: {len(filtered_df):,} / {len(recipes_df):,} "
          f"({len(filtered_df)/len(recipes_df)*100:.1f}%)")
    
    # Sample target number of samples (or all if less than target)
    if len(filtered_df) > target_samples:
        print(f"\nSampling {target_samples:,} recipes randomly...")
        sampled_df = filtered_df.sample(n=target_samples, random_state=42).reset_index(drop=True)
    else:
        print(f"\nUsing all {len(filtered_df):,} matching recipes...")
        sampled_df = filtered_df.copy()
    
    print(f"✓ Final sample size: {len(sampled_df):,}")
    
    # Add metadata
    sampled_df['domain'] = 'recipes'
    sampled_df['source'] = 'human_written'
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
    Save dataset in multiple formats.
    
    Args:
        df: DataFrame to save
        output_dir: Output directory path
    """
    print("\n" + "=" * 80)
    print("STEP 4: Saving Sampled Dataset")
    print("=" * 80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    csv_path = output_path / 'recipes_sampled.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved CSV: {csv_path}")
    print(f"  Size: {csv_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Save as JSON
    json_path = output_path / 'recipes_sampled.json'
    df.to_json(json_path, orient='records', lines=True)
    print(f"✓ Saved JSON: {json_path}")
    print(f"  Size: {json_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Create summary file
    summary_path = output_path / 'SAMPLING_SUMMARY.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RECIPES DATASET - RAID ALIGNED SAMPLING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Samples: {len(df):,}\n")
        f.write(f"Character Length: {df['text_length'].min()} - {df['text_length'].max()} "
                f"(mean: {df['text_length'].mean():.0f})\n")
        f.write(f"Word Count: {df['word_count'].min()} - {df['word_count'].max()} "
                f"(mean: {df['word_count'].mean():.0f})\n\n")
        f.write("Files:\n")
        f.write("  - recipes_sampled.csv\n")
        f.write("  - recipes_sampled.json\n\n")
        f.write("Sampling Strategy:\n")
        f.write("  Combined: title + ingredients + directions\n")
        f.write("  Filter: RAID mean ± 2σ\n")
        f.write("  Random seed: 42\n")
    
    print(f"✓ Created summary: {summary_path}")
    
    print(f"\n✅ Dataset saved successfully to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Process and sample recipes dataset based on RAID characteristics'
    )
    parser.add_argument(
        '--recipes-csv',
        type=str,
        default='/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/receipes/full_dataset.csv',
        help='Path to human-written recipes CSV file'
    )
    parser.add_argument(
        '--raid-csv',
        type=str,
        default='/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/raid/recipes.csv',
        help='Path to RAID recipes.csv file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/receipes',
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
        default=50000,
        help='Target number of recipe samples to extract (default: 50000)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("RECIPES DATASET PROCESSING AND SAMPLING SCRIPT")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Human Recipes CSV: {args.recipes_csv}")
    print(f"  RAID CSV:          {args.raid_csv}")
    print(f"  Output Dir:        {args.output_dir}")
    print(f"  RAID samples:      {args.raid_sample_size:,}")
    print(f"  Target samples:    {args.target_samples:,}")
    
    try:
        # Step 1: Analyze RAID characteristics
        raid_stats = analyze_raid_characteristics(args.raid_csv, args.raid_sample_size)
        
        # Step 2: Process human recipes (combine title + ingredients + directions)
        recipes_df = process_human_recipes(args.recipes_csv)
        
        # Step 3: Sample based on RAID
        sampled_df = sample_based_on_raid(recipes_df, raid_stats, args.target_samples)
        
        # Step 4: Save dataset
        save_dataset(sampled_df, args.output_dir)
        
        print("\n" + "=" * 80)
        print("✅ SCRIPT COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nDataset location: {args.output_dir}")
        print(f"Total samples: {len(sampled_df):,}")
        print("\nFiles created:")
        print(f"  - recipes_sampled.csv")
        print(f"  - recipes_sampled.json")
        print(f"  - SAMPLING_SUMMARY.txt")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

