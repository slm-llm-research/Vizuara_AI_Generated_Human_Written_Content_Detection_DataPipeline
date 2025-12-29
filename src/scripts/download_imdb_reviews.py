#!/usr/bin/env python3
"""
Script to process IMDb movie reviews and sample based on RAID reviews characteristics.

This script:
1. Reads all CSV files from IMDb reviews folder
2. Combines title + review into 'text' column for each file
3. Combines all files into single dataset
4. Analyzes RAID reviews dataset (title + generation)
5. Samples IMDb reviews based on RAID characteristics
6. Saves sampled dataset
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List
import warnings
import glob
warnings.filterwarnings('ignore')


def analyze_raid_characteristics(raid_csv_path: str, sample_size: int = 50000) -> Dict:
    """
    Analyze RAID reviews dataset to extract text characteristics.
    
    Args:
        raid_csv_path: Path to RAID reviews.csv file
        sample_size: Number of samples to analyze
    
    Returns:
        Dictionary with text characteristics
    """
    print("=" * 80)
    print("STEP 1: Analyzing RAID Reviews Dataset Characteristics")
    print("=" * 80)
    
    print(f"\nReading sample of {sample_size:,} rows from RAID reviews dataset...")
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
    print("RAID Reviews Dataset Text Characteristics (title + generation):")
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


def load_and_combine_imdb_reviews(reviews_dir: str) -> pd.DataFrame:
    """
    Load all CSV files from IMDb reviews directory and combine them.
    Combines title + review into single 'text' column.
    
    Args:
        reviews_dir: Directory containing review CSV files
    
    Returns:
        Combined DataFrame with all reviews
    """
    print("\n" + "=" * 80)
    print("STEP 2: Loading and Combining IMDb Movie Reviews")
    print("=" * 80)
    
    print(f"\nScanning directory: {reviews_dir}")
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(reviews_dir, "*.csv"))
    print(f"✓ Found {len(csv_files):,} CSV files")
    
    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found in {reviews_dir}")
    
    # Load and process each file
    all_reviews = []
    total_reviews = 0
    
    print("\n🔄 Processing files...")
    for i, csv_file in enumerate(csv_files, 1):
        try:
            # Read CSV
            df = pd.read_csv(csv_file)
            
            # Check for required columns
            if 'review' not in df.columns:
                # Try to find review column (might be named differently)
                possible_cols = [col for col in df.columns if 'review' in col.lower() or 'text' in col.lower()]
                if possible_cols:
                    review_col = possible_cols[0]
                else:
                    print(f"   ⚠️  Skipping {Path(csv_file).name} - no review column found")
                    continue
            else:
                review_col = 'review'
            
            # Check for title column
            if 'title' not in df.columns:
                # Try to find title column or use filename
                possible_cols = [col for col in df.columns if 'title' in col.lower()]
                if possible_cols:
                    title_col = possible_cols[0]
                else:
                    # Use filename as title
                    movie_name = Path(csv_file).stem
                    df['title'] = movie_name
                    title_col = 'title'
            else:
                title_col = 'title'
            
            # Combine title and review
            df['text'] = df[title_col].fillna('').astype(str) + ' ' + df[review_col].fillna('').astype(str)
            df['text_length'] = df['text'].str.len()
            df['word_count'] = df['text'].str.split().str.len()
            
            # Remove zero-length texts
            df = df[df['text_length'] > 0]
            
            # Add movie name from filename
            df['movie'] = Path(csv_file).stem
            
            all_reviews.append(df)
            total_reviews += len(df)
            
            if (i % 100 == 0) or (i == len(csv_files)):
                print(f"   Processed {i}/{len(csv_files)} files | Total reviews: {total_reviews:,}")
        
        except Exception as e:
            print(f"   ⚠️  Error processing {Path(csv_file).name}: {e}")
            continue
    
    # Combine all reviews
    print(f"\n📚 Combining all reviews...")
    combined_df = pd.concat(all_reviews, ignore_index=True)
    
    print(f"✓ Combined dataset created")
    print(f"✓ Total reviews: {len(combined_df):,}")
    print(f"✓ Movies: {combined_df['movie'].nunique():,}")
    
    print(f"\n📊 Combined IMDb Reviews Text Statistics:")
    print(f"   Char length - Min: {combined_df['text_length'].min()}, "
          f"Mean: {combined_df['text_length'].mean():.0f}, "
          f"Max: {combined_df['text_length'].max()}")
    print(f"   Word count  - Min: {combined_df['word_count'].min()}, "
          f"Mean: {combined_df['word_count'].mean():.0f}, "
          f"Max: {combined_df['word_count'].max()}")
    
    # Show sample
    print(f"\n📝 Sample review (first 300 chars):")
    sample = combined_df.iloc[0]['text']
    print(f"   {sample[:300]}...")
    
    return combined_df


def sample_based_on_raid(imdb_df: pd.DataFrame, raid_stats: Dict, 
                         target_samples: int = 100000) -> pd.DataFrame:
    """
    Sample IMDb reviews based on RAID characteristics.
    
    Args:
        imdb_df: IMDb reviews DataFrame
        raid_stats: Statistics from RAID dataset
        target_samples: Number of samples to extract
    
    Returns:
        Sampled DataFrame
    """
    print("\n" + "=" * 80)
    print("STEP 3: Sampling IMDb Reviews Based on RAID Characteristics")
    print("=" * 80)
    
    # Define filtering criteria based on RAID statistics
    char_min = max(0, raid_stats['char_length']['mean'] - 2 * raid_stats['char_length']['std'])
    char_max = raid_stats['char_length']['mean'] + 2 * raid_stats['char_length']['std']
    
    word_min = max(0, raid_stats['word_count']['mean'] - 2 * raid_stats['word_count']['std'])
    word_max = raid_stats['word_count']['mean'] + 2 * raid_stats['word_count']['std']
    
    print(f"\n📏 Sampling Criteria (based on RAID mean ± 2σ):")
    print(f"   Character length: {char_min:.0f} - {char_max:.0f} characters")
    print(f"   Word count:       {word_min:.0f} - {word_max:.0f} words")
    print(f"   Target samples:   {target_samples:,}")
    
    # Filter based on criteria
    print(f"\nFiltering reviews dataset...")
    filtered_df = imdb_df[
        (imdb_df['text_length'] >= char_min) &
        (imdb_df['text_length'] <= char_max) &
        (imdb_df['word_count'] >= word_min) &
        (imdb_df['word_count'] <= word_max)
    ].copy()
    
    print(f"✓ Reviews matching criteria: {len(filtered_df):,} / {len(imdb_df):,} "
          f"({len(filtered_df)/len(imdb_df)*100:.1f}%)")
    
    # Sample target number of samples (or all if less than target)
    if len(filtered_df) > target_samples:
        print(f"\nSampling {target_samples:,} reviews randomly...")
        sampled_df = filtered_df.sample(n=target_samples, random_state=42).reset_index(drop=True)
    else:
        print(f"\nUsing all {len(filtered_df):,} matching reviews...")
        sampled_df = filtered_df.copy()
    
    print(f"✓ Final sample size: {len(sampled_df):,}")
    
    # Add metadata
    sampled_df['domain'] = 'reviews'
    sampled_df['source'] = 'imdb_ieee'
    sampled_df['raid_aligned'] = True
    
    # Display final statistics
    print(f"\n📊 Sampled Dataset Statistics:")
    print(f"   Total samples: {len(sampled_df):,}")
    print(f"   Unique movies: {sampled_df['movie'].nunique():,}")
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
    print("STEP 4: Saving Dataset")
    print("=" * 80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    csv_path = output_path / 'imdb_reviews_sampled.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved CSV: {csv_path}")
    print(f"  Size: {csv_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Save as JSON
    json_path = output_path / 'imdb_reviews_sampled.json'
    df.to_json(json_path, orient='records', lines=True)
    print(f"✓ Saved JSON: {json_path}")
    print(f"  Size: {json_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Create summary file
    summary_path = output_path / 'SAMPLING_SUMMARY.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("IMDB MOVIE REVIEWS - RAID ALIGNED SAMPLING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Source: IEEE DataPort IMDb Movie Reviews Dataset\n")
        f.write(f"URL: https://ieee-dataport.org/open-access/imdb-movie-reviews-dataset\n\n")
        f.write(f"Total Samples: {len(df):,}\n")
        f.write(f"Unique Movies: {df['movie'].nunique():,}\n")
        f.write(f"Character Length: {df['text_length'].min()} - {df['text_length'].max()} "
                f"(mean: {df['text_length'].mean():.0f})\n")
        f.write(f"Word Count: {df['word_count'].min()} - {df['word_count'].max()} "
                f"(mean: {df['word_count'].mean():.0f})\n\n")
        f.write("Files:\n")
        f.write("  - imdb_reviews_sampled.csv\n")
        f.write("  - imdb_reviews_sampled.json\n\n")
        f.write("Sampling Strategy:\n")
        f.write("  Combined: title + review\n")
        f.write("  Filter: RAID mean ± 2σ\n")
        f.write("  Random seed: 42\n")
    
    print(f"✓ Created summary: {summary_path}")
    
    print(f"\n✅ Dataset saved successfully to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Process and sample IMDb reviews based on RAID characteristics'
    )
    parser.add_argument(
        '--reviews-dir',
        type=str,
        default='/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/reviews/2_reviews_per_movie_raw',
        help='Directory containing IMDb review CSV files'
    )
    parser.add_argument(
        '--raid-csv',
        type=str,
        default='/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/raid/reviews.csv',
        help='Path to RAID reviews.csv file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/reviews',
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
        help='Target number of review samples to extract (default: 100000)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("IMDB MOVIE REVIEWS PROCESSING AND SAMPLING SCRIPT")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Reviews Dir:     {args.reviews_dir}")
    print(f"  RAID CSV:        {args.raid_csv}")
    print(f"  Output Dir:      {args.output_dir}")
    print(f"  RAID samples:    {args.raid_sample_size:,}")
    print(f"  Target samples:  {args.target_samples:,}")
    
    try:
        # Step 1: Analyze RAID characteristics
        raid_stats = analyze_raid_characteristics(args.raid_csv, args.raid_sample_size)
        
        # Step 2: Load and combine all IMDb reviews
        combined_df = load_and_combine_imdb_reviews(args.reviews_dir)
        
        # Step 3: Sample based on RAID
        sampled_df = sample_based_on_raid(combined_df, raid_stats, args.target_samples)
        
        # Step 4: Save dataset
        save_dataset(sampled_df, args.output_dir)
        
        print("\n" + "=" * 80)
        print("✅ SCRIPT COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nDataset location: {args.output_dir}")
        print(f"Total samples: {len(sampled_df):,}")
        print(f"Unique movies: {sampled_df['movie'].nunique():,}")
        print("\nFiles created:")
        print(f"  - imdb_reviews_sampled.csv")
        print(f"  - imdb_reviews_sampled.json")
        print(f"  - SAMPLING_SUMMARY.txt")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

