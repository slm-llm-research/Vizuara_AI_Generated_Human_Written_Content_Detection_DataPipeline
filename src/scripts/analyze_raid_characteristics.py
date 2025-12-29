#!/usr/bin/env python3
"""
Generic script to analyze RAID dataset characteristics.

This script analyzes any RAID domain dataset (news, recipes, abstracts, reviews, etc.)
by combining 'title' + 'generation' columns and computing comprehensive text statistics.

Output includes:
- JSON file with statistics
- CSV file with detailed metrics
- Text report (markdown)
- Statistical visualizations (optional)
"""

import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
import argparse
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


def analyze_raid_dataset(csv_path: str, sample_size: int = None, 
                         domain_name: str = None) -> Dict:
    """
    Analyze RAID dataset to extract comprehensive text characteristics.
    
    Args:
        csv_path: Path to RAID CSV file
        sample_size: Number of samples to analyze (None = all)
        domain_name: Domain name (auto-detected if None)
    
    Returns:
        Dictionary with comprehensive statistics
    """
    print("=" * 80)
    print("RAID DATASET CHARACTERISTICS ANALYSIS")
    print("=" * 80)
    
    # Extract domain name from filename if not provided
    if domain_name is None:
        domain_name = Path(csv_path).stem  # e.g., 'news', 'recipes', etc.
    
    print(f"\nDataset: {csv_path}")
    print(f"Domain: {domain_name}")
    
    # Read dataset
    if sample_size:
        print(f"Loading {sample_size:,} samples...")
        df = pd.read_csv(csv_path, nrows=sample_size)
    else:
        print(f"Loading entire dataset...")
        df = pd.read_csv(csv_path)
    
    print(f"✓ Loaded {len(df):,} samples")
    print(f"✓ Columns: {list(df.columns)}")
    
    # Check for required columns
    if 'title' not in df.columns or 'generation' not in df.columns:
        print("\n⚠️  Warning: 'title' or 'generation' column not found")
        print(f"Available columns: {list(df.columns)}")
        raise ValueError("Dataset must contain 'title' and 'generation' columns")
    
    # Combine title and generation
    print("\n🔄 Combining 'title' + 'generation' columns...")
    df['combined_text'] = df['title'].fillna('') + ' ' + df['generation'].fillna('')
    df['text_length'] = df['combined_text'].str.len()
    df['word_count'] = df['combined_text'].str.split().str.len()
    
    # Remove any zero-length texts
    original_len = len(df)
    df = df[df['text_length'] > 0]
    if len(df) < original_len:
        print(f"⚠️  Removed {original_len - len(df)} samples with zero length")
    
    # Calculate comprehensive statistics
    print("\n📊 Computing statistics...")
    
    stats = {
        'dataset_info': {
            'file_path': str(csv_path),
            'domain': domain_name,
            'total_samples': len(df),
            'analysis_date': pd.Timestamp.now().isoformat()
        },
        'character_length': {
            'count': int(df['text_length'].count()),
            'min': int(df['text_length'].min()),
            'max': int(df['text_length'].max()),
            'mean': float(df['text_length'].mean()),
            'median': float(df['text_length'].median()),
            'std': float(df['text_length'].std()),
            'q01': float(df['text_length'].quantile(0.01)),
            'q05': float(df['text_length'].quantile(0.05)),
            'q10': float(df['text_length'].quantile(0.10)),
            'q25': float(df['text_length'].quantile(0.25)),
            'q75': float(df['text_length'].quantile(0.75)),
            'q90': float(df['text_length'].quantile(0.90)),
            'q95': float(df['text_length'].quantile(0.95)),
            'q99': float(df['text_length'].quantile(0.99)),
            'variance': float(df['text_length'].var()),
            'skewness': float(df['text_length'].skew()),
            'kurtosis': float(df['text_length'].kurtosis())
        },
        'word_count': {
            'count': int(df['word_count'].count()),
            'min': int(df['word_count'].min()),
            'max': int(df['word_count'].max()),
            'mean': float(df['word_count'].mean()),
            'median': float(df['word_count'].median()),
            'std': float(df['word_count'].std()),
            'q01': float(df['word_count'].quantile(0.01)),
            'q05': float(df['word_count'].quantile(0.05)),
            'q10': float(df['word_count'].quantile(0.10)),
            'q25': float(df['word_count'].quantile(0.25)),
            'q75': float(df['word_count'].quantile(0.75)),
            'q90': float(df['word_count'].quantile(0.90)),
            'q95': float(df['word_count'].quantile(0.95)),
            'q99': float(df['word_count'].quantile(0.99)),
            'variance': float(df['word_count'].var()),
            'skewness': float(df['word_count'].skew()),
            'kurtosis': float(df['word_count'].kurtosis())
        },
        'filtering_criteria': {
            'mean_minus_2std_chars': float(max(0, df['text_length'].mean() - 2 * df['text_length'].std())),
            'mean_plus_2std_chars': float(df['text_length'].mean() + 2 * df['text_length'].std()),
            'mean_minus_2std_words': float(max(0, df['word_count'].mean() - 2 * df['word_count'].std())),
            'mean_plus_2std_words': float(df['word_count'].mean() + 2 * df['word_count'].std()),
            'iqr_lower_chars': float(df['text_length'].quantile(0.25) - 1.5 * (df['text_length'].quantile(0.75) - df['text_length'].quantile(0.25))),
            'iqr_upper_chars': float(df['text_length'].quantile(0.75) + 1.5 * (df['text_length'].quantile(0.75) - df['text_length'].quantile(0.25))),
            'iqr_lower_words': float(df['word_count'].quantile(0.25) - 1.5 * (df['word_count'].quantile(0.75) - df['word_count'].quantile(0.25))),
            'iqr_upper_words': float(df['word_count'].quantile(0.75) + 1.5 * (df['word_count'].quantile(0.75) - df['word_count'].quantile(0.25)))
        },
        'sample_texts': {
            'shortest': {
                'text': str(df.loc[df['text_length'].idxmin(), 'combined_text'][:500]),
                'length': int(df['text_length'].min()),
                'words': int(df.loc[df['text_length'].idxmin(), 'word_count'])
            },
            'median': {
                'text': str(df.loc[(df['text_length'] - df['text_length'].median()).abs().idxmin(), 'combined_text'][:500]),
                'length': int(df.loc[(df['text_length'] - df['text_length'].median()).abs().idxmin(), 'text_length']),
                'words': int(df.loc[(df['text_length'] - df['text_length'].median()).abs().idxmin(), 'word_count'])
            },
            'longest': {
                'text': str(df.loc[df['text_length'].idxmax(), 'combined_text'][:500]),
                'length': int(df['text_length'].max()),
                'words': int(df.loc[df['text_length'].idxmax(), 'word_count'])
            }
        }
    }
    
    # Additional metadata if available
    if 'model' in df.columns:
        model_dist = df['model'].value_counts().to_dict()
        stats['metadata'] = {
            'models': {k: int(v) for k, v in model_dist.items()},
            'num_models': len(model_dist)
        }
    
    if 'domain' in df.columns:
        domain_dist = df['domain'].value_counts().to_dict()
        stats['metadata'] = stats.get('metadata', {})
        stats['metadata']['domains'] = {k: int(v) for k, v in domain_dist.items()}
    
    return stats, df


def print_statistics(stats: Dict):
    """Print formatted statistics to console."""
    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)
    
    info = stats['dataset_info']
    print(f"\n📁 Dataset Information:")
    print(f"   Domain:        {info['domain']}")
    print(f"   Total Samples: {info['total_samples']:,}")
    print(f"   Analysis Date: {info['analysis_date']}")
    
    chars = stats['character_length']
    print(f"\n📏 CHARACTER LENGTH STATISTICS:")
    print(f"   Count:    {chars['count']:>10,}")
    print(f"   Min:      {chars['min']:>10,} characters")
    print(f"   Q01:      {chars['q01']:>10,.0f} characters")
    print(f"   Q05:      {chars['q05']:>10,.0f} characters")
    print(f"   Q10:      {chars['q10']:>10,.0f} characters")
    print(f"   Q25:      {chars['q25']:>10,.0f} characters")
    print(f"   Median:   {chars['median']:>10,.0f} characters")
    print(f"   Mean:     {chars['mean']:>10,.0f} characters")
    print(f"   Q75:      {chars['q75']:>10,.0f} characters")
    print(f"   Q90:      {chars['q90']:>10,.0f} characters")
    print(f"   Q95:      {chars['q95']:>10,.0f} characters")
    print(f"   Q99:      {chars['q99']:>10,.0f} characters")
    print(f"   Max:      {chars['max']:>10,} characters")
    print(f"   Std Dev:  {chars['std']:>10,.0f} characters")
    print(f"   Variance: {chars['variance']:>10,.0f}")
    print(f"   Skewness: {chars['skewness']:>10,.2f}")
    print(f"   Kurtosis: {chars['kurtosis']:>10,.2f}")
    
    words = stats['word_count']
    print(f"\n📝 WORD COUNT STATISTICS:")
    print(f"   Count:    {words['count']:>10,}")
    print(f"   Min:      {words['min']:>10,} words")
    print(f"   Q01:      {words['q01']:>10,.0f} words")
    print(f"   Q05:      {words['q05']:>10,.0f} words")
    print(f"   Q10:      {words['q10']:>10,.0f} words")
    print(f"   Q25:      {words['q25']:>10,.0f} words")
    print(f"   Median:   {words['median']:>10,.0f} words")
    print(f"   Mean:     {words['mean']:>10,.0f} words")
    print(f"   Q75:      {words['q75']:>10,.0f} words")
    print(f"   Q90:      {words['q90']:>10,.0f} words")
    print(f"   Q95:      {words['q95']:>10,.0f} words")
    print(f"   Q99:      {words['q99']:>10,.0f} words")
    print(f"   Max:      {words['max']:>10,} words")
    print(f"   Std Dev:  {words['std']:>10,.0f} words")
    print(f"   Variance: {words['variance']:>10,.0f}")
    print(f"   Skewness: {words['skewness']:>10,.2f}")
    print(f"   Kurtosis: {words['kurtosis']:>10,.2f}")
    
    filt = stats['filtering_criteria']
    print(f"\n🔍 RECOMMENDED FILTERING CRITERIA:")
    print(f"\n   Method 1: Mean ± 2σ (captures ~95% of data)")
    print(f"   Character length: {filt['mean_minus_2std_chars']:.0f} - {filt['mean_plus_2std_chars']:.0f}")
    print(f"   Word count:       {filt['mean_minus_2std_words']:.0f} - {filt['mean_plus_2std_words']:.0f}")
    
    print(f"\n   Method 2: IQR-based (removes outliers)")
    print(f"   Character length: {max(0, filt['iqr_lower_chars']):.0f} - {filt['iqr_upper_chars']:.0f}")
    print(f"   Word count:       {max(0, filt['iqr_lower_words']):.0f} - {filt['iqr_upper_words']:.0f}")
    
    samples = stats['sample_texts']
    print(f"\n📋 SAMPLE TEXTS:")
    print(f"\n   Shortest ({samples['shortest']['length']} chars, {samples['shortest']['words']} words):")
    print(f"   {samples['shortest']['text'][:200]}...")
    
    print(f"\n   Median ({samples['median']['length']} chars, {samples['median']['words']} words):")
    print(f"   {samples['median']['text'][:200]}...")
    
    print(f"\n   Longest ({samples['longest']['length']} chars, {samples['longest']['words']} words):")
    print(f"   {samples['longest']['text'][:200]}...")
    
    if 'metadata' in stats and 'models' in stats['metadata']:
        print(f"\n🤖 MODEL DISTRIBUTION:")
        for model, count in sorted(stats['metadata']['models'].items(), 
                                   key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {model}: {count:,} samples")


def save_results(stats: Dict, df: pd.DataFrame, output_dir: str, domain_name: str):
    """Save analysis results in multiple formats."""
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Save JSON
    json_path = output_path / f'{domain_name}_characteristics.json'
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n✓ Saved JSON: {json_path}")
    
    # 2. Save detailed metrics CSV
    csv_path = output_path / f'{domain_name}_detailed_metrics.csv'
    metrics_df = pd.DataFrame({
        'metric': ['char_min', 'char_q01', 'char_q05', 'char_q10', 'char_q25', 
                   'char_median', 'char_mean', 'char_q75', 'char_q90', 'char_q95', 
                   'char_q99', 'char_max', 'char_std', 'char_variance',
                   'word_min', 'word_q01', 'word_q05', 'word_q10', 'word_q25',
                   'word_median', 'word_mean', 'word_q75', 'word_q90', 'word_q95',
                   'word_q99', 'word_max', 'word_std', 'word_variance'],
        'value': [
            stats['character_length']['min'], stats['character_length']['q01'],
            stats['character_length']['q05'], stats['character_length']['q10'],
            stats['character_length']['q25'], stats['character_length']['median'],
            stats['character_length']['mean'], stats['character_length']['q75'],
            stats['character_length']['q90'], stats['character_length']['q95'],
            stats['character_length']['q99'], stats['character_length']['max'],
            stats['character_length']['std'], stats['character_length']['variance'],
            stats['word_count']['min'], stats['word_count']['q01'],
            stats['word_count']['q05'], stats['word_count']['q10'],
            stats['word_count']['q25'], stats['word_count']['median'],
            stats['word_count']['mean'], stats['word_count']['q75'],
            stats['word_count']['q90'], stats['word_count']['q95'],
            stats['word_count']['q99'], stats['word_count']['max'],
            stats['word_count']['std'], stats['word_count']['variance']
        ]
    })
    metrics_df.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV: {csv_path}")
    
    # 3. Save markdown report
    md_path = output_path / f'{domain_name}_analysis_report.md'
    with open(md_path, 'w') as f:
        f.write(f"# RAID {domain_name.title()} Dataset - Characteristics Analysis\n\n")
        f.write(f"**Analysis Date**: {stats['dataset_info']['analysis_date']}\n")
        f.write(f"**Total Samples**: {stats['dataset_info']['total_samples']:,}\n\n")
        
        f.write("## Character Length Statistics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for key, value in stats['character_length'].items():
            if key not in ['variance', 'skewness', 'kurtosis']:
                f.write(f"| {key.replace('_', ' ').title()} | {value:,.0f} |\n")
        
        f.write("\n## Word Count Statistics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for key, value in stats['word_count'].items():
            if key not in ['variance', 'skewness', 'kurtosis']:
                f.write(f"| {key.replace('_', ' ').title()} | {value:,.0f} |\n")
        
        f.write("\n## Recommended Filtering Criteria\n\n")
        f.write("### Method 1: Mean ± 2σ (captures ~95% of data)\n\n")
        filt = stats['filtering_criteria']
        f.write(f"- **Character Length**: {filt['mean_minus_2std_chars']:.0f} - {filt['mean_plus_2std_chars']:.0f}\n")
        f.write(f"- **Word Count**: {filt['mean_minus_2std_words']:.0f} - {filt['mean_plus_2std_words']:.0f}\n\n")
        
        f.write("### Method 2: IQR-based (removes outliers)\n\n")
        f.write(f"- **Character Length**: {max(0, filt['iqr_lower_chars']):.0f} - {filt['iqr_upper_chars']:.0f}\n")
        f.write(f"- **Word Count**: {max(0, filt['iqr_lower_words']):.0f} - {filt['iqr_upper_words']:.0f}\n\n")
        
        f.write("## Sample Texts\n\n")
        samples = stats['sample_texts']
        f.write(f"### Shortest ({samples['shortest']['length']} chars, {samples['shortest']['words']} words)\n\n")
        f.write(f"```\n{samples['shortest']['text'][:300]}...\n```\n\n")
        
        f.write(f"### Median ({samples['median']['length']} chars, {samples['median']['words']} words)\n\n")
        f.write(f"```\n{samples['median']['text'][:300]}...\n```\n\n")
        
        f.write(f"### Longest ({samples['longest']['length']} chars, {samples['longest']['words']} words)\n\n")
        f.write(f"```\n{samples['longest']['text'][:300]}...\n```\n\n")
    
    print(f"✓ Saved Markdown Report: {md_path}")
    
    # 4. Save distribution data for plotting
    dist_path = output_path / f'{domain_name}_distribution_data.csv'
    dist_df = df[['text_length', 'word_count']].copy()
    dist_df.to_csv(dist_path, index=False)
    print(f"✓ Saved Distribution Data: {dist_path}")
    
    print(f"\n✅ All results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze RAID dataset characteristics (generic for all domains)'
    )
    parser.add_argument(
        'csv_path',
        type=str,
        help='Path to RAID CSV file (e.g., dataset/raid/recipes.csv)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/dataset_characteristics',
        help='Output directory for results'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Number of samples to analyze (None = all samples)'
    )
    parser.add_argument(
        '--domain-name',
        type=str,
        default=None,
        help='Domain name (auto-detected from filename if not provided)'
    )
    
    args = parser.parse_args()
    
    try:
        # Analyze dataset
        stats, df = analyze_raid_dataset(
            args.csv_path, 
            args.sample_size, 
            args.domain_name
        )
        
        # Print statistics
        print_statistics(stats)
        
        # Save results
        domain_name = stats['dataset_info']['domain']
        save_results(stats, df, args.output_dir, domain_name)
        
        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nResults saved to: {args.output_dir}")
        print(f"Domain: {domain_name}")
        print(f"Samples analyzed: {stats['dataset_info']['total_samples']:,}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

