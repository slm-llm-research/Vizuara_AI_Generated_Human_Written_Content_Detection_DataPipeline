#!/usr/bin/env python3
"""
Deterministic Dataset Builder for RAID-Aligned AI Detection Research

This script builds balanced datasets of various sizes (10K, 100K, 1M, 2M) with:
- 50% Human / 50% AI split
- Equal domain distribution
- Length-stratified sampling (short, medium, long buckets)
- Weighted model sampling for AI-generated content
- Full reproducibility (fixed random seeds)
"""

import os
import sys
import pandas as pd
import numpy as np
import random
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import warnings
import glob
warnings.filterwarnings('ignore')

# === STEP 1: SET UP DETERMINISM ===
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Model weights for AI sampling
MODEL_WEIGHTS = {
    "gpt4": 0.25,
    "chatgpt": 0.20,
    "gpt3": 0.15,
    "llama-chat": 0.10,
    "mistral-chat": 0.07,
    "mistral": 0.05,
    "mpt-chat": 0.07,
    "mpt": 0.05,
    "cohere-chat": 0.04,
    "cohere": 0.02,
    "gpt2": 0.00  # Excluded
}

# Dataset configurations
DOMAIN_CONFIGS = {
    'abstracts': {
        'raid': '/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/raid/abstracts.csv',
        'augmented': '/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/arxiv-abstracts-2021/csv/train.csv',
        'aug_text_col': 'abstract',  # Column name for text in augmented dataset
        'aug_title_col': 'title'
    },
    'books': {
        'raid': '/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/raid/books.csv',
        'augmented': '/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/booksum/csv/train.csv',
        'aug_text_col': 'summary_text',
        'aug_title_col': 'summary_name'
    },
    'news': {
        'raid': '/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/raid/news.csv',
        'augmented': '/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/news/cnn_dailymail_sampled.csv',
        'aug_text_col': 'combined_text',  # CNN/DailyMail uses combined_text
        'aug_title_col': None  # Already combined
    },
    'poetry': {
        'raid': '/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/raid/poetry.csv',
        'augmented': '/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/poetry_foundation_human_700_1600/poetry_foundation_human_700_1600.csv',
        'aug_text_col': 'text',
        'aug_title_col': 'title'
    },
    'code': {
        'raid': '/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/raid/code.csv',
        'augmented': '/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/the_stack_python_filtered/the_stack_python_filtered.csv',
        'aug_text_col': 'code',
        'aug_title_col': None
    },
    'recipes': {
        'raid': '/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/raid/recipes.csv',
        'augmented': '/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/receipes/recipes_sampled_combined.csv',
        'aug_text_col': 'text',
        'aug_title_col': None  # Already combined
    },
    'reddit': {
        'raid': '/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/raid/reddit.csv',
        'augmented': '/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/reddit/reddit_sampled.csv',
        'aug_text_col': 'text',
        'aug_title_col': None  # Already combined
    },
    'reviews': {
        'raid': '/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/raid/reviews.csv',
        'augmented': '/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/reviews/imdb_reviews_sampled.csv',
        'aug_text_col': 'text',
        'aug_title_col': None  # Already combined
    },
    'wiki': {
        'raid': '/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/raid/wiki.csv',
        'augmented': '/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/wiki/wikipedia_sampled.csv',
        'aug_text_col': 'text',
        'aug_title_col': None  # Already combined
    }
}


def log_progress(message: str, level: str = "INFO"):
    """Log progress with formatting."""
    prefix = {
        "INFO": "ℹ️ ",
        "SUCCESS": "✓",
        "WARNING": "⚠️ ",
        "ERROR": "❌"
    }.get(level, "  ")
    print(f"{prefix} {message}")


def load_raid_dataset(domain: str, config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load RAID dataset and split into Human and AI.
    
    Returns:
        (human_df, ai_df)
    """
    print(f"\n  Loading RAID {domain} dataset...")
    raid_df = pd.read_csv(config['raid'])
    
    # Combine title and generation
    raid_df['text'] = raid_df['title'].fillna('') + ' ' + raid_df['generation'].fillna('')
    raid_df['text_length'] = raid_df['text'].str.len()
    raid_df = raid_df[raid_df['text_length'] > 0]
    
    # Split by label
    if 'model' in raid_df.columns:
        human_mask = (raid_df['model'] == 'human') | (raid_df['model'].isna())
        ai_mask = ~human_mask
    else:
        # Fallback: assume all are AI if no model column
        human_mask = pd.Series([False] * len(raid_df))
        ai_mask = pd.Series([True] * len(raid_df))
    
    human_df = raid_df[human_mask].copy()
    ai_df = raid_df[ai_mask].copy()
    
    human_df['label'] = 'Human_Written'
    human_df['model'] = 'human'
    ai_df['label'] = 'AI_Generated'
    # AI model names already in 'model' column
    
    print(f"    RAID {domain}: {len(human_df):,} Human, {len(ai_df):,} AI")
    
    return human_df, ai_df


def load_augmented_dataset(domain: str, config: Dict) -> pd.DataFrame:
    """Load augmented dataset and create text column."""
    print(f"  Loading augmented {domain} dataset...")
    
    aug_df = pd.read_csv(config['augmented'])
    
    # Create text column
    if config['aug_text_col'] and config['aug_title_col']:
        # Combine title and text
        aug_df['text'] = (aug_df[config['aug_title_col']].fillna('').astype(str) + ' ' +
                         aug_df[config['aug_text_col']].fillna('').astype(str))
    elif config['aug_text_col']:
        # Text already combined
        aug_df['text'] = aug_df[config['aug_text_col']].fillna('').astype(str)
    else:
        raise ValueError(f"No text column configuration for {domain}")
    
    aug_df['text_length'] = aug_df['text'].str.len()
    aug_df = aug_df[aug_df['text_length'] > 0]
    
    aug_df['label'] = 'Human_Written'
    aug_df['model'] = 'human'
    
    print(f"    Augmented {domain}: {len(aug_df):,} Human")
    
    return aug_df


def assign_length_buckets(df: pd.DataFrame, domain: str, label: str) -> pd.DataFrame:
    """
    Assign samples to length buckets (short, medium, long) based on percentiles.
    
    Uses 33rd and 66th percentiles to create three roughly equal buckets.
    """
    if len(df) == 0:
        df['length_bucket'] = []
        return df
    
    # Compute percentiles
    p33 = df['text_length'].quantile(0.33)
    p66 = df['text_length'].quantile(0.66)
    
    # Assign buckets
    df['length_bucket'] = pd.cut(
        df['text_length'],
        bins=[0, p33, p66, float('inf')],
        labels=['short', 'medium', 'long'],
        include_lowest=True
    )
    
    bucket_counts = df['length_bucket'].value_counts()
    print(f"    {domain} {label} buckets: "
          f"short={bucket_counts.get('short', 0):,}, "
          f"medium={bucket_counts.get('medium', 0):,}, "
          f"long={bucket_counts.get('long', 0):,}")
    
    return df


def sample_human_balanced(df: pd.DataFrame, quota: int, domain: str) -> pd.DataFrame:
    """
    Sample human sequences with balanced length distribution.
    
    Args:
        df: DataFrame with length_bucket column
        quota: Total number of samples needed
        domain: Domain name
    
    Returns:
        Sampled DataFrame
    """
    if len(df) < quota:
        log_progress(f"Domain {domain}: Only {len(df):,} human samples available (need {quota:,})", "WARNING")
        return df.copy()
    
    # Target: roughly equal from each bucket
    per_bucket = quota // 3
    remainder = quota % 3
    
    sampled_dfs = []
    
    for bucket_idx, bucket in enumerate(['short', 'medium', 'long']):
        bucket_df = df[df['length_bucket'] == bucket]
        target = per_bucket + (1 if bucket_idx < remainder else 0)
        
        if len(bucket_df) >= target:
            sample = bucket_df.sample(n=target, random_state=RANDOM_SEED + bucket_idx)
        else:
            # Use all available from this bucket
            log_progress(f"{domain} {bucket}: only {len(bucket_df):,} available (need {target:,})", "WARNING")
            sample = bucket_df.copy()
        
        sampled_dfs.append(sample)
    
    result = pd.concat(sampled_dfs, ignore_index=True)
    
    # If we're short, fill from any available
    if len(result) < quota and len(df) > len(result):
        remaining = quota - len(result)
        available = df[~df.index.isin(result.index)]
        if len(available) > 0:
            additional = available.sample(n=min(remaining, len(available)), 
                                         random_state=RANDOM_SEED + 999)
            result = pd.concat([result, additional], ignore_index=True)
    
    return result


def sample_ai_weighted(df: pd.DataFrame, quota: int, domain: str) -> pd.DataFrame:
    """
    Sample AI sequences with weighted model sampling across length buckets.
    
    Args:
        df: DataFrame with length_bucket and model columns
        quota: Total number of AI samples needed
        domain: Domain name
    
    Returns:
        Sampled DataFrame
    """
    if len(df) < quota:
        log_progress(f"Domain {domain}: Only {len(df):,} AI samples available (need {quota:,})", "WARNING")
        return df.copy()
    
    # Get available models
    available_models = df['model'].unique()
    
    # Filter weights to available models with non-zero weight
    active_weights = {model: weight for model, weight in MODEL_WEIGHTS.items() 
                     if model in available_models and weight > 0}
    
    if not active_weights:
        log_progress(f"{domain}: No weighted models available, using uniform sampling", "WARNING")
        return df.sample(n=quota, random_state=RANDOM_SEED)
    
    # Normalize weights
    total_weight = sum(active_weights.values())
    normalized_weights = {model: weight / total_weight for model, weight in active_weights.items()}
    
    # Sample per bucket
    per_bucket = quota // 3
    remainder = quota % 3
    
    sampled_dfs = []
    
    for bucket_idx, bucket in enumerate(['short', 'medium', 'long']):
        bucket_df = df[df['length_bucket'] == bucket]
        bucket_quota = per_bucket + (1 if bucket_idx < remainder else 0)
        
        if len(bucket_df) == 0:
            log_progress(f"{domain} AI {bucket}: no samples", "WARNING")
            continue
        
        # Sample by model weights within this bucket
        bucket_samples = []
        allocated = 0
        
        # Sort models by weight (descending) for allocation
        sorted_models = sorted(normalized_weights.items(), key=lambda x: x[1], reverse=True)
        
        for model, weight in sorted_models:
            model_df = bucket_df[bucket_df['model'] == model]
            
            # Calculate target samples for this model
            target = int(bucket_quota * weight)
            
            if len(model_df) >= target:
                sample = model_df.sample(n=target, random_state=RANDOM_SEED + hash(f"{bucket}{model}") % 10000)
                bucket_samples.append(sample)
                allocated += target
            elif len(model_df) > 0:
                # Use all available and redistribute shortfall
                bucket_samples.append(model_df.copy())
                allocated += len(model_df)
        
        # If we haven't allocated enough, fill from any available in bucket
        if allocated < bucket_quota:
            remaining = bucket_quota - allocated
            already_sampled = pd.concat(bucket_samples) if bucket_samples else pd.DataFrame()
            available = bucket_df[~bucket_df.index.isin(already_sampled.index)] if len(already_sampled) > 0 else bucket_df
            
            if len(available) > 0:
                additional = available.sample(n=min(remaining, len(available)),
                                             random_state=RANDOM_SEED + bucket_idx + 100)
                bucket_samples.append(additional)
        
        if bucket_samples:
            sampled_dfs.append(pd.concat(bucket_samples, ignore_index=True))
    
    if not sampled_dfs:
        log_progress(f"{domain}: No AI samples collected, using fallback", "WARNING")
        return df.sample(n=min(quota, len(df)), random_state=RANDOM_SEED)
    
    result = pd.concat(sampled_dfs, ignore_index=True)
    
    # Ensure we don't exceed quota
    if len(result) > quota:
        result = result.sample(n=quota, random_state=RANDOM_SEED + 500)
    
    # If we're short, fill from any available
    elif len(result) < quota and len(df) > len(result):
        remaining = quota - len(result)
        available = df[~df.index.isin(result.index)]
        if len(available) > 0:
            additional = available.sample(n=min(remaining, len(available)),
                                         random_state=RANDOM_SEED + 600)
            result = pd.concat([result, additional], ignore_index=True)
    
    return result


def build_dataset(target_size: int, domains_to_use: List[str]) -> pd.DataFrame:
    """
    Build a balanced dataset of target size.
    
    Args:
        target_size: Total number of samples (must be even)
        domains_to_use: List of domain names to include
    
    Returns:
        Complete dataset DataFrame
    """
    print("=" * 80)
    print(f"BUILDING DATASET: {target_size:,} samples")
    print("=" * 80)
    
    num_domains = len(domains_to_use)
    quota_per_domain_per_label = target_size // (num_domains * 2)
    
    print(f"\nConfiguration:")
    print(f"  Target size: {target_size:,}")
    print(f"  Domains: {num_domains} ({', '.join(domains_to_use)})")
    print(f"  Quota per domain per label: {quota_per_domain_per_label:,}")
    print(f"  Split: 50% Human / 50% AI")
    
    all_samples = []
    
    # === STEP 2-4: Load and process each domain ===
    for domain in domains_to_use:
        print(f"\n{'─' * 80}")
        print(f"Processing Domain: {domain.upper()}")
        print(f"{'─' * 80}")
        
        config = DOMAIN_CONFIGS[domain]
        
        # Load RAID dataset
        raid_human, raid_ai = load_raid_dataset(domain, config)
        
        # Load augmented dataset
        aug_human = load_augmented_dataset(domain, config)
        
        # === STEP 4: Combine Human datasets ===
        print(f"  Combining human datasets...")
        all_human = pd.concat([raid_human, aug_human], ignore_index=True)
        all_human['domain_name'] = domain
        print(f"    Total human: {len(all_human):,}")
        
        # Prepare AI dataset
        raid_ai['domain_name'] = domain
        print(f"    Total AI: {len(raid_ai):,}")
        
        # === STEP 5: Assign length buckets ===
        print(f"  Assigning length buckets...")
        all_human = assign_length_buckets(all_human, domain, "Human")
        raid_ai = assign_length_buckets(raid_ai, domain, "AI")
        
        # === STEP 6: Sample Human sequences ===
        print(f"  Sampling {quota_per_domain_per_label:,} Human sequences...")
        human_sample = sample_human_balanced(all_human, quota_per_domain_per_label, domain)
        print(f"    Sampled: {len(human_sample):,}")
        
        # === STEP 7: Sample AI sequences with model weights ===
        print(f"  Sampling {quota_per_domain_per_label:,} AI sequences (weighted)...")
        ai_sample = sample_ai_weighted(raid_ai, quota_per_domain_per_label, domain)
        print(f"    Sampled: {len(ai_sample):,}")
        
        # Add to collection
        all_samples.append(human_sample)
        all_samples.append(ai_sample)
    
    # === STEP 8: Assemble final dataset ===
    print(f"\n{'=' * 80}")
    print("ASSEMBLING FINAL DATASET")
    print("=" * 80)
    
    final_df = pd.concat(all_samples, ignore_index=True)
    
    # Select and order columns
    columns_to_keep = ['domain_name', 'model', 'text', 'label', 'length_bucket', 'text_length']
    
    # Add source column if available
    if 'source' in final_df.columns:
        columns_to_keep.append('source')
    
    # Keep only specified columns that exist
    columns_to_keep = [col for col in columns_to_keep if col in final_df.columns]
    final_df = final_df[columns_to_keep]
    
    # Shuffle to mix domains and labels
    final_df = final_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Verify distribution
    print(f"\n✓ Final dataset size: {len(final_df):,}")
    print(f"\n📊 Label Distribution:")
    label_dist = final_df['label'].value_counts()
    for label, count in label_dist.items():
        print(f"  {label}: {count:,} ({count/len(final_df)*100:.1f}%)")
    
    print(f"\n📊 Domain Distribution:")
    domain_dist = final_df['domain_name'].value_counts()
    for domain, count in sorted(domain_dist.items()):
        print(f"  {domain}: {count:,}")
    
    print(f"\n📊 Length Bucket Distribution:")
    bucket_dist = final_df['length_bucket'].value_counts()
    for bucket, count in bucket_dist.items():
        print(f"  {bucket}: {count:,} ({count/len(final_df)*100:.1f}%)")
    
    return final_df


def save_dataset(df: pd.DataFrame, size: int, output_dir: str):
    """Save dataset to CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"datasets_ai_human_text_{size}.csv"
    filepath = output_path / filename
    
    df.to_csv(filepath, index=False)
    file_size = filepath.stat().st_size / 1024 / 1024
    
    log_progress(f"Saved: {filepath} ({file_size:.2f} MB)", "SUCCESS")
    
    return filepath


def upload_to_huggingface(filepath: Path, dataset_name: str, token: str = None):
    """
    Upload dataset to HuggingFace (optional).
    
    Args:
        filepath: Path to CSV file
        dataset_name: HuggingFace repo name (e.g., 'codefactory4791/dataset_10k')
        token: HuggingFace token (optional, will try to use cached token)
    """
    try:
        from huggingface_hub import HfApi, create_repo
        
        print(f"\n  Uploading to HuggingFace: {dataset_name}")
        
        api = HfApi(token=token)
        
        # Create repo if it doesn't exist
        try:
            create_repo(repo_id=dataset_name, repo_type="dataset", private=False, token=token)
            print(f"    Created repo: {dataset_name}")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"    Repo exists: {dataset_name}")
            else:
                raise
        
        # Upload file
        api.upload_file(
            path_or_fileobj=str(filepath),
            path_in_repo=filepath.name,
            repo_id=dataset_name,
            repo_type="dataset",
            token=token
        )
        
        log_progress(f"Uploaded to: https://huggingface.co/datasets/{dataset_name}", "SUCCESS")
        return True
        
    except ImportError:
        log_progress("HuggingFace Hub not installed. Install with: pip install huggingface_hub", "WARNING")
        return False
    except Exception as e:
        log_progress(f"Upload failed: {e}", "ERROR")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Build balanced AI detection datasets with RAID alignment'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/prepared_datasets',
        help='Output directory for prepared datasets'
    )
    parser.add_argument(
        '--sizes',
        type=int,
        nargs='+',
        default=[10000, 100000, 1000000, 2000000],
        help='Dataset sizes to build (default: 10000 100000 1000000 2000000)'
    )
    parser.add_argument(
        '--domains',
        type=str,
        nargs='+',
        default=list(DOMAIN_CONFIGS.keys()),
        help='Domains to include (default: all)'
    )
    parser.add_argument(
        '--upload',
        action='store_true',
        help='Upload datasets to HuggingFace after building'
    )
    parser.add_argument(
        '--hf-token',
        type=str,
        default=None,
        help='HuggingFace token for upload (optional, will use cached if available)'
    )
    parser.add_argument(
        '--hf-org',
        type=str,
        default='codefactory4791',
        help='HuggingFace organization/username (default: codefactory4791)'
    )
    parser.add_argument(
        '--skip-build',
        action='store_true',
        help='Skip building, only upload existing datasets'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("RAID-ALIGNED DATASET BUILDER")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Random Seed: {RANDOM_SEED}")
    print(f"  Output Dir: {args.output_dir}")
    print(f"  Dataset Sizes: {args.sizes}")
    print(f"  Domains: {len(args.domains)} ({', '.join(args.domains)})")
    print(f"  Upload: {args.upload}")
    if args.upload:
        print(f"  HF Organization: {args.hf_org}")
    
    built_files = []
    
    try:
        # === STEP 9: Build datasets of different sizes ===
        if not args.skip_build:
            for size in args.sizes:
                if size % 2 != 0:
                    log_progress(f"Size {size} must be even (for 50-50 split), skipping", "WARNING")
                    continue
                
                print(f"\n{'=' * 80}")
                print(f"BUILDING {size:,} SAMPLE DATASET")
                print(f"{'=' * 80}")
                
                # Build dataset
                dataset = build_dataset(size, args.domains)
                
                # Verify 50-50 split
                label_counts = dataset['label'].value_counts()
                human_pct = label_counts.get('Human_Written', 0) / len(dataset) * 100
                ai_pct = label_counts.get('AI_Generated', 0) / len(dataset) * 100
                
                print(f"\n✓ Final Split: {human_pct:.1f}% Human, {ai_pct:.1f}% AI")
                
                if abs(human_pct - 50.0) > 1.0:
                    log_progress(f"Split imbalance: {human_pct:.1f}% Human (target: 50%)", "WARNING")
                
                # Save dataset
                filepath = save_dataset(dataset, size, args.output_dir)
                built_files.append((filepath, size))
        else:
            # Load existing files
            print("\n  Skip-build mode: Loading existing datasets...")
            output_path = Path(args.output_dir)
            for size in args.sizes:
                filename = f"datasets_ai_human_text_{size}.csv"
                filepath = output_path / filename
                if filepath.exists():
                    built_files.append((filepath, size))
                    print(f"    Found: {filepath}")
                else:
                    log_progress(f"File not found: {filepath}", "WARNING")
        
        # === STEP 10: Upload to HuggingFace (optional) ===
        if args.upload and built_files:
            print(f"\n{'=' * 80}")
            print("UPLOADING TO HUGGINGFACE")
            print(f"{'=' * 80}")
            
            upload_success = []
            upload_failures = []
            
            for filepath, size in built_files:
                dataset_name = f"{args.hf_org}/raid_aligned_{size//1000}k"
                
                success = upload_to_huggingface(filepath, dataset_name, args.hf_token)
                
                if success:
                    upload_success.append(dataset_name)
                else:
                    upload_failures.append(dataset_name)
            
            print(f"\n{'─' * 80}")
            print(f"Upload Summary:")
            print(f"  Successful: {len(upload_success)}")
            print(f"  Failed: {len(upload_failures)}")
            
            if upload_failures:
                print(f"\n  Failed uploads:")
                for name in upload_failures:
                    print(f"    - {name}")
                print(f"\n  Datasets saved locally. You can upload later using --skip-build --upload")
        
        # Final summary
        print(f"\n{'=' * 80}")
        print("✅ DATASET BUILDER COMPLETED")
        print(f"{'=' * 80}")
        print(f"\nDatasets created: {len(built_files)}")
        for filepath, size in built_files:
            print(f"  - {filepath.name} ({size:,} samples)")
        
        print(f"\nLocation: {args.output_dir}")
        
        if args.upload:
            if upload_success:
                print(f"\nUploaded to HuggingFace:")
                for name in upload_success:
                    print(f"  - https://huggingface.co/datasets/{name}")
        
        return 0
        
    except Exception as e:
        log_progress(f"Fatal error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

