#!/usr/bin/env python3
"""
Build Train/Validation/Test Splits for RAID-Aligned Datasets

Creates deterministic, stratified splits for 10K, 100K, 1M, and 2M datasets.
For 2M dataset, creates additional 4 cross-validation folds for GPT prompt tuning.

Note: Validation/test size for 2M is capped at 30k each due to GPT evaluation
budget; sampling is stratified by domain×label×length_bucket to preserve
representativeness.

Author: Dataset Preparation Pipeline
Date: December 27, 2025
Seed: 42 (deterministic)
"""

import os
import sys
import pandas as pd
import numpy as np
import hashlib
import json
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Fixed seed for reproducibility
SEED = 42
np.random.seed(SEED)


def create_text_hash(row: pd.Series) -> str:
    """
    Create SHA256 hash for uniqueness checking.
    Hash = sha256(domain_name|label|model|text)
    """
    hash_input = f"{row['domain_name']}|{row['label']}|{row['model']}|{row['text']}"
    return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()


def compute_length_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute length buckets (short/medium/long) per domain and label.
    Uses 33rd and 66th percentiles.
    """
    if 'length_bucket' in df.columns:
        print("  ✓ length_bucket column already exists")
        return df
    
    print("  Computing length buckets...")
    
    # Add text length if missing
    if 'text_len_chars' not in df.columns:
        df['text_len_chars'] = df['text'].str.len()
    
    # Compute buckets per domain and label
    def assign_bucket(group):
        if len(group) < 3:
            group['length_bucket'] = 'medium'
            return group
        
        p33 = group['text_len_chars'].quantile(0.33)
        p66 = group['text_len_chars'].quantile(0.66)
        
        group['length_bucket'] = pd.cut(
            group['text_len_chars'],
            bins=[0, p33, p66, float('inf')],
            labels=['short', 'medium', 'long'],
            include_lowest=True
        )
        return group
    
    df = df.groupby(['domain_name', 'label'], group_keys=False).apply(assign_bucket)
    
    print(f"  ✓ Bucket distribution: {df['length_bucket'].value_counts().to_dict()}")
    
    return df


def stratified_split(df: pd.DataFrame, test_size: float, val_size: float, 
                     seed: int = SEED) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform stratified split into train/val/test.
    Stratifies by domain_name, label, and length_bucket.
    """
    from sklearn.model_selection import train_test_split
    
    # Create stratification key
    df['_strat_key'] = (df['domain_name'].astype(str) + '_' + 
                       df['label'].astype(str) + '_' + 
                       df['length_bucket'].astype(str))
    
    # First split: separate test set
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df['_strat_key'],
        random_state=seed
    )
    
    # Second split: separate validation from training
    val_ratio = val_size / (1 - test_size)  # Adjust ratio for remaining data
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        stratify=train_val['_strat_key'],
        random_state=seed + 1
    )
    
    # Drop stratification key
    train = train.drop('_strat_key', axis=1)
    val = val.drop('_strat_key', axis=1)
    test = test.drop('_strat_key', axis=1)
    
    return train, val, test


def create_cv_folds(validation_df: pd.DataFrame, n_folds: int = 4, 
                    seed: int = SEED) -> List[pd.DataFrame]:
    """
    Split validation set into n non-overlapping stratified folds.
    """
    from sklearn.model_selection import StratifiedKFold
    
    # Create stratification key
    strat_key = (validation_df['domain_name'].astype(str) + '_' + 
                validation_df['label'].astype(str) + '_' + 
                validation_df['length_bucket'].astype(str))
    
    # Create folds
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    folds = []
    for fold_idx, (_, fold_indices) in enumerate(skf.split(validation_df, strat_key)):
        fold_df = validation_df.iloc[fold_indices].copy()
        folds.append(fold_df)
    
    return folds


def validate_splits(splits_dict: Dict[str, pd.DataFrame], dataset_size: str):
    """
    Comprehensive validation of splits.
    """
    print(f"\n{'─' * 80}")
    print(f"VALIDATION CHECKS FOR {dataset_size}")
    print(f"{'─' * 80}")
    
    # Extract splits
    train = splits_dict['train']
    val = splits_dict['validation']
    test = splits_dict['test']
    
    all_splits = [train, val, test]
    split_names = ['train', 'validation', 'test']
    
    # Check 1: No overlap in text_hash
    print("\n1. Checking for overlaps...")
    train_hashes = set(train['text_hash'])
    val_hashes = set(val['text_hash'])
    test_hashes = set(test['text_hash'])
    
    train_val_overlap = train_hashes.intersection(val_hashes)
    train_test_overlap = train_hashes.intersection(test_hashes)
    val_test_overlap = val_hashes.intersection(test_hashes)
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print(f"  ❌ OVERLAP FOUND!")
        print(f"     Train-Val: {len(train_val_overlap)}")
        print(f"     Train-Test: {len(train_test_overlap)}")
        print(f"     Val-Test: {len(val_test_overlap)}")
        raise ValueError("Splits have overlapping samples!")
    else:
        print(f"  ✓ No overlaps between splits")
    
    # Check 2: Label balance
    print("\n2. Checking label balance...")
    for split_name, split_df in zip(split_names, all_splits):
        label_counts = split_df['label'].value_counts()
        human_pct = label_counts.get('Human_Written', 0) / len(split_df) * 100
        ai_pct = label_counts.get('AI_Generated', 0) / len(split_df) * 100
        print(f"  {split_name}: {human_pct:.1f}% Human, {ai_pct:.1f}% AI")
    
    # Check 3: Domain distribution
    print("\n3. Checking domain distribution...")
    for split_name, split_df in zip(split_names, all_splits):
        domain_counts = split_df['domain_name'].value_counts()
        print(f"  {split_name}: {len(domain_counts)} domains, "
              f"range: {domain_counts.min()}-{domain_counts.max()} samples per domain")
    
    # Check 4: Length bucket distribution
    print("\n4. Checking length bucket distribution...")
    for split_name, split_df in zip(split_names, all_splits):
        bucket_counts = split_df['length_bucket'].value_counts()
        total = len(split_df)
        print(f"  {split_name}: " + 
              ", ".join([f"{bucket}={count/total*100:.1f}%" 
                        for bucket, count in bucket_counts.items()]))
    
    # Check 5: For 2M, validate folds
    if 'val_fold_1' in splits_dict:
        print("\n5. Checking 2M validation folds...")
        folds = [splits_dict[f'val_fold_{i+1}'] for i in range(4)]
        
        # Check non-overlapping
        all_fold_hashes = []
        for i, fold in enumerate(folds, 1):
            fold_hashes = set(fold['text_hash'])
            all_fold_hashes.append(fold_hashes)
            print(f"  Fold {i}: {len(fold):,} samples")
        
        # Check overlaps
        for i in range(len(folds)):
            for j in range(i+1, len(folds)):
                overlap = all_fold_hashes[i].intersection(all_fold_hashes[j])
                if overlap:
                    print(f"  ❌ Overlap between fold {i+1} and {j+1}: {len(overlap)}")
                    raise ValueError(f"Folds {i+1} and {j+1} overlap!")
        
        print(f"  ✓ All folds are non-overlapping")
        
        # Check union equals validation_full
        all_fold_union = set().union(*all_fold_hashes)
        val_full_hashes = set(val['text_hash'])
        if all_fold_union == val_full_hashes:
            print(f"  ✓ Folds union equals validation_full exactly")
        else:
            diff = len(all_fold_union.symmetric_difference(val_full_hashes))
            print(f"  ❌ Folds union differs from validation_full by {diff} samples")
    
    print(f"\n✓ All validation checks passed")


def create_metadata(splits_dict: Dict[str, pd.DataFrame], dataset_size: str, 
                   split_config: Dict) -> Dict:
    """
    Create comprehensive metadata for splits.
    """
    metadata = {
        'dataset_size': dataset_size,
        'seed': SEED,
        'stratification_keys': ['domain_name', 'label', 'length_bucket'],
        'split_configuration': split_config,
        'splits': {}
    }
    
    for split_name, split_df in splits_dict.items():
        split_meta = {
            'total_samples': len(split_df),
            'label_distribution': split_df['label'].value_counts().to_dict(),
            'domain_distribution': split_df['domain_name'].value_counts().to_dict(),
            'length_bucket_distribution': split_df['length_bucket'].value_counts().to_dict(),
            'per_domain_label_counts': {},
            'ai_model_distribution': {}
        }
        
        # Per-domain label counts
        for domain in split_df['domain_name'].unique():
            domain_df = split_df[split_df['domain_name'] == domain]
            split_meta['per_domain_label_counts'][domain] = {
                'AI_Generated': len(domain_df[domain_df['label'] == 'AI_Generated']),
                'Human_Written': len(domain_df[domain_df['label'] == 'Human_Written'])
            }
        
        # AI model distribution
        ai_df = split_df[split_df['label'] == 'AI_Generated']
        if len(ai_df) > 0:
            split_meta['ai_model_distribution'] = ai_df['model'].value_counts().to_dict()
        
        metadata['splits'][split_name] = split_meta
    
    return metadata


def process_dataset(input_path: Path, output_dir: Path, dataset_size: str,
                   split_config: Dict, push_to_hub: bool = True,
                   hf_user: str = 'codefactory4791', hf_token: str = None):
    """
    Process one dataset: create splits, save, and optionally push to HuggingFace.
    """
    print(f"\n{'=' * 80}")
    print(f"PROCESSING {dataset_size.upper()} DATASET")
    print(f"{'=' * 80}")
    
    # Load dataset
    print(f"\n Loading {input_path.name}...")
    df = pd.read_csv(input_path)
    print(f"  ✓ Loaded {len(df):,} samples")
    
    # Add text_len_chars if missing
    if 'text_len_chars' not in df.columns:
        df['text_len_chars'] = df['text'].str.len()
    
    # Compute length buckets if missing
    df = compute_length_buckets(df)
    
    # Create text hash for uniqueness
    print(f"  Creating text hashes...")
    df['text_hash'] = df.apply(create_text_hash, axis=1)
    
    # Deduplicate by text_hash
    original_len = len(df)
    df = df.drop_duplicates(subset='text_hash', keep='first')
    if len(df) < original_len:
        print(f"  Removed {original_len - len(df):,} duplicates")
    
    # Create output directory
    output_path = output_dir / dataset_size
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Perform splits based on dataset size
    print(f"\n  Splitting dataset...")
    
    if dataset_size == '2m':
        # Special handling for 2M dataset
        test_size = 30000 / len(df)
        val_size = 30000 / len(df)
        
        train, val, test = stratified_split(df, test_size, val_size, SEED)
        
        splits_dict = {
            'train': train,
            'validation': val,
            'test': test,
            'validation_full': val.copy(),
            'test_full': test.copy()
        }
        
        # Create 4 CV folds from validation
        print(f"  Creating 4 cross-validation folds...")
        folds = create_cv_folds(val, n_folds=4, seed=SEED)
        for i, fold in enumerate(folds, 1):
            splits_dict[f'val_fold_{i}'] = fold
    
    else:
        # Standard splits
        test_pct = split_config['test']
        val_pct = split_config['validation']
        
        train, val, test = stratified_split(df, test_pct, val_pct, SEED)
        
        splits_dict = {
            'train': train,
            'validation': val,
            'test': test
        }
    
    # Print split sizes
    print(f"\n  Split sizes:")
    for split_name, split_df in splits_dict.items():
        print(f"    {split_name}: {len(split_df):,}")
    
    # Validate splits
    validate_splits(splits_dict, dataset_size)
    
    # Save splits locally
    print(f"\n  Saving splits to {output_path}...")
    for split_name, split_df in splits_dict.items():
        # Remove text_hash before saving (internal only)
        split_to_save = split_df.drop('text_hash', axis=1)
        
        csv_path = output_path / f"{split_name}.csv"
        split_to_save.to_csv(csv_path, index=False)
        file_size = csv_path.stat().st_size / 1024 / 1024
        print(f"    ✓ {split_name}.csv ({file_size:.2f} MB)")
    
    # Create and save metadata
    metadata = create_metadata(splits_dict, dataset_size, split_config)
    metadata_path = output_path / 'splits_metadata.json'
    
    # Add note about 2M validation/test cap
    if dataset_size == '2m':
        metadata['note'] = ("Validation/test size for 2M is capped at 30k each due to "
                           "GPT evaluation budget; sampling is stratified by "
                           "domain×label×length_bucket to preserve representativeness.")
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"    ✓ splits_metadata.json")
    
    # Push to HuggingFace if requested
    if push_to_hub and hf_token:
        push_to_huggingface(splits_dict, dataset_size, hf_user, hf_token, output_path)
    
    return splits_dict, metadata


def push_to_huggingface(splits_dict: Dict[str, pd.DataFrame], dataset_size: str,
                       hf_user: str, hf_token: str, local_path: Path):
    """
    Push dataset splits to HuggingFace Hub.
    """
    try:
        from datasets import Dataset, DatasetDict
        from huggingface_hub import HfApi
        
        print(f"\n  {'─' * 76}")
        print(f"  UPLOADING TO HUGGINGFACE")
        print(f"  {'─' * 76}")
        
        # Map size to repo name
        size_map = {
            '10k': '10k',
            '100k': '100k',
            '1m': '1000k',
            '2m': '2000k'
        }
        repo_name = f"{hf_user}/raid_aligned_{size_map[dataset_size]}"
        
        print(f"  Repository: {repo_name}")
        
        # Prepare DatasetDict
        dataset_dict = {}
        
        # Main splits
        for split_name in ['train', 'validation', 'test']:
            if split_name in splits_dict:
                # Remove text_hash
                df = splits_dict[split_name].drop('text_hash', axis=1, errors='ignore')
                dataset_dict[split_name] = Dataset.from_pandas(df, preserve_index=False)
                print(f"    Prepared {split_name}: {len(df):,} samples")
        
        # For 2M, add folds as separate splits
        if dataset_size == '2m':
            for i in range(1, 5):
                fold_name = f'val_fold_{i}'
                if fold_name in splits_dict:
                    df = splits_dict[fold_name].drop('text_hash', axis=1, errors='ignore')
                    dataset_dict[fold_name] = Dataset.from_pandas(df, preserve_index=False)
                    print(f"    Prepared {fold_name}: {len(df):,} samples")
        
        # Create DatasetDict
        ds_dict = DatasetDict(dataset_dict)
        
        # Push to hub
        print(f"\n  Pushing to HuggingFace Hub...")
        ds_dict.push_to_hub(
            repo_id=repo_name,
            token=hf_token,
            private=False
        )
        
        print(f"  ✓ Successfully uploaded to: https://huggingface.co/datasets/{repo_name}")
        
        return True
        
    except ImportError as e:
        print(f"  ⚠️  HuggingFace libraries not available: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Upload failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Build train/validation/test splits for RAID-aligned datasets'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/prepared_datasets',
        help='Directory containing prepared datasets'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/prepared_splits',
        help='Output directory for splits'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--push-to-hub',
        action='store_true',
        default=True,
        help='Push splits to HuggingFace Hub (default: True)'
    )
    parser.add_argument(
        '--no-push',
        action='store_true',
        help='Disable pushing to HuggingFace Hub'
    )
    parser.add_argument(
        '--hf-user',
        type=str,
        default='codefactory4791',
        help='HuggingFace username (default: codefactory4791)'
    )
    parser.add_argument(
        '--hf-token',
        type=str,
        default=None,
        help='HuggingFace token (or use HF_TOKEN env var)'
    )
    
    args = parser.parse_args()
    
    # Handle push flag
    push_to_hub = args.push_to_hub and not args.no_push
    
    # Get HF token
    hf_token = args.hf_token or os.environ.get('HF_TOKEN')
    
    if push_to_hub and not hf_token:
        print("⚠️  Warning: --push-to-hub enabled but no token provided")
        print("   Set HF_TOKEN environment variable or use --hf-token")
        push_to_hub = False
    
    # Update global seed
    global SEED
    SEED = args.seed
    np.random.seed(SEED)
    
    print("\n" + "=" * 80)
    print("RAID-ALIGNED DATASET SPLITS BUILDER")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Input Dir:  {args.input_dir}")
    print(f"  Output Dir: {args.output_dir}")
    print(f"  Seed:       {SEED}")
    print(f"  Push to Hub: {push_to_hub}")
    if push_to_hub:
        print(f"  HF User:    {args.hf_user}")
    
    # Dataset configurations
    datasets_config = {
        '10k': {
            'file': 'datasets_ai_human_text_10000.csv',
            'train': 0.80,
            'validation': 0.10,
            'test': 0.10
        },
        '100k': {
            'file': 'datasets_ai_human_text_100000.csv',
            'train': 0.85,
            'validation': 0.075,
            'test': 0.075
        },
        '1m': {
            'file': 'datasets_ai_human_text_1000000.csv',
            'train': 0.90,
            'validation': 0.05,
            'test': 0.05
        },
        '2m': {
            'file': 'datasets_ai_human_text_2000000.csv',
            'train': None,  # Computed after val/test
            'validation': 30000,  # Absolute count
            'test': 30000  # Absolute count
        }
    }
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each dataset
    all_results = {}
    
    for dataset_size, config in datasets_config.items():
        input_path = input_dir / config['file']
        
        if not input_path.exists():
            print(f"\n⚠️  Skipping {dataset_size}: {input_path.name} not found")
            continue
        
        try:
            splits_dict, metadata = process_dataset(
                input_path, output_dir, dataset_size, config,
                push_to_hub, args.hf_user, hf_token
            )
            all_results[dataset_size] = {
                'splits': splits_dict,
                'metadata': metadata,
                'success': True
            }
        except Exception as e:
            print(f"\n❌ Error processing {dataset_size}: {e}")
            import traceback
            traceback.print_exc()
            all_results[dataset_size] = {
                'success': False,
                'error': str(e)
            }
    
    # Final summary
    print(f"\n{'=' * 80}")
    print(f"FINAL SUMMARY")
    print(f"{'=' * 80}")
    
    successful = [k for k, v in all_results.items() if v.get('success')]
    failed = [k for k, v in all_results.items() if not v.get('success')]
    
    print(f"\n✓ Successfully processed: {len(successful)}/{len(datasets_config)}")
    for size in successful:
        print(f"  - {size}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}")
        for size in failed:
            print(f"  - {size}: {all_results[size].get('error', 'Unknown error')}")
    
    print(f"\nOutput location: {output_dir}")
    
    if push_to_hub and hf_token:
        print(f"\nHuggingFace Repositories:")
        size_map = {'10k': '10k', '100k': '100k', '1m': '1000k', '2m': '2000k'}
        for size in successful:
            repo_name = f"{args.hf_user}/raid_aligned_{size_map[size]}"
            print(f"  - https://huggingface.co/datasets/{repo_name}")
    
    print(f"\n{'=' * 80}")
    print(f"✅ SPLITS BUILDER COMPLETED")
    print(f"{'=' * 80}")
    
    return 0 if not failed else 1


if __name__ == '__main__':
    exit(main())

