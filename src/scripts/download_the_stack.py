#!/usr/bin/env python3
"""
Script to download and filter the bigcode/the-stack dataset.

Filters to Python code with 3-20 lines and 50-400 characters,
matching the RAID code domain distribution (MBPP average: 6.7 lines, 181.1 chars).
"""

import argparse
import uuid
from pathlib import Path
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and filter the-stack dataset for Python code"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/the_stack_python_filtered/the_stack_python_filtered.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50000,
        help="Maximum number of samples to collect (default: 50000)"
    )
    parser.add_argument(
        "--min-lines",
        type=int,
        default=3,
        help="Minimum number of lines (default: 3)"
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=20,
        help="Maximum number of lines (default: 20)"
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=50,
        help="Minimum number of characters (default: 50)"
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=400,
        help="Maximum number of characters (default: 400)"
    )
    return parser.parse_args()

def filter_code_sample(code, min_lines, max_lines, min_chars, max_chars):
    """
    Check if a code sample meets the filtering criteria.
    
    Args:
        code: The code string
        min_lines: Minimum number of lines
        max_lines: Maximum number of lines
        min_chars: Minimum number of characters
        max_chars: Maximum number of characters
    
    Returns:
        tuple: (passes_filter, num_lines, num_chars)
    """
    if not code or not isinstance(code, str):
        return False, 0, 0
    
    # Calculate metrics
    num_lines = len(code.splitlines())
    num_chars = len(code)
    
    # Check if it passes the filter
    passes = (
        min_lines <= num_lines <= max_lines and
        min_chars <= num_chars <= max_chars
    )
    
    return passes, num_lines, num_chars

def download_and_filter_the_stack(args):
    """
    Download and filter the-stack dataset for Python code.
    """
    print("=" * 80)
    print("Downloading and filtering bigcode/the-stack dataset")
    print("=" * 80)
    print(f"\nFiltering criteria:")
    print(f"  - Language: Python")
    print(f"  - Lines: {args.min_lines}-{args.max_lines} (MBPP avg: 6.7)")
    print(f"  - Characters: {args.min_chars}-{args.max_chars} (MBPP avg: 181.1)")
    print(f"  - Max samples: {args.max_samples}")
    print(f"\nOutput path: {args.output}")
    print("\n" + "=" * 80)
    
    # Load dataset with streaming (only Python files)
    print("\n📥 Loading dataset from HuggingFace (streaming mode)...")
    print("   Note: This is a 3TB dataset, so we'll stream and filter on-the-fly")
    
    # Try multiple dataset options in order of preference
    dataset_loaded = False
    is_python_only = False
    
    # Option 1: Try deduplicated version (smaller, faster)
    try:
        print("   Attempting to load the-stack-dedup (deduplicated, smaller version)...")
        ds = load_dataset(
            "bigcode/the-stack-dedup",
            data_dir="data/python",
            split="train",
            streaming=True
        )
        print("✓ Loaded the-stack-dedup successfully (Python subset)")
        is_python_only = True
        dataset_loaded = True
    except Exception as e:
        print(f"   Could not load the-stack-dedup Python subset: {e}")
    
    # Option 2: Try original the-stack with Python subset
    if not dataset_loaded:
        try:
            print("   Attempting to load the-stack Python-specific subset...")
            ds = load_dataset(
                "bigcode/the-stack",
                data_dir="data/python",
                split="train",
                streaming=True
            )
            print("✓ Loaded the-stack successfully (Python subset)")
            is_python_only = True
            dataset_loaded = True
        except Exception as e:
            print(f"   Could not load the-stack Python subset: {e}")
    
    # Option 3: Fall back to full dataset (slowest)
    if not dataset_loaded:
        print("   Loading full the-stack dataset (this may take several minutes)...")
        ds = load_dataset(
            "bigcode/the-stack",
            split="train",
            streaming=True
        )
        print("✓ Loaded the-stack successfully (will filter to Python)")
        is_python_only = False
    
    # Process samples
    filtered_samples = []
    total_processed = 0
    total_python = 0
    
    print(f"\n🔍 Processing samples (target: {args.max_samples} filtered samples)...")
    print("   This may take a while due to streaming...\n")
    
    # Use tqdm for progress bar
    pbar = tqdm(total=args.max_samples, desc="Collecting samples", unit="samples")
    
    for sample in ds:
        total_processed += 1
        
        # Check if it's Python (if we loaded the full dataset)
        if not is_python_only:
            # The field might be 'lang' or 'language'
            lang = sample.get('lang', sample.get('language', '')).lower()
            if 'python' not in lang and lang != 'py':
                continue
        
        total_python += 1
        
        # Get the code content
        code = sample.get('content', '')
        
        # Filter based on line and character count
        passes, num_lines, num_chars = filter_code_sample(
            code,
            args.min_lines,
            args.max_lines,
            args.min_chars,
            args.max_chars
        )
        
        if passes:
            # Generate unique ID
            sample_id = str(uuid.uuid4())
            
            filtered_samples.append({
                'id': sample_id,
                'code': code,
                'num_lines': num_lines,
                'num_chars': num_chars,
                'domain': 'code',
                'language': 'python',
                'source': 'the-stack'
            })
            
            pbar.update(1)
            
            # Stop when we reach the target
            if len(filtered_samples) >= args.max_samples:
                break
        
        # Print progress every 10000 samples
        if total_processed % 10000 == 0:
            pbar.set_postfix({
                'processed': total_processed,
                'python': total_python,
                'filtered': len(filtered_samples)
            })
    
    pbar.close()
    
    # Convert to DataFrame
    df = pd.DataFrame(filtered_samples)
    
    # Calculate statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Total samples processed: {total_processed:,}")
    print(f"Total Python samples: {total_python:,}")
    print(f"Samples retained after filtering: {len(df):,}")
    
    if len(df) > 0:
        print(f"\nCharacter length stats:")
        print(f"  min={df['num_chars'].min()}, max={df['num_chars'].max()}, mean={df['num_chars'].mean():.1f}")
        print(f"\nLine count stats:")
        print(f"  min={df['num_lines'].min()}, max={df['num_lines'].max()}, mean={df['num_lines'].mean():.1f}")
        
        # Create output directory
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"\n✅ Dataset saved to: {output_path}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
    else:
        print("\n⚠️  WARNING: No samples passed the filtering criteria!")
    
    print("=" * 80)
    
    return df

def main():
    """Main entry point."""
    args = parse_args()
    download_and_filter_the_stack(args)

if __name__ == "__main__":
    main()

