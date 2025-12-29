#!/usr/bin/env python3
"""
Script to download the BookSum dataset from HuggingFace.

BookSum is a collection of datasets for long-form narrative summarization
with summaries at paragraph, chapter, and book levels.
"""

import argparse
from pathlib import Path
from datasets import load_dataset
import pandas as pd

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download BookSum dataset from HuggingFace"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset/booksum",
        help="Output directory for the dataset"
    )
    parser.add_argument(
        "--save-format",
        type=str,
        default="all",
        choices=["arrow", "csv", "json", "all"],
        help="Format to save the dataset (default: all)"
    )
    return parser.parse_args()

def download_booksum(args):
    """
    Download the BookSum dataset from HuggingFace.
    """
    print("=" * 80)
    print("Downloading BookSum Dataset from HuggingFace")
    print("=" * 80)
    print(f"\nDataset: salesforce/booksum")
    print(f"Output directory: {args.output_dir}")
    print(f"Save format: {args.save_format}")
    print("\n" + "=" * 80)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n📥 Loading BookSum dataset from HuggingFace...")
    print("   Note: This dataset includes paragraph, chapter, and book-level summaries")
    
    # Load the dataset
    dataset = load_dataset("kmfoda/booksum")
    
    print(f"\n✓ Dataset loaded successfully!")
    print(f"\nDataset structure:")
    for split_name, split_data in dataset.items():
        print(f"  - {split_name}: {len(split_data):,} examples")
        if len(split_data) > 0:
            print(f"    Features: {list(split_data.features.keys())}")
    
    # Save in different formats
    if args.save_format in ["arrow", "all"]:
        print(f"\n💾 Saving in Arrow format...")
        arrow_path = output_path / "arrow"
        dataset.save_to_disk(str(arrow_path))
        print(f"✓ Saved in Arrow format to: {arrow_path}")
    
    if args.save_format in ["csv", "all"]:
        print(f"\n💾 Saving in CSV format...")
        csv_path = output_path / "csv"
        csv_path.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_data in dataset.items():
            csv_file = csv_path / f"{split_name}.csv"
            split_data.to_csv(str(csv_file))
            print(f"✓ Saved {split_name} split to: {csv_file}")
    
    if args.save_format in ["json", "all"]:
        print(f"\n💾 Saving in JSON format...")
        json_path = output_path / "json"
        json_path.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_data in dataset.items():
            json_file = json_path / f"{split_name}.json"
            split_data.to_json(str(json_file))
            print(f"✓ Saved {split_name} split to: {json_file}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    
    for split_name, split_data in dataset.items():
        print(f"\n{split_name.upper()} Split:")
        print(f"  Total examples: {len(split_data):,}")
        
        if len(split_data) > 0:
            # Try to get some statistics
            if 'chapter' in split_data[0]:
                print(f"  Contains: Chapter-level summaries")
            if 'summary_text' in split_data[0]:
                # Calculate average summary length
                sample_summaries = [ex.get('summary_text', '') for ex in split_data.select(range(min(100, len(split_data))))]
                avg_summary_len = sum(len(s) for s in sample_summaries) / len(sample_summaries)
                print(f"  Average summary length: {avg_summary_len:.0f} characters")
            
            # Print first example (truncated)
            print(f"\n  Sample record (first entry):")
            first_record = split_data[0]
            for key, value in first_record.items():
                if isinstance(value, str):
                    if len(value) > 100:
                        print(f"    {key}: {value[:100]}...")
                    else:
                        print(f"    {key}: {value}")
                else:
                    print(f"    {key}: {value}")
    
    print("\n" + "=" * 80)
    print("✅ Download complete!")
    print("=" * 80)
    
    return dataset

def main():
    """Main entry point."""
    args = parse_args()
    download_booksum(args)

if __name__ == "__main__":
    main()

