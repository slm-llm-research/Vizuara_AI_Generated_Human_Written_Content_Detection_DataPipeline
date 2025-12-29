#!/usr/bin/env python3
"""
Script to download the gfissore/arxiv-abstracts-2021 dataset from HuggingFace
and save it to the local dataset directory.
"""

import os
from pathlib import Path
from datasets import load_dataset

def download_arxiv_dataset():
    """
    Download the gfissore/arxiv-abstracts-2021 dataset from HuggingFace
    and save it to the dataset directory.
    """
    # Define paths
    dataset_name = "gfissore/arxiv-abstracts-2021"
    base_dir = Path("/Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Week2_V2/dataset")
    
    # Create folder name from dataset (using the last part of the dataset name)
    folder_name = "arxiv-abstracts-2021"
    save_path = base_dir / folder_name
    
    # Create directory if it doesn't exist
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading dataset: {dataset_name}")
    print(f"Save location: {save_path}")
    
    try:
        # Load the dataset from HuggingFace
        dataset = load_dataset(dataset_name)
        
        print(f"\nDataset loaded successfully!")
        print(f"Dataset structure: {dataset}")
        
        # Save the dataset in different formats
        print(f"\nSaving dataset to {save_path}...")
        
        # Save as Arrow format (native HuggingFace format - most efficient)
        arrow_path = save_path / "arrow"
        dataset.save_to_disk(str(arrow_path))
        print(f"✓ Saved in Arrow format to: {arrow_path}")
        
        # Optionally save each split as CSV for easy access
        csv_path = save_path / "csv"
        csv_path.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_data in dataset.items():
            csv_file = csv_path / f"{split_name}.csv"
            split_data.to_csv(str(csv_file))
            print(f"✓ Saved {split_name} split as CSV to: {csv_file}")
        
        # Optionally save as JSON for easy inspection
        json_path = save_path / "json"
        json_path.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_data in dataset.items():
            json_file = json_path / f"{split_name}.json"
            split_data.to_json(str(json_file))
            print(f"✓ Saved {split_name} split as JSON to: {json_file}")
        
        print(f"\n✅ Dataset download and save completed successfully!")
        print(f"\nDataset information:")
        for split_name, split_data in dataset.items():
            print(f"  - {split_name}: {len(split_data)} examples")
            if len(split_data) > 0:
                print(f"    Features: {split_data.features}")
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        raise

if __name__ == "__main__":
    download_arxiv_dataset()

