#!/usr/bin/env python3
"""
Register TokenSkip dataset in LLaMA-Factory's dataset_info.json.

This script adds the TokenSkip dataset entry to the dataset_info.json file
so it can be used for training.
"""

import json
import argparse
from pathlib import Path


def get_model_name(model_size):
    """
    Get the full model name based on model size.
    """
    return f"Qwen2.5-{model_size.upper()}-Instruct"


def register_dataset(model_size="7b", llamafactory_dir="./LLaMA-Factory", use_ner_enhanced=False):
    """
    Register the TokenSkip dataset in dataset_info.json.
    
    Args:
        model_size: Model size (e.g., "3b", "7b", "14b")
        llamafactory_dir: Path to LLaMA-Factory directory
        use_ner_enhanced: Whether to use NER enhanced compression data
    """
    # Get model name
    model_name = get_model_name(model_size)
    
    # Set paths based on compression type
    compression_type = "ner_enhanced" if use_ner_enhanced else "tokenskip"
    dataset_info_path = Path(llamafactory_dir) / "data" / "dataset_info.json"
    dataset_filename = f"mydataset_compressed_gsm8k_llmlingua2_qwen_{model_size.upper()}_{compression_type}.json"
    dataset_key = dataset_filename.replace('.json', '')  # Use filename without extension as key
    
    compression_type_display = "NER Enhanced" if use_ner_enhanced else "Standard TokenSkip"
    print("🚀 TokenSkip Dataset Registration")
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"Model size: {model_size}")
    print(f"Compression type: {compression_type_display}")
    print(f"Dataset file: {dataset_filename}")
    print(f"Dataset key: {dataset_key}")
    print(f"Dataset info path: {dataset_info_path.absolute()}")
    
    # Check if dataset_info.json exists
    if not dataset_info_path.exists():
        print(f"❌ Error: dataset_info.json not found: {dataset_info_path.absolute()}")
        return False
    
    # Check if dataset file exists
    dataset_file_path = Path(llamafactory_dir) / "data" / dataset_filename
    if not dataset_file_path.exists():
        print(f"❌ Error: Dataset file not found: {dataset_file_path.absolute()}")
        print("Please run copy_to_llamafactory.py first to copy the dataset.")
        return False
    
    # Read existing dataset_info.json
    try:
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)
    except Exception as e:
        print(f"❌ Error reading dataset_info.json: {str(e)}")
        return False
    
    # Check if dataset is already registered
    if dataset_key in dataset_info:
        print(f"⚠️  Dataset '{dataset_key}' is already registered.")
        response = input("Do you want to update it? (y/N): ")
        if response.lower() != 'y':
            print("Registration cancelled.")
            return True
    
    # Create dataset entry
    dataset_entry = {
        "file_name": dataset_filename,
        "columns": {
            "prompt": "instruction",
            "query": "input", 
            "response": "output"
        }
    }
    
    # Add to dataset_info
    dataset_info[dataset_key] = dataset_entry
    
    # Write back to file
    try:
        with open(dataset_info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        print(f"✅ Successfully registered dataset '{dataset_key}' in dataset_info.json")
        return True
    except Exception as e:
        print(f"❌ Error writing dataset_info.json: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Register TokenSkip dataset in LLaMA-Factory")
    parser.add_argument("--model-size", type=str, default="7b",
                       help="Qwen model size (e.g., 1.5b, 3b, 4b, 6b, 7b, 8b, 9b, 12b, 14b, 32b, etc.)")
    parser.add_argument("--llamafactory-dir", type=str, default="./LLaMA-Factory",
                       help="Path to LLaMA-Factory directory")
    parser.add_argument("--use-ner-enhanced", action="store_true",
                       help="Use NER enhanced compression data instead of standard TokenSkip compression")
    
    args = parser.parse_args()
    
    # Register the dataset
    success = register_dataset(
        model_size=args.model_size,
        llamafactory_dir=args.llamafactory_dir,
        use_ner_enhanced=args.use_ner_enhanced
    )
    
    if success:
        print("\n📋 Next steps:")
        print("1. Update your training config to use the dataset")
        print("2. Run the training command")
        compression_suffix = "ner_enhanced" if args.use_ner_enhanced else "tokenskip"
        print(f"\nExample training config entry:")
        print(f"dataset: mydataset_compressed_gsm8k_llmlingua2_qwen_{args.model_size.upper()}_{compression_suffix}")
    else:
        print("\n❌ Failed to register dataset. Please check the error messages above.")


if __name__ == "__main__":
    main()
