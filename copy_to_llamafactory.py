#!/usr/bin/env python3
"""
Copy TokenSkip dataset to LLaMA-Factory/data/ directory.

This script copies the generated dataset from outputs/ to LLaMA-Factory/data/
with support for different model names and sizes.
"""

import os
import shutil
import argparse
from pathlib import Path


def get_model_name(model_size):
    """
    Get the full model name based on model size.
    """
    return f"Qwen2.5-{model_size.upper()}-Instruct"


def copy_dataset_to_llamafactory(model_size="7b", llamafactory_dir=None, source_file=None, target_name=None, use_ner_enhanced=False):
    """
    Copy the dataset to LLaMA-Factory/data/ directory.
    
    Args:
        model_size: Model size (e.g., "3b", "7b", "14b")
        llamafactory_dir: Path to LLaMA-Factory directory (default: "../LLaMA-Factory")
        source_file: Source file path (auto-generated if None)
        target_name: Target filename in LLaMA-Factory/data/ (auto-generated if None)
        use_ner_enhanced: Whether to use NER enhanced compression data
    """
    # Get model name
    model_name = get_model_name(model_size)
    
    # Set default paths based on compression type
    if source_file is None:
        compression_type = "ner_enhanced" if use_ner_enhanced else "tokenskip"
        source_file = f"outputs/mydataset_compressed_gsm8k_llmlingua2_qwen_{model_size.upper()}_{compression_type}.json"
    
    if llamafactory_dir is None:
        llamafactory_dir = "./LLaMA-Factory"
    
    if target_name is None:
        compression_type = "ner_enhanced" if use_ner_enhanced else "tokenskip"
        target_name = f"mydataset_compressed_gsm8k_llmlingua2_qwen_{model_size.upper()}_{compression_type}.json"
    
    # Convert to Path objects
    source_path = Path(source_file)
    llamafactory_data_dir = Path(llamafactory_dir) / "data"
    target_path = llamafactory_data_dir / target_name
    
    compression_type = "NER Enhanced" if use_ner_enhanced else "Standard TokenSkip"
    print("🚀 TokenSkip Dataset Copier")
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"Model size: {model_size}")
    print(f"Compression type: {compression_type}")
    print(f"Source file: {source_path.absolute()}")
    print(f"Target file: {target_path.absolute()}")
    
    # Check if source file exists
    if not source_path.exists():
        print(f"❌ Error: Source file not found: {source_path.absolute()}")
        print("Please run get_llamafactory_input.py first to generate the dataset.")
        return False
    
    # Check if LLaMA-Factory directory exists
    if not Path(llamafactory_dir).exists():
        print(f"❌ Error: LLaMA-Factory directory not found: {Path(llamafactory_dir).absolute()}")
        print("Please clone LLaMA-Factory first:")
        print(f"git clone https://github.com/hiyouga/LLaMA-Factory.git {llamafactory_dir}")
        return False
    
    # Check if LLaMA-Factory/data directory exists
    if not llamafactory_data_dir.exists():
        print(f"❌ Error: LLaMA-Factory/data directory not found: {llamafactory_data_dir.absolute()}")
        print("Please make sure the data directory exists in your LLaMA-Factory installation.")
        return False
    
    # Copy the file
    try:
        shutil.copy2(source_path, target_path)
        print(f"✅ Successfully copied dataset to: {target_path.absolute()}")
        
        # Get file size
        file_size = target_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        print(f"📊 File size: {file_size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error copying file: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Copy TokenSkip dataset to LLaMA-Factory/data/")
    parser.add_argument("--model-size", type=str, default="7b",
                       help="Qwen model size (e.g., 1.5b, 3b, 4b, 6b, 7b, 8b, 9b, 12b, 14b, 32b, etc.)")
    parser.add_argument("--llamafactory-dir", type=str, default="./LLaMA-Factory",
                       help="Path to LLaMA-Factory directory")
    parser.add_argument("--source-file", type=str, default=None,
                       help="Source file path (auto-generated if not specified)")
    parser.add_argument("--target-name", type=str, default=None,
                       help="Target filename in LLaMA-Factory/data/ (auto-generated if not specified)")
    parser.add_argument("--use-ner-enhanced", action="store_true",
                       help="Use NER enhanced compression data instead of standard TokenSkip compression")
    
    args = parser.parse_args()
    
    # Copy the dataset
    success = copy_dataset_to_llamafactory(
        model_size=args.model_size,
        llamafactory_dir=args.llamafactory_dir,
        source_file=args.source_file,
        target_name=args.target_name,
        use_ner_enhanced=args.use_ner_enhanced
    )
    
    if success:
        print("\n📋 Next steps:")
        print("1. Register the dataset in LLaMA-Factory/data/dataset_info.json")
        print("2. Update your training config to use the dataset")
        print("3. Run the training command")
    else:
        print("\n❌ Failed to copy dataset. Please check the error messages above.")


if __name__ == "__main__":
    main()
