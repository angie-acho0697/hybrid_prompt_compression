#!/usr/bin/env python3
"""
Download script for TokenSkip project models and weights.

This script downloads:
1. Qwen2.5-7B-Instruct base model
2. LLMLingua-2 model weights
3. TokenSkip LoRA adapters for different model sizes
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
import subprocess
import shutil

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error:")
        print(f"Error: {e.stderr}")
        return False

def download_model_from_hf(model_id, local_dir, description):
    """Download a model from Hugging Face Hub."""
    print(f"\n🔄 Downloading {description}...")
    print(f"Model ID: {model_id}")
    print(f"Local directory: {local_dir}")
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"✅ {description} downloaded successfully to {local_dir}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {description}: {str(e)}")
        return False

def check_disk_space(directory, required_gb=50):
    """Check if there's enough disk space."""
    try:
        total, used, free = shutil.disk_usage(directory)
        free_gb = free // (1024**3)
        print(f"💾 Available disk space: {free_gb} GB")
        if free_gb < required_gb:
            print(f"⚠️  Warning: Only {free_gb} GB available, recommended: {required_gb} GB")
            return False
        return True
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
        return True

def main():
    parser = argparse.ArgumentParser(description="Download models for TokenSkip project")
    parser.add_argument("--base-dir", type=str, default="./your_model_path", 
                       help="Base directory to store all models")
    parser.add_argument("--qwen-only", action="store_true", 
                       help="Download only Qwen base model")
    parser.add_argument("--llmlingua-only", action="store_true", 
                       help="Download only LLMLingua-2 model")
    parser.add_argument("--adapters-only", action="store_true", 
                       help="Download only TokenSkip adapters")
    parser.add_argument("--model-size", type=str, default="7b",
                       help="Qwen model size to download (e.g., 1.5b, 3b, 4b, 6b, 7b, 8b, 9b, 12b, 14b, 32b, etc.)")
    parser.add_argument("--skip-disk-check", action="store_true",
                       help="Skip disk space check")
    
    args = parser.parse_args()
    
    # Create base directory
    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 TokenSkip Model Downloader")
    print("=" * 50)
    print(f"Base directory: {base_dir.absolute()}")
    print(f"Requested model size: {args.model_size}")
    
    # Inform about adapter availability
    available_adapters = ["3b", "7b", "14b"]
    if args.model_size.lower() not in available_adapters and not args.qwen_only and not args.llmlingua_only:
        print(f"ℹ️  Note: TokenSkip adapters are only available for {', '.join(available_adapters)} models.")
        print(f"   For {args.model_size} models, you'll need to train your own adapter.")
    
    # Check disk space
    if not args.skip_disk_check:
        check_disk_space(base_dir)
    
    success_count = 0
    total_count = 0
    
    # 1. Download Qwen base model
    if not args.llmlingua_only and not args.adapters_only:
        total_count += 1
        qwen_model_id = f"Qwen/Qwen2.5-{args.model_size.upper()}-Instruct"
        qwen_dir = base_dir / f"Qwen2.5-{args.model_size.upper()}-Instruct"
        
        if download_model_from_hf(qwen_model_id, qwen_dir, f"Qwen2.5-{args.model_size.upper()}-Instruct"):
            success_count += 1
            print(f"📝 Update your eval.sh with: MODEL_PATH=\"{qwen_dir.absolute()}\"")
    
    # 2. Download LLMLingua-2 model
    if not args.qwen_only and not args.adapters_only:
        total_count += 1
        llmlingua_model_id = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
        llmlingua_dir = base_dir / "llmlingua-2-xlm-roberta-large-meetingbank"
        
        if download_model_from_hf(llmlingua_model_id, llmlingua_dir, "LLMLingua-2 model"):
            success_count += 1
            print(f"📝 Update your LLMLingua.py with: llmlingua_path=\"{llmlingua_dir.absolute()}\"")
    
    # 3. Download TokenSkip adapters
    if not args.qwen_only and not args.llmlingua_only:
        # Define available adapter sizes
        available_adapters = ["3b", "7b", "14b"]
        
        # Check if user requested a specific size that has an adapter
        if args.model_size.lower() in available_adapters:
            # Download only the requested adapter
            adapter_id = f"hemingkx/TokenSkip-Qwen2.5-{args.model_size.upper()}-Instruct-GSM8K"
            adapter_dir = base_dir / f"TokenSkip-Qwen2.5-{args.model_size.upper()}-Instruct-GSM8K"
            
            total_count += 1
            if download_model_from_hf(adapter_id, adapter_dir, f"TokenSkip adapter ({args.model_size})"):
                success_count += 1
                print(f"📝 Update your eval.sh with: ADAPTER_PATH=\"{adapter_dir.absolute()}\"")
        else:
            # Download all available adapters if requested size doesn't have an adapter
            print(f"⚠️  No TokenSkip adapter available for {args.model_size}. Downloading all available adapters.")
            adapters = [
                ("hemingkx/TokenSkip-Qwen2.5-3B-Instruct-GSM8K", "3b"),
                ("hemingkx/TokenSkip-Qwen2.5-7B-Instruct-GSM8K", "7b"),
                ("hemingkx/TokenSkip-Qwen2.5-14B-Instruct-GSM8K", "14b"),
            ]
            
            for adapter_id, size in adapters:
                total_count += 1
                adapter_dir = base_dir / f"TokenSkip-Qwen2.5-{size.upper()}-Instruct-GSM8K"
                
                if download_model_from_hf(adapter_id, adapter_dir, f"TokenSkip adapter ({size})"):
                    success_count += 1
                    if size == args.model_size:
                        print(f"📝 Update your eval.sh with: ADAPTER_PATH=\"{adapter_dir.absolute()}\"")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Download Summary")
    print(f"✅ Successfully downloaded: {success_count}/{total_count} models")
    
    if success_count == total_count:
        print("🎉 All models downloaded successfully!")
        print("\n📋 Next steps:")
        print("1. Update the paths in eval.sh with the downloaded model paths")
        print("2. Update the llmlingua_path in LLMLingua.py")
        print("3. Run the evaluation script to generate original CoT outputs")
    else:
        print("⚠️  Some downloads failed. Please check the errors above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
