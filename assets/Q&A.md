# Q&A

This page contains frequently asked questions about the re-implementation of TokenSkip.

## Model Download Issues

#### 1. Download Script Fails with Network Error

**Q**: The download script fails with network timeout or connection errors. What should I do?

**A**: This is usually due to network instability or firewall restrictions. Try these solutions:

1. **Use a VPN** if you're behind a restrictive firewall
2. **Retry the download** - the script will automatically retry on network failures
3. **Download manually** from the Hugging Face links provided in the README
4. **Use a different network** or try during off-peak hours

#### 2. Insufficient Disk Space

**Q**: I don't have enough disk space for the models. What are my options?

**A**: You have several options:

1. **Free up space** by removing unnecessary files
2. **Use `--skip-disk-check`** flag to bypass the disk space check
3. **Download to external storage** using `--base-dir /path/to/external/drive`
4. **Download only essential models** using `--qwen-only` or `--llmlingua-only`

#### 3. Permission Errors During Download

**Q**: I get permission errors when trying to download models. How to fix this?

**A**: This is usually a file system permission issue:

1. **Check write permissions** to the download directory
2. **Run as administrator** (Windows) or use `sudo` (Linux/Mac) if needed
3. **Change download directory** to a location where you have write access
4. **Check antivirus software** - some may block large downloads

#### 4. Model Size Confusion

**Q**: The actual downloaded model size is different from what's listed. Why?

**A**: Model sizes can vary due to:

1. **Different model formats** (safetensors vs pytorch)
2. **Additional files** (tokenizers, configs, etc.)
3. **Compression** - some models are compressed during download
4. **Version differences** - newer model versions may have different sizes

#### 5. Hugging Face Authentication Issues

**Q**: I'm getting authentication errors when downloading models. What's wrong?

**A**: Some models may require authentication:

1. **Login to Hugging Face**: `huggingface-cli login`
2. **Accept model terms** on the Hugging Face website
3. **Use a token** if required: `export HF_TOKEN=your_token_here`
4. **Check model access** - ensure you have permission to download the model

#### 6. Model Size Flexibility

**Q**: Can I download Qwen models of sizes other than 3B, 7B, and 14B?

**A**: Yes! The download script supports any Qwen model size available on Hugging Face:

1. **Base models**: You can download any Qwen2.5 model size (1.5B, 3B, 4B, 6B, 7B, 8B, 9B, 12B, 14B, 32B, etc.)
2. **TokenSkip adapters**: Currently only available for 3B, 7B, and 14B models
3. **For other sizes**: You'll need to train your own TokenSkip adapter using the provided training pipeline

Example:
```bash
# Download 1.5B model
python download_models.py --model-size 1.5b

# Download 32B model
python download_models.py --model-size 32b
```

## Implementation Issues

#### 7. Choice of Delimiter

We use the `eos_token` from LLaMA-3.1-8B, which is `<|eot_id|>`, as the delimiter token for all experiments.  The format follows: `<|eot_id|>compression_ratio<|eot_id|>`. This delimiter clearly separates the compression ratio from the surrounding context.

The choice of delimiter does not affect the performance of TokenSkip, and you are free to select a unique delimiter of your own.

#### 8. Usage of vLLM

In our early-stage experiments, we observed that when using vLLM, the outputs of LLMs varied even when the same seed was used. To ensure stable reproducibility, we exclusively adopt the `transformers` implementation in our code. 

However, we note that TokenSkip only appends `compression_ratio` to the end of the input. Given this minimal modification, TokenSkip supports vLLM, and you can adapt it to [vLLM's implementation](https://github.com/deepseek-ai/DeepSeek-Math/blob/main/evaluation/infer/run_cot_eval.py#L75) as needed.

#### 9. Answer Format

TokenSkip is designed to controllably compress the *Chain-of-Thought* or *thinking* portion (enclosed within `<\think><\think>`) of LLMs while preserving the *summary/answer* part unchanged. 

In our experiments, we observed that most answer outputs from LLaMA-3.1-8B follow the format: `\n\nThe final answer is:`. To maintain consistency, we retain this pattern in our [implementation](https://github.com/hemingkx/TokenSkip/blob/main/LLMLingua.py#L51) and adopt the same format for the Qwen series. However, you are free to modify the answer format as needed.