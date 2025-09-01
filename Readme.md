<div align="center">
<h1>Hybrid Prompt Compression with Granularity Control</h1> 
</div>

<p align="center">
<a href="https://opensource.org/licenses/Apache-2.0">
  <img src="https://img.shields.io/badge/License-Apache_2.0-green.svg"></a> 
<a href="https://github.com/hemingkx/TokenSkip/pulls">
    <img src="https://img.shields.io/badge/Contributions-welcome-blue.svg?style=flat"></a>
</p>

## Introduction
TBD
## Overall System Flow Design:
The proposed sequence follows 
- Step 1: Input + NER-Enhanced TokenSkip (CoT)
- Step 2: Soft Prompt Compression → Model Inference → Output 

Which establishes a two-stage compression pipeline that processes generated CoT reasoning before final model inference.

### Stage 1: CoT-Specific NER-Enhanced TokenSkip:
- Algorithm Design: Extended tokenization with reasoning delimiter awareness, custom NER entities for mathematical expressions and logical connectors, and CoT-optimized importance scoring. Implement a hybrid approach combining Hugging Face transformers tokenization, Spacy+RoBERTA NER, and LLMLingua importance scoring
Reasoning Structure Preservation: Special boosting for logical connectors ("therefore", "because", "since"), numerical calculations, intermediate results, and causal relationships
- Compression Strategy: Higher retention ratio than input compression (potentially 60% vs 50%) to preserve reasoning integrity
Output: Hard-compressed CoT reasoning maintaining logical flow and factual accuracy

### Stage 2: Soft Prompt Compression Methodology:
Paper here: https://arxiv.org/pdf/2504.07109

## Model Selection: 
Use frozen Qwen2.5-3B, Qwen2.5-7B-Instruct-Instruct and Gemma with trainable LoRA parameters for encoder consistency


## Model Weights

Download corresponding model weights and modify the checkpoint path in `eval.sh`.

| LoRA Adapter                         | Link                                                         |
| ------------------------------------ | ------------------------------------------------------------ |
| TokenSkip-Qwen2.5-3B-Instruct-GSM8K  | [huggingface](https://huggingface.co/hemingkx/TokenSkip-Qwen2.5-3B-Instruct-GSM8K) |
| TokenSkip-Qwen2.5-7B-Instruct-GSM8K  | [huggingface](https://huggingface.co/hemingkx/TokenSkip-Qwen2.5-7B-Instruct-GSM8K) |
| TokenSkip-Qwen2.5-14B-Instruct-GSM8K | [huggingface](https://huggingface.co/hemingkx/TokenSkip-Qwen2.5-14B-Instruct-GSM8K) |

## Installation

```
conda create -n hybrid_prompt_compression python=3.12
conda activate hybrid_prompt_compression
cd hybrid_prompt_compression
pip install -r requirements.txt
```

## Model Download

We provide an automated script to download all required models and weights for the TokenSkip project. Ensure you have sufficient disk space (recommended: 50+ GB)

### Quick Start

Download all models (Qwen base model, LLMLingua-2, and TokenSkip adapters):

```bash
python download_models.py
```

This will download:
- Qwen2.5-7B-Instruct base model (~14 GB)
- LLMLingua-2 model weights (~1.5 GB)
- TokenSkip adapters for 3B, 7B, and 14B models (~1-2 GB each)

### Advanced Usage

#### Download specific components only:

```bash
# Download only Qwen base model
python download_models.py --qwen-only

# Download only LLMLingua-2 model
python download_models.py --llmlingua-only

# Download only TokenSkip adapters
python download_models.py --adapters-only
```

#### Choose different model size:

```bash
# Download 3B model instead of 7B
python download_models.py --model-size 3b

# Download 14B model
python download_models.py --model-size 14b

# Download any other size (e.g., 1.5b, 4b, 6b, 8b, 9b, 12b, 32b, etc.)
python download_models.py --model-size 1.5b
```

#### Custom download directory:

```bash
# Download to custom directory
python download_models.py --base-dir /path/to/your/models
```

#### Skip disk space check:

```bash
python download_models.py --skip-disk-check
```

### Model Details

#### 1. Qwen Base Models
- **Qwen2.5-3B-Instruct**: ~6 GB
- **Qwen2.5-7B-Instruct**: ~14 GB
- **Qwen2.5-14B-Instruct**: ~28 GB

#### 2. LLMLingua-2 Model
- **llmlingua-2-xlm-roberta-large-meetingbank**: ~1.5 GB
- Used for compressing Chain-of-Thought outputs

#### 3. TokenSkip Adapters
- **TokenSkip-Qwen2.5-3B-Instruct-GSM8K**: ~1 GB
- **TokenSkip-Qwen2.5-7B-Instruct-GSM8K**: ~1 GB
- **TokenSkip-Qwen2.5-14B-Instruct-GSM8K**: ~1 GB

> **Note**: TokenSkip adapters are currently available for 3B, 7B, and 14B models only. For other model sizes, you can train your own adapter using the provided training pipeline.

### After Download

Once the download is complete, update your configuration files:

#### 1. Update `eval.sh`:
```bash
MODEL_PATH="/path/to/models/Qwen2.5-7B-Instruct"
ADAPTER_PATH="/path/to/models/TokenSkip-Qwen2.5-7B-Instruct-GSM8K"
```

#### 2. Update `LLMLingua.py`:
```python
llmlingua_path="/path/to/models/llmlingua-2-xlm-roberta-large-meetingbank"
```

### Troubleshooting

#### Common Issues:

1. **Out of disk space**: Use `--skip-disk-check` or free up space
2. **Network timeout**: The script will retry automatically
3. **Permission errors**: Ensure write permissions to the download directory

#### Manual Download:

If the script fails, you can manually download models from Hugging Face:

- Qwen models: https://huggingface.co/Qwen
- LLMLingua-2: https://huggingface.co/microsoft/llmlingua-2-xlm-roberta-large-meetingbank
- TokenSkip adapters: https://huggingface.co/hemingkx

For more detailed troubleshooting, see the [Q&A section](./assets/Q&A.md).

## Token Pruning

**1.Obtain the original CoT outputs of the training data, using the target LLM**

Modify the command lines in `eval.sh` (e.g., set `DATA_TYPE` to `train`) and run `evaluation`.

```
python ./evaluation.py --output-dir "outputs/Qwen2.5-7B-Instruct/gsm8k/" \
    --model-path "your_model_path/Qwen2.5-7B-Instruct" --tokenizer-path "your_model_path/Qwen2.5-7B-Instruct" \
    --model-size "7b" --model-type "qwen" --data-type "train"  \
    --max_num_examples 100000000000000 --max_new_tokens 512 \
    --eval_batch_size 32 --temperature 0.0 --seed 42 --benchmark "gsm8k"
```

> The original CoT outputs of the target LLM will be stored in `outputs/.../Original`.

**2.Prune original CoTs using LLMLingua**

Download the [model weights](https://huggingface.co/microsoft/llmlingua-2-xlm-roberta-large-meetingbank) for [LLMLingua-2](https://github.com/microsoft/LLMLingua) and modify the checkpoint path in `LLMLingua.py`.

Run `LLMLingua with NER` to obtain compressed CoTs with various compression ratios.

### Basic Usage

```bash
# Use default 7B model
python ./llmlingua_ner.py

# Use 3B model
python ./llmlingua_ner.py --model-size 3b

```

### Advanced Usage

```bash
# Use custom LLMLingua + NER model path
python ./llmlingua_ner.py --llmlingua-path "/path/to/llmlingua-2-xlm-roberta-large-meetingbank"

# Use Llama3 model type for output formatting
python ./llmlingua_ner.py --model-size 7b --model-type llama3

# Combine multiple options
python ./llmlingua_ner.py --model-size 14b --model-type qwen --llmlingua-path "/custom/path/to/llmlingua"
```

### Command-line Arguments

- `--model-size`: Qwen model size (default: "7b", options: 1.5b, 3b, 4b, 6b, 7b, 8b, 9b, 12b, 14b, 32b, etc.)
- `--model-type`: Model type for output formatting (default: "qwen", options: "qwen", "llama3")
- `--llmlingua-path`: Path to LLMLingua-2 model (default: "/your_model_path/llmlingua-2-xlm-roberta-large-meetingbank")

The script automatically constructs file paths based on the model name and size, following the pattern: `outputs/{model_name}/gsm8k/{model_size}/...`

> The compressed CoTs will be stored in `outputs/.../NER_Enhanced_Compression`.

**3.Convert training data to LLaMA-Factory format**

Run `get_llamafactory_input` to convert the training data into the format of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

### Basic Usage

```bash
# Use default 7B model
python ./get_llamafactory_input.py

# Use 3B model
python ./get_llamafactory_input.py --model-size 3b

```

### Advanced Usage

```bash
# Use custom output file
python ./get_llamafactory_input.py --model-size 7b --output-file ./custom_output.json

# Combine multiple options
python ./get_llamafactory_input.py --model-size 14b --output-file ./my_custom_dataset.json
```

### Command-line Arguments

- `--model-size`: Qwen model size (default: "7b", options: 1.5b, 3b, 4b, 6b, 7b, 8b, 9b, 12b, 14b, 32b, etc.)
- `--output-file`: Output file path (default: auto-generated based on model size)

The script automatically constructs file paths based on the model name and size, following the pattern: `outputs/{model_name}/gsm8k/{model_size}/...`

> The converted data will be stored in `outputs/mydataset_compressed_gsm8k_llmlingua2_qwen_{MODEL_SIZE}.json` (auto-generated filename).
>
> For reference, we provide our processed training data in `datasets/gsm8k/llamafactory_inputs/`.

## Training

TokenSkip follows the general LoRA SFT pipeline of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Here's how to set it up:

1. Git clone [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and install the required environments.
2. Copy the training data to `LLaMA-Factory/data/` using the copy script and register it in `data/dataset_info.json`:

```bash
# Copy the dataset to LLaMA-Factory/data/
python ./copy_to_llamafactory.py --model-size 3b

# Register the dataset in dataset_info.json
python ./register_dataset.py --model-size 3b
```
3. To fine-tune the target LLM with LoRA, run the following command:

### Linux/Mac:
```bash
# For 3B model
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/examples/train_lora/myllama3_lora_sft_compressed_gsm8k_llmlingua2_qwen_3B.yaml

```

### Windows Command Prompt (cmd):
```cmd
# For 3B model
set CUDA_VISIBLE_DEVICES=0 && llamafactory-cli train configs/examples/train_lora/myllama3_lora_sft_compressed_gsm8k_llmlingua2_qwen_3B.yaml

```

### Alternative (Windows without GPU specification):
```cmd
# For 3B model
llamafactory-cli train configs/examples/train_lora/myllama3_lora_sft_compressed_gsm8k_llmlingua2_qwen_3B.yaml

```
unstate the comuntage the latest commmit
> We provide our training configs in `configs/examples/train_lora` for your reference.

## Inference

Modify and run command lines in `eval.sh`, the results will be stored in `outputs/`.

```
python ./evaluation.py --output-dir "outputs/Qwen2.5-3B-Instruct/gsm8k/" \
    --model-path "your_model_path/Qwen2.5-3B-Instruct" --tokenizer-path "your_model_path/Qwen2.5-3B-Instruct" \
    --model-size "3b" --model-type "qwen" --data-type "test"  \
    --max_num_examples 2000 --max_new_tokens 512 \
    --eval_batch_size 32 --temperature 0.0 --seed 42 --benchmark "gsm8k" \
    --adapter-path "your_model_path/TokenSkip-Qwen2.5-3B-Instruct-GSM8K" \
    --compression_ratio 0.5 --use_adapter
```

## Q&A

Frequently asked questions about the re-implementation of TokenSkip can be found in [Q&A](./assets/Q&A.md).

## Contributing

We warmly welcome contributions and discussions related to TokenSkip! If you have any suggestions for improvements or ideas you'd like to discuss, please don't hesitate to open an issue. This will allow us to collaborate and discuss your ideas in detail.

## Acknowledgments

This codebase is built from [DeepSeek-Math](https://github.com/deepseek-ai/DeepSeek-Math), [LLMLingua](https://github.com/microsoft/LLMLingua) and [TokenSkip](https://github.com/hemingkx/TokenSkip).

## Citation

If you find the resources in this repository useful, please cite the tokenskip paper:

```
@misc{xia2025tokenskip,
      title={TokenSkip: Controllable Chain-of-Thought Compression in LLMs}, 
      author={Heming Xia and Yongqi Li and Chak Tou Leong and Wenjie Wang and Wenjie Li},
      year={2025},
      eprint={2502.12067},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.12067}, 
}
```

