import os
import json
import random
import numpy as np
import argparse

def get_model_name(model_size):
    """
    Get the full model name based on model size.
    """
    return f"Qwen2.5-{model_size.upper()}-Instruct"

def load_json(file, encoding='utf-8'):
    data = []
    with open(file, 'r', encoding=encoding) as f:
        for j in f.readlines():
            j = json.loads(j)
            data.append(j)
    return data

def write_list_to_json(list, file_path):
    with open(file_path, 'w', encoding='utf-8') as  f:
        json.dump(list, f, ensure_ascii=False, indent=1)

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def load_all_data(model_name="Qwen2.5-7B-Instruct", model_size="7b", input_dir=None, use_ner_enhanced=False):
    if input_dir is None:
        input_dir = f"outputs/{model_name}/gsm8k/{model_size}/"
    
    # Check if train subdirectory exists
    train_original_path = os.path.join(input_dir, "Original/train/samples/predictions_formatted.jsonl")
    samples_original_path = os.path.join(input_dir, "Original/samples/predictions_formatted.jsonl")
    
    if os.path.exists(train_original_path):
        original_data = load_json(train_original_path)
    else:
        original_data = load_json(samples_original_path)
    
    # Choose compression directory and file naming pattern based on flag
    if use_ner_enhanced:
        compression_dir = "NER_Enhanced_Compression"
        file_prefix = "NER_ENHANCED_NEW_OUTPUT_compressed_ratio_"
        print("🧠 Using NER Enhanced Compression data")
    else:
        compression_dir = "Compression"
        file_prefix = "train_outputs_compressed_ratio_"
        print("⚡ Using standard TokenSkip compression data")
    
    compressed_data_0 = load_json(os.path.join(input_dir, f"{compression_dir}/{file_prefix}0.9.jsonl"))
    compressed_data_1 = load_json(os.path.join(input_dir, f"{compression_dir}/{file_prefix}0.8.jsonl"))
    compressed_data_2 = load_json(os.path.join(input_dir, f"{compression_dir}/{file_prefix}0.7.jsonl"))
    compressed_data_3 = load_json(os.path.join(input_dir, f"{compression_dir}/{file_prefix}0.6.jsonl"))
    compressed_data_4 = load_json(os.path.join(input_dir, f"{compression_dir}/{file_prefix}0.5.jsonl"))
    return [original_data, compressed_data_0, compressed_data_1, compressed_data_2, compressed_data_3, compressed_data_4]

def get_llamafactory_input(model_name="Qwen2.5-7B-Instruct", model_size="7b", output_file=None, use_ner_enhanced=False):
    compressed_data_list = load_all_data(model_name=model_name, model_size=model_size, use_ner_enhanced=use_ner_enhanced)
    original_data = compressed_data_list[0]
    compression_ratio_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    datalines = []
    for i in range(len(original_data)):
        data_index = random.choice([0,1,2,3,4,5])
        if data_index == 0:
            input_data = original_data[i]['messages'][0]['content']
            answer = original_data[i]['prediction']
            cot = original_data[i]['model_output']
            output_data = f"{cot}\n\nThe final answer is: " + "$\\boxed{" + answer + "}$"
        else:
            compression_ratio = compression_ratio_list[data_index]
            compressed_data = compressed_data_list[data_index]
            input_data = f"{compressed_data[i]['question']}<|eot_id|>{compression_ratio}<|eot_id|>"
            answer = compressed_data[i]['model_answer']
            cot = compressed_data[i]['compressed_cot']
            output_data = f"{cot}\n\nThe final answer is: " + "$\\boxed{" + answer + "}$"

        data = {
            "instruction": "Please reason step by step, and put your final answer within \\boxed{}.",
            "input": input_data,
            "output": output_data
        }
        datalines.append(data)
    print(len(datalines))
    random.shuffle(datalines)
    
    if output_file is None:
        compression_type = "ner_enhanced" if use_ner_enhanced else "tokenskip"
        output_file = f'./outputs/mydataset_compressed_gsm8k_llmlingua2_qwen_{model_size.upper()}_{compression_type}.json'
    
    write_list_to_json(datalines, output_file)


def main():
    parser = argparse.ArgumentParser(description="Convert TokenSkip data to LLaMA-Factory format")
    parser.add_argument("--model-size", type=str, default="7b",
                       help="Qwen model size (e.g., 1.5b, 3b, 4b, 6b, 7b, 8b, 9b, 12b, 14b, 32b, etc.)")
    parser.add_argument("--output-file", type=str, default=None,
                       help="Output file path (default: auto-generated based on model size)")
    parser.add_argument("--use-ner-enhanced", action="store_true",
                       help="Use NER enhanced compression data instead of standard TokenSkip compression")
    
    args = parser.parse_args()
    
    # Get model name based on size
    model_name = get_model_name(args.model_size)
    
    print("🚀 LLaMA-Factory Data Converter")
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"Model size: {args.model_size}")
    compression_type = "NER Enhanced" if args.use_ner_enhanced else "Standard TokenSkip"
    print(f"Compression type: {compression_type}")
    if args.output_file:
        print(f"Output file: {args.output_file}")
    else:
        compression_suffix = "ner_enhanced" if args.use_ner_enhanced else "tokenskip"
        print(f"Output file: auto-generated (mydataset_compressed_gsm8k_llmlingua2_qwen_{args.model_size.upper()}_{compression_suffix}.json)")
    
    # Set random seed and run conversion
    seed_everything(42)
    get_llamafactory_input(
        model_name=model_name,
        model_size=args.model_size,
        output_file=args.output_file,
        use_ner_enhanced=args.use_ner_enhanced
    )

if __name__ == '__main__':
    main()

