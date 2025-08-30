import os
import json
import argparse
from tqdm import tqdm
from llmlingua import PromptCompressor


def get_model_name(model_size):
    """
    Get the full model name based on model size.
    """
    return f"Qwen2.5-{model_size.upper()}-Instruct"


def load_jsonl(file, encoding='utf-8'):
    data = []
    with open(file, 'r', encoding=encoding) as f:
        for j in f.readlines():
            j = json.loads(j)
            data.append(j)
    return data

def save_jsonl(data, output_path):
    if os.path.exists(output_path):
        os.remove(output_path)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    for item in data:
        with open(output_path, 'a+', encoding='utf-8') as f:
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + '\n')

def filter_correct_outputs(model_name="Qwen2.5-7B-Instruct", model_size="7b", input_path=None, output_path=None):
    """
    Filter the correct outputs from the data.
    """
    if input_path is None:
        # Check if train subdirectory exists
        train_path = f"outputs/{model_name}/gsm8k/{model_size}/Original/train/samples/predictions.jsonl"
        samples_path = f"outputs/{model_name}/gsm8k/{model_size}/Original/samples/predictions.jsonl"
        
        if os.path.exists(train_path):
            input_path = train_path
        else:
            input_path = samples_path
            
    if output_path is None:
        # Use the same directory structure as input_path
        if "train" in input_path:
            output_path = f"outputs/{model_name}/gsm8k/{model_size}/Original/train/samples/predictions_correct.jsonl"
        else:
            output_path = f"outputs/{model_name}/gsm8k/{model_size}/Original/samples/predictions_correct.jsonl"
    data = load_jsonl(input_path)
    correct_data = []
    for i in range(len(data)):
        if data[i]['accuracy']:
            correct_data.append(data[i])
    print(f"Original Samples: {len(data)}, Correct Samples: {len(correct_data)}, Accuracy: {len(correct_data) / len(data)}")
    save_jsonl(correct_data, output_path)


def filter_formatted_outputs(model_name="Qwen2.5-7B-Instruct", model_size="7b", model_type="qwen", input_path=None, output_path=None):
    """
    Filter the formatted outputs from the data. Extract COT from th outputs.
    """
    if input_path is None:
        # Check if train subdirectory exists
        train_path = f"outputs/{model_name}/gsm8k/{model_size}/Original/train/samples/predictions_correct.jsonl"
        samples_path = f"outputs/{model_name}/gsm8k/{model_size}/Original/samples/predictions_correct.jsonl"
        
        if os.path.exists(train_path):
            input_path = train_path
        else:
            input_path = samples_path
            
    if output_path is None:
        # Use the same directory structure as input_path
        if "train" in input_path:
            output_path = f"outputs/{model_name}/gsm8k/{model_size}/Original/train/samples/predictions_formatted.jsonl"
        else:
            output_path = f"outputs/{model_name}/gsm8k/{model_size}/Original/samples/predictions_formatted.jsonl"
    
    data = load_jsonl(input_path)
    formatted_data = []
    for i in range(len(data)):
        if data[i]['cot_length'] > 500:
            continue
        if model_type == "llama3":
            spans = data[i]["output"].split('\n\nThe final answer is:')
            if len(spans) == 2:
                data[i]["cot"] = spans[0]
                formatted_data.append(data[i])
        elif model_type == "qwen":
            formatted_data.append(data[i])
        else:
            raise ValueError(f"Model Type {model_type} is not supported.")
    print(f"Original Samples: {len(data)}, Formatted Samples: {len(formatted_data)}")
    save_jsonl(formatted_data, output_path)

def LLMLingua(data, compression_ratio=0.5, model_type="qwen",
              llmlingua_path="your_model_path/llmlingua-2-xlm-roberta-large-meetingbank"):
    """
    Compress the CoT outputs with LLMLingua-2.
    """
    if model_type == "llama3":
        cot_type = "cot"
    elif model_type == "qwen":
        cot_type = "model_output"
    else:
        raise ValueError(f"Model Type {model_type} is not supported.")

    llm_lingua = PromptCompressor(
        model_name=llmlingua_path,
        use_llmlingua2=True,  # Whether to use llmlingua-2
    )
    compressed_data = []
    for i in tqdm(range(len(data))):
        cot_output = data[i][cot_type]
        if model_type == "llama3":
            compressed_prompt = llm_lingua.compress_prompt(cot_output, rate=compression_ratio, force_tokens=['Step', ':'], force_reserve_digit=True, drop_consecutive=True)
        elif model_type == "qwen":
            compressed_prompt = llm_lingua.compress_prompt(cot_output, rate=compression_ratio)
        else:
            raise ValueError(f"Model Type {model_type} is not supported.")
        compressed_data_line = {
            'question': data[i]['messages'][0]['content'],
            'input': data[i]['prompt'],
            'output': data[i]['model_output'],
            'answer': data[i]['answer'],
            'model_answer': data[i]['prediction'],
            'is_correct': data[i]['accuracy'],
            'cot': data[i][cot_type],
            'compressed_cot': compressed_prompt['compressed_prompt'],
            'original_cot_tokens': compressed_prompt['origin_tokens'],
            'compressed_cot_tokens': compressed_prompt['compressed_tokens'],
            'compression_rate': compressed_prompt['rate']
        }
        compressed_data.append(compressed_data_line)
    return compressed_data


def compress_cot_outputs(model_name="Qwen2.5-7B-Instruct", model_size="7b", model_type="qwen", llmlingua_path="your_model_path/llmlingua-2-xlm-roberta-large-meetingbank", input_path=None, output_dir=None):
    """
    Compress the CoT outputs with various compression ratios using LLMLingua-2.
    """
    if input_path is None:
        # Check if train subdirectory exists
        train_path = f"outputs/{model_name}/gsm8k/{model_size}/Original/train/samples/predictions_formatted.jsonl"
        samples_path = f"outputs/{model_name}/gsm8k/{model_size}/Original/samples/predictions_formatted.jsonl"
        
        if os.path.exists(train_path):
            input_path = train_path
        else:
            input_path = samples_path
            
    if output_dir is None:
        # Use the same directory structure as input_path
        if "train" in input_path:
            output_dir = f"outputs/{model_name}/gsm8k/{model_size}/Compression"
        else:
            output_dir = f"outputs/{model_name}/gsm8k/{model_size}/Compression"
    
    data = load_jsonl(input_path)
    ratio_list = [0.9, 0.8, 0.7, 0.6, 0.5]
    for compression_ratio in ratio_list:
        output_path = os.path.join(output_dir, f"train_outputs_compressed_ratio_{compression_ratio}.jsonl")
        compressed_data = LLMLingua(data, compression_ratio=compression_ratio, model_type=model_type, llmlingua_path=llmlingua_path)
        save_jsonl(compressed_data, output_path)
        get_average_compress_rate(compressed_data)

def get_average_compress_rate(data):
    compress_rate = 0
    for i in range(len(data)):
        compress_rate += data[i]['compressed_cot_tokens'] / data[i]['original_cot_tokens']
    compress_rate = compress_rate / len(data)
    print(f"Average Compression Rate: {compress_rate}")


def data_processing_gsm8k(model_name="Qwen2.5-7B-Instruct", model_size="7b", model_type="qwen", llmlingua_path="your_model_path/llmlingua-2-xlm-roberta-large-meetingbank"):
    """
    The overall pipeline to process the GSM8K data.
    """
    filter_correct_outputs(model_name=model_name, model_size=model_size)
    filter_formatted_outputs(model_name=model_name, model_size=model_size, model_type=model_type)
    compress_cot_outputs(model_name=model_name, model_size=model_size, model_type=model_type, llmlingua_path=llmlingua_path)

def main():
    parser = argparse.ArgumentParser(description="LLMLingua compression script for TokenSkip project")
    parser.add_argument("--model-size", type=str, default="7b",
                       help="Qwen model size (e.g., 1.5b, 3b, 4b, 6b, 7b, 8b, 9b, 12b, 14b, 32b, etc.)")
    parser.add_argument("--model-type", type=str, default="qwen", choices=["qwen", "llama3"],
                       help="Model type for output formatting")
    parser.add_argument("--llmlingua-path", type=str, default="your_model_path/llmlingua-2-xlm-roberta-large-meetingbank",
                       help="Path to LLMLingua-2 model")
    
    args = parser.parse_args()
    
    # Get model name based on size
    model_name = get_model_name(args.model_size)
    
    print("🚀 LLMLingua Compression Pipeline")
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"Model size: {args.model_size}")
    print(f"Model type: {args.model_type}")
    print(f"LLMLingua path: {args.llmlingua_path}")
    
    # Run the processing pipeline
    data_processing_gsm8k(
        model_name=model_name,
        model_size=args.model_size,
        model_type=args.model_type,
        llmlingua_path=args.llmlingua_path
    )

if __name__ == '__main__':
    main()


