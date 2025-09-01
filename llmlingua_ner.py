import os
import json
import re
import spacy
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, RobertaTokenizer, RobertaForTokenClassification
import torch
from llmlingua import PromptCompressor
import numpy as np
from datetime import datetime

class NEREnhancedTokenSkip:
    """
    NER-Enhanced TokenSkip implementation that preserves critical logical connections
    and factual elements during Chain-of-Thought compression.
    """
    
    def __init__(self, llmlingua_path="your_model_path/llmlingua-2-xlm-roberta-large-meetingbank", 
                 spacy_model="en_core_web_sm", roberta_model="roberta-base"):
        """
        Initialize NER-Enhanced TokenSkip with required models.
        
        Args:
            llmlingua_path: Path to LLMLingua-2 model
            spacy_model: SpaCy model for NER
            roberta_model: RoBERTa model for additional NER
        """
        # Initialize LLMLingua
        self.llm_lingua = PromptCompressor(
            model_name=llmlingua_path,
            use_llmlingua2=True,
        )
        
        # Initialize SpaCy for NER
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"SpaCy model {spacy_model} not found. Please install with:")
            print(f"python -m spacy download {spacy_model}")
            raise
        
        # Initialize RoBERTa tokenizer for token-level processing
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_model)
        
        # Define CoT-specific patterns and entities
        self.cot_patterns = self._initialize_cot_patterns()
        self.logical_connectors = {
            'therefore', 'because', 'since', 'thus', 'hence', 'so', 'as a result',
            'consequently', 'accordingly', 'given that', 'due to', 'owing to',
            'for this reason', 'this means', 'it follows that', 'we can conclude',
            'in conclusion', 'finally', 'ultimately', 'step', 'next', 'then',
            'first', 'second', 'third', 'lastly'
        }
        
    def _initialize_cot_patterns(self):
        """Initialize regex patterns for CoT-specific elements."""
        return {
            'step_markers': re.compile(r'(step\s+\d+|step\s+\w+)[:.]?', re.IGNORECASE),
            'mathematical_expressions': re.compile(r'[\d+\-*/=().\s]+[=][\d+\-*/().\s]+'),
            'numerical_results': re.compile(r'\b\d+(?:\.\d+)?(?:\s*[+\-*/]\s*\d+(?:\.\d+)?)*\s*=\s*\d+(?:\.\d+)?\b'),
            'calculations': re.compile(r'\$?\d+(?:,\d{3})*(?:\.\d{2})?\s*[+\-*/×÷]\s*\$?\d+(?:,\d{3})*(?:\.\d{2})?'),
            'logical_transitions': re.compile(r'\b(therefore|because|since|thus|hence|so|as a result)\b', re.IGNORECASE),
            'reasoning_conclusions': re.compile(r'\b(we can conclude|in conclusion|finally|the answer is)\b', re.IGNORECASE)
        }
    
    def extract_ner_entities(self, text):
        """
        Extract named entities using SpaCy with CoT-specific enhancements.
        
        Args:
            text: Input CoT text
            
        Returns:
            dict: Dictionary containing entity information and positions
        """
        doc = self.nlp(text)
        
        entities = {
            'spacy_entities': [],
            'cot_entities': [],
            'mathematical_expressions': [],
            'logical_connectors': [],
            'step_markers': [],
            'numerical_calculations': []
        }
        
        # Extract standard SpaCy entities
        for ent in doc.ents:
            entities['spacy_entities'].append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'importance_boost': self._get_entity_importance_boost(ent.label_)
            })
        
        # Extract CoT-specific patterns
        for pattern_name, pattern in self.cot_patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                entities['cot_entities'].append({
                    'text': match.group(),
                    'pattern': pattern_name,
                    'start': match.start(),
                    'end': match.end(),
                    'importance_boost': self._get_pattern_importance_boost(pattern_name)
                })
        
        # Extract logical connectors with context
        words = text.lower().split()
        for i, word in enumerate(words):
            if any(connector in word for connector in self.logical_connectors):
                # Get surrounding context
                start_idx = max(0, i - 2)
                end_idx = min(len(words), i + 3)
                context = ' '.join(words[start_idx:end_idx])
                
                entities['logical_connectors'].append({
                    'text': word,
                    'context': context,
                    'position': i,
                    'importance_boost': 2.0  # High importance for logical flow
                })
        
        return entities
    
    def _get_entity_importance_boost(self, entity_label):
        """Get importance boost factor based on SpaCy entity label."""
        importance_map = {
            'CARDINAL': 1.8,  # Numbers
            'QUANTITY': 1.8,  # Measurements
            'PERCENT': 1.8,   # Percentages
            'MONEY': 1.8,     # Money values
            'DATE': 1.5,      # Dates
            'TIME': 1.5,      # Times
            'PERSON': 1.3,    # Person names
            'ORG': 1.3,       # Organizations
            'GPE': 1.2,       # Geopolitical entities
        }
        return importance_map.get(entity_label, 1.0)
    
    def _get_pattern_importance_boost(self, pattern_name):
        """Get importance boost factor based on CoT pattern type."""
        boost_map = {
            'step_markers': 2.5,           # Highest for step structure
            'mathematical_expressions': 2.2, # High for calculations
            'numerical_results': 2.0,       # High for results
            'calculations': 2.0,           # High for calculations
            'logical_transitions': 2.3,    # Very high for logical flow
            'reasoning_conclusions': 2.4   # Very high for conclusions
        }
        return boost_map.get(pattern_name, 1.0)
    
    def create_token_importance_map(self, text, entities):
        """
        Create a token-level importance map based on NER entities and CoT patterns.
        
        Args:
            text: Input CoT text
            entities: Extracted entities dictionary
            
        Returns:
            dict: Token position to importance score mapping
        """
        # Tokenize the text
        tokens = self.roberta_tokenizer.tokenize(text)
        token_positions = []
        
        # Map tokens to character positions
        current_pos = 0
        for token in tokens:
            token_text = self.roberta_tokenizer.convert_tokens_to_string([token]).strip()
            if token_text:
                start_pos = text.find(token_text, current_pos)
                if start_pos != -1:
                    end_pos = start_pos + len(token_text)
                    token_positions.append((start_pos, end_pos))
                    current_pos = end_pos
                else:
                    token_positions.append((current_pos, current_pos))
            else:
                token_positions.append((current_pos, current_pos))
        
        # Initialize importance scores (base score = 1.0)
        importance_map = {i: 1.0 for i in range(len(tokens))}
        
        # Apply entity-based importance boosts
        all_entities = []
        
        # Collect all entities with their boosts
        for entity_list in ['spacy_entities', 'cot_entities']:
            for entity in entities[entity_list]:
                all_entities.append(entity)
        
        # Apply boosts to overlapping tokens
        for entity in all_entities:
            entity_start = entity['start']
            entity_end = entity['end']
            boost = entity['importance_boost']
            
            for i, (token_start, token_end) in enumerate(token_positions):
                # Check for overlap between token and entity
                if (token_start < entity_end and token_end > entity_start):
                    importance_map[i] = max(importance_map[i], boost)
        
        # Apply logical connector boosts
        for connector in entities['logical_connectors']:
            boost = connector['importance_boost']
            # Find tokens that match the connector context
            context_tokens = self.roberta_tokenizer.tokenize(connector['context'])
            for i in range(len(tokens) - len(context_tokens) + 1):
                if tokens[i:i+len(context_tokens)] == context_tokens:
                    for j in range(i, i + len(context_tokens)):
                        importance_map[j] = max(importance_map[j], boost)
        
        return importance_map
    
    def compress_with_ner_enhancement(self, text, compression_ratio=0.5, model_type="qwen"):
        """
        Compress CoT text with NER-enhanced importance scoring.
        
        Args:
            text: Input CoT text to compress
            compression_ratio: Target compression ratio
            model_type: Model type for compression settings
            
        Returns:
            dict: Compression results with NER enhancement information
        """
        print(f"Processing text with NER enhancement...")
        
        # Step 1: Extract NER entities and CoT patterns
        entities = self.extract_ner_entities(text)
        
        # Step 2: Create token importance map
        importance_map = self.create_token_importance_map(text, entities)
        
        # Step 3: Apply LLMLingua compression with custom force tokens
        force_tokens = self._get_force_tokens(entities, model_type)
        
        print(f"Found {len(entities['spacy_entities'])} SpaCy entities")
        print(f"Found {len(entities['cot_entities'])} CoT-specific entities")
        print(f"Found {len(entities['logical_connectors'])} logical connectors")
        print(f"Force preserving {len(force_tokens)} critical tokens")
        
        # Compress using LLMLingua with enhanced parameters
        if model_type == "llama3":
            compressed_result = self.llm_lingua.compress_prompt(
                text, 
                rate=compression_ratio, 
                force_tokens=force_tokens + ['Step', ':'],
                force_reserve_digit=True,
                drop_consecutive=True
            )
        elif model_type == "qwen":
            compressed_result = self.llm_lingua.compress_prompt(
                text, 
                rate=compression_ratio,
                force_tokens=force_tokens
            )
        else:
            raise ValueError(f"Model Type {model_type} is not supported.")
        
        # Step 4: Enhance result with NER information
        enhanced_result = {
            **compressed_result,
            'ner_entities': entities,
            'importance_map': importance_map,
            'force_tokens_used': force_tokens,
            'ner_preservation_rate': self._calculate_preservation_rate(text, compressed_result['compressed_prompt'], entities)
        }
        
        return enhanced_result
    
    def _get_force_tokens(self, entities, model_type):
        """Get tokens that should be force-preserved during compression."""
        force_tokens = []
        
        # Force preserve step markers
        for entity in entities['cot_entities']:
            if entity['pattern'] == 'step_markers':
                force_tokens.extend(entity['text'].split())
        
        # Force preserve high-importance logical connectors
        high_importance_connectors = [
            'therefore', 'because', 'thus', 'hence', 'conclusion'
        ]
        force_tokens.extend(high_importance_connectors)
        
        # Force preserve mathematical operators and equals signs
        math_tokens = ['=', '+', '-', '*', '/', '×', '÷', '$']
        force_tokens.extend(math_tokens)
        
        return list(set(force_tokens))  # Remove duplicates
    
    def _calculate_preservation_rate(self, original_text, compressed_text, entities):
        """Calculate how well critical entities were preserved."""
        if not entities['cot_entities']:
            return 1.0
        
        preserved_count = 0
        total_critical_entities = len(entities['cot_entities'])
        
        for entity in entities['cot_entities']:
            if entity['text'].lower() in compressed_text.lower():
                preserved_count += 1
        
        return preserved_count / total_critical_entities if total_critical_entities > 0 else 1.0


def load_jsonl(file, encoding='utf-8'):
    """Load data from JSONL file."""
    data = []
    with open(file, 'r', encoding=encoding) as f:
        for j in f.readlines():
            j = json.loads(j)
            data.append(j)
    return data


def save_jsonl(data, output_path):
    """Save data to JSONL file."""
    if os.path.exists(output_path):
        os.remove(output_path)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    for item in data:
        with open(output_path, 'a+', encoding='utf-8') as f:
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + '\n')


def compress_cot_with_ner_enhancement(model_name="Qwen2.5-7B-Instruct", model_size="7b", 
                                     model_type="qwen", llmlingua_path="your_model_path/llmlingua-2-xlm-roberta-large-meetingbank", 
                                     input_path=None, output_dir=None):
    """
    Compress CoT outputs using NER-Enhanced TokenSkip with various compression ratios.
    
    Args:
        model_name: Name of the model
        model_size: Size of the model
        model_type: Type of model for formatting
        llmlingua_path: Path to LLMLingua model
        input_path: Input file path
        output_dir: Output directory path
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
            output_dir = f"outputs/{model_name}/gsm8k/{model_size}/NER_Enhanced_Compression"
        else:
            output_dir = f"outputs/{model_name}/gsm8k/{model_size}/NER_Enhanced_Compression"
    
    print(f"Loading data from: {input_path}")
    data = load_jsonl(input_path)
    
    # Initialize NER-Enhanced TokenSkip
    ner_tokenskip = NEREnhancedTokenSkip(llmlingua_path=llmlingua_path)
    
    # Define compression ratios to test
    ratio_list = [0.9, 0.8, 0.7, 0.6, 0.5]
    
    for compression_ratio in ratio_list:
        print(f"\n🔄 Processing compression ratio: {compression_ratio}")
        output_path = os.path.join(output_dir, f"NER_ENHANCED_NEW_OUTPUT_compressed_ratio_{compression_ratio}.jsonl")
        
        compressed_data = []
        
        # Determine which field contains the CoT
        cot_type = "cot" if model_type == "llama3" else "model_output"
        
        for i in tqdm(range(len(data)), desc=f"Compressing with ratio {compression_ratio}"):
            cot_output = data[i][cot_type]
            
            # Apply NER-enhanced compression
            enhanced_result = ner_tokenskip.compress_with_ner_enhancement(
                cot_output, 
                compression_ratio=compression_ratio, 
                model_type=model_type
            )
            
            # Create enhanced data entry with clear NER enhancement indicators
            compressed_data_line = {
                'NER_ENHANCED_VERSION': 'NEW_ENHANCED_OUTPUT_v1.0',
                'enhancement_type': 'NER_Enhanced_TokenSkip',
                'generation_timestamp': datetime.now().isoformat(),
                'enhancement_metadata': {
                    'version': 'NEW_ENHANCED_OUTPUT_v1.0',
                    'description': 'NER-Enhanced TokenSkip with Entity Preservation',
                    'features': ['SpaCy NER', 'CoT Pattern Recognition', 'Logical Connector Preservation', 'Force Token Protection']
                },
                'question': data[i]['messages'][0]['content'],
                'input': data[i]['prompt'],
                'output': data[i]['model_output'],
                'answer': data[i]['answer'],
                'model_answer': data[i]['prediction'],
                'is_correct': data[i]['accuracy'],
                'cot': data[i][cot_type],
                'compressed_cot': enhanced_result['compressed_prompt'],
                'original_cot_tokens': enhanced_result['origin_tokens'],
                'compressed_cot_tokens': enhanced_result['compressed_tokens'],
                'compression_rate': enhanced_result['rate'],
                'ner_entities': enhanced_result['ner_entities'],
                'force_tokens_used': enhanced_result['force_tokens_used'],
                'ner_preservation_rate': enhanced_result['ner_preservation_rate'],
                'enhancement_features': {
                    'spacy_entities_preserved': len(enhanced_result['ner_entities']['spacy_entities']),
                    'cot_entities_preserved': len(enhanced_result['ner_entities']['cot_entities']),
                    'logical_connectors_preserved': len(enhanced_result['ner_entities']['logical_connectors']),
                    'force_tokens_count': len(enhanced_result['force_tokens_used'])
                }
            }
            compressed_data.append(compressed_data_line)
        
        # Save compressed data
        save_jsonl(compressed_data, output_path)
        
        # Print statistics
        avg_compression_rate = sum(item['compressed_cot_tokens'] / item['original_cot_tokens'] 
                                 for item in compressed_data) / len(compressed_data)
        avg_preservation_rate = sum(item['ner_preservation_rate'] for item in compressed_data) / len(compressed_data)
        
        print(f"✅ NER ENHANCED NEW OUTPUT Saved to: {output_path}")
        print(f"📊 Average Compression Rate: {avg_compression_rate:.4f}")
        print(f"🎯 Average NER Preservation Rate: {avg_preservation_rate:.4f}")
        print(f"🔍 NER Enhancement Features Applied:")
        print(f"   - SpaCy Entities: {len(compressed_data[0]['enhancement_features']['spacy_entities_preserved']) if compressed_data else 0}")
        print(f"   - CoT Entities: {len(compressed_data[0]['enhancement_features']['cot_entities_preserved']) if compressed_data else 0}")
        print(f"   - Logical Connectors: {len(compressed_data[0]['enhancement_features']['logical_connectors_preserved']) if compressed_data else 0}")
        print(f"   - Force Tokens: {len(compressed_data[0]['enhancement_features']['force_tokens_count']) if compressed_data else 0}")


def main():
    parser = argparse.ArgumentParser(description="NER-Enhanced TokenSkip compression script")
    parser.add_argument("--model-size", type=str, default="7b",
                       help="Qwen model size (e.g., 1.5b, 3b, 4b, 6b, 7b, 8b, 9b, 12b, 14b, 32b, etc.)")
    parser.add_argument("--model-type", type=str, default="qwen", choices=["qwen", "llama3"],
                       help="Model type for output formatting")
    parser.add_argument("--llmlingua-path", type=str, default="your_model_path/llmlingua-2-xlm-roberta-large-meetingbank",
                       help="Path to LLMLingua-2 model")
    parser.add_argument("--input-path", type=str, default=None,
                       help="Input file path (auto-detected if not specified)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory path (auto-generated if not specified)")
    
    args = parser.parse_args()
    
    # Get model name based on size
    model_name = f"Qwen2.5-{args.model_size.upper()}-Instruct"
    
    print("🚀 NER-Enhanced TokenSkip Compression Pipeline - NEW ENHANCED OUTPUT")
    print("=" * 80)
    print(f"🆕 VERSION: NEW_ENHANCED_OUTPUT_v1.0")
    print(f"🔧 ENHANCEMENT: NER-Enhanced TokenSkip with Entity Preservation")
    print(f"Model: {model_name}")
    print(f"Model size: {args.model_size}")
    print(f"Model type: {args.model_type}")
    print(f"LLMLingua path: {args.llmlingua_path}")
    print(f"Input path: {args.input_path or 'Auto-detected'}")
    print(f"Output dir: {args.output_dir or 'Auto-generated'}")
    print("=" * 80)
    
    # Run the NER-enhanced compression pipeline
    compress_cot_with_ner_enhancement(
        model_name=model_name,
        model_size=args.model_size,
        model_type=args.model_type,
        llmlingua_path=args.llmlingua_path,
        input_path=args.input_path,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()