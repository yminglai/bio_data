"""
Step 1: Collect Activations from 2-hop Model
Extract hidden states at the answer position for SAE training.

Adapted for 2-hop reasoning task with 22 relations.
"""
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import argparse
from tqdm import tqdm
import pickle

# 20 relations from 2-hop task (alphabetically sorted)
RELATIONS = [
    'accuses', 'admires', 'blames', 'boss_of', 'classmate_of',
    'competes_with', 'cousin_of', 'endorsed_by', 'follows', 'forgives',
    'friend_of', 'has_crush_on', 'mentor_of', 'neighbor_of', 'owes_debt_to',
    'protects', 'reports_to', 'subscribes_to', 'warns', 'works_with'
]

RELATION_TO_IDX = {rel: idx for idx, rel in enumerate(RELATIONS)}


def collect_activations(model, tokenizer, qa_file, layer_idx=-1, max_samples=None):
    """
    Collect activations from the model at the specified layer.
    
    For 2-hop reasoning, we need TWO activations per question:
    1. Activation when predicting E2 (should use R1)
    2. Activation when predicting E3 (should use R2)
    
    Args:
        layer_idx: Which transformer layer to extract from (-1 = last layer)
        
    Returns:
        List of activation dicts with:
        - h: hidden state (numpy array)
        - relation: the relation this activation should bind to
        - relation_idx: index of the relation
        - position: 'token_1' (for E2) or 'token_2' (for E3)
        - question: the question text
    """
    device = next(model.parameters()).device
    
    # Load data (JSONL format)
    qa_pairs = []
    with open(qa_file, 'r') as f:
        for line in f:
            qa_pairs.append(json.loads(line.strip()))
    
    if max_samples:
        qa_pairs = qa_pairs[:max_samples]
    
    activations = []
    
    model.eval()
    with torch.no_grad():
        for qa in tqdm(qa_pairs, desc="Collecting activations"):
            # Extract information from qa dict
            entity_1 = qa['entity_1']
            relation_1 = qa['relation_1']
            entity_2 = qa['entity_2']
            relation_2 = qa['relation_2']
            entity_3 = qa['entity_3']
            
            # Map relation names to indices
            relation_1_idx = RELATION_TO_IDX.get(relation_1, -1)
            relation_2_idx = RELATION_TO_IDX.get(relation_2, -1)
            
            if relation_1_idx == -1 or relation_2_idx == -1:
                print(f"Warning: Unknown relation {relation_1} or {relation_2}")
                continue
            
            # === Activation 1: When predicting E2 (should activate R1) ===
            question = qa['question']
            prompt = f"Question: {question}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_idx]  # [1, seq_len, d_model]
            last_position = inputs['input_ids'].shape[1] - 1
            h1 = hidden_states[0, last_position, :].cpu().numpy()  # [d_model]
            
            # Tokenize E2 to get the target token. Entities are dedicated
            # single tokens with no leading space (training concatenates them
            # directly after "Answer:"), so encode WITHOUT a space — with one,
            # the first token would be the space token for every sample.
            e2_tokens = tokenizer.encode(entity_2, add_special_tokens=False)
            e2_first_token = e2_tokens[0] if len(e2_tokens) > 0 else 0
            
            activations.append({
                'h': h1,
                'relation': relation_1,
                'relation_idx': relation_1_idx,
                'position': 'token_1',
                'entity_1': entity_1,
                'entity_2': entity_2,
                'entity_3': entity_3,
                'question': question,
                'answer': qa['output'],
                'target_token': e2_first_token,  # First token of E2
            })
            
            # === Activation 2: When predicting E3 (should activate R2) ===
            # Training concatenated the E2 token directly after "Answer:"
            # (no space), so build the context the same way instead of
            # tokenizing "Answer: {entity_2}", which inserts a space token.
            input_ids_e2 = torch.cat(
                [inputs['input_ids'], torch.tensor([e2_tokens], device=device)], dim=1)
            attention_mask_e2 = torch.ones_like(input_ids_e2)

            outputs_with_e2 = model(input_ids=input_ids_e2,
                                    attention_mask=attention_mask_e2,
                                    output_hidden_states=True)
            hidden_states_with_e2 = outputs_with_e2.hidden_states[layer_idx]
            last_position_e2 = input_ids_e2.shape[1] - 1
            h2 = hidden_states_with_e2[0, last_position_e2, :].cpu().numpy()  # [d_model]

            # Tokenize E3 to get the target token (no leading space, as above)
            e3_tokens = tokenizer.encode(entity_3, add_special_tokens=False)
            e3_first_token = e3_tokens[0] if len(e3_tokens) > 0 else 0
            
            activations.append({
                'h': h2,
                'relation': relation_2,
                'relation_idx': relation_2_idx,
                'position': 'token_2',
                'entity_1': entity_1,
                'entity_2': entity_2,
                'entity_3': entity_3,
                'question': question,
                'answer': qa['output'],
                'target_token': e3_first_token,  # First token of E3
            })
    
    return activations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained 2-hop model checkpoint')
    parser.add_argument('--val_data', type=str, 
                        default='2hop/_dataset/_gen/val_two_hop_qa_data_4k.jsonl',
                        help='Path to validation QA data JSONL')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for activations pickle file')
    parser.add_argument('--layer', type=int, default=-1, 
                        help='Layer to extract (-1 = last)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to collect')
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")
    print(f"Model hidden size: {model.config.hidden_size}")
    
    # Collect activations
    print(f"Collecting activations from layer {args.layer}")
    activations = collect_activations(
        model, tokenizer, args.val_data,
        layer_idx=args.layer,
        max_samples=args.max_samples
    )
    
    print(f"Collected {len(activations)} activations")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(activations, f)
    
    print(f"Saved activations to {output_path}")


if __name__ == '__main__':
    main()
