"""
Step 3: Collect Activations from SFT Model
Extract hidden states at the answer position for SAE training.
"""
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import argparse
from tqdm import tqdm
import pickle

def collect_activations(model, tokenizer, qa_file, layer_idx=6, max_samples=None):
    """
    Collect activations from the model at the specified layer.
    Uses PURE QA format (no biography context) - knowledge retrieval mode:
    the biographies are only used during SFT (step 2) so facts live in the
    weights; here the model must retrieve them from parametric memory.

    Args:
        layer_idx: Which hidden_states index to extract (6 = the mid-layer
                   used in the paper; -1 = last layer)
    """
    device = next(model.parameters()).device
    
    # Load data
    with open(qa_file, 'r') as f:
        qa_pairs = [json.loads(line) for line in f]
    
    if max_samples:
        qa_pairs = qa_pairs[:max_samples]
    
    activations = []
    
    model.eval()
    with torch.no_grad():
        for qa in tqdm(qa_pairs, desc="Collecting activations"):
            # Format input: Pure QA (no biography context!)
            prompt = f"Q: {qa['question']}\nA:"
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            
            # Forward pass with hidden states
            outputs = model(**inputs, output_hidden_states=True)
            
            # Extract activation from the last token position of the prompt
            # This is where the model "decides" what to answer
            hidden_states = outputs.hidden_states[layer_idx]  # [1, seq_len, d_model]
            last_position = inputs['input_ids'].shape[1] - 1
            h = hidden_states[0, last_position, :].cpu().numpy()  # [d_model]
            
            # Tokenize answer to get target tokens
            answer_tokens = tokenizer.encode(f" {qa['answer']}", add_special_tokens=False)
            
            activations.append({
                'h': h,
                'rule_idx': qa['rule_idx'],
                'rule_name': qa.get('rule_name'),
                'answer': qa['answer'],
                'answer_tokens': answer_tokens,
                'person_id': qa['person_id'],
                'question': qa['question'],
                'template_idx': qa.get('template_idx'),
                'split': qa.get('split'),
            })
    
    return activations

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/base_sft/final')
    parser.add_argument('--train_qa', type=str, default='data/generated/qa_train.jsonl')
    parser.add_argument('--output', type=str, default='data/activations/train_activations.pkl')
    parser.add_argument('--layer', type=int, default=6, help='hidden_states index to extract (paper uses 6)')
    parser.add_argument('--max_samples', type=int, default=None)
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
    
    # Collect activations
    print(f"Collecting activations from layer {args.layer}")
    activations = collect_activations(
        model, tokenizer, args.train_qa,
        layer_idx=args.layer,
        max_samples=args.max_samples
    )
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(activations, f)
    
    print(f"Saved {len(activations)} activations to {output_path}")
    print(f"Activation shape: {activations[0]['h'].shape}")
    
    # Print statistics
    rule_counts = {}
    for act in activations:
        rule_name = act['rule_name']
        rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1
    
    print("\nActivations per rule:")
    for rule_name, count in sorted(rule_counts.items()):
        print(f"  {rule_name}: {count}")

if __name__ == "__main__":
    main()
