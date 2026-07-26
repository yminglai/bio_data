#!/usr/bin/env python3
"""
Step 2: Swap evaluation for unsupervised SAE using discovered relation features.
Can be run in parallel for different alpha values.
"""
import argparse
import json
import random
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM


class TraditionalSAE(nn.Module):
    """Traditional Sparse Autoencoder."""
    
    def __init__(self, d_model, n_features):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.encoder = nn.Linear(d_model, n_features, bias=True)
        self.decoder = nn.Linear(n_features, d_model, bias=True)
        
    def forward(self, x):
        z = self.encoder(x)
        z = F.relu(z)
        x_hat = self.decoder(z)
        return z, x_hat
    
    def encode(self, x):
        z = self.encoder(x)
        z = F.relu(z)
        return z


def load_facts_database(facts_file):
    """Load facts database: entity -> {relation: target_entity}"""
    facts_db = {}
    with open(facts_file, 'r') as f:
        for line in f:
            fact = json.loads(line)
            entity1 = fact['entity_1']
            relation = fact['relation']
            entity2 = fact['entity_2']
            
            if entity1 not in facts_db:
                facts_db[entity1] = {}
            facts_db[entity1][relation] = entity2
    return facts_db


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sae_checkpoint', type=str, required=True)
    parser.add_argument('--lm_model_path', type=str, required=True)
    parser.add_argument('--val_qa_file', type=str, required=True)
    parser.add_argument('--facts_db', type=str, required=True)
    parser.add_argument('--relation_features', type=str, required=True,
                       help='JSON file with relation-to-features mapping')
    parser.add_argument('--alpha', type=float, required=True,
                       help='Single alpha value for this run')
    parser.add_argument('--layer_idx', type=int, default=6)
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--max_new_tokens', type=int, default=10)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load SAE
    checkpoint = torch.load(args.sae_checkpoint, map_location=device)
    sae = TraditionalSAE(d_model=checkpoint['d_model'], n_features=checkpoint['n_features'])
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae = sae.to(device)
    sae.eval()
    
    # Load LM
    tokenizer = AutoTokenizer.from_pretrained(args.lm_model_path)
    lm_model = AutoModelForCausalLM.from_pretrained(args.lm_model_path)
    lm_model = lm_model.to(device)
    lm_model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load relation features
    with open(args.relation_features, 'r') as f:
        relation_to_features = json.load(f)
    
    # Load facts database
    facts_db = load_facts_database(args.facts_db)
    
    # Load val QA
    with open(args.val_qa_file, 'r') as f:
        qa_pairs = [json.loads(line) for line in f]
    
    if args.num_samples and args.num_samples < len(qa_pairs):
        random.seed(1234)
        qa_pairs = random.sample(qa_pairs, args.num_samples)
    
    print(f"Alpha={args.alpha}: Evaluating {len(qa_pairs)} samples on {device}")
    
    results = []
    
    with torch.no_grad():
        for qa in tqdm(qa_pairs, desc=f"Alpha {args.alpha}"):
            e1 = qa['entity_1']
            e2 = qa['entity_2']
            r1 = qa['relation_1']
            r2_true = qa['relation_2']
            e3_true = qa['entity_3']
            
            # Baseline: Q: E1 [R1] [R2] → A: E2 [E3]
            prompt = f"Q: {e1} [{r1}] [{r2_true}]\nA:"
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            
            baseline_out = lm_model.generate(**inputs, max_new_tokens=args.max_new_tokens, 
                                            do_sample=False, pad_token_id=tokenizer.eos_token_id,
                                            eos_token_id=tokenizer.eos_token_id)
            baseline_text = tokenizer.decode(baseline_out[0][inputs['input_ids'].shape[1]:], 
                                            skip_special_tokens=True).strip()
            
            # Get hidden states
            outputs = lm_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[args.layer_idx]
            
            # Try swapping R2 to other relations
            r2_features_dict = relation_to_features.get('r2_when_generating_e3', {})
            
            for r2_swap in r2_features_dict.keys():
                if r2_swap == r2_true:
                    continue
                
                # Check if we have ground truth for E2 -> R2_swap
                if e2 not in facts_db or r2_swap not in facts_db[e2]:
                    continue
                
                e3_expected = facts_db[e2][r2_swap]
                
                # Use hook to intervene during generation
                intervention_applied = [False]
                z_swapped = [None]
                
                def intervention_hook(module, input, output):
                    if intervention_applied[0]:
                        return output
                    
                    hidden = output[0]  # [batch, seq_len, hidden_dim]
                    last_pos = hidden.shape[1] - 1
                    h = hidden[0, last_pos, :]
                    
                    # Encode with SAE
                    z = sae.encode(h.unsqueeze(0))[0]
                    z_mod = z.clone()
                    
                    # Suppress R2_true features
                    if r2_true in r2_features_dict:
                        for feat_idx in r2_features_dict[r2_true]['feature_indices']:
                            z_mod[feat_idx] = 0.0
                    
                    # Boost R2_swap features  
                    if r2_swap in r2_features_dict:
                        for feat_idx in r2_features_dict[r2_swap]['feature_indices']:
                            z_mod[feat_idx] = args.alpha
                    
                    # Decode back
                    h_swapped = sae.decoder(z_mod.unsqueeze(0))[0]
                    hidden[0, last_pos, :] = h_swapped
                    intervention_applied[0] = True
                    
                    return (hidden,) + output[1:]
                
                # For GPT2, hook the output of block layer_idx-1, which is
                # hidden_states[layer_idx] — the representation the SAE
                # features were extracted from (01_find_unsupervised_features).
                hook = lm_model.transformer.h[args.layer_idx - 1].register_forward_hook(intervention_hook)
                
                try:
                    swapped_out = lm_model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                                   do_sample=False, pad_token_id=tokenizer.eos_token_id,
                                                   eos_token_id=tokenizer.eos_token_id)
                    swapped_text = tokenizer.decode(swapped_out[0][inputs['input_ids'].shape[1]:],
                                                   skip_special_tokens=True).strip()
                except Exception as e:
                    swapped_text = f"ERROR: {str(e)}"
                finally:
                    hook.remove()
                
                # Check if swap was successful
                success = (e3_expected.lower() in swapped_text.lower()) if swapped_text and not swapped_text.startswith('ERROR') else False
                
                results.append({
                    'e1': e1,
                    'e2': e2,
                    'r1': r1,
                    'r2_true': r2_true,
                    'r2_swap': r2_swap,
                    'e3_true': e3_true,
                    'e3_expected': e3_expected,
                    'baseline_gen': baseline_text,
                    'swapped_gen': swapped_text,
                    'success': success
                })
    
    # Compute metrics
    total = len(results)
    successful = sum(1 for r in results if r['success'])
    success_rate = successful / total if total > 0 else 0
    
    output_data = {
        'alpha': args.alpha,
        'total_swaps': total,
        'successful_swaps': successful,
        'success_rate': success_rate,
        'details': results
    }
    
    # Save
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nAlpha {args.alpha}: {successful}/{total} = {success_rate*100:.2f}% success")
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    main()
