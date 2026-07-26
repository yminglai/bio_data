#!/usr/bin/env python3
"""
Swap controllability evaluator for AlignSAE using hook-based intervention.
Uses the same intervention method as G-SAE (hook on transformer layer).
"""
import argparse
import json
import random
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM


def normalize_date(date_str):
    import re
    date_str = (date_str or '').strip().lower()
    match = re.match(r"(\d{1,2}),\s*([a-z]+),\s*(\d{4})", date_str)
    if match:
        day, month, year = match.groups()
        return (int(day), month.lower(), int(year))
    match = re.match(r"([a-z]+)\s+(\d{1,2}),?\s+(\d{4})", date_str)
    if match:
        month, day, year = match.groups()
        return (int(day), month.lower(), int(year))
    match = re.match(r"(\d{1,2})\s+([a-z]+)\s+(\d{4})", date_str)
    if match:
        day, month, year = match.groups()
        return (int(day), month.lower(), int(year))
    match = re.match(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", date_str)
    if match:
        year, month_num, day = match.groups()
        months = ['', 'january', 'february', 'march', 'april', 'may', 'june',
                  'july', 'august', 'september', 'october', 'november', 'december']
        month = months[int(month_num)]
        return (int(day), month, int(year))
    return None


def compare_answers(gold_answer, gen_answer, rule_name):
    gold_answer = (gold_answer or '').strip().lower()
    gen_answer = (gen_answer or '').strip().lower()
    if not gen_answer:
        return False
    if rule_name == 'birth_date':
        gold_date = normalize_date(gold_answer)
        gen_date = normalize_date(gen_answer)
        if gold_date and gen_date:
            return gold_date == gen_date
    return (gold_answer in gen_answer) or (gen_answer in gold_answer) or (gold_answer == gen_answer)


class LargeSupervisedSAE(nn.Module):
    """AlignSAE: 100,000 free slots + 6 relation slots."""
    def __init__(self, d_model, n_free=100000, n_relation=6, vocab_size=50257):
        super().__init__()
        self.n_free = n_free
        self.n_relation = n_relation
        self.n_slots = n_free + n_relation
        self.d_model = d_model
        
        self.encoder = nn.Linear(d_model, self.n_slots, bias=True)
        self.decoder = nn.Linear(self.n_slots, d_model, bias=True)
        
        self.value_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 256),
                nn.ReLU(),
                nn.Linear(256, vocab_size)
            )
            for _ in range(n_relation)
        ])
    
    def forward(self, h, temperature=1.0, hard=False):
        z = torch.relu(self.encoder(h))  # z = ReLU(W_e h + b_e)
        h_recon = self.decoder(z)
        return z, h_recon


def load_alignsae(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    d_model = ckpt['d_model']
    args = ckpt.get('args', {})
    n_free = args.get('n_free', 100000)
    n_relation = args.get('n_relation', 6)
    
    model = LargeSupervisedSAE(d_model=d_model, n_free=n_free, n_relation=n_relation)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model, n_relation


def swap_controllability_alignsae(sae, n_relation, lm_model, tokenizer, qa_file, kg_file, layer_idx=-1,
                                   alphas=(1.0, 10.0, 100.0), num_samples=100, max_new_tokens=50,
                                   output_dir='results/alignsae_swap_eval', steering_method='decoder_column'):
    """
    steering_method: 
        'decoder_column' - Use decoder column as steering vector (like G-SAE paper)
        'latent_decode' - Manipulate latent then decode (original AlignSAE method)
    """
    device = next(lm_model.parameters()).device
    rule_names = ['birth_date', 'birth_city', 'university', 'major', 'employer', 'work_city']
    qa_to_kg_mapping = {'birth_date': 'birth_date', 'birth_city': 'birth_city', 'university': 'university',
                        'major': 'major', 'employer': 'employer', 'company_city': 'work_city', 'work_city': 'work_city'}

    # load KG
    with open(kg_file, 'r') as f:
        kg = json.load(f)
    person_to_attrs = {p['person_id']: {
        'birth_date': p.get('birth_date'), 'birth_city': p.get('birth_city'),
        'university': p.get('university'), 'major': p.get('major'),
        'employer': p.get('employer'), 'work_city': p.get('work_city')
    } for p in kg}

    # load QA and sample
    with open(qa_file, 'r') as f:
        qa_pairs = [json.loads(line) for line in f]
    if num_samples is not None and num_samples < len(qa_pairs):
        random.seed(1234)
        qa_pairs = random.sample(qa_pairs, num_samples)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for alpha in alphas:
        print(f"\nRunning swaps for alpha={alpha} over {len(qa_pairs)} QA samples (layer={layer_idx})")
        records = []
        lm_model.eval()
        sae.eval()

        with torch.no_grad():
            for qa in tqdm(qa_pairs, desc=f"swap alpha={alpha}"):
                pid = qa.get('person_id')
                true_rule_idx = qa.get('rule_idx')
                true_rule_name = qa.get('rule_name')
                if pid not in person_to_attrs:
                    continue
                person_attrs = person_to_attrs[pid]

                prompt = f"Q: {qa['question']}\nA:"
                inputs = tokenizer(prompt, return_tensors='pt').to(device)

                # baseline generation
                baseline_gen = lm_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                                 pad_token_id=tokenizer.eos_token_id,
                                                 eos_token_id=tokenizer.eos_token_id)
                baseline_text = tokenizer.decode(baseline_gen[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                if '<end_of_text>' in baseline_text:
                    baseline_text = baseline_text.split('<end_of_text>')[0]
                baseline_text = baseline_text.strip().split('\n')[0]

                # hidden states
                outputs = lm_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[layer_idx]
                last_pos = inputs['input_ids'].shape[1] - 1
                h = hidden_states[0, last_pos, :].unsqueeze(0)  # [1, d]

                # encode
                z, _ = sae(h)
                z = z[0]  # [n_slots]

                for target_rule_idx in range(n_relation):
                    if target_rule_idx == true_rule_idx:
                        continue
                    target_rule_name = rule_names[target_rule_idx]
                    target_answer = person_attrs.get(qa_to_kg_mapping.get(target_rule_name, target_rule_name))
                    if not target_answer or str(target_answer).lower() in ['none', 'unknown', '']:
                        continue

                    if steering_method == 'decoder_column':
                        # G-SAE style: use decoder column as steering vector
                        # β_i = ||x||_2 / ||D_{:,i}||_2
                        # steering = α * β * D_{:,i}
                        # Only activate target (same as G-SAE paper)
                        
                        x_norm = h.norm(dim=1, keepdim=True)  # [1, 1]
                        decoder_weight = sae.decoder.weight  # [d_model, n_slots]
                        
                        # Relation slots are the last n_relation columns
                        target_col_idx = sae.n_free + target_rule_idx
                        
                        d_target = decoder_weight[:, target_col_idx]  # [d_model]
                        
                        beta_target = x_norm / (d_target.norm() + 1e-8)
                        
                        # Only activate target (like G-SAE)
                        steering = alpha * beta_target * d_target
                        steering = steering.squeeze()  # [d_model]
                    
                    elif steering_method == 'decoder_column_both':
                        # Activate target AND suppress source
                        x_norm = h.norm(dim=1, keepdim=True)
                        decoder_weight = sae.decoder.weight
                        
                        target_col_idx = sae.n_free + target_rule_idx
                        source_col_idx = sae.n_free + true_rule_idx
                        
                        d_target = decoder_weight[:, target_col_idx]
                        d_source = decoder_weight[:, source_col_idx]
                        
                        beta_target = x_norm / (d_target.norm() + 1e-8)
                        beta_source = x_norm / (d_source.norm() + 1e-8)
                        
                        steering = alpha * (beta_target * d_target - beta_source * d_source)
                        steering = steering.squeeze()
                    
                    elif steering_method == 'orthogonal':
                        # Orthogonalize target direction w.r.t. source
                        x_norm = h.norm(dim=1, keepdim=True)
                        decoder_weight = sae.decoder.weight
                        
                        target_col_idx = sae.n_free + target_rule_idx
                        source_col_idx = sae.n_free + true_rule_idx
                        
                        d_target = decoder_weight[:, target_col_idx].clone()
                        d_source = decoder_weight[:, source_col_idx]
                        
                        # Orthogonalize: remove source component from target
                        proj = (d_target @ d_source) / (d_source @ d_source + 1e-8)
                        d_target_orth = d_target - proj * d_source
                        
                        beta_target = x_norm / (d_target_orth.norm() + 1e-8)
                        beta_source = x_norm / (d_source.norm() + 1e-8)
                        
                        steering = alpha * (beta_target * d_target_orth - beta_source * d_source)
                        steering = steering.squeeze()
                    
                    elif steering_method == 'mean_diff':
                        # Use mean activation difference as steering
                        # This requires precomputed mean activations per rule
                        x_norm = h.norm(dim=1, keepdim=True)
                        decoder_weight = sae.decoder.weight
                        
                        target_col_idx = sae.n_free + target_rule_idx
                        source_col_idx = sae.n_free + true_rule_idx
                        
                        d_target = decoder_weight[:, target_col_idx]
                        d_source = decoder_weight[:, source_col_idx]
                        
                        # Difference vector
                        diff = d_target - d_source
                        beta = x_norm / (diff.norm() + 1e-8)
                        
                        steering = alpha * beta * diff
                        steering = steering.squeeze()
                        
                    else:  # latent_decode
                        # Original AlignSAE: manipulate latent then decode
                        z_swapped = z.clone()
                        z_swapped[-(n_relation - true_rule_idx)] = 0.0
                        z_swapped[-(n_relation - target_rule_idx)] = float(alpha)
                        h_swapped = sae.decoder(z_swapped.unsqueeze(0))
                        steering = (h_swapped[0] - h[0])  # [d_model]

                    # Use hook to modify hidden state at layer_idx
                    def make_hook(steer_vec, pos):
                        def hook_fn(module, input, output):
                            hidden = output[0]  # [batch, seq, d_model]
                            if hidden.shape[1] > pos:
                                hidden = hidden.clone()
                                hidden[:, pos, :] = hidden[:, pos, :] + steer_vec
                            return (hidden,) + output[1:] if len(output) > 1 else (hidden,)
                        return hook_fn
                    
                    # hidden_states[layer_idx] (where h was read) is the OUTPUT
                    # of block layer_idx-1, so steer at that same block.
                    if layer_idx < 1:
                        raise ValueError("layer_idx must be >= 1 (hidden_states[0] is the embedding output)")
                    layer_module = lm_model.transformer.h[layer_idx - 1]
                    hook = layer_module.register_forward_hook(make_hook(steering, last_pos))
                    
                    try:
                        prompt_len = inputs['input_ids'].shape[1]
                        out = lm_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                                pad_token_id=tokenizer.eos_token_id,
                                                eos_token_id=tokenizer.eos_token_id)
                        gen_text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
                    finally:
                        hook.remove()
                    
                    if '<end_of_text>' in gen_text:
                        gen_text = gen_text.split('<end_of_text>')[0]
                    gen_text = gen_text.strip().split('\n')[0]

                    is_correct = compare_answers(str(target_answer), gen_text, target_rule_name)

                    records.append({
                        'alpha': float(alpha),
                        'person_id': pid,
                        'question': qa['question'],
                        'original_relation': true_rule_name,
                        'swap_relation': target_rule_name,
                        'original_answer': person_attrs.get(qa_to_kg_mapping.get(true_rule_name, true_rule_name)),
                        'target_answer': target_answer,
                        'baseline': baseline_text,
                        'generated': gen_text,
                        'is_correct': bool(is_correct)
                    })

        df = pd.DataFrame(records)
        overall = df['is_correct'].mean() if len(df) > 0 else 0.0
        per_pair = {}
        for orig in rule_names:
            for targ in rule_names:
                if orig == targ:
                    continue
                subset = df[(df['original_relation'] == orig) & (df['swap_relation'] == targ)]
                if len(subset) > 0:
                    per_pair[f"{orig}_to_{targ}"] = float(subset['is_correct'].mean())

        res = {
            'overall_success_rate': float(overall),
            'per_rule_success': per_pair,
            'total_swaps_tested': int(len(records)),
            'detailed_results': records
        }
        all_results[f'alpha_{alpha}'] = res
        print(f"Alpha {alpha}: success={overall:.3f} tested={len(records)}")

        with open(output_dir / f'swap_alpha_{alpha}.json', 'w') as f:
            json.dump(res, f, indent=2)

    # Save combined results
    best_alpha = max(all_results.keys(), key=lambda k: all_results[k]['overall_success_rate'])
    best_rate = all_results[best_alpha]['overall_success_rate']
    
    summary = {
        'best_alpha': best_alpha,
        'best_success_rate': best_rate,
        'all_alphas': {k: v['overall_success_rate'] for k, v in all_results.items()}
    }
    with open(output_dir / 'swap_controllability_all_alphas.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nFinished swap sweep. Best {best_alpha} rate= {best_rate}")
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sae_checkpoint', type=str, required=True)
    parser.add_argument('--lm_model', type=str, required=True)
    parser.add_argument('--qa_file', type=str, required=True)
    parser.add_argument('--kg_file', type=str, required=True)
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--alphas', type=str, default='0.5,1,2,5,10')
    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='results/alignsae_swap_hook')
    parser.add_argument('--steering_method', type=str, default='decoder_column',
                        choices=['decoder_column', 'decoder_column_both', 'latent_decode', 'orthogonal', 'mean_diff'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading AlignSAE from {args.sae_checkpoint}")
    sae, n_relation = load_alignsae(args.sae_checkpoint, device)

    print(f"Loading LM from {args.lm_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.lm_model)
    lm_model = AutoModelForCausalLM.from_pretrained(args.lm_model).to(device)
    lm_model.eval()

    alphas = [float(a) for a in args.alphas.split(',')]

    swap_controllability_alignsae(
        sae=sae,
        n_relation=n_relation,
        lm_model=lm_model,
        tokenizer=tokenizer,
        qa_file=args.qa_file,
        kg_file=args.kg_file,
        layer_idx=args.layer,
        alphas=alphas,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
        steering_method=args.steering_method
    )


if __name__ == '__main__':
    main()
