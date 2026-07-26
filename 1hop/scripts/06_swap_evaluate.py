#!/usr/bin/env python3
"""
Focused swap controllability evaluator.

Creates a standalone, configurable evaluation that:
- loads a trained SAE checkpoint
- loads a causal LM (SFT) checkpoint
- samples QA pairs and KG entries
- for a grid of alpha values, intervenes on SAE relation slots by swapping
  (suppress true slot, boost target slot to alpha)
- generates up to `max_new_tokens` tokens and stops at the literal
  marker "<end_of_text>" if present in the generation
- saves detailed results and a summary JSON

This script intentionally does not modify any existing evaluation code.
"""
import argparse
import json
import random
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM


def normalize_date(date_str):
    import re
    date_str = date_str.strip().lower()
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


def _patch_hook(h_new, position):
    """Forward hook that replaces one position of a block's output hidden state."""
    def hook_fn(module, module_in, module_out):
        hidden = module_out[0]
        # Apply only on the prefill pass; later generation steps have seq_len 1
        if hidden.shape[1] > position:
            hidden = hidden.clone()
            hidden[0, position, :] = h_new
            return (hidden,) + module_out[1:]
        return module_out
    return hook_fn


def generate_with_patched_hidden(lm_model, tokenizer, inputs, h_new, position, layer_idx, max_new_tokens=50):
    """
    Patch the residual stream at layer_idx / position, then continue the
    forward pass through the remaining layers and generate autoregressively
    (paper Appendix D.3). hidden_states[k] is the output of transformer block
    k-1 (index 0 is the embedding output), so this hooks block layer_idx-1.
    """
    if layer_idx < 1:
        raise ValueError("layer_idx must be >= 1 (hidden_states[0] is the embedding output)")
    hook = lm_model.transformer.h[layer_idx - 1].register_forward_hook(_patch_hook(h_new, position))
    try:
        generated = lm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    finally:
        hook.remove()
    return tokenizer.decode(
        generated[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
    ).strip()


def load_sae(checkpoint_path, device):
    # Lazy import of SAE training module structures
    import importlib.util
    from pathlib import Path

    cp = torch.load(checkpoint_path, map_location=device)
    d_model = cp['d_model']
    # detect args/n_free
    args = cp.get('args', {})
    n_free = args.get('n_free', 16)
    n_relation = args.get('n_relation', 6) if 'n_relation' in args else 6

    # Try to load LargeSupervisedSAE from 04_train_sae.py if available
    train_sae_path = Path(__file__).parent / '04_train_sae.py'
    if train_sae_path.exists():
        spec = importlib.util.spec_from_file_location('train_sae', train_sae_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        SAEClass = getattr(mod, 'LargeSupervisedSAE', None) or getattr(mod, 'SupervisedSAE', None)
        if SAEClass is None:
            raise RuntimeError('Could not find SAE class in scripts/04_train_sae.py')
        model = SAEClass(d_model=d_model, n_free=n_free, n_relation=n_relation)
    else:
        raise RuntimeError(f'Cannot find training module to construct SAE at {train_sae_path}')

    model.load_state_dict(cp['model_state_dict'])
    model.to(device)
    model.eval()
    return model, args


def swap_controllability_only(sae, lm_model, tokenizer, qa_file, kg_file, layer_idx=-1,
                             alphas=(1.0, 10.0, 100.0), num_samples=100, max_new_tokens=50,
                             output_dir='results/swap_eval'):
    device = next(lm_model.parameters()).device
    n_relation = 6

    # load KG
    with open(kg_file, 'r') as f:
        kg = json.load(f)
    person_to_attrs = {p['person_id']:{'birth_date':p.get('birth_date'), 'birth_city':p.get('birth_city'),
                                        'university':p.get('university'),'major':p.get('major'),
                                        'employer':p.get('employer'),'work_city':p.get('work_city')}
                       for p in kg}

    # load QA and sample
    with open(qa_file, 'r') as f:
        qa_pairs = [json.loads(line) for line in f]

    if num_samples is not None and num_samples < len(qa_pairs):
        random.seed(1234)
        qa_pairs = random.sample(qa_pairs, num_samples)

    rule_names = ['birth_date', 'birth_city', 'university', 'major', 'employer', 'work_city']
    qa_to_kg_mapping = {'birth_date':'birth_date','birth_city':'birth_city','university':'university',
                        'major':'major','employer':'employer','company_city':'work_city','work_city':'work_city'}

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

                # baseline prompt
                prompt = f"Q: {qa['question']}\nA:"
                inputs = tokenizer(prompt, return_tensors='pt').to(device)

                # baseline generation (generate full sequence)
                baseline_gen = lm_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                                 pad_token_id=tokenizer.eos_token_id,
                                                 eos_token_id=tokenizer.eos_token_id)
                baseline_text = tokenizer.decode(baseline_gen[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                if '<end_of_text>' in baseline_text:
                    baseline_text = baseline_text.split('<end_of_text>')[0]
                baseline_text = baseline_text.strip().split('\n')[0]

                # get hidden states to compute activation
                outputs = lm_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[layer_idx]
                last_pos = inputs['input_ids'].shape[1] - 1
                h = hidden_states[0, last_pos, :]

                z, _ = sae(h.unsqueeze(0))
                z = z[0]

                for target_rule_idx in range(n_relation):
                    if target_rule_idx == true_rule_idx:
                        continue
                    target_rule_name = rule_names[target_rule_idx]
                    target_answer = person_attrs.get(qa_to_kg_mapping.get(target_rule_name, target_rule_name))
                    if not target_answer or str(target_answer).lower() in ['none', 'unknown', '']:
                        continue

                    z_swapped = z.clone()
                    # suppress original slot
                    z_swapped[-(n_relation - true_rule_idx)] = 0.0
                    # boost target
                    z_swapped[-(n_relation - target_rule_idx)] = float(alpha)

                    h_swapped = sae.decoder(z_swapped.unsqueeze(0))

                    # Patch the residual stream at layer_idx and generate
                    # autoregressively (paper Appendix D.3)
                    gen_text = generate_with_patched_hidden(
                        lm_model, tokenizer, inputs, h_swapped[0], last_pos,
                        layer_idx, max_new_tokens=max_new_tokens)
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
        all_results[float(alpha)] = res

        # save per-alpha file to avoid losing progress
        with open(output_dir / f'swap_alpha_{alpha:.6g}.json', 'w') as f:
            json.dump(res, f, indent=2)

        print(f"Alpha {alpha}: success={overall:.3f} tested={len(records)}")

    # summary
    summary = {str(k): float(v['overall_success_rate']) for k, v in all_results.items()}
    best_alpha = max(all_results.keys(), key=lambda a: all_results[a]['overall_success_rate'])

    final = {
        'all_results': all_results,
        'summary': summary,
        'best_alpha': float(best_alpha),
        'best_success_rate': float(all_results[best_alpha]['overall_success_rate'])
    }

    with open(output_dir / 'swap_controllability_all_alphas.json', 'w') as f:
        json.dump(final, f, indent=2)

    print('\nFinished swap sweep. Best alpha=', final['best_alpha'], 'rate=', final['best_success_rate'])
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sae_checkpoint', type=str, default='models/sae_large/sae_final.pt')
    parser.add_argument('--lm_model', type=str, default='models/base_sft/final')
    parser.add_argument('--qa_file', type=str, default='data/generated/qa_test_ood.jsonl')
    parser.add_argument('--kg_file', type=str, default='data/generated/test_kg.json')
    parser.add_argument('--layer', type=int, default=6, help='hidden_states index (must match 03/04)')
    parser.add_argument('--alphas', type=str, default='0.1,0.5,1,2,5,10,20,50,100,200,500,1000')
    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='results/sae_eval_swap_extended')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    print('Loading SAE from', args.sae_checkpoint)
    sae, sae_args = load_sae(args.sae_checkpoint, device)

    print('Loading LM from', args.lm_model)
    tokenizer = AutoTokenizer.from_pretrained(args.lm_model)
    lm_model = AutoModelForCausalLM.from_pretrained(args.lm_model)
    lm_model.to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    alphas = [float(x) for x in args.alphas.split(',') if x.strip()]

    swap_controllability_only(sae, lm_model, tokenizer,
                             args.qa_file, args.kg_file,
                             layer_idx=args.layer, alphas=alphas,
                             num_samples=args.num_samples, max_new_tokens=args.max_new_tokens,
                             output_dir=args.output_dir)


if __name__ == '__main__':
    main()
