"""
Step 5: Evaluate SAE - Verify 1-to-1 Mapping with Binding Accuracy
Tests: 
1. Question → Relation binding (does question activate correct slot?)
2. Relation → Answer binding (does slot predict correct answer?)
3. OOD generalization (in-distribution vs out-of-distribution templates)
"""
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re
import random
import pandas as pd

def normalize_date(date_str):
    """
    Normalize date strings to a common format for comparison.
    Handles formats like: "15,March,1985", "March 15, 1985", "1985-03-15", "15 March 1985"
    Returns: (day, month_name, year) tuple or None if parsing fails
    """
    date_str = date_str.strip().lower()
    
    # Try to extract day, month, year using various patterns
    # Pattern 1: "15,March,1985" or "15,march,1985"
    match = re.match(r'(\d{1,2}),\s*([a-z]+),\s*(\d{4})', date_str)
    if match:
        day, month, year = match.groups()
        return (int(day), month.lower(), int(year))
    
    # Pattern 2: "March 15, 1985" or "march 15 1985"
    match = re.match(r'([a-z]+)\s+(\d{1,2}),?\s+(\d{4})', date_str)
    if match:
        month, day, year = match.groups()
        return (int(day), month.lower(), int(year))
    
    # Pattern 3: "15 March 1985"
    match = re.match(r'(\d{1,2})\s+([a-z]+)\s+(\d{4})', date_str)
    if match:
        day, month, year = match.groups()
        return (int(day), month.lower(), int(year))
    
    # Pattern 4: "1985-03-15" or "1985/03/15"
    match = re.match(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', date_str)
    if match:
        year, month_num, day = match.groups()
        months = ['', 'january', 'february', 'march', 'april', 'may', 'june', 
                  'july', 'august', 'september', 'october', 'november', 'december']
        month = months[int(month_num)]
        return (int(day), month, int(year))
    
    return None

def compare_answers(gold_answer, gen_answer, rule_name):
    """
    Compare gold and generated answers with rule-specific handling.
    For dates: normalize and compare components.
    For others: flexible string matching.
    Returns False for empty generated answers.
    """
    gold_answer = gold_answer.strip().lower()
    gen_answer = gen_answer.strip().lower()
    
    # Return False for empty generated answers
    if not gen_answer:
        return False
    
    # Special handling for dates
    if rule_name == 'birth_date':
        gold_date = normalize_date(gold_answer)
        gen_date = normalize_date(gen_answer)
        
        if gold_date and gen_date:
            # Compare day, month, year
            return gold_date == gen_date
        # Fallback to string matching if parsing fails
    
    # For non-dates or failed date parsing: flexible matching
    return (
        gold_answer in gen_answer or 
        gen_answer in gold_answer or
        gold_answer == gen_answer
    )

# Import SAE model from 04_train_sae.py (numeric filename, so use importlib)
import importlib.util
spec = importlib.util.spec_from_file_location("train_sae", Path(__file__).parent / "04_train_sae.py")
train_sae_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_sae_module)
LargeSupervisedSAE = train_sae_module.LargeSupervisedSAE


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


def forward_with_patched_hidden(lm_model, inputs, h_new, position, layer_idx):
    """
    One forward pass with hidden_states[layer_idx][0, position] replaced by
    h_new. hidden_states[k] is the output of transformer block k-1 (index 0
    is the embedding output), so this hooks block layer_idx-1 and requires
    layer_idx >= 1.
    """
    if layer_idx < 1:
        raise ValueError("layer_idx must be >= 1 (hidden_states[0] is the embedding output)")
    hook = lm_model.transformer.h[layer_idx - 1].register_forward_hook(_patch_hook(h_new, position))
    try:
        return lm_model(**inputs)
    finally:
        hook.remove()


def generate_with_patched_hidden(lm_model, tokenizer, inputs, h_new, position, layer_idx, max_new_tokens=30):
    """
    Patch the residual stream at layer_idx / position, then continue the
    forward pass through the remaining layers and generate autoregressively
    (paper Appendix D.3). Returns the decoded continuation.
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
    """Load trained SAE model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    d_model = checkpoint['d_model']
    n_free = checkpoint['args']['n_free']
    n_relation = 6
    
    model = LargeSupervisedSAE(d_model=d_model, n_free=n_free, n_relation=n_relation)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint['args']

def evaluate_binding_accuracy(sae, lm_model, tokenizer, qa_file, kg_file, layer_idx=-1, split_name="test"):
    """
    Evaluate binding accuracy: Question → Relation → Answer
    Uses PURE QA format (no biography context) - knowledge retrieval mode.
    
    Returns:
        - slot_binding_acc: Does the question activate the correct slot?
        - answer_acc: Does the model generate the correct answer?
        - per_rule_metrics: Breakdown by each rule
    """
    device = next(lm_model.parameters()).device
    
    # Load data
    with open(qa_file, 'r') as f:
        qa_pairs = [json.loads(line) for line in f]
    
    n_relation = 6
    results = []
    rule_names = ['birth_date', 'birth_city', 'university', 'major', 'employer', 'work_city']
    
    # Map QA rule names to KG attribute names for per-rule metrics
    qa_to_kg_mapping = {
        'birth_date': 'birth_date',
        'birth_city': 'birth_city',
        'university': 'university',
        'major': 'major',
        'employer': 'employer',
        'company_city': 'work_city',  # QA uses 'company_city', KG uses 'work_city'
    }
    
    # Metrics
    total = 0
    slot_correct = 0
    slot_top3_correct = 0
    slot_top5_correct = 0
    answer_correct = 0
    total_margin = 0.0
    
    per_rule = defaultdict(lambda: {'total': 0, 'slot_correct': 0, 'slot_top3_correct': 0, 'slot_top5_correct': 0, 'answer_correct': 0, 'margin_sum': 0.0})
    
    lm_model.eval()
    sae.eval()
    
    with torch.no_grad():
        for qa in tqdm(qa_pairs, desc=f"Evaluating {split_name}"):
            # Pure QA format (no biography!)
            prompt = f"Q: {qa['question']}\nA:"
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            
            # Get model output with activations
            outputs = lm_model(**inputs, output_hidden_states=True)
            
            # Extract activation from answer position
            hidden_states = outputs.hidden_states[layer_idx]
            last_position = inputs['input_ids'].shape[1] - 1
            h = hidden_states[0, last_position, :]  # [d_model]
            
            # Pass through SAE
            z, h_recon = sae(h.unsqueeze(0))
            predicted_slot = z[0, -n_relation:].argmax(dim=-1).item()
            
            # Get top-k predictions for binding accuracy
            slot_logits = z[0, -n_relation:]
            top1_slot = slot_logits.argmax(dim=-1).item()
            top3_slots = slot_logits.topk(3, dim=-1).indices.tolist()
            top5_slots = slot_logits.topk(5, dim=-1).indices.tolist()
            
            # Margin: difference between top1 and top2
            sorted_logits = slot_logits.sort(descending=True).values
            margin = (sorted_logits[0] - sorted_logits[1]).item() if len(sorted_logits) > 1 else 0.0
            
            # Check slot binding: Does question activate correct slot?
            # (per-sample flags; do NOT reuse the accumulator names above)
            true_rule = qa['rule_idx']
            slot_is_correct = (predicted_slot == true_rule)
            is_top3_correct = true_rule in top3_slots
            is_top5_correct = true_rule in top5_slots
            
            # Generate answer
            generated = lm_model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(
                generated[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Clean up generated answer (remove everything after .<|endoftext|> or newline)
            if '.' in generated_text:
                generated_text = generated_text.split('.')[0].strip()
            if '\n' in generated_text:
                generated_text = generated_text.split('\n')[0].strip()
            
            # Check answer correctness (with rule-specific comparison)
            answer_is_correct = compare_answers(qa['answer'], generated_text, qa['rule_name'])
            
            # Update metrics
            total += 1
            if slot_is_correct:
                slot_correct += 1
            if is_top3_correct:
                slot_top3_correct += 1
            if is_top5_correct:
                slot_top5_correct += 1
            if answer_is_correct:
                answer_correct += 1
            total_margin += margin
            
            # Per-rule metrics - use KG attribute name (not QA rule name)
            kg_rule_name = qa_to_kg_mapping.get(qa['rule_name'], qa['rule_name'])
            per_rule[kg_rule_name]['total'] += 1
            if slot_is_correct:
                per_rule[kg_rule_name]['slot_correct'] += 1
            if is_top3_correct:
                per_rule[kg_rule_name]['slot_top3_correct'] += 1
            if is_top5_correct:
                per_rule[kg_rule_name]['slot_top5_correct'] += 1
            if answer_is_correct:
                per_rule[kg_rule_name]['answer_correct'] += 1
            per_rule[kg_rule_name]['margin_sum'] += margin
            
            results.append({
                'question': qa['question'],
                'gold_answer': qa['answer'],
                'generated_answer': generated_text,
                'true_rule': true_rule,
                'predicted_slot': predicted_slot,
                'slot_correct': slot_is_correct,
                'answer_correct': answer_is_correct,
                'rule_name': qa['rule_name'],  # Keep original QA rule name for results
            })
    
    # Calculate overall metrics
    slot_binding_acc = slot_correct / total if total > 0 else 0
    slot_top3_acc = slot_top3_correct / total if total > 0 else 0
    slot_top5_acc = slot_top5_correct / total if total > 0 else 0
    answer_acc = answer_correct / total if total > 0 else 0
    avg_margin = total_margin / total if total > 0 else 0.0
    
    # Calculate per-rule metrics
    per_rule_metrics = {}
    for rule_name in rule_names:
        if per_rule[rule_name]['total'] > 0:
            count = per_rule[rule_name]['total']
            per_rule_metrics[rule_name] = {
                'slot_binding_acc': per_rule[rule_name]['slot_correct'] / count,
                'slot_top3_acc': per_rule[rule_name]['slot_top3_correct'] / count,
                'slot_top5_acc': per_rule[rule_name]['slot_top5_correct'] / count,
                'answer_acc': per_rule[rule_name]['answer_correct'] / count,
                'avg_margin': per_rule[rule_name]['margin_sum'] / count,
                'count': count
            }
        else:
            per_rule_metrics[rule_name] = {
                'slot_binding_acc': 0.0,
                'slot_top3_acc': 0.0,
                'slot_top5_acc': 0.0,
                'answer_acc': 0.0,
                'avg_margin': 0.0,
                'count': 0
            }
    
    return {
        'slot_binding_acc': slot_binding_acc,
        'slot_top3_acc': slot_top3_acc,
        'slot_top5_acc': slot_top5_acc,
        'answer_acc': answer_acc,
        'avg_margin': avg_margin,
        'total': total,
        'per_rule_metrics': per_rule_metrics,
        'detailed_results': results
    }

def test_slot_assignment(sae, lm_model, tokenizer, qa_file, kg_file, layer_idx=-1):
    """
    Test 2: Check if slots are assigned correctly to rules.
    Returns confusion matrix: [predicted_slot, true_rule] for relation slots only.
    Uses pure QA format (no biography context).
    """
    device = next(lm_model.parameters()).device
    n_relation = 6
    
    # Load data
    with open(qa_file, 'r') as f:
        qa_pairs = [json.loads(line) for line in f]
    
    confusion = np.zeros((n_relation, n_relation))
    
    lm_model.eval()
    sae.eval()
    
    with torch.no_grad():
        for qa in tqdm(qa_pairs, desc="Testing slot assignment"):
            # Pure QA format (no biography!)
            prompt = f"Q: {qa['question']}\nA:"
            
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            outputs = lm_model(**inputs, output_hidden_states=True)
            
            hidden_states = outputs.hidden_states[layer_idx]
            last_position = inputs['input_ids'].shape[1] - 1
            h = hidden_states[0, last_position, :].unsqueeze(0)
            
            z, _ = sae(h)
            predicted_slot = z[0, -n_relation:].argmax(dim=-1).item()
            
            confusion[predicted_slot, qa['rule_idx']] += 1
    
    # Normalize by true rule counts
    rule_counts = confusion.sum(axis=0, keepdims=True)
    confusion_norm = confusion / (rule_counts + 1e-9)
    
    return confusion, confusion_norm

def causal_edits_evaluation(sae, lm_model, tokenizer, qa_file, layer_idx=6, num_samples=200):
    """
    Evaluate causal edits: Knock-Out (KO) and Knock-In (KI) on relation slots.
    Measures logit deltas and odds multipliers when intervening on slots.
    """
    device = next(lm_model.parameters()).device
    n_relation = 6

    # Load QA pairs and subsample for tractability
    with open(qa_file, 'r') as f:
        qa_pairs = [json.loads(line) for line in f]
    if num_samples is not None and num_samples < len(qa_pairs):
        random.seed(1234)
        qa_pairs = random.sample(qa_pairs, num_samples)
    results = []
    
    lm_model.eval()
    sae.eval()
    
    with torch.no_grad():
        for qa in tqdm(qa_pairs, desc="Causal edits evaluation"):
            prompt = f"Q: {qa['question']}\nA:"
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            
            # Get baseline logits (no intervention)
            outputs = lm_model(**inputs, output_hidden_states=True)
            baseline_logits = outputs.logits[0, -1, :]  # Last token logits
            
            # Get SAE activations
            hidden_states = outputs.hidden_states[layer_idx]
            last_position = inputs['input_ids'].shape[1] - 1
            h = hidden_states[0, last_position, :].unsqueeze(0)
            
            z, _ = sae(h)
            z = z[0]  # [n_slots]
            
            # For each relation slot, do KO and KI
            for slot_idx in range(n_relation):
                # Knock-Out: Set slot to 0
                z_ko = z.clone()
                z_ko[-(n_relation - slot_idx)] = 0.0  # Last 6 slots, indexed from end
                
                # Reconstruct activation with KO
                h_ko = sae.decoder(z_ko.unsqueeze(0))

                # Patch the residual stream at layer_idx and rerun the forward pass
                outputs_ko = forward_with_patched_hidden(
                    lm_model, inputs, h_ko[0], last_position, layer_idx)
                ko_logits = outputs_ko.logits[0, -1, :]

                # Logit delta for KO
                delta_ko = ko_logits - baseline_logits

                # Knock-In: Set slot to high value (e.g., 10)
                z_ki = z.clone()
                z_ki[-(n_relation - slot_idx)] = 10.0

                h_ki = sae.decoder(z_ki.unsqueeze(0))

                outputs_ki = forward_with_patched_hidden(
                    lm_model, inputs, h_ki[0], last_position, layer_idx)
                ki_logits = outputs_ki.logits[0, -1, :]
                
                delta_ki = ki_logits - baseline_logits
                
                # Get answer token logit changes
                answer_text = qa['answer']
                answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)[:5]  # First 5 tokens
                for token_idx, token in enumerate(answer_tokens):
                    if token_idx < len(delta_ko) and token_idx < len(delta_ki):
                        ko_delta = delta_ko[token].item()
                        ki_delta = delta_ki[token].item()
                        
                        results.append({
                            'question': qa['question'],
                            'rule_idx': qa['rule_idx'],
                            'slot_idx': slot_idx,
                            'token_idx': token_idx,
                            'token': token,
                            'ko_logit_delta': ko_delta,
                            'ki_logit_delta': ki_delta,
                            'ko_odds_mult': np.exp(ko_delta),
                            'ki_odds_mult': np.exp(ki_delta),
                        })
    
    # Aggregate results
    df = pd.DataFrame(results)
    
    # Per-slot averages
    per_slot = {}
    for slot_idx in range(n_relation):
        slot_data = df[df['slot_idx'] == slot_idx]
        per_slot[slot_idx] = {
            'ko_avg_delta': slot_data['ko_logit_delta'].mean(),
            'ki_avg_delta': slot_data['ki_logit_delta'].mean(),
            'ko_avg_odds': slot_data['ko_odds_mult'].mean(),
            'ki_avg_odds': slot_data['ki_odds_mult'].mean(),
        }
    
    return {
        'per_slot_summary': per_slot,
        'detailed_results': results
    }

def swap_controllability_evaluation(sae, lm_model, tokenizer, qa_file, kg_file, layer_idx=6,
                                    alphas=[1.0, 10.0, 100.0, 1000.0], num_samples=100):
    """
    Evaluate swap controllability: Test if we can control answer generation by swapping activated slots.
    For each question about attribute A, activate slot for attribute B and check if we get answer for B.
    Tests different alpha values for target slot activation strength.

    Protocol (paper Appendix D.3): suppress the original slot, set the target
    slot to alpha, decode, patch the residual stream at layer_idx, and let the
    model generate autoregressively.
    """
    device = next(lm_model.parameters()).device
    n_relation = 6
    
    # Load KG data
    with open(kg_file, 'r') as f:
        kg_data = json.load(f)
    
    # Create person_id to attributes mapping
    person_to_attrs = {}
    for person in kg_data:
        person_to_attrs[person['person_id']] = {
            'birth_date': person['birth_date'],
            'birth_city': person['birth_city'], 
            'university': person['university'],
            'major': person['major'],
            'employer': person['employer'],
            'work_city': person['work_city']
        }
    
    # Load QA data and subsample for tractability
    with open(qa_file, 'r') as f:
        qa_pairs = [json.loads(line) for line in f]
    if num_samples is not None and num_samples < len(qa_pairs):
        random.seed(1234)
        qa_pairs = random.sample(qa_pairs, num_samples)
    
    rule_names = ['birth_date', 'birth_city', 'university', 'major', 'employer', 'work_city']
    # Map QA rule names to KG attribute names
    qa_to_kg_mapping = {
        'birth_date': 'birth_date',
        'birth_city': 'birth_city',
        'university': 'university',
        'major': 'major',
        'employer': 'employer',
        'company_city': 'work_city',  # QA uses 'company_city', KG uses 'work_city'
        'work_city': 'work_city'      # Also handle direct mapping
    }
    
    all_results = {}
    
    for alpha in alphas:
        print(f"\nTesting alpha = {alpha}")
        results = []
        
        lm_model.eval()
        sae.eval()
        
        with torch.no_grad():
            for qa in tqdm(qa_pairs, desc=f"Swap evaluation (alpha={alpha})"):
                person_id = qa['person_id']
                true_rule_idx = qa['rule_idx']
                true_rule_name = qa['rule_name']
                
                if person_id not in person_to_attrs:
                    continue
                    
                person_attrs = person_to_attrs[person_id]
                
                # For each possible swap target (different from true rule)
                for target_rule_idx in range(n_relation):
                    if target_rule_idx == true_rule_idx:
                        continue
                        
                    target_rule_name = rule_names[target_rule_idx]
                    target_answer = person_attrs[qa_to_kg_mapping[target_rule_name]]
                    
                    # Get original answer for the true rule
                    original_answer = person_attrs[qa_to_kg_mapping[true_rule_name]]
                    
                    # Skip if target answer is empty
                    if not target_answer or target_answer.lower() in ['none', 'unknown', '']:
                        continue
                    
                    # Create prompt for the original question
                    prompt = f"Q: {qa['question']}\nA:"
                    
                    # Tokenize
                    inputs = tokenizer(prompt, return_tensors='pt').to(device)
                    
                    # BASELINE: Generate without any intervention
                    baseline_generated = lm_model.generate(
                        **inputs,
                        max_new_tokens=30,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    baseline_text = tokenizer.decode(
                        baseline_generated[0][inputs['input_ids'].shape[1]:], 
                        skip_special_tokens=True
                    ).strip()
                    if '.' in baseline_text:
                        baseline_text = baseline_text.split('.')[0].strip()
                    if '\n' in baseline_text:
                        baseline_text = baseline_text.split('\n')[0].strip()

                    # Get activations for intervention
                    outputs = lm_model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[layer_idx]
                    last_position = inputs['input_ids'].shape[1] - 1
                    h = hidden_states[0, last_position, :]
                    
                    # Get original SAE activations
                    z, _ = sae(h.unsqueeze(0))
                    z = z[0]  # [n_slots]
                    
                    # Swap: Suppress original slot and boost target slot
                    z_swapped = z.clone()
                    # Suppress the original slot (set to 0)
                    z_swapped[-(n_relation - true_rule_idx)] = 0.0
                    # Boost the target slot (set to alpha value)
                    z_swapped[-(n_relation - target_rule_idx)] = alpha
                    
                    # Reconstruct activation with swapped slots
                    h_swapped = sae.decoder(z_swapped.unsqueeze(0))

                    # INTERVENTION: patch the residual stream at layer_idx and
                    # generate autoregressively (paper Appendix D.3)
                    generated_text = generate_with_patched_hidden(
                        lm_model, tokenizer, inputs, h_swapped[0], last_position, layer_idx)
                    if '.' in generated_text:
                        generated_text = generated_text.split('.')[0].strip()
                    if '\n' in generated_text:
                        generated_text = generated_text.split('\n')[0].strip()

                    # Check if generated answer matches target answer
                    is_correct = compare_answers(target_answer, generated_text, target_rule_name)

                    results.append({
                        'alpha': alpha,
                        'person_id': person_id,
                        'question': qa['question'],
                        'original_relation': true_rule_name,
                        'swap_relation': target_rule_name,
                        'original_answer': original_answer,
                        'target_answer': target_answer,
                        'baseline_answer': baseline_text,
                        'generated_answer': generated_text,
                        'is_correct': is_correct
                    })
        
        # Calculate success rates for this alpha
        df = pd.DataFrame(results)
        
        # Overall success rate
        overall_success = df['is_correct'].mean()
        
        # Per-rule swap success rates
        per_rule_success = {}
        for orig_rule in rule_names:
            for target_rule in rule_names:
                if orig_rule != target_rule:
                    subset = df[(df['original_relation'] == orig_rule) & (df['swap_relation'] == target_rule)]
                    if len(subset) > 0:
                        success_rate = subset['is_correct'].mean()
                        key = f"{orig_rule}_to_{target_rule}"
                        per_rule_success[key] = success_rate
        
        all_results[alpha] = {
            'overall_success_rate': overall_success,
            'per_rule_success': per_rule_success,
            'total_swaps_tested': len(results),
            'detailed_results': results
        }
        
        print(f"Alpha {alpha}: Success rate = {overall_success:.3f} ({len(results)} swaps)")
    
    # Find best alpha
    best_alpha = max(all_results.keys(), key=lambda a: all_results[a]['overall_success_rate'])
    
    return {
        'all_results': all_results,
        'best_alpha': best_alpha,
        'best_success_rate': all_results[best_alpha]['overall_success_rate'],
        'summary': {alpha: all_results[alpha]['overall_success_rate'] for alpha in alphas}
    }

def evaluate_reconstruction_mse(sae, activation_file):
    """
    Evaluate SAE reconstruction MSE on held-out activations.
    """
    # Load activations
    with open(activation_file, 'rb') as f:
        activations = pickle.load(f)
    
    # Take a subset for efficiency
    activations = activations
    
    device = next(sae.parameters()).device
    sae.eval()
    
    total_mse = 0.0
    count = 0
    
    with torch.no_grad():
        for item in activations:
            h = torch.from_numpy(item['h']).float().to(device).unsqueeze(0)
            z, h_recon = sae(h)
            mse = F.mse_loss(h_recon, h).item()
            total_mse += mse
            count += 1
    
    avg_mse = total_mse / count if count > 0 else 0.0
    return avg_mse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sae_checkpoint', type=str, default='models/sae_large/sae_final.pt')
    parser.add_argument('--lm_model', type=str, default='models/base_sft/final')
    parser.add_argument('--activation_file', type=str, default='data/activations/train_activations.pkl',
                       help='Activation file for reconstruction MSE evaluation')
    parser.add_argument('--train_qa', type=str, default='data/generated/qa_train.jsonl',
                       help='Training QA for baseline')
    parser.add_argument('--train_kg', type=str, default='data/generated/train_kg.json')
    parser.add_argument('--test_qa_id', type=str, default='data/generated/qa_test_ood.jsonl',
                       help='QA file for the confusion matrix (held-out templates)')
    parser.add_argument('--test_qa_ood', type=str, default='data/generated/qa_test_ood.jsonl')
    parser.add_argument('--test_kg', type=str, default='data/generated/test_kg.json')
    parser.add_argument('--output_dir', type=str, default='results/sae_eval')
    parser.add_argument('--layer', type=int, default=6, help='hidden_states index (must match 03/04)')
    parser.add_argument('--causal_num_samples', type=int, default=200,
                       help='QA samples for the causal-edits (KO/KI) evaluation')
    parser.add_argument('--swap_num_samples', type=int, default=100,
                       help='QA samples for the swap-controllability evaluation')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load SAE
    print(f"Loading SAE from {args.sae_checkpoint}")
    sae, sae_args = load_sae(args.sae_checkpoint, device)
    
    # Load LM model
    print(f"Loading LM model from {args.lm_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.lm_model)
    lm_model = AutoModelForCausalLM.from_pretrained(args.lm_model)
    lm_model.to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    rule_names = ['birth_date', 'birth_city', 'university', 'major', 'employer', 'work_city']
    
    # ===================================================================
    # BINDING ACCURACY EVALUATION
    # ===================================================================
    
    print("\n" + "="*60)
    print("BINDING ACCURACY EVALUATION")
    print("="*60)
    
    # Test on TRAIN set (sanity check - should be very high)
    print("\n=== Train Set (templates 0-1, train persons) ===")
    train_results = evaluate_binding_accuracy(
        sae, lm_model, tokenizer, args.train_qa, args.train_kg, args.layer, split_name="Train"
    )
    
    print(f"\nTrain Slot Binding Accuracy: {train_results['slot_binding_acc']:.3f}")
    print(f"Train Slot Top-3 Accuracy: {train_results['slot_top3_acc']:.3f}")
    print(f"Train Slot Top-5 Accuracy: {train_results['slot_top5_acc']:.3f}")
    print(f"Train Average Margin: {train_results['avg_margin']:.3f}")
    print(f"Train Answer Accuracy: {train_results['answer_acc']:.3f}")
    print("\nPer-rule (Train):")
    for rule_name in rule_names:
        metrics = train_results['per_rule_metrics'][rule_name]
        print(f"  {rule_name:20s}: Slot={metrics['slot_binding_acc']:.3f}, Top3={metrics['slot_top3_acc']:.3f}, Top5={metrics['slot_top5_acc']:.3f}, Margin={metrics['avg_margin']:.3f}, Ans={metrics['answer_acc']:.3f} (n={metrics['count']})")
    
    # Test on OUT-OF-DISTRIBUTION test set (unseen templates 2-3, different persons)
    print("\n=== Out-of-Distribution Test Set (templates 2-3, test persons) ===")
    ood_results = evaluate_binding_accuracy(
        sae, lm_model, tokenizer, args.test_qa_ood, args.test_kg, args.layer, split_name="Test-OOD"
    )
    
    print(f"\nTest-OOD Slot Binding Accuracy: {ood_results['slot_binding_acc']:.3f}")
    print(f"Test-OOD Slot Top-3 Accuracy: {ood_results['slot_top3_acc']:.3f}")
    print(f"Test-OOD Slot Top-5 Accuracy: {ood_results['slot_top5_acc']:.3f}")
    print(f"Test-OOD Average Margin: {ood_results['avg_margin']:.3f}")
    print(f"Test-OOD Answer Accuracy: {ood_results['answer_acc']:.3f}")
    print("\nPer-rule (Test-OOD):")
    for rule_name in rule_names:
        metrics = ood_results['per_rule_metrics'][rule_name]
        print(f"  {rule_name:20s}: Slot={metrics['slot_binding_acc']:.3f}, Top3={metrics['slot_top3_acc']:.3f}, Top5={metrics['slot_top5_acc']:.3f}, Margin={metrics['avg_margin']:.3f}, Ans={metrics['answer_acc']:.3f} (n={metrics['count']})")
    
    # ===================================================================
    # SAE RECONSTRUCTION QUALITY
    # ===================================================================
    
    print("\n=== SAE Reconstruction Quality ===")
    recon_mse = evaluate_reconstruction_mse(sae, args.activation_file)
    print(f"Reconstruction MSE: {recon_mse:.6f}")
    
    # ===================================================================
    # CONFUSION MATRIX (Slot Assignment)
    # ===================================================================
    
    print("\n=== Slot Assignment Confusion Matrix (held-out templates) ===")
    confusion, confusion_norm = test_slot_assignment(
        sae, lm_model, tokenizer, args.test_qa_id, args.test_kg, args.layer
    )
    
    print("\nConfusion Matrix (rows=predicted slot, cols=true rule):")
    print(confusion_norm)
    
    diagonal_acc = np.trace(confusion_norm) / 6
    print(f"\nDiagonal accuracy: {diagonal_acc:.3f} (1.0 = perfect 1-to-1)")
    
    # ===================================================================
    # CAUSAL EDITS EVALUATION
    # ===================================================================
    
    print("\n=== Causal Edits Evaluation (KO/KI) ===")
    causal_results = causal_edits_evaluation(
        sae, lm_model, tokenizer, args.train_qa, args.layer,
        num_samples=args.causal_num_samples
    )
    
    print("\nPer-slot causal effects:")
    for slot_idx in range(6):
        summary = causal_results['per_slot_summary'][slot_idx]
        print(f"  Slot {slot_idx}: KO Δ={summary['ko_avg_delta']:.3f} (×{summary['ko_avg_odds']:.2f}), "
              f"KI Δ={summary['ki_avg_delta']:.3f} (×{summary['ki_avg_odds']:.2f})")
    
    # ===================================================================
    # SWAP CONTROLLABILITY EVALUATION
    # ===================================================================
    
    print("\n=== Swap Controllability Evaluation ===")
    swap_results = swap_controllability_evaluation(
        sae, lm_model, tokenizer, args.train_qa, args.train_kg, args.layer,
        num_samples=args.swap_num_samples
    )
    
    print(f"Best alpha: {swap_results['best_alpha']} (success rate: {swap_results['best_success_rate']:.3f})")
    print("\nAlpha comparison:")
    for alpha, success_rate in swap_results['summary'].items():
        print(f"  Alpha {alpha:4.0f}: {success_rate:.3f}")
    
    # Use best alpha results for detailed analysis
    best_results = swap_results['all_results'][swap_results['best_alpha']]
    print(f"\nBest alpha ({swap_results['best_alpha']}) results:")
    print(f"Swap Controllability Success Rate: {best_results['overall_success_rate']:.3f}")
    print(f"Total swaps tested: {best_results['total_swaps_tested']}")
    
    # Show some examples from best alpha
    print("\nExample swaps (best alpha):")
    for i, result in enumerate(best_results['detailed_results'][:5]):
        status = "✅" if result['is_correct'] else "❌"
        print(f"  {status} {result['original_relation']} → {result['swap_relation']}: '{result['generated_answer']}' (expected: '{result['target_answer']}')")
    
    # ===================================================================
    # VISUALIZATIONS
    # ===================================================================
    
    print("\n=== Generating Visualizations ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot 1: Slot assignment confusion matrix
    sns.heatmap(
        confusion_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=rule_names,
        yticklabels=[f'Slot {i}' for i in range(6)],
        ax=axes[0, 0],
        cbar_kws={'label': 'Fraction'},
        vmin=0, vmax=1
    )
    axes[0, 0].set_title('Slot Assignment Confusion (held-out templates)\n(Should be diagonal for 1-to-1)')
    axes[0, 0].set_xlabel('True Rule')
    axes[0, 0].set_ylabel('Predicted Slot')
    
    # Plot 2: Binding accuracy comparison
    splits = ['Train', 'Test-OOD']
    slot_accs = [
        train_results['slot_binding_acc'],
        ood_results['slot_binding_acc']
    ]
    answer_accs = [
        train_results['answer_acc'],
        ood_results['answer_acc']
    ]
    
    x = np.arange(len(splits))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, slot_accs, width, label='Slot Binding Acc', alpha=0.8)
    axes[0, 1].bar(x + width/2, answer_accs, width, label='Answer Acc', alpha=0.8)
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Binding Accuracy: Question → Relation → Answer')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(splits)
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 1.0])
    axes[0, 1].axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='Target')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Per-rule binding accuracy (Train)
    rule_slot_accs = [train_results['per_rule_metrics'][r]['slot_binding_acc'] for r in rule_names]
    rule_answer_accs = [train_results['per_rule_metrics'][r]['answer_acc'] for r in rule_names]
    
    x_rules = np.arange(len(rule_names))
    axes[1, 0].bar(x_rules - width/2, rule_slot_accs, width, label='Slot Binding', alpha=0.8)
    axes[1, 0].bar(x_rules + width/2, rule_answer_accs, width, label='Answer Acc', alpha=0.8)
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Per-Rule Accuracy (Train)')
    axes[1, 0].set_xticks(x_rules)
    axes[1, 0].set_xticklabels(rule_names, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].set_ylim([0, 1.0])
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Per-rule binding accuracy (Test-OOD)
    ood_slot_accs = [ood_results['per_rule_metrics'][r]['slot_binding_acc'] for r in rule_names]
    ood_answer_accs = [ood_results['per_rule_metrics'][r]['answer_acc'] for r in rule_names]
    
    axes[1, 1].bar(x_rules - width/2, ood_slot_accs, width, label='Slot Binding', alpha=0.8)
    axes[1, 1].bar(x_rules + width/2, ood_answer_accs, width, label='Answer Acc', alpha=0.8)
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Per-Rule Accuracy (Test-OOD)')
    axes[1, 1].set_xticks(x_rules)
    axes[1, 1].set_xticklabels(rule_names, rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].set_ylim([0, 1.0])
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'binding_accuracy_evaluation.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_dir / 'binding_accuracy_evaluation.png'}")
    plt.close()
    
    # ===================================================================
    # SAVE RESULTS
    # ===================================================================
    
    results = {
        'train': {
            'slot_binding_acc': float(train_results['slot_binding_acc']),
            'slot_top3_acc': float(train_results['slot_top3_acc']),
            'slot_top5_acc': float(train_results['slot_top5_acc']),
            'avg_margin': float(train_results['avg_margin']),
            'answer_acc': float(train_results['answer_acc']),
            'per_rule': {r: {k: float(v) for k, v in train_results['per_rule_metrics'][r].items()} 
                        for r in rule_names}
        },
        'test_ood': {
            'slot_binding_acc': float(ood_results['slot_binding_acc']),
            'slot_top3_acc': float(ood_results['slot_top3_acc']),
            'slot_top5_acc': float(ood_results['slot_top5_acc']),
            'avg_margin': float(ood_results['avg_margin']),
            'answer_acc': float(ood_results['answer_acc']),
            'per_rule': {r: {k: float(v) for k, v in ood_results['per_rule_metrics'][r].items()} 
                        for r in rule_names}
        },
        'diagonal_accuracy': float(diagonal_acc),
        'confusion_matrix': confusion_norm.tolist(),
        'reconstruction_mse': float(recon_mse),
        'causal_edits': causal_results['per_slot_summary'],
        'swap_controllability': {
            'best_alpha': swap_results['best_alpha'],
            'best_success_rate': float(swap_results['best_success_rate']),
            'alpha_comparison': {str(k): float(v) for k, v in swap_results['summary'].items()},
            'best_results': {
                'overall_success_rate': float(best_results['overall_success_rate']),
                'total_swaps_tested': best_results['total_swaps_tested'],
                'per_rule_success': {k: float(v) for k, v in best_results['per_rule_success'].items()}
            }
        }
    }
    
    with open(output_dir / 'binding_accuracy_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save sample predictions
    sample_predictions = {
        'train_samples': train_results['detailed_results'][:20],
        'test_ood_samples': ood_results['detailed_results'][:20]
    }
    
    with open(output_dir / 'sample_predictions.json', 'w') as f:
        json.dump(sample_predictions, f, indent=2)
    
    # Save detailed swap results (all alphas)
    with open(output_dir / 'swap_controllability_detailed.json', 'w') as f:
        json.dump(swap_results['all_results'], f, indent=2)
    
    print(f"\nResults saved to {output_dir}/")
    
    # ===================================================================
    # SUMMARY
    # ===================================================================
    
    print("\n" + "="*60)
    print("SUMMARY: 1-to-1 SAE BINDING ACCURACY")
    print("="*60)
    print(f"\nQuestion → Relation Binding (Slot Activation):")
    print(f"  Train:    {train_results['slot_binding_acc']:.3f}")
    print(f"  Test-OOD: {ood_results['slot_binding_acc']:.3f}")
    
    print(f"\nRelation → Answer Binding (Answer Generation):")
    print(f"  Train:    {train_results['answer_acc']:.3f}")
    print(f"  Test-OOD: {ood_results['answer_acc']:.3f}")
    
    print(f"\nDiagonal Accuracy (1-to-1 mapping): {diagonal_acc:.3f}")
    
    # Success criteria
    success = (
        train_results['slot_binding_acc'] >= 0.85 and
        ood_results['slot_binding_acc'] >= 0.75 and
        diagonal_acc >= 0.85
    )
    
    print(f"\nOverall Assessment: {'✅ SUCCESSFUL' if success else '⚠️  NEEDS IMPROVEMENT'}")
    
    if not success:
        print("\nSuggestions:")
        if train_results['slot_binding_acc'] < 0.85:
            print("  - Increase lambda_align in SAE training")
            print("  - Train SAE for more epochs")
        if ood_results['slot_binding_acc'] < 0.75:
            print("  - Add more diverse question templates")
            print("  - Increase training data size")
        if diagonal_acc < 0.85:
            print("  - Check for slot collapse (multiple rules → same slot)")
            print("  - Increase lambda_indep to decorrelate slots")
    
    print("="*60)

if __name__ == "__main__":
    main()
