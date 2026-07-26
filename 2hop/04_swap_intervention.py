"""
Step 4: Swap Intervention Experiment for 2-hop Reasoning
Test if SAE learned causal relation binding by swapping R2 to alternative relations.

Intervention method:
1. At Token-2 position (after generating E2)
2. Suppress R2 supervised slot: z[R2_idx] = 0
3. Activate alternative R_swap slot: z[R_swap_idx] = alpha
4. Generate E3 and check if it matches the swapped relation's target

Layer: 6 (where concept binding occurs)
"""
import torch
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
from pathlib import Path


class LargeSupervisedSAE(nn.Module):
    """Large SAE: 100,000 free + 20 supervised relation slots"""
    def __init__(self, d_model, n_free=100000, n_relation=20, vocab_size=50383):
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
        
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.T.clone()
    
    def forward(self, h):
        z = torch.relu(self.encoder(h))  # z = ReLU(W_e h + b_e)
        h_recon = self.decoder(z)
        return z, h_recon
    
    def get_relation_activations(self, z):
        """Extract supervised relation slot activations (last 20 slots)"""
        return z[:, -self.n_relation:]


RELATIONS = [
    'accuses', 'admires', 'blames', 'boss_of', 'classmate_of',
    'competes_with', 'cousin_of', 'endorsed_by', 'follows', 'forgives',
    'friend_of', 'has_crush_on', 'mentor_of', 'neighbor_of', 'owes_debt_to',
    'protects', 'reports_to', 'subscribes_to', 'warns', 'works_with'
]

RELATION_TO_IDX = {rel: idx for idx, rel in enumerate(RELATIONS)}


def load_models(lm_path, sae_path, device):
    """Load language model and SAE"""
    tokenizer = AutoTokenizer.from_pretrained(lm_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    lm_model = AutoModelForCausalLM.from_pretrained(lm_path).to(device)
    lm_model.eval()
    
    d_model = lm_model.config.hidden_size
    sae = LargeSupervisedSAE(d_model=d_model, n_free=100000, n_relation=20).to(device)
    checkpoint = torch.load(sae_path, map_location=device, weights_only=False)
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.eval()
    
    return tokenizer, lm_model, sae


def load_facts_database(facts_db_path):
    """Load entity relations mapping"""
    with open(facts_db_path, 'r') as f:
        entity_relations = json.load(f)
    return entity_relations


def get_valid_swap_relations(entity, original_rel, entity_relations):
    """Get all valid alternative relations for an entity (excluding original)"""
    if entity not in entity_relations:
        return []
    all_relations = entity_relations[entity]
    return [(rel, target) for rel, target in all_relations if rel != original_rel]


def intervention_forward_pass(
    tokenizer, lm_model, sae, question, entity_2,
    original_rel_idx, swap_rel_idx, alpha, layer_idx, device, max_new_tokens=1
):
    """
    Run forward pass with relation swap intervention at Token-2 position.
    
    Intervention on supervised relation slots:
    - Suppress original R2 slot to 0
    - Activate swap relation slot to alpha
    """
    # Match the training format exactly: a real newline before "Answer:" and
    # the E2 token concatenated directly after it (no space; entities are
    # dedicated single tokens).
    prompt_ids = tokenizer(f"Question: {question}\nAnswer:",
                           return_tensors='pt')['input_ids'].to(device)
    e2_ids = tokenizer.encode(entity_2, add_special_tokens=False)
    input_ids = torch.cat([prompt_ids, torch.tensor([e2_ids], device=device)], dim=1)
    inputs = {'input_ids': input_ids, 'attention_mask': torch.ones_like(input_ids)}

    intervention_applied = False
    
    def intervention_hook(module, input, output):
        nonlocal intervention_applied
        if not intervention_applied:
            hidden = output[0]
            h_last = hidden[0, -1, :].clone()
            
            with torch.no_grad():
                # Get all slot activations (same encoding as sae.forward)
                z = torch.relu(sae.encoder(h_last))  # [n_slots]
                z_intervened = z.clone()
                
                # Intervene on supervised relation slots (last 20)
                relation_slot_offset = sae.n_free
                z_intervened[relation_slot_offset + original_rel_idx] = 0.0
                z_intervened[relation_slot_offset + swap_rel_idx] = alpha
                
                # Reconstruct
                h_intervened = sae.decoder(z_intervened)
                
                hidden_modified = hidden.clone()
                hidden_modified[0, -1, :] = h_intervened
                intervention_applied = True
                return (hidden_modified,) + output[1:]
        return output
    
    # 01_collect reads outputs.hidden_states[layer_idx], which is the OUTPUT
    # of block layer_idx-1 (hidden_states[0] is the embedding output). Hook
    # the same block so the SAE sees the representation it was trained on.
    if layer_idx < 1:
        raise ValueError("layer_idx must be >= 1 (hidden_states[0] is the embedding output)")
    hook_handle = lm_model.transformer.h[layer_idx - 1].register_forward_hook(intervention_hook)
    
    with torch.no_grad():
        generated = lm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    hook_handle.remove()
    predicted_text = tokenizer.decode(generated[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return predicted_text.strip()


def evaluate_single_alpha(
    tokenizer, lm_model, sae, val_data_path, entity_relations,
    alpha, layer_idx, device, max_samples=None, max_new_tokens=1
):
    """Evaluate swap intervention for ONE alpha value"""
    qa_pairs = []
    with open(val_data_path, 'r') as f:
        for line in f:
            qa_pairs.append(json.loads(line.strip()))
    
    if max_samples:
        qa_pairs = qa_pairs[:max_samples]
    
    result = {
        'alpha': alpha,
        'total': 0,
        'success': 0,
        'by_relation': {},
        'examples': []
    }
    
    for qa in tqdm(qa_pairs, desc=f"Alpha={alpha}"):
        question = qa['question']
        e1 = qa['entity_1']
        r1 = qa['relation_1']
        e2 = qa['entity_2']
        r2 = qa['relation_2']
        e3 = qa['entity_3']
        
        swap_options = get_valid_swap_relations(e2, r2, entity_relations)
        if not swap_options:
            continue
        
        for swap_rel, swap_target in swap_options:
            r2_idx = RELATION_TO_IDX[r2]
            swap_rel_idx = RELATION_TO_IDX[swap_rel]
            
            predicted = intervention_forward_pass(
                tokenizer, lm_model, sae, question, e2,
                r2_idx, swap_rel_idx, alpha, layer_idx, device, max_new_tokens
            )
            
            success = swap_target in predicted
            
            result['total'] += 1
            if success:
                result['success'] += 1
            
            if swap_rel not in result['by_relation']:
                result['by_relation'][swap_rel] = {'total': 0, 'success': 0}
            result['by_relation'][swap_rel]['total'] += 1
            if success:
                result['by_relation'][swap_rel]['success'] += 1
            
            if len(result['examples']) < 100:
                result['examples'].append({
                    'question': question,
                    'e1': e1,
                    'r1': r1,
                    'e2': e2,
                    'original_r2': r2,
                    'original_e3': e3,
                    'swap_rel': swap_rel,
                    'swap_target': swap_target,
                    'alpha': alpha,
                    'predicted': predicted,
                    'success': success
                })
    
    result['success_rate'] = result['success'] / result['total'] if result['total'] > 0 else 0.0
    
    for rel in result['by_relation']:
        rel_total = result['by_relation'][rel]['total']
        rel_success = result['by_relation'][rel]['success']
        result['by_relation'][rel]['success_rate'] = rel_success / rel_total if rel_total > 0 else 0.0
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Swap intervention for ONE alpha value")
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--lm_model', type=str,
                        default='2hop/_trained_model_4k_mixed/checkpoint_epoch_100')
    parser.add_argument('--sae_checkpoint', type=str,
                        default='2hop/sae_large/sae_best.pt')
    parser.add_argument('--val_data', type=str,
                        default='2hop/_dataset/_gen/val_two_hop_qa_data_4k.jsonl')
    parser.add_argument('--facts_db', type=str,
                        default='2hop/facts_database/entity_relations.json')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=1)
    parser.add_argument('--output_dir', type=str,
                        default='2hop/swap_results')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"Evaluating Alpha = {args.alpha}")
    print(f"Layer: {args.layer}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"{'='*60}\n")
    
    tokenizer, lm_model, sae = load_models(args.lm_model, args.sae_checkpoint, device)
    entity_relations = load_facts_database(args.facts_db)
    
    result = evaluate_single_alpha(
        tokenizer, lm_model, sae,
        args.val_data, entity_relations,
        args.alpha, args.layer, device,
        args.max_samples, args.max_new_tokens
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"alpha_{args.alpha}.json"
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✓ Alpha={args.alpha}: {result['success']}/{result['total']} ({result['success_rate']*100:.1f}%)")
    print(f"✓ Saved to {output_file}\n")


if __name__ == '__main__':
    main()
