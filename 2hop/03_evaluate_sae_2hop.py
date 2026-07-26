"""
Step 3: Evaluate Large SAE for 2-hop Reasoning
Verify binding accuracy of supervised relation slots.

This evaluates:
1. Slot binding accuracy: Does the question activate the correct relation slot?
2. Visualization of binding matrix (20 relations × 20 supervised slots)
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

# 20 relations (alphabetically sorted)
RELATIONS = [
    'accuses', 'admires', 'blames', 'boss_of', 'classmate_of',
    'competes_with', 'cousin_of', 'endorsed_by', 'follows', 'forgives',
    'friend_of', 'has_crush_on', 'mentor_of', 'neighbor_of', 'owes_debt_to',
    'protects', 'reports_to', 'subscribes_to', 'warns', 'works_with'
]

# Import SAE model
import sys
sys.path.append(str(Path(__file__).parent))
try:
    from train_sae_2hop import LargeSupervisedSAE
except ImportError:
    # Direct import from file
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_sae_2hop", 
        Path(__file__).parent / "02_train_sae_2hop.py"
    )
    train_sae_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_sae_module)
    LargeSupervisedSAE = train_sae_module.LargeSupervisedSAE


def load_sae(checkpoint_path, device):
    """Load trained SAE model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    d_model = checkpoint['d_model']
    n_free = checkpoint.get('n_free', 100000)
    n_relation = checkpoint.get('n_relation', 20)
    vocab_size = 50383  # GPT-2 vocab size + custom tokens (entities/relations)
    
    model = LargeSupervisedSAE(d_model=d_model, n_free=n_free, n_relation=n_relation, vocab_size=vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint


def evaluate_binding_accuracy(sae, lm_model, tokenizer, qa_file, layer_idx=-1):
    """
    Evaluate slot binding accuracy: Question → Relation slot
    
    For 2-hop reasoning, we evaluate TWO activations per question:
    1. When predicting E2: should activate R1 slot
    2. When predicting E3: should activate R2 slot
    
    We only look at the last 20 supervised slots for binding.
    """
    device = next(lm_model.parameters()).device
    
    # Load data (JSONL format)
    qa_pairs = []
    with open(qa_file, 'r') as f:
        for line in f:
            qa_pairs.append(json.loads(line.strip()))
    
    # Metrics for both positions
    metrics = {
        'token_1': {'total': 0, 'correct': 0, 'top3': 0, 'top5': 0},
        'token_2': {'total': 0, 'correct': 0, 'top3': 0, 'top5': 0}
    }
    
    # Per-relation metrics (combined across both positions)
    per_relation = defaultdict(lambda: {'total': 0, 'correct': 0, 'top3': 0, 'top5': 0})
    
    # Confusion matrices for both positions
    confusion_matrix_t1 = np.zeros((len(RELATIONS), sae.n_relation))
    confusion_matrix_t2 = np.zeros((len(RELATIONS), sae.n_relation))
    
    # Binding matrices for both positions
    binding_matrix_t1 = np.zeros((len(RELATIONS), sae.n_relation))
    binding_matrix_t2 = np.zeros((len(RELATIONS), sae.n_relation))
    relation_counts_t1 = np.zeros(len(RELATIONS))
    relation_counts_t2 = np.zeros(len(RELATIONS))
    
    lm_model.eval()
    sae.eval()
    
    with torch.no_grad():
        for qa in tqdm(qa_pairs, desc="Evaluating"):
            # Extract from new data format
            entity_1 = qa['entity_1']
            relation_1 = qa['relation_1']
            entity_2 = qa['entity_2']
            relation_2 = qa['relation_2']
            entity_3 = qa['entity_3']
            
            relation_1_idx = RELATIONS.index(relation_1)
            relation_2_idx = RELATIONS.index(relation_2)
            
            # === Evaluate position 1: predicting E2 (should activate R1) ===
            question = qa['question']
            prompt = f"Question: {question}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            
            outputs = lm_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_idx]
            last_position = inputs['input_ids'].shape[1] - 1
            h1 = hidden_states[0, last_position, :]
            
            # Get SAE activations - only look at supervised relation slots (last 20)
            z1, _ = sae(h1.unsqueeze(0))
            z1_relation = sae.get_relation_activations(z1)  # [1, 20]
            slot_logits_t1 = z1_relation[0]  # [20]
            
            top1_t1 = slot_logits_t1.argmax(dim=-1).item()
            top3_t1 = slot_logits_t1.topk(3, dim=-1).indices.tolist()
            top5_t1 = slot_logits_t1.topk(5, dim=-1).indices.tolist()
            
            metrics['token_1']['total'] += 1
            if top1_t1 == relation_1_idx:
                metrics['token_1']['correct'] += 1
            if relation_1_idx in top3_t1:
                metrics['token_1']['top3'] += 1
            if relation_1_idx in top5_t1:
                metrics['token_1']['top5'] += 1
            
            per_relation[relation_1]['total'] += 1
            if top1_t1 == relation_1_idx:
                per_relation[relation_1]['correct'] += 1
            if relation_1_idx in top3_t1:
                per_relation[relation_1]['top3'] += 1
            if relation_1_idx in top5_t1:
                per_relation[relation_1]['top5'] += 1
            
            confusion_matrix_t1[relation_1_idx, top1_t1] += 1
            binding_matrix_t1[relation_1_idx, :] += slot_logits_t1.cpu().numpy()
            relation_counts_t1[relation_1_idx] += 1
            
            # === Evaluate position 2: predicting E3 (should activate R2) ===
            # Training concatenated the E2 token directly after "Answer:"
            # (no space; entities are dedicated single tokens), so build the
            # context the same way.
            e2_ids = tokenizer.encode(entity_2, add_special_tokens=False)
            input_ids_e2 = torch.cat(
                [inputs['input_ids'], torch.tensor([e2_ids], device=device)], dim=1)
            inputs_with_e2 = {'input_ids': input_ids_e2,
                              'attention_mask': torch.ones_like(input_ids_e2)}

            outputs_with_e2 = lm_model(**inputs_with_e2, output_hidden_states=True)
            hidden_states_with_e2 = outputs_with_e2.hidden_states[layer_idx]
            last_position_e2 = inputs_with_e2['input_ids'].shape[1] - 1
            h2 = hidden_states_with_e2[0, last_position_e2, :]
            
            # Get SAE activations - only look at supervised relation slots (last 20)
            z2, _ = sae(h2.unsqueeze(0))
            z2_relation = sae.get_relation_activations(z2)  # [1, 20]
            slot_logits_t2 = z2_relation[0]  # [20]
            
            top1_t2 = slot_logits_t2.argmax(dim=-1).item()
            top3_t2 = slot_logits_t2.topk(3, dim=-1).indices.tolist()
            top5_t2 = slot_logits_t2.topk(5, dim=-1).indices.tolist()
            
            metrics['token_2']['total'] += 1
            if top1_t2 == relation_2_idx:
                metrics['token_2']['correct'] += 1
            if relation_2_idx in top3_t2:
                metrics['token_2']['top3'] += 1
            if relation_2_idx in top5_t2:
                metrics['token_2']['top5'] += 1
            
            per_relation[relation_2]['total'] += 1
            if top1_t2 == relation_2_idx:
                per_relation[relation_2]['correct'] += 1
            if relation_2_idx in top3_t2:
                per_relation[relation_2]['top3'] += 1
            if relation_2_idx in top5_t2:
                per_relation[relation_2]['top5'] += 1
            
            confusion_matrix_t2[relation_2_idx, top1_t2] += 1
            binding_matrix_t2[relation_2_idx, :] += slot_logits_t2.cpu().numpy()
            relation_counts_t2[relation_2_idx] += 1

    # Normalize binding matrices
    for i in range(len(RELATIONS)):
        if relation_counts_t1[i] > 0:
            binding_matrix_t1[i, :] /= relation_counts_t1[i]
        if relation_counts_t2[i] > 0:
            binding_matrix_t2[i, :] /= relation_counts_t2[i]
    
    # Compute overall metrics
    print(f"\n{'='*60}")
    print(f"Binding Accuracy by Position")
    print(f"{'='*60}")
    
    for pos in ['token_1', 'token_2']:
        total = metrics[pos]['total']
        correct = metrics[pos]['correct']
        top3 = metrics[pos]['top3']
        top5 = metrics[pos]['top5']
        
        print(f"\n{pos.upper()} (predicting {'E2 using R1' if pos == 'token_1' else 'E3 using R2'}):")
        print(f"  Top-1 Accuracy: {correct/total:.2%} ({correct}/{total})")
        print(f"  Top-3 Accuracy: {top3/total:.2%} ({top3}/{total})")
        print(f"  Top-5 Accuracy: {top5/total:.2%} ({top5}/{total})")
    
    # Overall accuracy (average of both positions)
    total_all = metrics['token_1']['total'] + metrics['token_2']['total']
    correct_all = metrics['token_1']['correct'] + metrics['token_2']['correct']
    top3_all = metrics['token_1']['top3'] + metrics['token_2']['top3']
    
    print(f"\n{'='*60}")
    print(f"Overall Binding Accuracy (Both Positions)")
    print(f"{'='*60}")
    print(f"Top-1 Accuracy: {correct_all/total_all:.2%} ({correct_all}/{total_all})")
    print(f"Top-3 Accuracy: {top3_all/total_all:.2%} ({top3_all}/{total_all})")
    
    print(f"\n{'='*60}")
    print(f"Per-Relation Binding Accuracy")
    print(f"{'='*60}")
    for relation in RELATIONS:
        if per_relation[relation]['total'] > 0:
            rel_acc = per_relation[relation]['correct'] / per_relation[relation]['total']
            rel_top3 = per_relation[relation]['top3'] / per_relation[relation]['total']
            print(f"{relation:20s}: Top-1={rel_acc:.2%}, Top-3={rel_top3:.2%} "
                  f"({per_relation[relation]['total']} samples)")
    
    return {
        'metrics_by_position': metrics,
        'overall_top1_acc': correct_all / total_all,
        'overall_top3_acc': top3_all / total_all,
        'per_relation': dict(per_relation),
        'confusion_matrix_t1': confusion_matrix_t1,
        'confusion_matrix_t2': confusion_matrix_t2,
        'binding_matrix_t1': binding_matrix_t1,
        'binding_matrix_t2': binding_matrix_t2,
    }


def visualize_binding_matrix(binding_matrix, title, save_path):
    """Visualize relation-to-slot binding matrix as heatmap."""
    plt.figure(figsize=(14, 12))
    
    # Normalize each row for better visualization
    binding_matrix_norm = binding_matrix / (np.abs(binding_matrix).max(axis=1, keepdims=True) + 1e-10)
    
    sns.heatmap(binding_matrix_norm,
                xticklabels=[f'Slot {i}' for i in range(binding_matrix.shape[1])],
                yticklabels=RELATIONS,
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'Normalized Activation'},
                fmt='.2f',
                linewidths=0.5)
    
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('SAE Slot', fontsize=12)
    plt.ylabel('Relation', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved binding matrix to {save_path}")


def visualize_confusion_matrix(confusion_matrix, title, save_path):
    """Visualize confusion matrix."""
    plt.figure(figsize=(14, 12))
    
    # Normalize by row (true relation)
    confusion_norm = confusion_matrix / (confusion_matrix.sum(axis=1, keepdims=True) + 1e-10)
    
    sns.heatmap(confusion_norm,
                xticklabels=[f'Slot {i}' for i in range(confusion_matrix.shape[1])],
                yticklabels=RELATIONS,
                cmap='Blues',
                cbar_kws={'label': 'Probability'},
                fmt='.2f',
                linewidths=0.5,
                vmin=0, vmax=1)
    
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Predicted Slot', fontsize=12)
    plt.ylabel('True Relation', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved confusion matrix to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sae_checkpoint', type=str, required=True,
                        help='Path to trained SAE checkpoint')
    parser.add_argument('--lm_model_path', type=str, required=True,
                        help='Path to trained 2-hop LM checkpoint')
    parser.add_argument('--val_data', type=str, required=True,
                        help='Path to validation QA data (JSON)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--layer', type=int, default=-1,
                        help='Layer to extract (-1 = last)')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print("\nLoading SAE...")
    sae, sae_checkpoint = load_sae(args.sae_checkpoint, device)
    print(f"SAE: {sae.n_relation} slots")
    
    print("\nLoading LM...")
    tokenizer = AutoTokenizer.from_pretrained(args.lm_model_path)
    lm_model = AutoModelForCausalLM.from_pretrained(args.lm_model_path).to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Evaluate
    print("\nEvaluating binding accuracy...")
    results = evaluate_binding_accuracy(
        sae, lm_model, tokenizer, args.val_data, layer_idx=args.layer
    )
    
    # Visualize binding matrices for both positions
    binding_viz_t1_path = output_dir / "binding_matrix_token1.png"
    visualize_binding_matrix(
        results['binding_matrix_t1'],
        title="Relation-to-Slot Binding (Token 1: Predicting E2 using R1)",
        save_path=binding_viz_t1_path
    )
    
    binding_viz_t2_path = output_dir / "binding_matrix_token2.png"
    visualize_binding_matrix(
        results['binding_matrix_t2'],
        title="Relation-to-Slot Binding (Token 2: Predicting E3 using R2)",
        save_path=binding_viz_t2_path
    )
    
    # Visualize confusion matrices
    confusion_viz_t1_path = output_dir / "confusion_matrix_token1.png"
    visualize_confusion_matrix(
        results['confusion_matrix_t1'],
        title="Slot Prediction Confusion (Token 1: E2 using R1)",
        save_path=confusion_viz_t1_path
    )
    
    confusion_viz_t2_path = output_dir / "confusion_matrix_token2.png"
    visualize_confusion_matrix(
        results['confusion_matrix_t2'],
        title="Slot Prediction Confusion (Token 2: E3 using R2)",
        save_path=confusion_viz_t2_path
    )
    
    # Save results
    results_path = output_dir / "results.json"
    results_serializable = {
        'overall_top1_acc': results['overall_top1_acc'],
        'overall_top3_acc': results['overall_top3_acc'],
        'metrics_by_position': results['metrics_by_position'],
        'per_relation': results['per_relation'],
    }
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nSaved results to {results_path}")
    
    # Save numpy arrays
    np.save(output_dir / "binding_matrix_t1.npy", results['binding_matrix_t1'])
    np.save(output_dir / "binding_matrix_t2.npy", results['binding_matrix_t2'])
    np.save(output_dir / "confusion_matrix_t1.npy", results['confusion_matrix_t1'])
    np.save(output_dir / "confusion_matrix_t2.npy", results['confusion_matrix_t2'])
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
