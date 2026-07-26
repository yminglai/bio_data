"""
Step 6: Grokking Analysis for 2-hop Reasoning with Large SAE
Analyze how relation binding emerges across training epochs using 100k+20 SAE.

For each epoch checkpoint:
1. Load activations from that epoch
2. Measure binding accuracy (supervised slots)
3. Measure slot specialization and purity
4. Track emergence of relation-specific features
"""
import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse
from collections import defaultdict


class LargeSupervisedSAE(nn.Module):
    """Large SAE: 100,000 free + 20 supervised relation slots"""
    def __init__(self, d_model, n_free=100000, n_relation=20):
        super().__init__()
        self.n_free = n_free
        self.n_relation = n_relation
        self.n_slots = n_free + n_relation
        self.d_model = d_model
        
        self.encoder = nn.Linear(d_model, self.n_slots, bias=True)
        self.decoder = nn.Linear(self.n_slots, d_model, bias=True)
        
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


def analyze_epoch_activations(sae, activations_file, device):
    """
    Analyze binding accuracy for one epoch using the trained SAE.
    
    Returns:
        - top1_accuracy: Top-1 binding accuracy
        - top3_accuracy: Top-3 binding accuracy  
        - top5_accuracy: Top-5 binding accuracy
        - binding_matrix: 20x20 confusion matrix
        - slot_stats: Statistics per slot
    """
    with open(activations_file, 'rb') as f:
        data = pickle.load(f)
    
    n_slots = 20
    binding_matrix = np.zeros((n_slots, n_slots))
    
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    total = 0
    
    slot_activations = defaultdict(list)
    
    sae.eval()
    with torch.no_grad():
        for item in tqdm(data, desc="Analyzing", leave=False):
            h = torch.from_numpy(item['h']).float().to(device)
            relation_idx = item['relation_idx']
            
            # Get SAE activations
            z, _ = sae(h.unsqueeze(0))
            z_relation = sae.get_relation_activations(z)  # Extract supervised slots
            slot_logits = z_relation[0]  # [20]
            
            # Track activations per slot
            for slot_idx in range(n_slots):
                slot_activations[slot_idx].append(slot_logits[slot_idx].item())
            
            # Top-k predictions
            top_slots = torch.topk(slot_logits, k=5).indices.cpu().numpy()
            
            if top_slots[0] == relation_idx:
                correct_top1 += 1
            if relation_idx in top_slots[:3]:
                correct_top3 += 1
            if relation_idx in top_slots[:5]:
                correct_top5 += 1
            
            # Update binding matrix
            predicted_slot = top_slots[0]
            binding_matrix[relation_idx, predicted_slot] += 1
            
            total += 1
    
    # Calculate accuracies
    top1_acc = correct_top1 / total if total > 0 else 0.0
    top3_acc = correct_top3 / total if total > 0 else 0.0
    top5_acc = correct_top5 / total if total > 0 else 0.0
    
    # Normalize binding matrix (row-wise)
    row_sums = binding_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    binding_matrix_norm = binding_matrix / row_sums
    
    # Calculate slot statistics
    slot_stats = {}
    for slot_idx in range(n_slots):
        activations = slot_activations[slot_idx]
        slot_stats[slot_idx] = {
            'mean': float(np.mean(activations)),
            'std': float(np.std(activations)),
            'max': float(np.max(activations)),
            'min': float(np.min(activations)),
        }
    
    return {
        'top1_accuracy': top1_acc,
        'top3_accuracy': top3_acc,
        'top5_accuracy': top5_acc,
        'binding_matrix': binding_matrix_norm.tolist(),
        'slot_stats': slot_stats,
        'total_samples': total
    }


def plot_grokking_curves(results, output_dir):
    """Generate grokking analysis plots"""
    epochs = sorted([r['epoch'] for r in results])
    top1_accs = [r['top1_accuracy'] for r in results]
    top3_accs = [r['top3_accuracy'] for r in results]
    top5_accs = [r['top5_accuracy'] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Plot 1: Binding accuracy over epochs
    axes[0].plot(epochs, top1_accs, marker='o', linewidth=2, label='Top-1', color='#2E86AB')
    axes[0].plot(epochs, top3_accs, marker='s', linewidth=2, label='Top-3', color='#A23B72')
    axes[0].plot(epochs, top5_accs, marker='^', linewidth=2, label='Top-5', color='#F18F01')
    axes[0].set_xlabel('Training Epoch', fontsize=12)
    axes[0].set_ylabel('Binding Accuracy', fontsize=12)
    axes[0].set_title('Grokking: Emergence of Relation Binding\n(100k+20 Large SAE)', fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.0])
    
    # Plot 2: Diagonal strength in binding matrix
    diagonal_strengths = []
    for r in results:
        binding_matrix = np.array(r['binding_matrix'])
        diagonal_strength = np.mean(np.diag(binding_matrix))
        diagonal_strengths.append(diagonal_strength)
    
    axes[1].plot(epochs, diagonal_strengths, marker='o', linewidth=2, color='#06A77D')
    axes[1].set_xlabel('Training Epoch', fontsize=12)
    axes[1].set_ylabel('Diagonal Strength (1-to-1 Binding)', fontsize=12)
    axes[1].set_title('Specialization: Supervised Slots Align to Relations', fontsize=13)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.0])
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'grokking_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Grokking plot saved to {output_path}")
    plt.close()


def plot_binding_matrices(results, output_dir):
    """Plot binding matrices for selected epochs"""
    # Select key epochs to visualize
    key_epochs = [5, 20, 40, 60, 80, 100]
    available_epochs = {r['epoch']: r for r in results}
    
    epochs_to_plot = [e for e in key_epochs if e in available_epochs]
    n_plots = len(epochs_to_plot)
    
    if n_plots == 0:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, epoch in enumerate(epochs_to_plot):
        result = available_epochs[epoch]
        binding_matrix = np.array(result['binding_matrix'])
        
        im = axes[idx].imshow(binding_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        axes[idx].set_title(f'Epoch {epoch} (Top-1: {result["top1_accuracy"]:.1%})', fontsize=11)
        axes[idx].set_xlabel('Predicted Slot', fontsize=10)
        axes[idx].set_ylabel('True Relation', fontsize=10)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        
        # Mark diagonal
        for i in range(20):
            axes[idx].plot([i-0.5, i+0.5], [i-0.5, i+0.5], 'b-', linewidth=1, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Binding Matrix Evolution: Grokking to 1-to-1 Alignment', fontsize=14, y=1.00)
    plt.tight_layout()
    output_path = Path(output_dir) / 'binding_matrices_evolution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Binding matrices saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Grokking analysis for 2-hop with large SAE")
    parser.add_argument('--grokking_activations_dir', type=str, required=True,
                        help='Directory with epoch_X_activations.pkl files')
    parser.add_argument('--sae_checkpoint', type=str, required=True,
                        help='Path to trained SAE checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for analysis results')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load SAE
    print("Loading SAE...")
    checkpoint = torch.load(args.sae_checkpoint, map_location=device, weights_only=False)
    d_model = checkpoint.get('d_model', 768)
    n_free = checkpoint.get('args', {}).get('n_free', 100000)
    sae = LargeSupervisedSAE(d_model=d_model, n_free=n_free, n_relation=20).to(device)
    # strict=False: the checkpoint also contains value-head weights (an
    # auxiliary training component this analysis does not use).
    sae.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"✓ Loaded SAE with {sae.n_slots} total slots ({n_free} free + 20 supervised)")
    
    # Find all activation files
    activations_dir = Path(args.grokking_activations_dir)
    activation_files = sorted(activations_dir.glob('epoch_*_activations.pkl'))
    
    if not activation_files:
        print(f"❌ No activation files found in {activations_dir}")
        return
    
    print(f"Found {len(activation_files)} epoch checkpoints")
    
    # Analyze each epoch
    results = []
    for act_file in activation_files:
        # Extract epoch number from filename
        epoch_num = int(act_file.stem.split('_')[1])
        print(f"\nAnalyzing epoch {epoch_num}...")
        
        result = analyze_epoch_activations(sae, act_file, device)
        result['epoch'] = epoch_num
        result['activation_file'] = str(act_file)
        results.append(result)
        
        print(f"  Top-1: {result['top1_accuracy']:.1%}")
        print(f"  Top-3: {result['top3_accuracy']:.1%}")
        print(f"  Top-5: {result['top5_accuracy']:.1%}")
    
    # Sort by epoch
    results.sort(key=lambda x: x['epoch'])
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'grokking_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to {output_dir / 'grokking_results.json'}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_grokking_curves(results, output_dir)
    plot_binding_matrices(results, output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("GROKKING ANALYSIS SUMMARY")
    print("="*60)
    for r in results:
        print(f"Epoch {r['epoch']:3d}: Top-1={r['top1_accuracy']:.1%}  "
              f"Top-3={r['top3_accuracy']:.1%}  Top-5={r['top5_accuracy']:.1%}")
    print("="*60)


if __name__ == '__main__':
    main()
