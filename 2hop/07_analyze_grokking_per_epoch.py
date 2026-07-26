"""
Step 7: Analyze Grokking Per-Epoch SAEs
For each epoch, analyze the SAE trained on that epoch's activations.
Track how binding emerges as training progresses.
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
        return z[:, -self.n_relation:]


RELATIONS = [
    'accuses', 'admires', 'blames', 'boss_of', 'classmate_of',
    'competes_with', 'cousin_of', 'endorsed_by', 'follows', 'forgives',
    'friend_of', 'has_crush_on', 'mentor_of', 'neighbor_of', 'owes_debt_to',
    'protects', 'reports_to', 'subscribes_to', 'warns', 'works_with'
]


def load_sae(sae_path, device):
    """Load trained SAE from checkpoint"""
    checkpoint = torch.load(sae_path, map_location=device, weights_only=False)
    d_model = checkpoint.get('d_model', 768)
    n_free = checkpoint.get('n_free', 100000)
    sae = LargeSupervisedSAE(d_model=d_model, n_free=n_free, n_relation=20).to(device)
    sae.load_state_dict(checkpoint['model_state_dict'], strict=False)
    sae.eval()
    return sae


def analyze_sae_binding(sae, activations_file, device):
    """
    Analyze binding accuracy for one SAE using its corresponding activations.
    
    Returns binding metrics and statistics.
    """
    with open(activations_file, 'rb') as f:
        data = pickle.load(f)
    
    n_slots = 20
    binding_matrix = np.zeros((n_slots, n_slots))
    
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    total = 0
    
    sae.eval()
    with torch.no_grad():
        for item in tqdm(data, desc="Analyzing", leave=False):
            h = torch.from_numpy(item['h']).float().to(device)
            relation_idx = item['relation_idx']
            
            # Get SAE activations (supervised slots only)
            z, _ = sae(h.unsqueeze(0))
            z_relation = sae.get_relation_activations(z)
            slot_logits = z_relation[0]
            
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
    
    # Normalize binding matrix
    row_sums = binding_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    binding_matrix_norm = binding_matrix / row_sums
    
    # Diagonal strength (how much binding is 1-to-1)
    diagonal_strength = np.mean(np.diag(binding_matrix_norm))
    
    return {
        'top1_accuracy': top1_acc,
        'top3_accuracy': top3_acc,
        'top5_accuracy': top5_acc,
        'binding_matrix': binding_matrix_norm.tolist(),
        'diagonal_strength': diagonal_strength,
        'total_samples': total
    }


def plot_grokking_curves(results, output_dir):
    """Generate grokking analysis plots"""
    epochs = sorted([r['epoch'] for r in results])
    top1_accs = [r['top1_accuracy'] for r in results]
    top3_accs = [r['top3_accuracy'] for r in results]
    top5_accs = [r['top5_accuracy'] for r in results]
    diagonal_strengths = [r['diagonal_strength'] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Plot 1: Binding accuracy over epochs
    axes[0].plot(epochs, top1_accs, marker='o', linewidth=2, label='Top-1', color='#2E86AB', markersize=8)
    axes[0].plot(epochs, top3_accs, marker='s', linewidth=2, label='Top-3', color='#A23B72', markersize=6)
    axes[0].plot(epochs, top5_accs, marker='^', linewidth=2, label='Top-5', color='#F18F01', markersize=6)
    axes[0].set_xlabel('Training Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Binding Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Grokking: Emergence of Relation Binding\n(100k+20 Large SAE)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11, framealpha=0.9)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_ylim([0, 1.0])
    
    # Plot 2: Diagonal strength (1-to-1 binding)
    axes[1].plot(epochs, diagonal_strengths, marker='o', linewidth=2.5, color='#06A77D', markersize=8)
    axes[1].set_xlabel('Training Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Diagonal Strength (1-to-1 Binding)', fontsize=12, fontweight='bold')
    axes[1].set_title('Specialization: Supervised Slots Align to Relations', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_ylim([0, 1.0])
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'grokking_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Grokking curves saved to {output_path}")
    plt.close()


def plot_binding_matrices(results, output_dir):
    """Plot binding matrices for selected epochs"""
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
        axes[idx].set_title(f'Epoch {epoch} (Top-1: {result["top1_accuracy"]:.1%})', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted Slot', fontsize=10)
        axes[idx].set_ylabel('True Relation', fontsize=10)
        
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        
        # Mark diagonal
        for i in range(20):
            axes[idx].plot([i-0.5, i+0.5], [i-0.5, i+0.5], 'b-', linewidth=1.5, alpha=0.4)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Binding Matrix Evolution: Grokking to 1-to-1 Alignment', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    output_path = Path(output_dir) / 'binding_matrices_evolution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Binding matrices saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze grokking with per-epoch SAEs")
    parser.add_argument('--sae_base_dir', type=str, required=True,
                        help='Base directory containing sae_epoch_X folders')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for analysis results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--plots', action='store_true', help='If set, also generate plots')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Find all SAE checkpoints (robust to log/extra files)
    sae_base = Path(args.sae_base_dir)
    raw_dirs = sorted(sae_base.glob('sae_epoch_*'))
    sae_dirs = []
    for d in raw_dirs:
        # Expect suffix after last underscore to be an int (epoch)
        try:
            _ = int(d.name.split('_')[-1])
            sae_dirs.append(d)
        except ValueError:
            print(f"⚠ Skipping non-epoch entry: {d}")

    if not sae_dirs:
        print(f"❌ No SAE directories found in {sae_base}")
        return

    print(f"Found {len(sae_dirs)} SAE checkpoints (filtered)")
    
    # Analyze each epoch's SAE
    results = []
    for sae_dir in sae_dirs:
        # Extract epoch number (already validated)
        epoch_num = int(sae_dir.name.split('_')[-1])
        
        sae_path = sae_dir / 'sae_best.pt'
        act_path = Path('2hop/grokking_activations') / f'epoch_{epoch_num}_activations.pkl'
        
        if not sae_path.exists():
            print(f"⚠ SAE checkpoint not found: {sae_path}")
            continue

        if not act_path.exists():
            print(f"⚠ Activation file not found: {act_path}")
            continue
        
        print(f"\nAnalyzing epoch {epoch_num}...")
        sae = load_sae(sae_path, device)
        result = analyze_sae_binding(sae, act_path, device)
        result['epoch'] = epoch_num
        results.append(result)
        
        print(f"  Top-1: {result['top1_accuracy']:.1%}")
        print(f"  Top-3: {result['top3_accuracy']:.1%}")
        print(f"  Diagonal: {result['diagonal_strength']:.3f}")
    
    # Sort by epoch
    results.sort(key=lambda x: x['epoch'])
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'grokking_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_dir / 'grokking_results.json'}")
    
    # Optional plots (default off)
    if args.plots:
        print("\nGenerating plots...")
        plot_grokking_curves(results, output_dir)
        plot_binding_matrices(results, output_dir)

    # Summary (print all numbers)
    print("\n" + "="*72)
    print("GROKKING ANALYSIS SUMMARY (ALL EPOCHS)")
    print("="*72)
    header = f"{'Epoch':>6} | {'Top-1':>8} | {'Top-3':>8} | {'Top-5':>8} | {'Diagonal':>9} | {'Samples':>7}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['epoch']:6d} | {r['top1_accuracy']*100:8.2f} | {r['top3_accuracy']*100:8.2f} | "
              f"{r['top5_accuracy']*100:8.2f} | {r['diagonal_strength']:9.4f} | {r['total_samples']:7d}")
    print("="*72)


if __name__ == '__main__':
    main()
