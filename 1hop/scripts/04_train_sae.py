"""
Step 4: Train Supervised 1-to-1 SAE
Large SAE with K free slots + 6 relation slots (paper: K=10,000 for 1-hop).
Supports staged training: reconstruction first, then alignment.
Supports separate and joint training modes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
import json

class ActivationDataset(Dataset):
    def __init__(self, activations_file):
        with open(activations_file, 'rb') as f:
            self.data = pickle.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'h': torch.from_numpy(item['h']).float(),
            'rule_idx': item['rule_idx'],
            'answer_tokens': item['answer_tokens'],
        }

class LargeSupervisedSAE(nn.Module):
    """
    Large SAE: n_free free slots + 6 relation slots.
    The last 6 slots are supervised to bind one relation each.
    """
    def __init__(self, d_model, n_free=10000, n_relation=6, vocab_size=50257):
        super().__init__()
        self.n_free = n_free
        self.n_relation = n_relation
        self.n_slots = n_free + n_relation
        self.d_model = d_model
        
        # Encoder: maps activations to slot activations (pre-activation)
        self.encoder = nn.Linear(d_model, self.n_slots, bias=True)
        
        # Decoder: reconstructs activations from all slots
        self.decoder = nn.Linear(self.n_slots, d_model, bias=True)
        
        # Value heads: predict answer tokens from relation slots
        self.value_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 256),
                nn.ReLU(),
                nn.Linear(256, vocab_size)
            )
            for _ in range(n_relation)
        ])
        
        # Initialize decoder to be close to identity (via transpose of encoder)
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data[:,:d_model].T.clone()
    
    def forward(self, h, temperature=1.0, hard=False):
        """
        Args:
            h: [batch, d_model] - input activations
            temperature: Gumbel-Softmax temperature (lower = more one-hot)
            hard: If True, use hard one-hot (straight-through estimator)
        Returns:
            z: [batch, n_slots] - slot activations (pre-activations)
            h_recon: [batch, d_model] - reconstructed activations
        """
        z = F.relu(self.encoder(h))  # [batch, n_slots], z = ReLU(W_e h + b_e)
        h_recon = self.decoder(z)
        return z, h_recon
    
    def predict_values(self, z, rule_idx):
        """
        Predict answer token logits using the active relation slot's value head.
        Args:
            z: [batch, n_slots]
            rule_idx: [batch] - which rule this is
        Returns:
            logits: [batch, vocab_size]
        """
        batch_size = z.shape[0]
        device = z.device
        all_logits = []
        for i in range(batch_size):
            # Use the corresponding relation slot (last 6 slots)
            rel_slot_val = z[i, -(self.n_relation):][rule_idx[i]].unsqueeze(0).unsqueeze(1)  # [1, 1]
            logits = self.value_heads[rule_idx[i]](rel_slot_val)
            all_logits.append(logits)
        return torch.cat(all_logits, dim=0)  # [batch, vocab_size]

def compute_loss(model, h, rule_idx, answer_tokens, stage=1, mode='joint', 
                 lambda_recon=1.0, lambda_sparse=1e-3, lambda_align=1.0, lambda_indep=1e-2, lambda_ortho=1e-2, lambda_value=0.5):
    """
    Combined loss for large SAE.
    Stage 1: Only reconstruction
    Stage 2: Add alignment, independence, value prediction
    Mode: 'joint' (train all together), 'separate_activation' (train encoder only), 'separate_value' (train value heads only)
    """
    batch_size = h.shape[0]
    device = h.device
    n_relation = model.n_relation
    n_free = model.n_free
    n_slots = model.n_slots

    # Forward pass
    z, h_recon = model(h)

    # 1. Reconstruction loss - use MSE for consistency
    L_recon = F.mse_loss(h_recon, h)

    # Initialize losses
    L_sparse = 0.0
    L_align = 0.0
    L_indep = 0.0
    L_ortho = 0.0
    L_value = 0.0

    if stage >= 2:
        # 2. Sparsity loss (L1 on free slots)
        L_sparse = z[:, :n_free].abs().mean()

        # 3. Alignment loss: cross-entropy over the 6 relation slots so the
        # gold relation's slot dominates (paper Eq. 9-10)
        rel_slots = z[:, -n_relation:]
        L_align = F.cross_entropy(rel_slots, rule_idx)

        # 4. Independence loss (decorrelate free slots within themselves)
        z_free = z[:, :n_free]
        if lambda_indep > 0:
            z_centered = z_free - z_free.mean(dim=0, keepdim=True)
            cov = (z_centered.T @ z_centered) / batch_size
            off_diag = cov - torch.eye(n_free, device=device) * cov
            L_indep = (off_diag ** 2).sum()
        else:
            L_indep = 0.0

        # 4b. Orthogonality loss (relation slots orthogonal to free slots)
        z_rel = z[:, -n_relation:]
        # Compute correlation between relation and free slots
        z_rel_centered = z_rel - z_rel.mean(dim=0, keepdim=True)
        z_free_centered = z_free - z_free.mean(dim=0, keepdim=True)
        cross_cov = (z_rel_centered.T @ z_free_centered) / batch_size
        L_ortho = (cross_cov ** 2).sum()

        # 5. Value prediction loss (use relation slots)
        if mode in ['joint', 'separate_value']:
            answer_first_tokens = torch.tensor(
                [tokens[0] if len(tokens) > 0 else 0 for tokens in answer_tokens],
                device=device
            )
            value_logits = model.predict_values(z, rule_idx)
            L_value = F.cross_entropy(value_logits, answer_first_tokens)

    # Total loss based on mode
    if mode == 'separate_activation':
        # Only train encoder (reconstruction + alignment + independence + orthogonality)
        total_loss = (
            lambda_recon * L_recon +
            lambda_sparse * L_sparse +
            lambda_align * L_align +
            lambda_indep * L_indep +
            lambda_ortho * L_ortho
        )
    elif mode == 'separate_value':
        # Only train value heads
        total_loss = lambda_value * L_value
    else:  # joint
        total_loss = (
            lambda_recon * L_recon +
            lambda_sparse * L_sparse +
            lambda_align * L_align +
            lambda_indep * L_indep +
            lambda_ortho * L_ortho +
            lambda_value * L_value
        )

    # Compute accuracy for monitoring (relation slot)
    slot_acc = 0.0
    value_acc = 0.0
    if stage >= 2:
        rel_pred = rel_slots.argmax(dim=-1)
        slot_acc = (rel_pred == rule_idx).float().mean()
        
        if mode in ['joint', 'separate_value']:
            value_pred = model.predict_values(z, rule_idx).argmax(dim=-1)
            value_acc = (value_pred == answer_first_tokens).float().mean()

    return {
        'loss': total_loss,
        'L_recon': float(L_recon),
        'L_sparse': float(L_sparse) if stage >= 2 else 0.0,
        'L_align': float(L_align) if stage >= 2 else 0.0,
        'L_indep': float(L_indep) if stage >= 2 else 0.0,
        'L_ortho': float(L_ortho) if stage >= 2 else 0.0,
        'L_value': float(L_value) if stage >= 2 and mode in ['joint', 'separate_value'] else 0.0,
        'slot_acc': float(slot_acc) if stage >= 2 else 0.0,
        'value_acc': float(value_acc) if stage >= 2 and mode in ['joint', 'separate_value'] else 0.0,
    }

def collate_fn(batch):
    """Custom collate function to handle variable-length answer tokens."""
    h = torch.stack([item['h'] for item in batch])
    rule_idx = torch.tensor([item['rule_idx'] for item in batch])
    answer_tokens = [item['answer_tokens'] for item in batch]
    
    return {
        'h': h,
        'rule_idx': rule_idx,
        'answer_tokens': answer_tokens,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation_file', type=str, default='data/activations/train_activations.pkl')
    parser.add_argument('--output_dir', type=str, default='models/sae_large')
    parser.add_argument('--epochs_stage1', type=int, default=100, help='Epochs for stage 1 (reconstruction only)')
    parser.add_argument('--epochs_stage2', type=int, default=500, help='Epochs for stage 2 (full training)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_free', type=int, default=10000, help='Number of free slots (paper: 10,000 for 1-hop)')
    parser.add_argument('--mode', type=str, default='joint', choices=['joint', 'separate_activation', 'separate_value'])
    parser.add_argument('--lambda_recon', type=float, default=1.0)
    parser.add_argument('--lambda_sparse', type=float, default=1e-3)
    parser.add_argument('--lambda_align', type=float, default=1.0)
    parser.add_argument('--lambda_indep', type=float, default=0.0,
                       help='Weight for free-slot independence loss. Not part of the paper objective '
                            '(used only as a diagnostic there); >0 builds an n_free x n_free covariance matrix')
    parser.add_argument('--lambda_ortho', type=float, default=1e-2, help='Weight for orthogonality loss between relation and free slots')
    parser.add_argument('--lambda_value', type=float, default=0.5)
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading activations from {args.activation_file}")
    dataset = ActivationDataset(args.activation_file)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )

    # Get dimensionality from first sample
    d_model = dataset[0]['h'].shape[0]
    print(f"Activation dimension: {d_model}")
    print(f"Dataset size: {len(dataset)}")

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = LargeSupervisedSAE(d_model=d_model, n_free=args.n_free, n_relation=6)
    model.to(device)

    # Paper Appendix C.5: weight decay disabled so L2 regularization does not
    # interfere with the explicit sparsity penalty
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    start_epoch = 0

    # Resume from checkpoint if specified
    if args.resume and Path(args.resume).exists():
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}")

    # Training loop with stages
    print(f"Starting training...")
    history = []

    # Stage 1: Reconstruction only
    if start_epoch < args.epochs_stage1:
        print(f"\n=== Stage 1: Reconstruction Only ({args.epochs_stage1} epochs) ===")
        for epoch in range(start_epoch, min(args.epochs_stage1, start_epoch + args.epochs_stage1)):
            model.train()
            epoch_metrics = {
                'loss': 0, 'L_recon': 0, 'L_sparse': 0, 'L_align': 0,
                'L_indep': 0, 'L_ortho': 0, 'L_value': 0, 'slot_acc': 0, 'value_acc': 0
            }

            pbar = tqdm(dataloader, desc=f"Stage 1 Epoch {epoch+1}/{args.epochs_stage1}")
            for batch in pbar:
                h = batch['h'].to(device)
                rule_idx = batch['rule_idx'].to(device)
                answer_tokens = batch['answer_tokens']

                optimizer.zero_grad()

                metrics = compute_loss(
                    model, h, rule_idx, answer_tokens, stage=1, mode=args.mode,
                    lambda_recon=args.lambda_recon,
                    lambda_sparse=args.lambda_sparse,
                    lambda_align=args.lambda_align,
                    lambda_indep=args.lambda_indep,
                    lambda_ortho=args.lambda_ortho,
                    lambda_value=args.lambda_value
                )

                metrics['loss'].backward()
                optimizer.step()

                # Accumulate metrics
                for key in epoch_metrics:
                    epoch_metrics[key] += float(metrics[key]) if key != 'loss' else float(metrics['loss'])

                pbar.set_postfix({
                    'loss': f"{float(metrics['loss']):.4f}",
                    'L_recon': f"{float(metrics['L_recon']):.4f}"
                })

            # Average metrics
            for key in epoch_metrics:
                epoch_metrics[key] /= len(dataloader)

            epoch_metrics['epoch'] = epoch + 1
            epoch_metrics['stage'] = 1
            history.append(epoch_metrics)

            print(f"Stage 1 Epoch {epoch+1} summary:")
            print(f"  Loss: {epoch_metrics['loss']:.4f}")
            print(f"  L_recon: {epoch_metrics['L_recon']:.4f}")

    # Save after stage 1 if completed
    if start_epoch >= args.epochs_stage1:
        print(f"Stage 1 already completed, loading from checkpoint")
    else:
        stage1_checkpoint_path = output_dir / "sae_stage1.pt"
        torch.save({
            'epoch': args.epochs_stage1,
            'stage': 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': vars(args),
            'd_model': d_model,
            'n_free': args.n_free,
            'n_relation': 6,
        }, stage1_checkpoint_path)
        print(f"✓ Saved stage 1 checkpoint to {stage1_checkpoint_path}")

    # Stage 2: Full training
    print(f"\n=== Stage 2: Full Training ({args.epochs_stage2} epochs) ===")
    start_epoch_stage2 = max(0, start_epoch - args.epochs_stage1)
    
    # Track best checkpoints
    best_slot_acc = 0.0
    best_loss = float('inf')
    
    for epoch in range(start_epoch_stage2, args.epochs_stage2):
        model.train()
        epoch_metrics = {
            'loss': 0, 'L_recon': 0, 'L_sparse': 0, 'L_align': 0,
            'L_indep': 0, 'L_ortho': 0, 'L_value': 0, 'slot_acc': 0, 'value_acc': 0
        }

        pbar = tqdm(dataloader, desc=f"Stage 2 Epoch {epoch+1}/{args.epochs_stage2}")
        for batch in pbar:
            h = batch['h'].to(device)
            rule_idx = batch['rule_idx'].to(device)
            answer_tokens = batch['answer_tokens']

            optimizer.zero_grad()

            metrics = compute_loss(
                model, h, rule_idx, answer_tokens, stage=2, mode=args.mode,
                lambda_recon=args.lambda_recon,
                lambda_sparse=args.lambda_sparse,
                lambda_align=args.lambda_align,
                lambda_indep=args.lambda_indep,
                lambda_ortho=args.lambda_ortho,
                lambda_value=args.lambda_value
            )

            metrics['loss'].backward()
            optimizer.step()

            # Accumulate metrics
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key] if key != 'loss' else metrics['loss'].item()

            pbar.set_postfix({
                'loss': f"{metrics['loss'].item():.4f}",
                'slot_acc': f"{metrics['slot_acc']:.3f}",
                'value_acc': f"{metrics['value_acc']:.3f}"
            })

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= len(dataloader)

        epoch_metrics['epoch'] = args.epochs_stage1 + epoch + 1
        epoch_metrics['stage'] = 2
        history.append(epoch_metrics)

        print(f"Stage 2 Epoch {epoch+1} summary:")
        print(f"  Loss: {epoch_metrics['loss']:.4f}")
        print(f"  Slot Acc: {epoch_metrics['slot_acc']:.3f}")
        print(f"  Value Acc: {epoch_metrics['value_acc']:.3f}")
        print(f"  L_recon: {epoch_metrics['L_recon']:.4f}")
        print(f"  L_align: {epoch_metrics['L_align']:.4f}")
        print(f"  L_ortho: {epoch_metrics['L_ortho']:.4f}")

        # Save best checkpoint based on slot accuracy
        current_slot_acc = epoch_metrics['slot_acc']
        current_loss = epoch_metrics['loss']
        
        save_best = False
        if current_slot_acc > best_slot_acc:
            best_slot_acc = current_slot_acc
            save_best = True
            reason = f"slot_acc_{current_slot_acc:.3f}"
        elif current_slot_acc == best_slot_acc and current_loss < best_loss:
            best_loss = current_loss
            save_best = True
            reason = f"loss_{current_loss:.4f}"
            
        if save_best:
            best_checkpoint_path = output_dir / "sae_best.pt"
            torch.save({
                'epoch': args.epochs_stage1 + epoch + 1,
                'stage': 2,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args),
                'best_slot_acc': best_slot_acc,
                'best_loss': best_loss,
                'd_model': d_model,
                'n_free': args.n_free,
                'n_relation': 6,
            }, best_checkpoint_path)
            print(f"  ✓ Saved best checkpoint ({reason}) to {best_checkpoint_path}")



    # Save final model
    final_path = output_dir / "sae_final.pt"
    torch.save({
        'epoch': args.epochs_stage1 + args.epochs_stage2,
        'stage': 2,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
        'd_model': d_model,
        'n_free': args.n_free,
        'n_relation': 6,
    }, final_path)
    print(f"\nSaved final model to {final_path}")

    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_path}")

if __name__ == "__main__":
    main()
