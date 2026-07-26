"""
Step 2: Train Large Supervised SAE for 2-hop Reasoning
SAE with 100,000 free slots + 20 supervised relation slots.

Following the framework from scripts/04_train_sae.py but adapted for 2-hop task.
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
            'relation_idx': item['relation_idx'],  # Single relation for this activation
            'position': item['position'],  # 'token_1' or 'token_2'
            'target_token': item['target_token'],  # First token of target entity
        }


class LargeSupervisedSAE(nn.Module):
    """
    Large SAE: 100,000 free slots + 20 relation slots = 100,020 total.
    The last 20 slots are supervised to match the relation label.
    Each relation slot has a value head to predict entity tokens.
    """
    def __init__(self, d_model, n_free=100000, n_relation=20, vocab_size=50383):
        super().__init__()
        self.n_free = n_free
        self.n_relation = n_relation
        self.n_slots = n_free + n_relation
        self.d_model = d_model
        
        # Encoder: maps activations to relation slots
        self.encoder = nn.Linear(d_model, self.n_slots, bias=True)
        
        # Decoder: reconstructs activations from slots
        self.decoder = nn.Linear(self.n_slots, d_model, bias=True)
        
        # Value heads: predict entity tokens from relation slots (20 heads, one per relation)
        self.value_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 256),
                nn.ReLU(),
                nn.Linear(256, vocab_size)
            )
            for _ in range(n_relation)
        ])
        
        # Initialize decoder to be close to identity (transpose of encoder)
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.T.clone()
    
    def forward(self, h):
        """
        Args:
            h: [batch, d_model] - input activations
        Returns:
            z: [batch, n_slots] - slot activations (all slots)
            h_recon: [batch, d_model] - reconstructed activations
        """
        z = torch.relu(self.encoder(h))  # z = ReLU(W_e h + b_e)  # [batch, n_slots]
        h_recon = self.decoder(z)
        return z, h_recon
    
    def get_relation_activations(self, z):
        """Extract supervised relation slot activations (last 20 slots)"""
        return z[:, -self.n_relation:]
    
    def predict_values(self, z, relation_idx):
        """
        Predict entity token logits using the active relation slot's value head.
        Args:
            z: [batch, n_slots]
            relation_idx: [batch] - which relation this is (0-19)
        Returns:
            logits: [batch, vocab_size]
        """
        batch_size = z.shape[0]
        all_logits = []
        for i in range(batch_size):
            # Use the corresponding relation slot (last 20 slots)
            rel_slot_val = z[i, -(self.n_relation):][relation_idx[i]].unsqueeze(0).unsqueeze(1)  # [1, 1]
            logits = self.value_heads[relation_idx[i]](rel_slot_val)  # [1, vocab_size]
            all_logits.append(logits)
        return torch.cat(all_logits, dim=0)  # [batch, vocab_size]


def compute_loss(model, h, relation_idx, target_token,
                 lambda_recon=1.0, lambda_sparse=1e-3, lambda_align=1.0, lambda_ortho=1e-2, lambda_value=0.5):
    """
    Compute training loss with five components:
    1. Reconstruction loss: ||h_recon - h||^2
    2. Sparsity loss: L1 on FREE slot activations only
    3. Alignment loss: Cross-entropy for supervised relation slots
    4. Orthogonality loss: Supervised slots orthogonal to free slots
    5. Value prediction loss: Predict entity token from relation slot
    
    Args:
        h: [batch, d_model]
        relation_idx: [batch] - ground truth relation index
        target_token: [batch] - first token of target entity (E2 or E3)
    """
    z, h_recon = model(h)
    batch_size = h.shape[0]
    
    # 1. Reconstruction loss
    recon_loss = F.mse_loss(h_recon, h)
    
    # 2. Sparsity loss (L1 on FREE slots only, not supervised slots)
    z_free = z[:, :model.n_free]
    sparse_loss = torch.mean(torch.abs(z_free))
    
    # 3. Alignment loss (cross-entropy on supervised relation slots)
    # Last 20 slots should match the relation
    z_relation = z[:, -model.n_relation:]
    align_loss = F.cross_entropy(z_relation, relation_idx)
    
    # 4. Orthogonality loss (relation slots should be orthogonal to free slots)
    z_rel_centered = z_relation - z_relation.mean(dim=0, keepdim=True)
    z_free_centered = z_free - z_free.mean(dim=0, keepdim=True)
    cross_cov = (z_rel_centered.T @ z_free_centered) / batch_size
    ortho_loss = (cross_cov ** 2).sum()
    
    # 5. Value prediction loss (predict entity token from relation slot)
    value_logits = model.predict_values(z, relation_idx)  # [batch, vocab_size]
    value_loss = F.cross_entropy(value_logits, target_token)
    
    # Total loss
    total_loss = (lambda_recon * recon_loss + 
                  lambda_sparse * sparse_loss + 
                  lambda_align * align_loss +
                  lambda_ortho * ortho_loss +
                  lambda_value * value_loss)
    
    return total_loss, {
        'recon': recon_loss.item(),
        'sparse': sparse_loss.item(),
        'align': align_loss.item(),
        'ortho': ortho_loss.item(),
        'value': value_loss.item(),
        'total': total_loss.item()
    }


def train_sae(model, train_loader, val_loader, args, device):
    """Train the SAE model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            h = batch['h'].to(device)
            relation_idx = batch['relation_idx'].to(device)
            target_token = batch['target_token'].to(device)
            
            # Forward pass
            loss, loss_dict = compute_loss(
                model, h, relation_idx, target_token,
                lambda_recon=args.lambda_recon,
                lambda_sparse=args.lambda_sparse,
                lambda_align=args.lambda_align,
                lambda_ortho=args.lambda_ortho,
                lambda_value=args.lambda_value
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss_dict)
        
        # Average training losses
        avg_train_loss = {
            key: np.mean([d[key] for d in train_losses])
            for key in train_losses[0].keys()
        }
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                h = batch['h'].to(device)
                relation_idx = batch['relation_idx'].to(device)
                target_token = batch['target_token'].to(device)
                
                loss, loss_dict = compute_loss(
                    model, h, relation_idx, target_token,
                    lambda_recon=args.lambda_recon,
                    lambda_sparse=args.lambda_sparse,
                    lambda_align=args.lambda_align,
                    lambda_ortho=args.lambda_ortho,
                    lambda_value=args.lambda_value
                )
                
                val_losses.append(loss_dict)
        
        avg_val_loss = {
            key: np.mean([d[key] for d in val_losses])
            for key in val_losses[0].keys()
        }
        
        # Print progress
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train - Total: {avg_train_loss['total']:.4f}, "
              f"Recon: {avg_train_loss['recon']:.4f}, "
              f"Sparse: {avg_train_loss['sparse']:.4f}, "
              f"Align: {avg_train_loss['align']:.4f}, "
              f"Ortho: {avg_train_loss['ortho']:.4f}, "
              f"Value: {avg_train_loss['value']:.4f}")
        print(f"  Val   - Total: {avg_val_loss['total']:.4f}, "
              f"Recon: {avg_val_loss['recon']:.4f}, "
              f"Sparse: {avg_val_loss['sparse']:.4f}, "
              f"Align: {avg_val_loss['align']:.4f}, "
              f"Ortho: {avg_val_loss['ortho']:.4f}, "
              f"Value: {avg_val_loss['value']:.4f}")
        
        # Save best model
        if avg_val_loss['total'] < best_val_loss:
            best_val_loss = avg_val_loss['total']
            
            checkpoint_path = Path(args.output_dir) / f"sae_best.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'd_model': model.d_model,
                'n_free': model.n_free,
                'n_relation': model.n_relation,
                'args': vars(args),
                'epoch': epoch,
                'val_loss': best_val_loss
            }, checkpoint_path)
            print(f"  Saved best model (val_loss={best_val_loss:.4f})")
    
    # Save final model
    final_path = Path(args.output_dir) / f"sae_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'd_model': model.d_model,
        'n_free': model.n_free,
        'n_relation': model.n_relation,
        'args': vars(args),
        'epoch': epoch,
        'val_loss': avg_val_loss['total']
    }, final_path)
    print(f"\nSaved final model to {final_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_activations', type=str, required=True,
                        help='Path to training activations pickle')
    parser.add_argument('--val_activations', type=str, required=True,
                        help='Path to validation activations pickle')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for SAE checkpoints')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lambda_recon', type=float, default=1.0)
    parser.add_argument('--lambda_sparse', type=float, default=1e-3)
    parser.add_argument('--lambda_align', type=float, default=1.0)
    parser.add_argument('--lambda_ortho', type=float, default=1e-2,
                        help='Weight for orthogonality loss between relation and free slots')
    parser.add_argument('--lambda_value', type=float, default=0.5,
                        help='Weight for value prediction loss (entity tokens)')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = ActivationDataset(args.train_activations)
    val_dataset = ActivationDataset(args.val_activations)
    
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    
    # Get d_model from first sample
    sample = train_dataset[0]
    d_model = sample['h'].shape[0]
    print(f"d_model: {d_model}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=0)
    
    # Initialize model
    print("Initializing Large SAE model...")
    model = LargeSupervisedSAE(d_model=d_model, n_free=100000, n_relation=20).to(device)
    print(f"Model: {model.n_free} free slots + {model.n_relation} supervised slots = {model.n_slots} total")
    
    # Train
    print("\nStarting training...")
    train_sae(model, train_loader, val_loader, args, device)
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
