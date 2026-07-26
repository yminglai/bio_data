"""
Step 2: Fine-tune Base LLM on Biography QA
Light SFT to ensure the model can answer the 6 rule types consistently.
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
)
from pathlib import Path
import argparse
from tqdm import tqdm
import os
import math

class BioQADataset(Dataset):
    def __init__(self, qa_file, kg_file, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos_token = tokenizer.eos_token  # Use model's EOS token
        
        # Load QA pairs
        with open(qa_file, 'r') as f:
            self.qa_pairs = [json.loads(line) for line in f]
        
        # Load biographies for memorization training
        with open(kg_file, 'r') as f:
            persons = json.load(f)
            self.person_bios = {
                p['person_id']: p['biographies'][0]  # Use first bio variant
                for p in persons
            }
    
    def __len__(self):
        # Each person contributes: 1 biography + multiple QA pairs
        # For memorization: we need biography instances + QA instances
        return len(self.person_bios) + len(self.qa_pairs)
    
    def __getitem__(self, idx):
        # First part: biography memorization (learns to store facts)
        if idx < len(self.person_bios):
            person_id = list(self.person_bios.keys())[idx]
            bio = self.person_bios[person_id]
            
            # Format: biography + EOS token for proper sequence ending
            full_text = f"{bio}{self.eos_token}"
            
            encoding = self.tokenizer(
                full_text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            labels = input_ids.clone()  # Train on entire biography
            labels[attention_mask == 0] = -100  # No loss on padding

        # Second part: pure QA (learns to retrieve stored facts)
        else:
            qa_idx = idx - len(self.person_bios)
            qa = self.qa_pairs[qa_idx]
            
            # Format: Q: {question}\nA: {answer} + EOS (NO BIOGRAPHY!)
            prompt = f"Q: {qa['question']}\nA:"
            full_text = f"{prompt} {qa['answer']}{self.eos_token}"
            
            encoding = self.tokenizer(
                full_text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            
            # Labels: mask the prompt and padding, only train on answer
            labels = input_ids.clone()
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
            labels[:len(prompt_tokens)] = -100  # Ignore loss on prompt
            labels[attention_mask == 0] = -100  # No loss on padding
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2', 
                       help='Base model (gpt2, gpt2-medium, etc.)')
    parser.add_argument('--train_qa', type=str, default='data/generated/qa_train.jsonl')
    parser.add_argument('--train_kg', type=str, default='data/generated/train_kg.json')
    parser.add_argument('--output_dir', type=str, default='models/base_sft')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (will be calculated from max_steps)')
    parser.add_argument('--max_steps', type=int, default=80000, help='Total training steps')
    parser.add_argument('--batch_size', type=int, default=96, help='Total batch size (will be split across GPUs)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-4, help='Minimum learning rate for cosine decay')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Linear warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--adam_epsilon', type=float, default=1e-6)
    parser.add_argument('--max_length', type=int, default=512)
    args = parser.parse_args()
    
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    
    # Load dataset
    print(f"Loading training data from {args.train_qa}")
    train_dataset = BioQADataset(
        args.train_qa,
        args.train_kg,
        tokenizer,
        args.max_length
    )
    
    print(f"Training on {len(train_dataset)} instances (biographies + QA pairs)")
    
    # Setup device and multi-GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"Using device: {device}")
    print(f"Number of GPUs: {num_gpus}")
    
    if num_gpus > 0:
        for i in range(num_gpus):
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({gpu_mem:.1f} GB)")
    
    # Multi-GPU support
    if num_gpus > 1:
        print(f"Using DataParallel across {num_gpus} GPUs!")
        model = torch.nn.DataParallel(model)
    
    model.to(device)
    
    # Calculate per-device batch size to achieve total batch size of 96
    per_device_batch_size = max(1, args.batch_size // num_gpus)
    gradient_accumulation_steps = max(1, args.batch_size // (per_device_batch_size * num_gpus))
    effective_batch_size = per_device_batch_size * num_gpus * gradient_accumulation_steps
    
    print(f"\nBatch size configuration:")
    print(f"  Target total batch size: {args.batch_size}")
    print(f"  Per-device batch size: {per_device_batch_size}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Number of GPUs: {num_gpus}")
    print(f"  Effective batch size: {effective_batch_size}")
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Calculate epochs from max_steps
    steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    if args.epochs is None:
        args.epochs = (args.max_steps + steps_per_epoch - 1) // steps_per_epoch
    
    total_training_steps = args.max_steps
    
    print(f"\nTraining configuration:")
    print(f"  Max training steps: {total_training_steps}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total epochs: {args.epochs}")
    print(f"  Warmup steps: {args.warmup_steps}")
    
    # Optimizer: AdamW with weight_decay=0.1, eps=1e-6
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        eps=args.adam_epsilon
    )
    
    # Custom cosine schedule with warmup: linear warmup (0 → 1e-3) then cosine decay (1e-3 → 1e-4)
    def lr_lambda(current_step):
        if current_step < args.warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, args.warmup_steps))
        # Cosine decay
        progress = float(current_step - args.warmup_steps) / float(max(1, total_training_steps - args.warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scale from 1.0 to (min_lr / lr)
        min_lr_ratio = args.min_lr / args.lr
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"  LR schedule: {args.lr} (warmup) → cosine decay → {args.min_lr}")
    
    # Training loop
    print("\nStarting training...")
    global_step = 0
    optimizer.zero_grad()
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step, batch in enumerate(pbar):
            # Stop if we've reached max_steps
            if global_step >= args.max_steps:
                print(f"\nReached max_steps ({args.max_steps}). Stopping training.")
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            if num_gpus > 1:
                loss = loss.mean()  # Average across GPUs
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            epoch_loss += loss.item() * gradient_accumulation_steps
            
            # Update weights every gradient_accumulation_steps
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Log every 100 steps
                if global_step % 100 == 0:
                    avg_loss = epoch_loss / (step + 1)
                    lr = scheduler.get_last_lr()[0]
                    progress = global_step / args.max_steps * 100
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{lr:.2e}',
                        'step': f'{global_step}/{args.max_steps}',
                        'progress': f'{progress:.1f}%'
                    })
                    
                # Save checkpoint every 10000 steps
                if global_step % 10000 == 0:
                    checkpoint_dir = Path(args.output_dir) / f"checkpoint-step-{global_step}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    print(f"\nSaved checkpoint at step {global_step} to {checkpoint_dir}")
        
        # Check if we should stop
        if global_step >= args.max_steps:
            break
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} completed. Avg Loss: {avg_epoch_loss:.4f}, Global Step: {global_step}/{args.max_steps}")
    
    # Save final model
    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving final model to {final_dir}")
    
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    print("Training complete!")

if __name__ == "__main__":
    main()
