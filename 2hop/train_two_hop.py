"""
2-Hop Reasoning Training Pipeline with 1-to-1 SAE Interface

This script trains a causal language model on a synthetic 2-hop reasoning task.
The model learns to predict two entity tokens (entity_2, entity_3) given a
question template and supporting relation sentences.

The training is structured to support later integration with a 1-to-1 SAE
(Sparse Autoencoder) where each SAE slot corresponds to a specific relation.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from tqdm import tqdm
import csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TwoHopDataset(Dataset):
    """
    Dataset for 2-hop reasoning task with mixed training support.
    
    Supports three modes:
    1. 'qa': Question-answering only - predicts exactly 2 entity tokens
    2. 'paragraph': Paragraph (relation sentences) - standard language modeling
    3. 'mixed': Both paragraph (100%) + qa (50%) for training
    
    For QA mode:
    - Input: "Question: <question>\\nAnswer:"
    - Output: Two entity tokens (entity_2, entity_3)
    - Only the 2 answer tokens are supervised
    
    For paragraph mode:
    - Input: All relation sentences as continuous text
    - Output: Standard next-token prediction (all tokens supervised)
    
    The dataset formats the data so that:
    - QA: Prompt tokens are masked with -100, only 2 answer tokens supervised
    - Paragraph: All tokens supervised (standard LM)
    """
    
    def __init__(
        self,
        path: str,
        tokenizer: GPT2Tokenizer,
        max_length: int = 512,
        num_sentences_per_relation: int = 8,
        mode: str = 'mixed',  # 'qa', 'paragraph', or 'mixed'
        qa_ratio: float = 0.5  # For mixed mode: ratio of QA samples to include
    ):
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_sentences_per_relation = num_sentences_per_relation
        self.mode = mode
        self.qa_ratio = qa_ratio
        
        if self.mode not in ['qa', 'paragraph', 'mixed']:
            raise ValueError(f"Mode must be 'qa', 'paragraph', or 'mixed', got {self.mode}")
        
        # Load data
        base_examples = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    base_examples.append(json.loads(line))
        
        # Build dataset based on mode
        self.examples = []
        
        if self.mode == 'qa':
            # QA mode only
            self.examples = [{'type': 'qa', 'data': ex} for ex in base_examples]
        
        elif self.mode == 'paragraph':
            # Paragraph mode only
            self.examples = [{'type': 'paragraph', 'data': ex} for ex in base_examples]
        
        elif self.mode == 'mixed':
            # Mixed mode: 100% paragraph + qa_ratio of QA
            import random
            
            # Add all as paragraph samples
            for ex in base_examples:
                self.examples.append({'type': 'paragraph', 'data': ex})
            
            # Add qa_ratio of samples as QA
            num_qa = int(len(base_examples) * self.qa_ratio)
            qa_samples = random.sample(base_examples, num_qa)
            for ex in qa_samples:
                self.examples.append({'type': 'qa', 'data': ex})
            
            # Shuffle the mixed dataset
            random.shuffle(self.examples)
            
            logger.info(f"Mixed mode: {len(base_examples)} paragraph + {num_qa} QA = {len(self.examples)} total")
        
        logger.info(f"Loaded {len(self.examples)} examples from {path} (mode={self.mode})")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example_wrapper = self.examples[idx]
        example_type = example_wrapper['type']
        example = example_wrapper['data']
        
        # Extract fields
        question_template = example['question_template']
        entity_1 = example['entity_1']
        relation_1 = example['relation_1']
        relation_2 = example['relation_2']
        entity_2 = example['entity_2']
        entity_3 = example['entity_3']
        relation_1_sentences = example['relation_1_sentences']
        relation_2_sentences = example['relation_2_sentences']
        
        # Build question by replacing placeholders
        question = question_template.replace('[ENTITY_1]', entity_1)
        question = question.replace('[RELATION_1]', relation_1.replace('_', ' '))
        question = question.replace('[RELATION_2]', relation_2.replace('_', ' '))
        
        # Select sentences (all sentences for paragraph mode)
        selected_sentences_1 = relation_1_sentences[:self.num_sentences_per_relation]
        selected_sentences_2 = relation_2_sentences[:self.num_sentences_per_relation]
        
        # Build prompt and target based on example type
        if example_type == 'qa':
            # QA mode: Question + Answer with 2 entity tokens
            prompt = f"Question: {question}\nAnswer:"
            target = f"{entity_2}{entity_3}"
            
            return self._build_qa_sample(prompt, target, entity_2, entity_3)
        else:  # 'paragraph'
            # Paragraph mode: Standard language modeling on relation sentences
            all_sentences = selected_sentences_1 + selected_sentences_2
            text = ' '.join(all_sentences)
            
            return self._build_paragraph_sample(text)
    
    def _build_qa_sample(self, prompt: str, target: str, entity_2: str, entity_3: str) -> Dict[str, torch.Tensor]:
        """
        Build QA sample where only the 2 answer entity tokens are supervised.
        """
        # Tokenize prompt
        prompt_encoding = self.tokenizer(
            prompt,
            add_special_tokens=True,
            max_length=self.max_length - 2,  # Reserve space for 2 target tokens
            truncation=True,
            return_tensors='pt'
        )
        
        prompt_input_ids = prompt_encoding['input_ids'][0]
        
        # Tokenize target entities separately to ensure they're single tokens
        entity_2_encoding = self.tokenizer(entity_2, add_special_tokens=False)
        entity_3_encoding = self.tokenizer(entity_3, add_special_tokens=False)
        
        entity_2_ids = entity_2_encoding['input_ids']
        entity_3_ids = entity_3_encoding['input_ids']
        
        # Verify that each entity is a single token
        if len(entity_2_ids) != 1:
            logger.warning(f"Entity '{entity_2}' tokenizes to {len(entity_2_ids)} tokens: {entity_2_ids}")
        if len(entity_3_ids) != 1:
            logger.warning(f"Entity '{entity_3}' tokenizes to {len(entity_3_ids)} tokens: {entity_3_ids}")
        
        # Take the first token of each entity (should be the only token)
        entity_2_id = entity_2_ids[0]
        entity_3_id = entity_3_ids[0]
        
        # Concatenate: prompt + entity_2 + entity_3
        input_ids = torch.cat([
            prompt_input_ids,
            torch.tensor([entity_2_id]),
            torch.tensor([entity_3_id])
        ])
        
        # Create attention mask (all ones)
        attention_mask = torch.ones_like(input_ids)
        
        # Create labels: -100 for prompt tokens, actual token IDs for target tokens
        labels = torch.full_like(input_ids, -100)
        labels[-2] = entity_2_id  # Second-to-last position
        labels[-1] = entity_3_id  # Last position
        
        # Pad if necessary
        seq_len = input_ids.size(0)
        if seq_len < self.max_length:
            padding_length = self.max_length - seq_len
            input_ids = torch.cat([input_ids, torch.zeros(padding_length, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((padding_length,), -100, dtype=torch.long)])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'entity_2': entity_2,
            'entity_3': entity_3,
            'prompt_length': len(prompt_input_ids),
            'sample_type': 'qa'
        }
    
    def _build_paragraph_sample(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Build paragraph sample with standard language modeling (all tokens supervised).
        """
        # Tokenize the entire text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]
        
        # For language modeling: labels = input_ids shifted by 1
        # The model predicts the next token at each position
        labels = input_ids.clone()
        
        # Pad if necessary
        seq_len = input_ids.size(0)
        if seq_len < self.max_length:
            padding_length = self.max_length - seq_len
            input_ids = torch.cat([input_ids, torch.zeros(padding_length, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((padding_length,), -100, dtype=torch.long)])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'entity_2': '',  # Not applicable for paragraph
            'entity_3': '',  # Not applicable for paragraph
            'prompt_length': 0,  # Not applicable for paragraph
            'sample_type': 'paragraph'
        }


def build_tokenizer_and_model(
    tokenizer_path: str = None,
    model_name: str = 'gpt2',
    vocab_size: int = None
) -> Tuple[GPT2Tokenizer, GPT2LMHeadModel]:
    """
    Build tokenizer and model for 2-hop training.
    
    Args:
        tokenizer_path: Path to pretrained tokenizer (with special entity tokens)
        model_name: Base model name (default: gpt2)
        vocab_size: Vocabulary size (if None, use tokenizer's vocab size)
    
    Returns:
        tokenizer, model
    """
    # Load tokenizer
    if tokenizer_path:
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer not found at {tokenizer_path}. Training requires the "
                f"tokenizer with the added entity/relation tokens (2hop/_base_model/); "
                f"the plain '{model_name}' tokenizer would silently break single-token "
                f"entity supervision."
            )
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    else:
        logger.info(f"Loading tokenizer from {model_name}")
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get vocab size
    if vocab_size is None:
        vocab_size = len(tokenizer)
    
    # Load or create model
    logger.info(f"Initializing GPT2 model with vocab_size={vocab_size}")
    config = GPT2Config.from_pretrained(model_name)
    config.vocab_size = vocab_size
    
    # Use standard GPT-2 (124M) configuration: 12-layer, 12-head, 768-dim
    config.n_layer = 12
    config.n_head = 12
    config.n_embd = 768
    
    model = GPT2LMHeadModel(config)
    
    # Resize token embeddings to match vocabulary
    model.resize_token_embeddings(len(tokenizer))
    
    logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    return tokenizer, model


def get_layer_activations(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int
) -> torch.Tensor:
    """
    Extract hidden states at a specific transformer layer.
    
    This function is designed to support SAE integration by providing
    clean access to intermediate layer activations.
    
    Args:
        model: GPT2 model
        input_ids: Input token IDs [batch, seq_len]
        attention_mask: Attention mask [batch, seq_len]
        layer_idx: Layer index to extract (0-indexed)
    
    Returns:
        Hidden states at specified layer [batch, seq_len, hidden_dim]
    """
    # Run model with output_hidden_states=True
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True
    )
    
    # Extract hidden states at specified layer
    # hidden_states is a tuple of (num_layers + 1) tensors
    # Index 0 is embeddings, index 1 is layer 0, etc.
    hidden_states = outputs.hidden_states[layer_idx + 1]
    
    return hidden_states


def compute_accuracy(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    batch: Dict[str, torch.Tensor],
    device: torch.device
) -> Tuple[float, float]:
    """
    Compute 1-hop and 2-hop accuracy for a batch.
    Only applies to QA samples (paragraph samples return 0, 0).
    
    Args:
        model: GPT2 model
        tokenizer: Tokenizer
        batch: Batch dictionary with 'input_ids', 'attention_mask', 'entity_2', 'entity_3', 'prompt_length', 'sample_type'
        device: Device to run on
    
    Returns:
        (accuracy_1hop, accuracy_2hop)
    """
    model.eval()
    
    batch_size = batch['input_ids'].size(0)
    correct_1hop = 0
    correct_2hop = 0
    num_qa_samples = 0
    
    with torch.no_grad():
        for i in range(batch_size):
            # Skip paragraph samples
            if batch['sample_type'][i] == 'paragraph':
                continue
            
            num_qa_samples += 1
            
            # Get prompt input_ids (exclude the target tokens)
            prompt_len = batch['prompt_length'][i].item()
            prompt_input_ids = batch['input_ids'][i, :prompt_len].unsqueeze(0).to(device)
            
            # Generate 2 tokens
            generated = model.generate(
                prompt_input_ids,
                max_new_tokens=2,
                min_new_tokens=2,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=None,  # Don't stop early
                num_beams=1
            )
            
            # Extract the 2 generated tokens
            generated_tokens = generated[0, prompt_len:prompt_len+2]
            
            if len(generated_tokens) >= 2:
                pred_entity_2_id = generated_tokens[0].item()
                pred_entity_3_id = generated_tokens[1].item()
                
                # Decode predictions
                pred_entity_2 = tokenizer.decode([pred_entity_2_id], skip_special_tokens=True).strip()
                pred_entity_3 = tokenizer.decode([pred_entity_3_id], skip_special_tokens=True).strip()
                
                # Ground truth
                gt_entity_2 = batch['entity_2'][i]
                gt_entity_3 = batch['entity_3'][i]
                
                # Check accuracy
                if pred_entity_2 == gt_entity_2:
                    correct_1hop += 1
                if pred_entity_3 == gt_entity_3:
                    correct_2hop += 1
    
    # Return accuracy only for QA samples
    if num_qa_samples == 0:
        return 0.0, 0.0
    
    acc_1hop = correct_1hop / num_qa_samples
    acc_2hop = correct_2hop / num_qa_samples
    
    return acc_1hop, acc_2hop


def evaluate_two_hop_model(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Evaluate model on validation/test set.
    
    Args:
        model: GPT2 model
        tokenizer: Tokenizer
        dataloader: DataLoader for evaluation
        device: Device to run on
    
    Returns:
        (avg_loss, avg_acc_1hop, avg_acc_2hop)
    """
    model.eval()
    
    total_loss = 0.0
    total_acc_1hop = 0.0
    total_acc_2hop = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Compute accuracy
            acc_1hop, acc_2hop = compute_accuracy(model, tokenizer, batch, device)
            total_acc_1hop += acc_1hop
            total_acc_2hop += acc_2hop
            
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_acc_1hop = total_acc_1hop / num_batches
    avg_acc_2hop = total_acc_2hop / num_batches
    
    return avg_loss, avg_acc_1hop, avg_acc_2hop


def train_two_hop_model(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    train_dataloader: DataLoader,
    val_train_dataloader: DataLoader,
    val_eval_dataloader: DataLoader,
    device: torch.device,
    output_dir: str,
    num_epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    save_every: int = 5,
    log_every: int = 200,
    mode: str = 'qa'
):
    """
    Train the 2-hop reasoning model.
    
    Args:
        model: GPT2 model
        tokenizer: Tokenizer
        train_dataloader: Training data loader (from train set)
        val_train_dataloader: Validation data loader for training (paragraph mode)
        val_eval_dataloader: Validation data loader for evaluation (QA mode)
        device: Device to train on
        output_dir: Directory to save checkpoints and logs
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        warmup_ratio: Ratio of total steps for warmup
        save_every: Save checkpoint every N epochs
        log_every: Log training metrics every N steps
        mode: Training mode ('qa', 'paragraph', or 'mixed')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Training mode: {mode}")
    
    # Setup optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Calculate total steps and warmup steps
    total_steps = len(train_dataloader) * num_epochs + len(val_train_dataloader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    
    # Setup learning rate scheduler with warmup
    from torch.optim.lr_scheduler import LambdaLR
    
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
        )
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    logger.info(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    # Setup CSV logging
    log_file = os.path.join(output_dir, 'training_log.csv')
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'epoch', 'split', 'loss', 'acc_1hop', 'acc_2hop', 'lr'])
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    # For smooth loss tracking
    running_loss = 0.0
    running_loss_window = 100  # Average over 100 steps
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        model.train()
        epoch_loss = 0.0
        num_train_batches = 0
        
        # Process train dataloader
        progress_bar = tqdm(
            total=len(train_dataloader) + len(val_train_dataloader),
            desc=f"Training Epoch {epoch + 1}"
        )
        
        # Train on train set (mixed mode)
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate
            
            epoch_loss += loss.item()
            num_train_batches += 1
            global_step += 1
            
            # Update running loss for smooth tracking
            running_loss = 0.9 * running_loss + 0.1 * loss.item() if running_loss > 0 else loss.item()
            
            # Update progress bar with smoothed loss
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.update(1)
            progress_bar.set_postfix({'loss': f'{running_loss:.4f}', 'lr': f'{current_lr:.2e}'})
            
            # Log training metrics periodically with smoothed loss
            if global_step % log_every == 0:
                # Compute accuracy on current batch (only for QA samples)
                acc_1hop, acc_2hop = compute_accuracy(model, tokenizer, batch, device)
                current_lr = scheduler.get_last_lr()[0]
                
                with open(log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    # Log smoothed loss instead of single batch loss
                    writer.writerow([global_step, epoch, 'train', running_loss, acc_1hop, acc_2hop, current_lr])
                
                logger.info(
                    f"Step {global_step}: loss={running_loss:.4f}, "
                    f"acc_1hop={acc_1hop:.4f}, acc_2hop={acc_2hop:.4f}, lr={current_lr:.2e}"
                )
                
                model.train()  # Set back to training mode
        
        # Train on val set (paragraph mode)
        for batch in val_train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate
            
            epoch_loss += loss.item()
            num_train_batches += 1
            global_step += 1
            
            # Update running loss for smooth tracking
            running_loss = 0.9 * running_loss + 0.1 * loss.item() if running_loss > 0 else loss.item()
            
            # Update progress bar with smoothed loss
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.update(1)
            progress_bar.set_postfix({'loss': f'{running_loss:.4f}', 'lr': f'{current_lr:.2e}'})
            
            # Log training metrics periodically with smoothed loss
            if global_step % log_every == 0:
                # Compute accuracy on current batch (only for QA samples)
                acc_1hop, acc_2hop = compute_accuracy(model, tokenizer, batch, device)
                current_lr = scheduler.get_last_lr()[0]
                
                with open(log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    # Log smoothed loss instead of single batch loss
                    writer.writerow([global_step, epoch, 'train', running_loss, acc_1hop, acc_2hop, current_lr])
                
                logger.info(
                    f"Step {global_step}: loss={running_loss:.4f}, "
                    f"acc_1hop={acc_1hop:.4f}, acc_2hop={acc_2hop:.4f}, lr={current_lr:.2e}"
                )
                
                model.train()  # Set back to training mode
        
        progress_bar.close()
        avg_train_loss = epoch_loss / num_train_batches
        
        # Validation phase - evaluate on QA mode
        logger.info("Running validation...")
        val_loss, val_acc_1hop, val_acc_2hop = evaluate_two_hop_model(
            model, tokenizer, val_eval_dataloader, device
        )
        
        # Log epoch metrics
        logger.info(
            f"Epoch {epoch + 1} Summary:\n"
            f"  Train Loss: {avg_train_loss:.4f}\n"
            f"  Val Loss: {val_loss:.4f}\n"
            f"  Val Acc 1-hop: {val_acc_1hop:.4f}\n"
            f"  Val Acc 2-hop: {val_acc_2hop:.4f}"
        )
        
        current_lr = scheduler.get_last_lr()[0]
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([global_step, epoch, 'train_epoch', avg_train_loss, -1, -1, current_lr])
            writer.writerow([global_step, epoch, 'val', val_loss, val_acc_1hop, val_acc_2hop, current_lr])
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_dir = os.path.join(output_dir, f'checkpoint_epoch_{epoch + 1}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            
            # Save optimizer state
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'val_acc_1hop': val_acc_1hop,
                'val_acc_2hop': val_acc_2hop
            }, os.path.join(checkpoint_dir, 'training_state.pt'))
            
            logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_dir = os.path.join(output_dir, 'best_model')
            os.makedirs(best_checkpoint_dir, exist_ok=True)
            
            model.save_pretrained(best_checkpoint_dir)
            tokenizer.save_pretrained(best_checkpoint_dir)
            
            logger.info(f"Saved best model to {best_checkpoint_dir} (val_loss={val_loss:.4f})")
    
    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(description='Train 2-hop reasoning model')
    
    # Data paths
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to training JSONL file')
    parser.add_argument('--val_path', type=str, required=True,
                        help='Path to validation JSONL file')
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='Path to pretrained tokenizer (optional)')
    
    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save checkpoints and logs')
    
    # Model settings
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for optimizer')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio for learning rate scheduler')
    parser.add_argument('--mode', type=str, default='mixed', choices=['qa', 'paragraph', 'mixed'],
                        help='Training mode: qa, paragraph, or mixed (100%% paragraph + 50%% qa)')
    parser.add_argument('--qa_ratio', type=float, default=0.5,
                        help='For mixed mode: ratio of QA samples to include (default: 0.5)')
    parser.add_argument('--num_sentences_per_relation', type=int, default=8,
                        help='Number of sentences to use per relation')
    
    # Checkpointing
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_every', type=int, default=200,
                        help='Log training metrics every N steps')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Build tokenizer and model
    tokenizer, model = build_tokenizer_and_model(
        tokenizer_path=args.tokenizer_path,
    )
    model = model.to(device)
    
    # Create datasets
    # Training dataset: Use specified mode (mixed by default)
    train_dataset = TwoHopDataset(
        path=args.train_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        num_sentences_per_relation=args.num_sentences_per_relation,
        mode=args.mode,
        qa_ratio=args.qa_ratio
    )
    
    # Validation dataset for training: paragraph mode only (sentences participate in training)
    val_train_dataset = TwoHopDataset(
        path=args.val_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        num_sentences_per_relation=args.num_sentences_per_relation,
        mode='paragraph',  # Only use sentences for training
        qa_ratio=0.0
    )
    
    # Validation dataset for evaluation: QA mode (for consistent evaluation)
    val_eval_dataset = TwoHopDataset(
        path=args.val_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        num_sentences_per_relation=args.num_sentences_per_relation,
        mode='qa',  # Use QA mode for evaluation
        qa_ratio=1.0
    )
    
    # Create dataloaders
    # Training dataloader (from train set)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Validation training dataloader (val sentences for training)
    val_train_dataloader = DataLoader(
        val_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Validation evaluation dataloader (val QA for evaluation)
    val_eval_dataloader = DataLoader(
        val_eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Train model
    train_two_hop_model(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_train_dataloader=val_train_dataloader,
        val_eval_dataloader=val_eval_dataloader,
        device=device,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        save_every=args.save_every,
        log_every=args.log_every,
        mode=args.mode
    )


if __name__ == '__main__':
    main()
