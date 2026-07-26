# 2-Hop Training Implementation Notes (English)

## Data strategy

**Key principle:** every single-hop fact is verbalized into sentences and placed in the training text.
- No hidden facts
- Each relation has 8 paraphrased sentences
- During training the model sees all sentences for all relations

**Splits**
- Train (`train_two_hop_qa_data.jsonl`):
    - 100% paragraph mode: all 16 sentences (8 for relation_1 + 8 for relation_2) → LM objective
    - 50% QA mode: question + 2 answer tokens supervised
- Val (`val_two_hop_qa_data.jsonl`):
    - During training: 100% paragraph mode (sentences only)
    - During evaluation: 100% QA mode (questions)

## Hyperparameters (used in reference code)
```python
batch_size = 128
max_length = 128
num_epochs = 100
lr = 1e-4
weight_decay = 0.01
warmup_ratio = 0.1

# Model (standard GPT-2 124M, hardcoded in train_two_hop.py)
n_layer = 12
n_head = 12
n_embd = 768
```

Learning rate schedule: linear warmup (10% steps) + linear decay.

Optimizer: AdamW(lr=1e-4, weight_decay=0.01).

## Training loop
- Train phase each epoch:
    - iterate `train_dataloader` (mixed: paragraph + QA)
    - iterate `val_train_dataloader` (paragraph only)
    - per batch: forward → backward → optimizer.step() → scheduler.step()
- Eval phase each epoch:
    - `val_eval_dataloader` (QA mode) → loss, acc_1hop, acc_2hop

Logging (`training_log.csv`):
```
step, epoch, split, loss, acc_1hop, acc_2hop, lr
```
- `train`: every 200 steps
- `val`: end of each epoch

## Recommended command (mixed mode)
```bash
bash 2hop/run_train_two_hop.sh
```

Equivalent:
```bash
python 2hop/train_two_hop.py \
    --train_path 2hop/_dataset/_gen/train_two_hop_qa_data.jsonl \
    --val_path 2hop/_dataset/_gen/val_two_hop_qa_data.jsonl \
    --tokenizer_path 2hop/_base_model \
    --output_dir 2hop/_trained_model_mixed \
    --mode mixed \
    --qa_ratio 0.5 \
    --batch_size 128 \
    --max_length 128 \
    --num_epochs 100 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --num_sentences_per_relation 8 \
    --save_every 5 \
    --log_every 200 \
    --device cuda
```

## Expected behavior
- Grokking-like: loss drops early; val accuracy jumps mid/late training.
- Metrics: `acc_1hop` (entity_2), `acc_2hop` (entity_3); target ≥ 90% each.

## Next steps
1) Run training: `bash 2hop/run_train_two_hop.sh`
2) Monitor `training_log.csv` (acc_2hop)
3) Plots auto-saved under `_trained_model*/plots` (if enabled)
4) If 100 epochs insufficient, extend to 200–400.

## SAE integration
Use a hook to grab hidden activations for SAE training:
```python
activations = get_layer_activations(model, input_ids, attention_mask, layer_idx=3)
```
