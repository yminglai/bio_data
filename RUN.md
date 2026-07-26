# Quick Run Guide

## Environments
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## 1-hop (bio QA) pipeline

All 1-hop scripts use paths relative to `1hop/`, so `cd 1hop` first
(or just run `bash 1hop/scripts/run_all_onehop.sh`, which does this for you).

1) Generate data (QA pairs + train/test knowledge graphs)
   ```bash
   cd 1hop
   python scripts/01_generate_dataset.py
   ```
2) Fine-tune base model
   ```bash
   python scripts/02_sft_base_model.py
   ```
3) Collect activations (layer 6 by default, matching the paper)
   ```bash
   python scripts/03_collect_activations.py
   ```
4) Train supervised SAE
   ```bash
   python scripts/04_train_sae.py
   ```
5) Evaluate & swap
   ```bash
   python scripts/05_evaluate_sae.py
   python scripts/06_swap_evaluate.py
   ```

## 2-hop pipeline (relations)
1) Generate two-hop QA in new format
   ```bash
   # from repo root
   python 2hop/_gen_data/generate_two_hop.py --path 2hop/_dataset/_org/train_qa_data.json
   python 2hop/_gen_data/generate_two_hop.py --path 2hop/_dataset/_org/val_qa_data.json
   # outputs to 2hop/_dataset/_gen/{train,val}_two_hop_qa_data.jsonl
   ```
   Downsample to 4k/4k (required: the runners read the *_4k.jsonl files):
   ```bash
   python 2hop/split_dataset.py
   # uses _gen files as input, writes *_4k.jsonl
   ```
2) Train 2-hop model (example: mixed 30%)
   ```bash
   bash 2hop/run_train_two_hop.sh
   ```
3) Collect activations & train SAE
   ```bash
   bash 2hop/run_full_pipeline.sh
   # runs: collect activations -> train SAE -> evaluate -> swap -> grokking analysis
   ```
4) Swap interventions (supervised SAE)
   ```bash
   bash 2hop/run_swap_layer6.sh
   ```

## Artifacts (ignored by git)
- models/, 2hop/_trained_model_*/, 2hop/sae_*, 2hop/activations/, 2hop/grokking_activations/, 2hop/swap_results/, generated datasets in 2hop/_dataset/_gen/. Keep generation scripts and README tracked.

## Visualization
- Main figure: `plots/main.png`.
- Plotting helper scripts are gitignored; regenerate plots locally if needed.
