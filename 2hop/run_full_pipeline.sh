#!/bin/bash

# Complete pipeline for 2-hop SAE training and analysis with 100k+20 architecture
set -e

echo "=========================================="
echo "2-Hop SAE Complete Pipeline"
echo "100,000 free + 20 supervised slots"
echo "=========================================="

# Configuration
TRAIN_DATA="2hop/_dataset/_gen/train_two_hop_qa_data_4k.jsonl"
VAL_DATA="2hop/_dataset/_gen/val_two_hop_qa_data_4k.jsonl"
MODEL_PATH="2hop/_trained_model_4k_mixed/checkpoint_epoch_100"
OUTPUT_DIR="2hop/sae_large"
LAYER=6

# Epoch checkpoints for grokking analysis
EPOCHS=(5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100)

echo ""
echo "=========================================="
echo "Step 1: Collect Activations from Epoch 100"
echo "=========================================="
mkdir -p 2hop/activations

python 2hop/01_collect_activations_2hop.py \
  --model_path "$MODEL_PATH" \
  --val_data "$TRAIN_DATA" \
  --output "2hop/activations/train_activations.pkl" \
  --layer "$LAYER"

python 2hop/01_collect_activations_2hop.py \
  --model_path "$MODEL_PATH" \
  --val_data "$VAL_DATA" \
  --output "2hop/activations/val_activations.pkl" \
  --layer "$LAYER"

echo "✓ Collected train and val activations"

echo ""
echo "=========================================="
echo "Step 2: Collect Activations for Grokking"
echo "=========================================="
mkdir -p 2hop/grokking_activations

for epoch in "${EPOCHS[@]}"; do
    echo "Collecting activations from epoch $epoch..."
    CHECKPOINT="2hop/_trained_model_4k_mixed/checkpoint_epoch_${epoch}"
    
    if [ -d "$CHECKPOINT" ]; then
        python 2hop/01_collect_activations_2hop.py \
          --model_path "$CHECKPOINT" \
          --val_data "$VAL_DATA" \
          --output "2hop/grokking_activations/epoch_${epoch}_activations.pkl" \
          --layer "$LAYER" \
          --max_samples 500
        echo "✓ Epoch $epoch done"
    else
        echo "⚠ Checkpoint not found: $CHECKPOINT"
    fi
done

echo ""
echo "=========================================="
echo "Step 3: Train Large SAE (100k+20 slots)"
echo "=========================================="

python 2hop/02_train_sae_2hop.py \
  --train_activations "2hop/activations/train_activations.pkl" \
  --val_activations "2hop/activations/val_activations.pkl" \
  --output_dir "$OUTPUT_DIR" \
  --epochs 200 \
  --batch_size 128 \
  --lr 1e-3 \
  --lambda_recon 1.0 \
  --lambda_sparse 1e-3 \
  --lambda_align 1.0

echo "✓ SAE training complete"

echo ""
echo "=========================================="
echo "Step 4: Evaluate SAE Binding Accuracy"
echo "=========================================="

python 2hop/03_evaluate_sae_2hop.py \
  --sae_checkpoint "$OUTPUT_DIR/sae_best.pt" \
  --lm_model_path "$MODEL_PATH" \
  --val_data "$VAL_DATA" \
  --layer "$LAYER" \
  --output_dir "$OUTPUT_DIR/evaluation"

echo "✓ Evaluation complete"

echo ""
echo "=========================================="
echo "Step 5: Grokking Analysis Across Epochs"
echo "=========================================="

python 2hop/06_analyze_grokking.py \
  --grokking_activations_dir "2hop/grokking_activations" \
  --sae_checkpoint "$OUTPUT_DIR/sae_best.pt" \
  --output_dir "2hop/grokking_analysis_large"

echo "✓ Grokking analysis complete"

echo ""
echo "=========================================="
echo "Step 6: Swap Intervention (1-token)"
echo "=========================================="

bash 2hop/run_swap_layer6.sh

echo "✓ 1-token swap complete"

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Results:"
echo "  - SAE checkpoints: $OUTPUT_DIR"
echo "  - SAE evaluation: $OUTPUT_DIR/evaluation"
echo "  - Grokking analysis: 2hop/grokking_analysis_large"
echo "  - Swap results: 2hop/swap_results/layer6_1token"
echo "=========================================="
