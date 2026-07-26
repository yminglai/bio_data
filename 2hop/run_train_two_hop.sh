#!/bin/bash
#
# Train 2-hop reasoning model in MIXED mode with 30% QA (70% OOD)
# Training: 100% paragraph (all 8000 samples) + 30% QA (train 1200 only)
# Validation: 100% QA (val 4000) - 70% are OOD questions
# Dataset: 4000 train + 4000 val = 8000 total
#

# Set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DATA="${SCRIPT_DIR}/_dataset/_gen/train_two_hop_qa_data_4k.jsonl"
VAL_DATA="${SCRIPT_DIR}/_dataset/_gen/val_two_hop_qa_data_4k.jsonl"
TOKENIZER_PATH="${SCRIPT_DIR}/_base_model"
OUTPUT_DIR="${SCRIPT_DIR}/_trained_model_4k_mixed"

# Training hyperparameters (based on reference code)
BATCH_SIZE=128          # Increased from 16
MAX_LENGTH=128          # Reduced from 512 (most samples are shorter)
NUM_EPOCHS=200          # Increased to 200 for better convergence
LR=1e-4                 # Increased from 5e-5
WEIGHT_DECAY=0.01       # Added weight decay
WARMUP_RATIO=0.1        # 10% warmup steps
NUM_SENTENCES=8         # Use all 8 sentences per relation
SAVE_EVERY=5            # Save less frequently
LOG_EVERY=200           # Log every 200 steps
MODE="mixed"            # Mixed mode: 100% paragraph + 30% QA
QA_RATIO=0.3            # 30% of train examples also as QA (70% OOD)

# Device (set CUDA_VISIBLE_DEVICES yourself to pick a specific GPU)
DEVICE="cuda"

echo "========================================="
echo "Training 2-hop Reasoning Model (30% QA, 70% OOD)"
echo "========================================="
echo "Train data: ${TRAIN_DATA}"
echo "Val data: ${VAL_DATA}"
echo "Tokenizer: ${TOKENIZER_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Mode: ${MODE} (100% paragraph + ${QA_RATIO}*100% QA = 30% QA, 70% OOD)"
echo "Batch size: ${BATCH_SIZE}"
echo "Max length: ${MAX_LENGTH}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Learning rate: ${LR}"
echo "Weight decay: ${WEIGHT_DECAY}"
echo "Warmup ratio: ${WARMUP_RATIO}"
echo "Num sentences: ${NUM_SENTENCES}"
echo "Device: ${DEVICE}"
echo "========================================="

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run training
python "${SCRIPT_DIR}/train_two_hop.py" \
    --train_path "${TRAIN_DATA}" \
    --val_path "${VAL_DATA}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size ${BATCH_SIZE} \
    --max_length ${MAX_LENGTH} \
    --num_epochs ${NUM_EPOCHS} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_ratio ${WARMUP_RATIO} \
    --num_sentences_per_relation ${NUM_SENTENCES} \
    --save_every ${SAVE_EVERY} \
    --log_every ${LOG_EVERY} \
    --mode ${MODE} \
    --qa_ratio ${QA_RATIO} \
    --device ${DEVICE}

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Training completed successfully!"
    echo "========================================="
    
    echo ""
    echo "Model checkpoints saved to: ${OUTPUT_DIR}"
    echo "Training log: ${OUTPUT_DIR}/training_log.csv"
else
    echo ""
    echo "========================================="
    echo "Training failed!"
    echo "========================================="
    exit 1
fi
