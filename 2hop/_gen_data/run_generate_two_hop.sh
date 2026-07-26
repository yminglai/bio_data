#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/generate_two_hop.py" \
  --path "${SCRIPT_DIR}/../_dataset/_org/val_qa_data.json"

python "${SCRIPT_DIR}/generate_two_hop.py" \
  --path "${SCRIPT_DIR}/../_dataset/_org/train_qa_data.json"
