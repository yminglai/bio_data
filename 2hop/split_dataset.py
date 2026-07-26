"""
Split the 2-hop dataset into 4000 train + 4000 validation.

This script:
1. Loads the original 6400 train + 1600 val data
2. Combines them and randomly splits into 4000 train + 4000 val
3. Saves the new splits
"""

import json
import random
from pathlib import Path

def load_jsonl(path):
    """Load JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data, path):
    """Save data to JSONL file."""
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def main():
    # Paths (relative to this script so it works from any cwd)
    gen_dir = Path(__file__).resolve().parent / '_dataset' / '_gen'
    original_train = gen_dir / 'train_two_hop_qa_data.jsonl'
    original_val = gen_dir / 'val_two_hop_qa_data.jsonl'

    # Output paths
    new_train = gen_dir / 'train_two_hop_qa_data_4k.jsonl'
    new_val = gen_dir / 'val_two_hop_qa_data_4k.jsonl'
    
    print("Loading original data...")
    train_data = load_jsonl(original_train)
    val_data = load_jsonl(original_val)
    
    print(f"Original train: {len(train_data)} samples")
    print(f"Original val: {len(val_data)} samples")
    print(f"Total: {len(train_data) + len(val_data)} samples")
    
    # Combine all data
    all_data = train_data + val_data
    
    # Shuffle
    random.seed(42)  # For reproducibility
    random.shuffle(all_data)
    
    # Split into 4000 train + 4000 val
    new_train_data = all_data[:4000]
    new_val_data = all_data[4000:8000]
    
    print(f"\nNew split:")
    print(f"Train: {len(new_train_data)} samples")
    print(f"Validation: {len(new_val_data)} samples")
    
    # Save
    print(f"\nSaving to:")
    print(f"  {new_train}")
    print(f"  {new_val}")
    
    save_jsonl(new_train_data, new_train)
    save_jsonl(new_val_data, new_val)
    
    print("\n✓ Done! New datasets created.")
    print("\nTo use the new split, update your training command:")
    print("  --train_path 2hop/_dataset/_gen/train_two_hop_qa_data_4k.jsonl")
    print("  --val_path 2hop/_dataset/_gen/val_two_hop_qa_data_4k.jsonl")

if __name__ == '__main__':
    main()
