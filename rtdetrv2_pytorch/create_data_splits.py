#!/usr/bin/env python3
"""
Create train/val splits from the test.txt file for UAV temporal training
"""

import os
import random

def create_train_val_splits():
    """Split the test.txt into train/val sets"""
    
    sequences_dir = r"c:\data\processed_anti_uav_v2\sequences"
    test_file = os.path.join(sequences_dir, "test.txt")
    
    print("Reading sequences from test.txt...")
    
    # Read all sequences
    with open(test_file, 'r') as f:
        all_sequences = [line.strip() for line in f if line.strip()]
    
    print(f"Total sequences: {len(all_sequences)}")
    
    # Shuffle for random split
    random.seed(42)  # For reproducible splits
    random.shuffle(all_sequences)
    
    # Split: 80% train, 20% val
    split_idx = int(0.8 * len(all_sequences))
    train_sequences = all_sequences[:split_idx]
    val_sequences = all_sequences[split_idx:]
    
    print(f"Train sequences: {len(train_sequences)}")
    print(f"Val sequences: {len(val_sequences)}")
    
    # Write train file
    train_file = os.path.join(sequences_dir, "train.txt")
    with open(train_file, 'w') as f:
        for seq in train_sequences:
            f.write(seq + '\n')
    
    # Write val file  
    val_file = os.path.join(sequences_dir, "val.txt")
    with open(val_file, 'w') as f:
        for seq in val_sequences:
            f.write(seq + '\n')
    
    print(f"✅ Created {train_file}")
    print(f"✅ Created {val_file}")
    
    # Check a few sample sequences
    print(f"\nSample train sequences:")
    for i, seq in enumerate(train_sequences[:3]):
        frames = seq.split()
        print(f"  {i+1}: {len(frames)} frames - {frames[0]} ... {frames[-1]}")
    
    return len(train_sequences), len(val_sequences)

if __name__ == "__main__":
    create_train_val_splits()
