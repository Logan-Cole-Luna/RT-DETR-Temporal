#!/usr/bin/env python3

"""
Test to debug the _load_sequences issue
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data.uav_temporal.uav_temporal_dataset import UAVTemporalDataset

def debug_dataset():
    """Debug dataset creation"""
    
    print("🔍 Debugging UAV Dataset Creation...")
    
    try:
        print("1. Creating dataset instance...")
        
        # Check if the class has the method before creating instance
        print(f"2. Class has _load_sequences: {hasattr(UAVTemporalDataset, '_load_sequences')}")
        
        # Try to create with dummy paths first
        print("3. Creating with dummy paths...")
        dataset = UAVTemporalDataset(
            img_folder="dummy",
            seq_file="dummy.txt",
            seq_len=5,
            transforms=None
        )
        
        print(f"✅ Dataset created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        
        # Check instance methods
        print(f"\n🔍 Debugging instance...")
        try:
            # Create a partial instance to check methods
            instance = object.__new__(UAVTemporalDataset)
            print(f"Instance has _load_sequences: {hasattr(instance, '_load_sequences')}")
            print(f"Instance methods: {[m for m in dir(instance) if m.startswith('_load')]}")
        except Exception as e2:
            print(f"Error checking instance: {e2}")

if __name__ == "__main__":
    debug_dataset()
