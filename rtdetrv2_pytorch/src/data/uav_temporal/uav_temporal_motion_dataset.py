"""
Enhanced UAV Temporal Dataset with Motion Strength
Loads full temporal sequences and computes motion maps
"""

import os
import json
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple, Optional

from src.core import register


@register
class UAVTemporalMotionDataset(Dataset):
    """
    Enhanced dataset for loading temporal sequences of UAV frames with motion computation.
    
    Returns full temporal sequences that can be processed by MotionStrengthModule.
    """
    __inject__ = ['transforms']
    
    def __init__(
        self,
        img_folder: str,
        seq_file: str,
        seq_len: int = 5,
        transforms=None,
        return_masks: bool = False,
        motion_enabled: bool = True,
    ):
        """
        Args:
            img_folder: Path to images base folder
            seq_file: Path to sequence file
            seq_len: Number of frames per sequence
            transforms: Data transforms to apply
            return_masks: Whether to return segmentation masks
            motion_enabled: Whether to compute motion maps
        """
        self.img_folder = Path(img_folder)
        self.seq_file = Path(seq_file)
        self.seq_len = seq_len
        self._transforms = transforms
        self.return_masks = return_masks
        self.motion_enabled = motion_enabled
        
        # Load sequence information
        self.sequences = self._load_sequences()
        
        print(f"Loaded {len(self.sequences)} temporal sequences from {seq_file}")

    def _load_sequences(self) -> List[List[str]]:
        """Load sequences from the sequence file"""
        sequences = []
        
        if not self.seq_file.exists():
            print(f"Warning: Sequence file {self.seq_file} not found.")
            return []
        
        with open(self.seq_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Each line contains space-separated frame names for one sequence
                    frame_names = line.split()
                    if len(frame_names) == self.seq_len:
                        sequences.append(frame_names)
                    elif self.seq_len == 1:
                        # For non-temporal mode, only take the middle frame of each sequence
                        mid = len(frame_names) // 2
                        sequences.append([frame_names[mid]])
                    else:
                        print(f"Warning: Sequence has {len(frame_names)} frames, expected {self.seq_len}")
        
        return sequences
    
    def _load_image(self, frame_name: str) -> np.ndarray:
        """Load a single image"""
        # Handle path resolution
        if frame_name.startswith('images/'):
            relative_path = frame_name[7:]  # Remove 'images/'
            img_path = self.img_folder / relative_path
        else:
            img_path = self.img_folder / frame_name
        
        if not img_path.exists():
            print(f"Warning: Image {img_path} not found")
            return np.zeros((512, 640, 3), dtype=np.uint8)
        
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not load image {img_path}")
            return np.zeros((512, 640, 3), dtype=np.uint8)
        
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def _load_annotation(self, img_name: str) -> Dict[str, Any]:
        """Load annotation for a single frame"""
        # Extract filename and split info
        if img_name.startswith('images/'):
            filename = img_name.split('/')[-1]
            split_name = img_name.split('/')[1]
        else:
            filename = img_name
            split_name = "test"
        
        # Construct label path
        label_name = filename.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = self.img_folder.parent / "labels" / split_name / label_name
        
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:5])
                            boxes.append([cx, cy, w, h])
                            labels.append(class_id)
        
        # Convert to tensors
        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        
        return {
            'boxes': boxes,
            'labels': labels,
        }

    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get a temporal sequence of frames
        
        Returns:
            frames: Tensor of shape [T, C, H, W] - full temporal sequence
            target: Dict with 'boxes' and 'labels' for the middle frame
        """
        sequence = self.sequences[idx]
        
        # Load all frames in the sequence
        frames = []
        for frame_name in sequence:
            image = self._load_image(frame_name)
            
            # Convert to tensor and normalize
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            frames.append(image_tensor)
        
        # Stack frames: [T, C, H, W]
        frames_tensor = torch.stack(frames)
        
        # Get annotation for the middle frame (used as target)
        middle_idx = len(sequence) // 2
        target = self._load_annotation(sequence[middle_idx])
        
        # Apply transforms if specified (only to middle frame for now)
        if self._transforms is not None:
            # Convert middle frame back to PIL for transforms
            from PIL import Image
            middle_frame = frames_tensor[middle_idx].permute(1, 2, 0).numpy()
            middle_frame = (middle_frame * 255).astype(np.uint8)
            middle_frame_pil = Image.fromarray(middle_frame)
            
            # Apply transforms
            transformed_frame, target = self._transforms(middle_frame_pil, target)
            
            # Convert back to tensor if needed
            if not isinstance(transformed_frame, torch.Tensor):
                transformed_frame = torch.from_numpy(np.array(transformed_frame)).permute(2, 0, 1).float() / 255.0
            
            # Replace middle frame with transformed version
            frames_tensor[middle_idx] = transformed_frame
        
        return frames_tensor, target


@register
def uav_temporal_motion_collate_fn(batch):
    """
    Collate function for temporal motion dataset
    
    Args:
        batch: List of (frames, target) tuples
        
    Returns:
        frames: Tensor [B, T, C, H, W] - batch of temporal sequences
        targets: List of target dicts (for middle frames)
    """
    frames_list = []
    targets = []
    
    for frames, target in batch:
        frames_list.append(frames)  # [T, C, H, W]
        targets.append(target)
    
    # Stack frames: [B, T, C, H, W]
    frames_batch = torch.stack(frames_list)
    
    return frames_batch, targets
