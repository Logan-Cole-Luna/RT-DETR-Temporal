"""
Enhanced UAV Temporal Dataset with Motion Strength
Loads full temporal sequences and computes motion maps
"""

import os
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple, Optional
from torchvision.io import read_image
from torchvision.transforms import ToTensor

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
                    frame_names = line.split()
                    if len(frame_names) == self.seq_len:
                        sequences.append(frame_names)
                    elif self.seq_len == 1:
                        mid = len(frame_names) // 2
                        sequences.append([frame_names[mid]])
                    else:
                        print(f"Warning: Sequence has {len(frame_names)} frames, expected {self.seq_len}")
        
        return sequences
    
    def _load_image(self, frame_name: str) -> torch.Tensor:
        """Load a single image as torch.Tensor [C,H,W]"""
        # resolve path
        if frame_name.startswith('images/'):
            img_path = self.img_folder / frame_name[7:]
        else:
            img_path = self.img_folder / frame_name
        if not img_path.exists():
            print(f"Warning: Image {img_path} not found")
            # return zero tensor [3,H,W]
            return torch.zeros((3, 512, 640), dtype=torch.float32)
        # use read_image to avoid numpy
        img_t = read_image(str(img_path)).float() / 255.0  # [C, H, W]
        # if grayscale or wrong channels, pad or replicate
        if img_t.size(0) == 1:
            img_t = img_t.repeat(3, 1, 1)
        return img_t
    
    def _load_annotation(self, img_name: str) -> Dict[str, Any]:
        """Load annotation for a single frame"""
        if img_name.startswith('images/'):
            filename = img_name.split('/')[-1]
            split_name = img_name.split('/')[1]
        else:
            filename = img_name
            split_name = "test"
        
        label_name = filename.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = self.img_folder.parent / "labels" / split_name / label_name
        
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        boxes.append([cx, cy, w, h])
                        labels.append(class_id)
        
        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        
        return {'boxes': boxes, 'labels': labels}

    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Log data loading
        #print(f"[UAVTemporalMotionDataset] __getitem__ idx={idx}")
        sequence = self.sequences[idx]
        #print(f"[UAVTemporalMotionDataset] sequence names: {sequence}")
        frames = []
        for frame_name in sequence:
            img_t = self._load_image(frame_name)
            frames.append(img_t)
        frames_tensor = torch.stack(frames)  # [T,C,H,W]
        middle_idx = len(sequence) // 2
        target = self._load_annotation(sequence[middle_idx])
        #print(f"[UAVTemporalMotionDataset] raw target boxes: {target.get('boxes').shape}")
        # Add image_id for COCO API compatibility
        target['image_id'] = torch.tensor([idx], dtype=torch.int64)

        if self._transforms:
            mf = frames_tensor[middle_idx]  # [C,H,W]
            try:
                tf_frame, target = self._transforms(mf, target)
                # Ensure tensor
                if not isinstance(tf_frame, torch.Tensor):
                    tf_frame = ToTensor()(tf_frame)
                # Resize entire sequence to match transformed frame
                new_h, new_w = tf_frame.shape[1], tf_frame.shape[2]
                try:
                    # Attempt to interpolate full temporal sequence
                    frames_tensor = F.interpolate(frames_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    frames_tensor[middle_idx] = tf_frame
                except ValueError:
                    # Fallback: when dimensions mismatch, replace with single-frame sequence
                    frames_tensor = tf_frame.unsqueeze(0)
            except TypeError as e:
                print(f"[UAVTemporalMotionDataset] Skipping transforms for idx={idx} due to: {e}")
        # Log post-transform shapes
        # print(f"[UAVTemporalMotionDataset] frames_tensor shape: {frames_tensor.shape}, labels: {len(target.get('labels', []))}")
        
        return frames_tensor, target


@register
def uav_temporal_motion_collate_fn(batch):
    """
    Collate function for temporal motion dataset
    """
    frames_list, targets = zip(*batch)
    frames_batch = torch.stack(list(frames_list), dim=0)
    return frames_batch, list(targets)
