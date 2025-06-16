"""
UAV Temporal Dataset for RT-DETR
Loads sequences of frames as prepared by prepare_uav_data_v2.py
"""

import os
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple, Optional

from src.core import register


@register
class UAVTemporalDataset(Dataset):
    """
    Dataset for loading temporal sequences of UAV frames.
    
    Expected structure from prepare_uav_data_v2.py:
    - images/{split}/*.jpg: Individual frames
    - labels/{split}/*.txt: YOLO format labels (class cx cy w h)
    - sequences/{split}.txt: List of sequence groups
    """
    __inject__ = ['transforms']
    
    def __init__(
        self,
        img_folder: str,
        seq_file: str,
        seq_len: int = 5,
        transforms=None,
        return_masks: bool = False,
    ):
        """
        Args:
            img_folder: Path to images base folder
            seq_file: Path to sequence file
            seq_len: Number of frames per sequence
            transforms: Data transforms to apply
            return_masks: Whether to return segmentation masks
        """
        self.img_folder = Path(img_folder)
        self.seq_file = Path(seq_file)
        self.seq_len = seq_len
        self._transforms = transforms
        self.return_masks = return_masks
        
        # Load sequence information
        self.sequences = self._load_sequences()
        
        print(f"Loaded {len(self.sequences)} sequences from {seq_file}")

    def _load_sequences(self) -> List[List[str]]:
        """Load sequences from the sequence file"""
        sequences = []
        
        if not self.seq_file.exists():
            print(f"Warning: Sequence file {self.seq_file} not found. Creating from image folder...")
            return self._create_sequences_from_images()
        
        with open(self.seq_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Each line contains space-separated frame names for one sequence
                    frame_names = line.split()
                    if len(frame_names) == self.seq_len:
                        sequences.append(frame_names)
                    else:
                        print(f"Warning: Sequence has {len(frame_names)} frames, expected {self.seq_len}")
        
        return sequences
        
    def _create_sequences_from_images(self) -> List[List[str]]:
        """Create sequences by grouping images by prefix"""
        if not self.img_folder.exists():
            print(f"Error: Image folder {self.img_folder} not found")
            return []
        
        # Group images by sequence prefix
        sequences_dict = {}
        for img_path in self.img_folder.glob("*.jpg"):
            # Assuming format: seqname_00000.jpg
            parts = img_path.stem.split('_')
            if len(parts) >= 2:
                seq_name = '_'.join(parts[:-1])
                frame_num = parts[-1]
                
                if seq_name not in sequences_dict:
                    sequences_dict[seq_name] = []
                sequences_dict[seq_name].append(img_path.name)
        
        # Sort each sequence and create fixed-length sequences
        sequences = []
        for seq_name, frame_list in sequences_dict.items():
            frame_list.sort()
            
            # Create overlapping sequences of seq_len
            for i in range(0, len(frame_list) - self.seq_len + 1, self.seq_len):
                sequence = frame_list[i:i + self.seq_len]
                if len(sequence) == self.seq_len:
                    sequences.append(sequence)
        
        return sequences
    
    def _load_annotation(self, img_name: str) -> Dict[str, Any]:
        """Load annotation for a single frame"""
        # img_name format: "images/test/filename.jpg" or just "filename.jpg"
        if img_name.startswith('images/'):
            # Extract just the filename
            filename = img_name.split('/')[-1]
            split_name = img_name.split('/')[1]  # "test", "train", or "val"
        else:
            filename = img_name
            split_name = "test"  # default
        
        # Replace .jpg with .txt for label file
        label_name = filename.replace('.jpg', '.txt').replace('.png', '.txt')
        
        # Construct label path: labels/{split}/filename.txt
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
                            
                            # Keep in normalized YOLO format for now
                            boxes.append([cx, cy, w, h])
                            labels.append(class_id)
        
        # If no annotations, create empty ones
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
        Get a sequence of frames and return the middle frame for training
        
        Returns:
            image: Tensor of shape [C, H, W] - middle frame of sequence
            target: Dict with 'boxes' and 'labels' for the middle frame
        """
        sequence = self.sequences[idx]
        
        # For now, we'll use the middle frame of each sequence for training
        # TODO: Extend RT-DETR to handle full temporal sequences
        middle_idx = len(sequence) // 2
        frame_name = sequence[middle_idx]
        
        # Load image - frame_name includes path like "images/test/filename.jpg"
        # We need to resolve this relative to the base img_folder parent
        if frame_name.startswith('images/'):
            # Remove the 'images/' prefix since img_folder already points to images
            relative_path = frame_name[7:]  # Remove 'images/'
            img_path = self.img_folder / relative_path
        else:
            # Fallback for direct filename
            img_path = self.img_folder / frame_name
        
        if not img_path.exists():
            print(f"Warning: Image {img_path} not found")
            # Create dummy image
            image = np.zeros((640, 640, 3), dtype=np.uint8)
        else:
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Could not load image {img_path}")
                image = np.zeros((640, 640, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotations - use the base filename for label lookup
        target = self._load_annotation(frame_name)
        
        # Apply transforms if specified
        if self._transforms is not None:
            # Convert to PIL Image or format expected by transforms
            from PIL import Image
            image = Image.fromarray(image)
            
            # Apply transforms (this might modify both image and target)
            image, target = self._transforms(image, target)
        
        # Ensure image is tensor
        if not isinstance(image, torch.Tensor):
            if hasattr(image, 'numpy'):  # PIL Image
                image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            else:
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, target


@register
def uav_single_frame_collate_fn(batch):
    """
    Simple collate function for single-frame UAV dataset
    
    Args:
        batch: List of (image, target) tuples
    
    Returns:
        Collated batch suitable for RT-DETR training
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    # Stack images
    images = torch.stack(images)  # [batch_size, C, H, W]
    
    return images, targets


@register
def uav_temporal_collate_fn(batch):
    """
    Custom collate function for temporal sequences
    
    Args:
        batch: List of (images, targets) tuples
    
    Returns:
        Collated batch suitable for RT-DETR training
    """
    images = []
    all_targets = []
    
    for seq_images, seq_targets in batch:
        # seq_images: [seq_len, C, H, W]
        # seq_targets: List of target dicts (length = seq_len)
        
        batch_size, seq_len = len(batch), seq_images.size(0)
        
        # For now, we'll use the middle frame of each sequence for training
        # TODO: Extend RT-DETR to handle full temporal sequences
        middle_idx = seq_len // 2
        
        images.append(seq_images[middle_idx])  # [C, H, W]
        all_targets.append(seq_targets[middle_idx])
    
    # Stack images
    images = torch.stack(images)  # [batch_size, C, H, W]
    
    return images, all_targets