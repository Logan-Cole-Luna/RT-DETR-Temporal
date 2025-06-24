import os
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from src.core import register, create
from src.data.transforms import Compose


@register
class UAVDataset(Dataset):
    def __init__(self, img_dir, label_dir, sequence_file, transforms=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transforms = transforms
        # Build transforms pipeline if config dict is provided
        if isinstance(self.transforms, dict) and 'type' in self.transforms:
            t_cfg = self.transforms
            # Extract ops list under 'ops' or 'transforms'
            ops = t_cfg.get('ops', t_cfg.get('transforms', [])) or []
            self.transforms = Compose(ops)
        
        with open(sequence_file, 'r') as f:
            self.img_paths = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_dir / self.img_paths[idx]
        
        # The image is BGR: B=gray, G=motion, R=0.
        # We want RGB: R=gray, G=motion, B=0.
        # So we swap B and R channels.
        img = cv2.imread(str(img_path))
        img = img[:, :, [2, 1, 0]] # BGR to RGB -> R=0, G=motion, B=gray
        # now we want R=gray, G=motion, B=0
        img = img[:, :, [2, 1, 0]] # BGR to RGB, but our channels are custom.
        
        # Correct channel mapping:
        # cv2 reads as BGR.
        # B is last_r (gray), G is mot_r (motion), R is zero.
        # We want to create an RGB-like image. Let's make R=gray, G=motion, B=0.
        # So we need to map B -> R, G -> G, R -> B.
        # This is equivalent to swapping R and B.
        img = img[:, :, [2, 1, 0]]

        # Replicate gray channel (now R) to B channel.
        img[:, :, 2] = img[:, :, 0]

        h, w, _ = img.shape
        
        label_path = self.label_dir / Path(self.img_paths[idx]).with_suffix('.txt')
        
        # Read boxes and labels
        boxes_list = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    cls, cx, cy, wn, hn = map(float, line.strip().split())
                    x1 = (cx - wn / 2) * w
                    y1 = (cy - hn / 2) * h
                    x2 = (cx + wn / 2) * w
                    y2 = (cy + hn / 2) * h
                    boxes_list.append([cls, x1, y1, x2, y2])
        # Convert box list to tensor, ensure shape even if empty
        arr = np.array(boxes_list, dtype=np.float32).reshape(-1, 5)
        boxes = torch.as_tensor(arr[:, 1:], dtype=torch.float32)
        labels = torch.as_tensor(arr[:, 0], dtype=torch.int64)
        # compute area and iscrowd for coco api compatibility
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # image_id
        image_id = torch.tensor([idx])
        target = {
            'boxes': boxes,
            'labels': labels,
            'area': area,
            'iscrowd': iscrowd,
            'image_id': image_id,
            'orig_size': torch.as_tensor([h, w]),
            'size': torch.as_tensor([h, w]),
        }

        # Apply transforms if specified and callable
        if callable(self.transforms):
            # convert numpy HWC array to channel-first tensor
            img_t = torch.from_numpy(img).permute(2, 0, 1)
            # wrap as datapoints.Image for v2 transforms
            try:
                from torchvision import datapoints
                img_t = datapoints.Image(img_t)
            except ImportError:
                pass
            # attempt full transform (image + target)
            try:
                img_transformed, target_transformed = self.transforms(img_t, target)
                img, target = img_transformed, target_transformed
            except TypeError:
                # fallback to image-only transforms
                img_transformed = self.transforms(img_t)
                img = img_transformed
            # unwrap datapoints.Image back to Tensor if needed
            try:
                from torchvision import datapoints
                if isinstance(img, datapoints.Image):
                    img = torch.tensor(img)
            except ImportError:
                pass
            # ensure image_id remains
            if 'image_id' not in target:
                target['image_id'] = image_id

        return img, target
