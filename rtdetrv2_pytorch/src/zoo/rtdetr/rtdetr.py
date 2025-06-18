"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register


__all__ = ['RTDETR', ]


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', 'motion_strength_module']

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None, motion_strength_module=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        self.motion_strength_module = motion_strength_module
        # If temporal motion module provided, setup adapter to project 4-channel input to 3 channels
        self.input_adapter = nn.Conv2d(4, 3, kernel_size=1, bias=False) if motion_strength_module else None

    def forward(self, x, targets=None):
        # If temporal motion module is enabled, x is a frame cuboid [B, T, C, H, W]
        if self.motion_strength_module is not None:
            # Compute motion strength map [B,1,H,W]
            motion_map = self.motion_strength_module(x)
            # Extract current frame (middle of sequence)
            mid = x.shape[1] // 2
            x_current = x[:, mid]  # [B, C, H, W]
            # Concatenate motion map as 4th channel
            x = torch.cat([x_current, motion_map], dim=1)
            # Project to 3 channels for backbone input
            x = self.input_adapter(x)
        # If input has temporal dimension but no motion module, select middle frame
        elif x.dim() == 5:
            mid = x.shape[1] // 2
            x = x[:, mid]  # collapse temporal dim

        # Apply multi-scale interpolation if configured (only for 4D tensors)
        if self.multi_scale and self.training and x.dim() == 4:
            sz = np.random.choice(self.multi_scale)
            try:
                x = F.interpolate(x, size=[sz, sz])
            except ValueError as e:
                # Skip interpolation if spatial dims mismatch
                print(f"[RTDETR] Skipping multi-scale interpolation due to: {e}")
            
        x = self.backbone(x)
        x = self.encoder(x)        
        x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
