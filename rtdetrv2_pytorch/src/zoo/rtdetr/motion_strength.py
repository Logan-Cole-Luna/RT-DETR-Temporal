"""Motion Strength Module inspired by MG-VTOD.
Processes a sequence of frames to extract a motion map for the current frame."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core import register


@register
class MotionStrengthModule(nn.Module):
    """
    Motion Strength Module inspired by MG-VTOD.
    Input: Frame cuboid [B, T, C, H, W] (normalized RGB frames)
    Output: Motion map [B, 1, H, W] (normalized to [0,1])
    """
    def __init__(self, t_window=3, in_channels=3):
        super().__init__()
        # Temporal high-pass filter (Magno-like stage)
        self.temp_conv = nn.Conv3d(in_channels, 1, 
                                   kernel_size=(t_window, 1, 1),
                                   padding=(t_window // 2, 0, 0), 
                                   bias=False)
        
        # Spatial low-pass filter (Ganglion-like stage)
        self.spat_conv = nn.Conv2d(1, 1, 
                                   kernel_size=5, 
                                   padding=2,
                                   bias=False)
        
        # Initialize filters for motion detection
        self._init_temporal_filter()
        self._init_spatial_filter()

    def _init_temporal_filter(self):
        """Initialize temporal convolution for motion detection"""
        with torch.no_grad():
            # Initialize all weights to zero first
            self.temp_conv.weight.zero_()
            
            if self.temp_conv.weight.shape[2] == 3:  # t_window = 3
                # Create temporal difference filter: next - previous
                for c in range(self.temp_conv.weight.shape[1]):
                    self.temp_conv.weight[0, c, 0, 0, 0] = -1.0  # t-1 (previous)
                    self.temp_conv.weight[0, c, 1, 0, 0] = 0.0   # t (current)
                    self.temp_conv.weight[0, c, 2, 0, 0] = 1.0   # t+1 (next)
                    
            elif self.temp_conv.weight.shape[2] == 5:  # t_window = 5
                for c in range(self.temp_conv.weight.shape[1]):
                    self.temp_conv.weight[0, c, 0, 0, 0] = -0.5  # t-2
                    self.temp_conv.weight[0, c, 1, 0, 0] = -0.5  # t-1
                    self.temp_conv.weight[0, c, 2, 0, 0] = 0.0   # t (current)
                    self.temp_conv.weight[0, c, 3, 0, 0] = 0.5   # t+1
                    self.temp_conv.weight[0, c, 4, 0, 0] = 0.5   # t+2

    def _init_spatial_filter(self):
        """Initialize spatial convolution as averaging filter"""
        with torch.no_grad():
            # Initialize as uniform averaging filter
            nn.init.constant_(self.spat_conv.weight, 1.0 / (5 * 5))

    def forward(self, frames_cuboid):
        """
        Args:
            frames_cuboid: [B, T, C, H, W] - temporal sequence of frames
        Returns:
            motion_map: [B, 1, H, W] - motion strength map for current frame
        """
        # Permute to [B, C, T, H, W] for Conv3d
        x = frames_cuboid.permute(0, 2, 1, 3, 4)
        m = self.temp_conv(x)  # [B, 1, T, H, W]

        middle_idx = m.shape[2] // 2
        m_current = F.relu(m[:, :, middle_idx, :, :])  # [B,1,H,W]
        m_smooth = self.spat_conv(m_current)
        motion_map = torch.sigmoid(m_smooth)
        return motion_map
