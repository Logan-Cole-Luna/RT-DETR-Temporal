"""
Motion Strength Module and Enhanced RT-DETR for Temporal UAV Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from src.core import register


@register
class MotionStrengthModule(nn.Module):
    """
    Motion Strength Module inspired by MG-VTOD.
    Processes a sequence of frames to extract a motion map for the current frame.
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
                for c in range(self.temp_conv.weight.shape[1]):  # For each input channel
                    self.temp_conv.weight[0, c, 0, 0, 0] = -1.0  # t-1 (previous)
                    self.temp_conv.weight[0, c, 1, 0, 0] = 0.0   # t (current)
                    self.temp_conv.weight[0, c, 2, 0, 0] = 1.0   # t+1 (next)
                    
            elif self.temp_conv.weight.shape[2] == 5:  # t_window = 5
                # More sophisticated temporal filter for 5-frame sequences
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
        # Permute to [B, C, T, H, W] for Conv3d compatibility
        x = frames_cuboid.permute(0, 2, 1, 3, 4)        # Apply temporal convolution
        m = self.temp_conv(x)  # Output: [B, 1, T, H, W]

        # For motion detection, use the middle time slice for better temporal context
        # or aggregate across multiple time slices
        middle_idx = m.shape[2] // 2  # Middle time slice
        m_current_time_slice = F.relu(m[:, :, middle_idx, :, :])  # Output: [B, 1, H, W]
        
        # Apply spatial smoothing
        m_smooth = self.spat_conv(m_current_time_slice)  # Output: [B, 1, H, W]
        
        # Normalize to [0,1] to produce the motion strength map
        motion_map = torch.sigmoid(m_smooth)  # Output: [B, 1, H, W]
        
        return motion_map


@register 
class TemporalRTDETR(nn.Module):
    """
    Enhanced RT-DETR that processes temporal sequences with motion information
    """
    __inject__ = ['backbone', 'encoder', 'decoder']
    
    def __init__(self, 
                 backbone: nn.Module,
                 encoder, 
                 decoder,
                 motion_module=None,
                 use_motion=True,
                 motion_t_window=5,
                 motion_in_channels=3,
                 multi_scale=None):
        super().__init__()
        
        self.backbone = backbone
        self.encoder = encoder 
        self.decoder = decoder
        self.use_motion = use_motion
        self.multi_scale = multi_scale
        
        if self.use_motion:
            # If motion_module is not provided but use_motion is True,
            # create it automatically using the provided parameters
            if motion_module is None:
                motion_module = MotionStrengthModule(
                    t_window=motion_t_window,
                    in_channels=motion_in_channels
                )
            self.motion_module = motion_module
            # Modify backbone input channels from 3 to 4 (RGB + Motion)
            self._modify_backbone_input_channels()
    
    def _modify_backbone_input_channels(self):
        """Modify the first layer of backbone to accept 4 channels instead of 3"""
        modified = False
        
        # Handle DLANet wrapper structure
        if hasattr(self.backbone, 'model') and hasattr(self.backbone.model, 'base_layer'):
            base_layer = self.backbone.model.base_layer
            if isinstance(base_layer, nn.Sequential) and len(base_layer) > 0:
                old_conv = base_layer[0]  # First layer is Conv2d
                if isinstance(old_conv, nn.Conv2d):
                    new_conv = nn.Conv2d(
                        4,  # 4 input channels (RGB + Motion)
                        old_conv.out_channels,
                        kernel_size=old_conv.kernel_size,
                        stride=old_conv.stride,
                        padding=old_conv.padding,
                        bias=old_conv.bias is not None
                    )
                    
                    # Initialize new weights
                    with torch.no_grad():
                        # Copy RGB weights
                        new_conv.weight[:, :3, :, :] = old_conv.weight
                        # Initialize motion channel weights as small random values
                        new_conv.weight[:, 3:, :, :] = torch.randn_like(old_conv.weight[:, :1, :, :]) * 0.01
                        
                        if old_conv.bias is not None:
                            new_conv.bias = old_conv.bias
                    
                    # Replace the conv layer
                    base_layer[0] = new_conv
                    modified = True
                    
        # Handle direct DLA structure
        elif hasattr(self.backbone, 'base_layer'):
            if isinstance(self.backbone.base_layer, nn.Sequential):
                old_conv = self.backbone.base_layer[0]
                if isinstance(old_conv, nn.Conv2d):
                    new_conv = nn.Conv2d(
                        4,  # 4 input channels (RGB + Motion)
                        old_conv.out_channels,
                        kernel_size=old_conv.kernel_size,
                        stride=old_conv.stride,
                        padding=old_conv.padding,
                        bias=old_conv.bias is not None
                    )
                    
                    # Initialize new weights
                    with torch.no_grad():
                        # Copy RGB weights
                        new_conv.weight[:, :3, :, :] = old_conv.weight
                        # Initialize motion channel weights as small random values
                        new_conv.weight[:, 3:, :, :] = torch.randn_like(old_conv.weight[:, :1, :, :]) * 0.01
                        
                        if old_conv.bias is not None:
                            new_conv.bias = old_conv.bias
                    
                    # Replace the conv layer
                    self.backbone.base_layer[0] = new_conv
                    modified = True
                    
        # Handle ResNet-style backbone
        elif hasattr(self.backbone, 'conv1'):
            old_conv = self.backbone.conv1
            new_conv = nn.Conv2d(
                4,
                old_conv.out_channels, 
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = old_conv.weight
                new_conv.weight[:, 3:, :, :] = torch.randn_like(old_conv.weight[:, :1, :, :]) * 0.01
                
                if old_conv.bias is not None:
                    new_conv.bias = old_conv.bias
            
            self.backbone.conv1 = new_conv
            modified = True
        
        if modified:
            print("✅ Successfully modified backbone for 4-channel input")
        else:
            print("⚠️ Warning: Could not automatically modify backbone input channels")
    
    def forward(self, x, targets=None):
        """
        Args:
            x: Input tensor
               - If temporal: [B, T, C, H, W] - temporal sequence
               - If single frame: [B, C, H, W] - single frame
            targets: Ground truth targets (during training)
            
        Returns:
            Model outputs compatible with RT-DETR
        """
        # Handle different input formats
        if x.dim() == 5:  # Temporal input [B, T, C, H, W]
            if self.use_motion:
                # Compute motion map
                motion_map = self.motion_module(x)  # [B, 1, H, W]
                
                # Use middle frame + motion map
                middle_idx = x.size(1) // 2
                current_frame = x[:, middle_idx, :, :, :]  # [B, C, H, W]
                
                # Concatenate RGB + Motion: [B, 4, H, W]
                x_input = torch.cat([current_frame, motion_map], dim=1)
            else:
                # Just use middle frame
                middle_idx = x.size(1) // 2
                x_input = x[:, middle_idx, :, :, :]  # [B, C, H, W]
        
        elif x.dim() == 4:  # Single frame input [B, C, H, W]
            if self.use_motion:
                # Cannot compute motion from single frame - pad with zeros
                batch_size, channels, height, width = x.shape
                motion_map = torch.zeros(batch_size, 1, height, width, 
                                       device=x.device, dtype=x.dtype)
                x_input = torch.cat([x, motion_map], dim=1)
            else:
                x_input = x
        
        else:
            raise ValueError(f"Unexpected input dimensions: {x.shape}")
        
        # Standard RT-DETR forward pass with potentially 4-channel input
        x = self.backbone(x_input)
        x = self.encoder(x)
        x = self.decoder(x, targets)
        
        return x
