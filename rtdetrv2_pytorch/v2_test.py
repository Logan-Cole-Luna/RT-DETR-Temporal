#!/usr/bin/env python3
"""
Test the trained temporal motion RT-DETR model and generate a video of predictions.
V2: Processes a folder of frames and outputs a video.
"""

import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import glob

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core import YAMLConfig
from src.zoo.rtdetr.temporal_rtdetr_fixed import MotionStrengthModule, TemporalRTDETR

# Default configuration (can be overridden by command-line arguments)
DEFAULT_IMG_HEIGHT = 256
DEFAULT_IMG_WIDTH = 256
DEFAULT_SEQ_LEN = 5
DEFAULT_CONF_THRESHOLD = 0.5

# --- Configuration Start ---
class ScriptConfig:
    INPUT_FOLDER = Path(r"C:\\Users\\Logan\\Downloads\\train_uav\\train\\3700000000002_113556_2")
    OUTPUT_VIDEO = Path("./output/v2_temporal_output.mp4")
    CHECKPOINT_PATH = Path("./output/uav_temporal_motion_training/best_temporal_model.pth")
    CHECKPOINT_PATH = Path("output/uav_temporal_motion_training_v2/epoch_24_temporal_model.pth")
    CHECKPOINT_PATH = Path("output/uav_temporal_training_v2_train/epoch_60_temporal_model.pth")
    CONFIG_PATH = Path('./configs/rtdetr/rtdetr_dla34_6x_uav_temporal_fixed.yml')
    
    SEQ_LEN = DEFAULT_SEQ_LEN
    IMG_WIDTH = 640  # Changed from DEFAULT_IMG_WIDTH (256)
    IMG_HEIGHT = 512 # Changed from DEFAULT_IMG_HEIGHT (256)
    CONF_THRESHOLD = DEFAULT_CONF_THRESHOLD
    FPS = 10
# --- Configuration End ---

def load_trained_temporal_model(checkpoint_path, seq_len, config_path='./configs/rtdetr/rtdetr_dla34_6x_uav_temporal_fixed.yml'):
    """Load the trained temporal motion model."""
    print(f"Loading temporal model from: {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"✅ Checkpoint loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    base_cfg = YAMLConfig(config_path)
    backbone = base_cfg.model.backbone
    encoder = base_cfg.model.encoder
    decoder = base_cfg.model.decoder
    
    motion_module = MotionStrengthModule(t_window=seq_len, in_channels=3)
    
    model = TemporalRTDETR(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        motion_module=motion_module,
        use_motion=True  # Assuming motion was used during training
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Temporal model loaded successfully")
    if 'train_loss' in checkpoint and 'val_loss' in checkpoint:
        print(f"   Train Loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
        print(f"   Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    return model

def preprocess_frame(frame_bgr, target_size=(DEFAULT_IMG_WIDTH, DEFAULT_IMG_HEIGHT)):
    """Preprocess a single frame (BGR numpy array) for the model."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, target_size)
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1)  # HWC to CHW
    return frame_tensor

def draw_predictions(frame, boxes, scores, conf_threshold=DEFAULT_CONF_THRESHOLD):
    """Draws bounding boxes and scores on a frame."""
    h, w = frame.shape[:2]
    for box, score in zip(boxes, scores):
        if score < conf_threshold:
            continue
        
        # Convert normalized [cx, cy, w, h] to [x1, y1, x2, y2]
        cx, cy, bw, bh = box
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def main_video_processing(): # Removed args parameter
    """Main function to process frames and generate video."""
    print("Starting temporal model video processing...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = ScriptConfig() # Use the ScriptConfig class

    model = load_trained_temporal_model(config.CHECKPOINT_PATH, config.SEQ_LEN, config.CONFIG_PATH)
    if model is None:
        return
    model.to(device)

    image_files = sorted(glob.glob(os.path.join(config.INPUT_FOLDER, '*.jpg')) + 
                       glob.glob(os.path.join(config.INPUT_FOLDER, '*.png')))
    
    if not image_files:
        print(f"No image files found in {config.INPUT_FOLDER}")
        return

    # Determine video properties from the first frame
    first_frame_bgr = cv2.imread(image_files[0])
    if first_frame_bgr is None:
        print(f"Could not read the first image: {image_files[0]}")
        return
        
    original_h, original_w = first_frame_bgr.shape[:2]
    
    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'XVID'
    out_video = cv2.VideoWriter(str(config.OUTPUT_VIDEO), fourcc, config.FPS, (original_w, original_h))

    frame_buffer = [] # To store preprocessed frames for a sequence
    original_frame_buffer = [] # To store original frames for display

    print(f"Processing {len(image_files)} frames...")
    for i in tqdm(range(len(image_files))):
        frame_bgr = cv2.imread(image_files[i])
        if frame_bgr is None:
            print(f"Warning: Could not read frame {image_files[i]}. Skipping.")
            continue

        # Store original frame for potential display (if it's a middle frame)
        original_frame_buffer.append(frame_bgr.copy())
        
        # Preprocess and add to buffer
        processed_frame = preprocess_frame(frame_bgr, (config.IMG_WIDTH, config.IMG_HEIGHT))
        frame_buffer.append(processed_frame)

        if len(frame_buffer) == config.SEQ_LEN:
            # We have a full sequence
            sequence_tensor = torch.stack(frame_buffer).unsqueeze(0).to(device) # [1, seq_len, C, H, W]
            
            with torch.no_grad():
                outputs = model(sequence_tensor)
            
            pred_logits = outputs['pred_logits']  # [1, num_queries, num_classes]
            pred_boxes = outputs['pred_boxes']    # [1, num_queries, 4]
            
            # Assuming single class detection (UAV) and taking the first class logit
            confidence_scores = torch.sigmoid(pred_logits[0, :, 0]) # [num_queries]
            boxes_normalized = pred_boxes[0] # [num_queries, 4]
            
            # Get the middle frame of the original buffer to draw on
            # The prediction corresponds to the context of these seq_len frames.
            # We typically draw on the most recent or middle frame.
            # For this setup, let's draw on the last frame of the current input sequence.
            frame_to_display = original_frame_buffer[-1] 
            
            drawn_frame = draw_predictions(frame_to_display, 
                                           boxes_normalized.cpu().numpy(), 
                                           confidence_scores.cpu().numpy(),
                                           config.CONF_THRESHOLD)
            out_video.write(drawn_frame)
            
            # Slide the window: remove the oldest frame
            frame_buffer.pop(0)
            original_frame_buffer.pop(0)
        elif i >= config.SEQ_LEN -1 : # If buffer is not full yet, but we have enough past frames to write something
            # This handles the case where we are at the end of the video and the buffer isn't full
            # but we still want to write the frames that have been processed.
            # For simplicity, if the buffer isn't full at the start, we just write original frames until it is.
            # Here, we are past the initial buffer filling stage.
            # The current logic with pop(0) means we always try to maintain seq_len frames.
            # If we are at the very end and len(frame_buffer) < args.seq_len,
            # the current loop structure won't make a prediction.
            # This is generally fine, as we need a full sequence.
            pass

    # Ensure output directory exists
    config.OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    out_video.release()
    print(f"✅ Video saved to: {config.OUTPUT_VIDEO}")

if __name__ == "__main__":
    # Remove argparse and directly call main_video_processing
    # The ScriptConfig class at the top now holds the configuration
    main_video_processing() # Removed None argument
