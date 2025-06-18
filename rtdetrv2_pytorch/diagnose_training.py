#!/usr/bin/env python3
"""
Diagnostic script to analyze model, data, and training issues
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import config
from src.core import YAMLConfig
from src.data.uav_temporal import UAVTemporalMotionDataset, uav_temporal_motion_collate_fn
from src.zoo.rtdetr.temporal_rtdetr_fixed import MotionStrengthModule, TemporalRTDETR


def analyze_dataset(dataset, name, max_samples=100):
    """Analyze dataset statistics"""
    print(f"\n=== {name} Dataset Analysis ===")
    print(f"Total samples: {len(dataset)}")
    
    bbox_areas = []
    bbox_aspect_ratios = []
    num_objects_per_image = []
    
    for i in range(min(len(dataset), max_samples)):
        try:
            frames, targets = dataset[i]
            if len(targets) > 0:
                target = targets[0]  # Take first frame's targets
                boxes = target.get('boxes', torch.tensor([]))
                if len(boxes) > 0:
                    # boxes are in normalized cxcywh format
                    w = boxes[:, 2]
                    h = boxes[:, 3]
                    areas = w * h
                    aspect_ratios = w / (h + 1e-8)
                    
                    bbox_areas.extend(areas.tolist())
                    bbox_aspect_ratios.extend(aspect_ratios.tolist())
                    num_objects_per_image.append(len(boxes))
                else:
                    num_objects_per_image.append(0)
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    if bbox_areas:
        print(f"Bounding box areas - Mean: {np.mean(bbox_areas):.4f}, Std: {np.std(bbox_areas):.4f}")
        print(f"Bounding box areas - Min: {np.min(bbox_areas):.4f}, Max: {np.max(bbox_areas):.4f}")
        print(f"Aspect ratios - Mean: {np.mean(bbox_aspect_ratios):.4f}, Std: {np.std(bbox_aspect_ratios):.4f}")
        print(f"Objects per image - Mean: {np.mean(num_objects_per_image):.2f}, Max: {np.max(num_objects_per_image)}")
        
        # Check for very small objects (potential issue)
        small_objects = [a for a in bbox_areas if a < 0.001]  # Less than 0.1% of image
        print(f"Very small objects (area < 0.001): {len(small_objects)} ({len(small_objects)/len(bbox_areas)*100:.1f}%)")
    else:
        print("No bounding boxes found!")


def analyze_model_outputs(model, criterion, data_loader, device, max_batches=5):
    """Analyze model outputs and loss components"""
    print(f"\n=== Model Output Analysis ===")
    
    model.eval()
    loss_components = {}
    prediction_stats = {}
    
    with torch.no_grad():
        for batch_idx, (frames_batch, targets) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
                
            frames_batch = frames_batch.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in target.items()} for target in targets]
            
            if frames_batch.dim() == 5 and frames_batch.size(1) == 1:
                frames_batch = frames_batch.squeeze(1)
            
            # Forward pass
            outputs = model(frames_batch, targets)
            loss_dict = criterion(outputs, targets)
            
            # Analyze outputs
            if 'pred_logits' in outputs:
                logits = outputs['pred_logits']  # [batch, queries, classes]
                probs = F.sigmoid(logits)
                
                # Check prediction confidence distribution
                max_probs = probs.max(dim=-1)[0]  # Max prob per query
                avg_confidence = max_probs.mean().item()
                
                if 'avg_confidence' not in prediction_stats:
                    prediction_stats['avg_confidence'] = []
                prediction_stats['avg_confidence'].append(avg_confidence)
                
                # Check how many queries are "active" (confident predictions)
                active_queries = (max_probs > 0.5).sum().item()
                if 'active_queries' not in prediction_stats:
                    prediction_stats['active_queries'] = []
                prediction_stats['active_queries'].append(active_queries)
            
            if 'pred_boxes' in outputs:
                boxes = outputs['pred_boxes']  # [batch, queries, 4]
                
                # Check for degenerate boxes
                w = boxes[:, :, 2]
                h = boxes[:, :, 3]
                areas = w * h
                
                degenerate_boxes = (areas < 1e-6).sum().item()
                very_large_boxes = (areas > 0.9).sum().item()
                
                if 'degenerate_boxes' not in prediction_stats:
                    prediction_stats['degenerate_boxes'] = []
                    prediction_stats['very_large_boxes'] = []
                prediction_stats['degenerate_boxes'].append(degenerate_boxes)
                prediction_stats['very_large_boxes'].append(very_large_boxes)
            
            # Collect loss components
            for k, v in loss_dict.items():
                if k not in loss_components:
                    loss_components[k] = []
                loss_components[k].append(v.item() if torch.is_tensor(v) else v)
    
    # Print loss component analysis
    print("Loss Components:")
    for k, values in loss_components.items():
        print(f"  {k}: Mean={np.mean(values):.4f}, Std={np.std(values):.4f}, Min={np.min(values):.4f}, Max={np.max(values):.4f}")
    
    # Print prediction statistics
    print("\nPrediction Statistics:")
    for k, values in prediction_stats.items():
        print(f"  {k}: Mean={np.mean(values):.2f}, Std={np.std(values):.2f}")


def analyze_gradients(model, criterion, data_loader, device, max_batches=3):
    """Analyze gradient flow and magnitudes"""
    print(f"\n=== Gradient Analysis ===")
    
    model.train()
    grad_norms = {}
    
    for batch_idx, (frames_batch, targets) in enumerate(data_loader):
        if batch_idx >= max_batches:
            break
            
        frames_batch = frames_batch.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in target.items()} for target in targets]
        
        if frames_batch.dim() == 5 and frames_batch.size(1) == 1:
            frames_batch = frames_batch.squeeze(1)
        
        model.zero_grad()
        
        # Forward and backward pass
        outputs = model(frames_batch, targets)
        loss_dict = criterion(outputs, targets)
        
        # Calculate total loss properly
        total_loss = 0
        for k, v in loss_dict.items():
            if k in criterion.weight_dict:
                total_loss += loss_dict[k] * criterion.weight_dict[k]
        
        total_loss.backward()
        
        # Collect gradient norms
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if name not in grad_norms:
                    grad_norms[name] = []
                grad_norms[name].append(grad_norm)
    
    # Print gradient statistics
    print("Gradient Norms (Top 10 largest):")
    avg_grad_norms = {name: np.mean(norms) for name, norms in grad_norms.items()}
    sorted_grads = sorted(avg_grad_norms.items(), key=lambda x: x[1], reverse=True)
    
    for name, avg_norm in sorted_grads[:10]:
        print(f"  {name}: {avg_norm:.6f}")
    
    # Check for gradient issues
    zero_grads = [name for name, avg_norm in avg_grad_norms.items() if avg_norm < 1e-8]
    large_grads = [name for name, avg_norm in avg_grad_norms.items() if avg_norm > 1.0]
    
    if zero_grads:
        print(f"\nParameters with very small gradients ({len(zero_grads)}):")
        for name in zero_grads[:5]:  # Show first 5
            print(f"  {name}")
    
    if large_grads:
        print(f"\nParameters with large gradients ({len(large_grads)}):")
        for name in large_grads:
            print(f"  {name}: {avg_grad_norms[name]:.6f}")


def main():
    print("RT-DETR Temporal Training Diagnostic Tool")
    print("=" * 50)
    
    # Configuration
    model_is_temporal = config.MODEL_IS_TEMPORAL
    actual_seq_len = config.BASE_TEMPORAL_SEQ_LEN if model_is_temporal else 1
    actual_motion_enabled = model_is_temporal
    
    print(f"Model type: {'Temporal' if model_is_temporal else 'Non-temporal'}")
    print(f"Sequence length: {actual_seq_len}")
    print(f"Motion enabled: {actual_motion_enabled}")
    
    # Setup datasets
    train_dataset = UAVTemporalMotionDataset(
        img_folder=config.IMG_FOLDER,
        seq_file=config.TRAIN_SEQ_FILE,
        seq_len=actual_seq_len,
        motion_enabled=actual_motion_enabled
    )
    
    val_dataset = UAVTemporalMotionDataset(
        img_folder=config.IMG_FOLDER,
        seq_file=config.VAL_SEQ_FILE,
        seq_len=actual_seq_len,
        motion_enabled=actual_motion_enabled
    )
    
    # Analyze datasets
    analyze_dataset(train_dataset, "Training")
    analyze_dataset(val_dataset, "Validation")
    
    # Setup model
    config_basename = config.TEMPORAL_CONFIG_BASENAME if model_is_temporal else config.NON_TEMPORAL_CONFIG_BASENAME
    config_file_path = str(Path(script_dir) / 'configs' / 'rtdetr' / config_basename)
    
    cfg = YAMLConfig(config_file_path, resume_from=None)
    
    if model_is_temporal:
        motion_module = MotionStrengthModule(t_window=actual_seq_len, in_channels=config.MOTION_CALC_IN_CHANNELS)
        model = TemporalRTDETR(
            backbone=cfg.model.backbone,
            encoder=cfg.model.encoder,
            decoder=cfg.model.decoder,
            motion_module=motion_module,
            use_motion=actual_motion_enabled
        )
    else:
        from src.zoo.rtdetr.rtdetr import RTDETR
        model = RTDETR(backbone=cfg.model.backbone, encoder=cfg.model.encoder, decoder=cfg.model.decoder)
    
    criterion = cfg.criterion
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    
    print(f"\nUsing device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"Criterion losses: {criterion.losses}")
    print(f"Criterion weight_dict: {criterion.weight_dict}")
    
    # Setup data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=False,  # Small batch for debugging
        collate_fn=uav_temporal_motion_collate_fn, num_workers=0,  # No workers for debugging
        drop_last=True
    )
    
    # Perform analyses
    analyze_model_outputs(model, criterion, train_loader, device)
    analyze_gradients(model, criterion, train_loader, device)
    
    print("\n" + "=" * 50)
    print("Diagnostic completed!")
    print("\nRecommendations:")
    print("1. Check if very small objects are causing training issues")
    print("2. Monitor for gradient explosion/vanishing")
    print("3. Ensure loss components are balanced")
    print("4. Use fixed training script v2-train_fixed.py")


if __name__ == "__main__":
    main()
