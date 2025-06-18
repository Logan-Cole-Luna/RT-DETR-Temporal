#!/usr/bin/env python3
"""
v4-train.py - Enhanced RT-DETR Training with Proper Object Detection Metrics
- Fixed validation issues with comprehensive error handling
- Validation runs every 20% of epochs (configurable)
- Blue training progress bars, green validation progress bars
- Proper object detection metrics (IoU-based P/R/F1)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import time
from pathlib import Path
import json
import traceback
from tqdm import tqdm
import math
from collections import defaultdict
import numpy as np

import config

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from src.core import YAMLConfig
from src.data.uav_temporal import UAVTemporalMotionDataset, uav_temporal_motion_collate_fn
from src.zoo.rtdetr.temporal_rtdetr_fixed import MotionStrengthModule, TemporalRTDETR
from src.zoo.rtdetr.rtdetr import RTDETR


class EnhancedMonitor:
    """Enhanced monitoring with proper object detection metrics"""
    
    def __init__(self, output_dir, model_type_str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_type_str = model_type_str
        
        self.metrics_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_precision': [],
            'train_recall': [],
            'train_f1': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        self.log_file = self.output_dir / "training_log.txt"
        self.metrics_file = self.output_dir / "metrics.json"
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("UAV RT-DETR Training Log - v4 Enhanced with Proper Detection Metrics\n")
            f.write("=" * 80 + "\n")
    
    def log(self, message):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"{timestamp} - {message}")
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{timestamp} - {message}\n")
    
    def update_epoch(self, epoch, train_metrics, val_metrics, current_lr):
        """Update metrics and check for improvements"""
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['train_loss'].append(train_metrics['loss'])
        self.metrics_history['val_loss'].append(val_metrics['loss'])
        self.metrics_history['train_precision'].append(train_metrics.get('precision', 0))
        self.metrics_history['train_recall'].append(train_metrics.get('recall', 0))
        self.metrics_history['train_f1'].append(train_metrics.get('f1', 0))
        self.metrics_history['val_precision'].append(val_metrics.get('precision', 0))
        self.metrics_history['val_recall'].append(val_metrics.get('recall', 0))
        self.metrics_history['val_f1'].append(val_metrics.get('f1', 0))
        self.metrics_history['learning_rate'].append(current_lr)
        
        # Save metrics to JSON
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # Check for improvement (prioritize F1 score, fallback to loss)
        val_f1 = val_metrics.get('f1', 0)
        val_loss = val_metrics['loss']
        
        is_best = False
        
        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.patience_counter = 0
            is_best = True
        elif val_f1 == self.best_val_f1 and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.patience_counter = 0
            is_best = True
        else:
            self.patience_counter += 1
        
        return is_best


def calculate_detection_metrics(outputs, targets, confidence_threshold=0.5, iou_threshold=0.5):
    """
    Calculate proper object detection metrics (Precision, Recall, F1)
    """
    def box_iou_single(box1, box2):
        """Calculate IoU between two boxes in cxcywh format"""
        # Convert to xyxy
        def cxcywh_to_xyxy(box):
            x_c, y_c, w, h = box
            return [x_c - w/2, y_c - h/2, x_c + w/2, y_c + h/2]
        
        box1_xyxy = cxcywh_to_xyxy(box1)
        box2_xyxy = cxcywh_to_xyxy(box2)
        
        # Calculate intersection
        x1 = max(box1_xyxy[0], box2_xyxy[0])
        y1 = max(box1_xyxy[1], box2_xyxy[1])
        x2 = min(box1_xyxy[2], box2_xyxy[2])
        y2 = min(box1_xyxy[3], box2_xyxy[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        area1 = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
        area2 = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    batch_size = outputs['pred_logits'].shape[0]
    
    for i in range(batch_size):
        # Get predictions
        pred_logits = outputs['pred_logits'][i]  # [num_queries, num_classes]
        pred_boxes = outputs['pred_boxes'][i]    # [num_queries, 4]
        
        # Convert to probabilities
        pred_probs = F.sigmoid(pred_logits)
        
        # Get confidence scores (for single class)
        if pred_probs.shape[1] == 1:
            confidence_scores = pred_probs[:, 0]
        else:
            confidence_scores = pred_probs.max(dim=1)[0]
        
        # Filter confident predictions
        confident_mask = confidence_scores > confidence_threshold
        pred_boxes_confident = pred_boxes[confident_mask]
        pred_scores_confident = confidence_scores[confident_mask]
        
        # Get ground truth
        if i < len(targets):
            target = targets[i] if isinstance(targets[i], dict) else targets[i][0]
            gt_boxes = target.get('boxes', torch.tensor([]))
            
            if len(gt_boxes) == 0:
                gt_boxes = torch.tensor([])
        else:
            gt_boxes = torch.tensor([])
        
        num_pred = len(pred_boxes_confident)
        num_gt = len(gt_boxes)
        
        if num_pred == 0 and num_gt == 0:
            # True negative case
            continue
        elif num_pred == 0 and num_gt > 0:
            # All ground truth are false negatives
            total_fn += num_gt
            continue
        elif num_pred > 0 and num_gt == 0:
            # All predictions are false positives
            total_fp += num_pred
            continue
        
        # Match predictions to ground truth using IoU
        matched_gt = set()
        
        # Sort predictions by confidence (highest first)
        if len(pred_scores_confident) > 0:
            sorted_indices = torch.argsort(pred_scores_confident, descending=True)
            
            for pred_idx in sorted_indices:
                pred_box = pred_boxes_confident[pred_idx].cpu().numpy()
                
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx in range(len(gt_boxes)):
                    if gt_idx in matched_gt:
                        continue
                    
                    gt_box = gt_boxes[gt_idx].cpu().numpy()
                    iou = box_iou_single(pred_box, gt_box)
                    
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_gt_idx >= 0:
                    # True positive
                    total_tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    # False positive
                    total_fp += 1
        
        # Unmatched ground truth boxes are false negatives
        total_fn += num_gt - len(matched_gt)
    
    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1_score,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn
    }


def calculate_loss_with_weights(loss_dict, criterion):
    """Calculate properly weighted loss"""
    total_loss = 0
    loss_components = {}
    
    for k, v in loss_dict.items():
        if k in criterion.weight_dict:
            weighted_loss = v * criterion.weight_dict[k]
            total_loss += weighted_loss
            loss_components[k] = v.item() if hasattr(v, 'item') else v
    
    return total_loss, loss_components


def train_epoch(model, criterion, optimizer, train_loader, device, monitor, model_is_temporal, grad_clip_max_norm):
    """Training epoch with proper object detection metrics"""
    model.train()
    
    total_weighted_loss = 0
    num_batches = 0
    all_metrics = {'precision': [], 'recall': [], 'f1': []}
    loss_components_sum = defaultdict(float)
    gradient_norms = []
    
    # Blue training progress bar
    progress_bar = tqdm(train_loader, desc="Training", unit="batch", leave=False, colour="blue")
    
    for batch_idx, batch in enumerate(progress_bar):
        frames_batch, targets = batch
        
        frames_batch = frames_batch.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in target.items()} for target in targets]
        
        if not model_is_temporal and frames_batch.dim() == 5 and frames_batch.size(1) == 1:
            frames_batch = frames_batch.squeeze(1)

        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(frames_batch, targets)
        
        # Calculate loss
        loss_dict = criterion(outputs, targets)
        total_loss, loss_components = calculate_loss_with_weights(loss_dict, criterion)
        
        # Check for NaN/Inf
        if not math.isfinite(total_loss.item()):
            monitor.log(f"WARNING: Non-finite loss detected: {total_loss.item()}")
            continue

        # Backward pass
        total_loss.backward()
        
        # Gradient monitoring
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        gradient_norms.append(total_norm)
        
        if grad_clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
        
        optimizer.step()
        
        # Calculate detection metrics (for training monitoring)
        with torch.no_grad():
            # Use inference mode for metrics
            model.eval()
            inference_outputs = model(frames_batch)
            metrics = calculate_detection_metrics(inference_outputs, targets, confidence_threshold=0.3)
            model.train()
            
            for k in ['precision', 'recall', 'f1']:
                all_metrics[k].append(metrics[k])
        
        # Accumulate metrics
        total_weighted_loss += total_loss.item()
        num_batches += 1
        
        for k, v in loss_components.items():
            loss_components_sum[k] += v
        
        # Update progress bar
        avg_loss = total_weighted_loss / num_batches
        avg_f1 = np.mean(all_metrics['f1']) if all_metrics['f1'] else 0
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}", f1=f"{avg_f1:.3f}")
    
    # Calculate final metrics
    avg_loss = total_weighted_loss / num_batches if num_batches > 0 else 0
    avg_metrics = {k: np.mean(v) if v else 0 for k, v in all_metrics.items()}
    avg_loss_components = {k: v / num_batches for k, v in loss_components_sum.items()}
    avg_grad_norm = np.mean(gradient_norms) if gradient_norms else 0
    
    return {
        'loss': avg_loss,
        'precision': avg_metrics['precision'],
        'recall': avg_metrics['recall'],
        'f1': avg_metrics['f1'],
        'loss_components': avg_loss_components,
        'grad_norm': avg_grad_norm
    }


def validate_epoch(model, criterion, val_loader, device, model_is_temporal):
    """FIXED validation epoch with proper error handling"""
    model.eval()
    
    total_weighted_loss = 0
    num_batches = 0
    all_metrics = {'precision': [], 'recall': [], 'f1': []}
    
    # Green validation progress bar
    progress_bar = tqdm(val_loader, desc="Validation", leave=False, colour="green")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            try:
                frames_batch, targets = batch
                
                frames_batch = frames_batch.to(device)
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                           for k, v in target.items()} for target in targets]

                if not model_is_temporal and frames_batch.dim() == 5 and frames_batch.size(1) == 1:
                    frames_batch = frames_batch.squeeze(1)
                
                # FIXED: Loss calculation with proper error handling
                try:
                    outputs_with_targets = model(frames_batch, targets)
                    loss_dict = criterion(outputs_with_targets, targets)
                    total_loss, _ = calculate_loss_with_weights(loss_dict, criterion)
                    
                    if math.isfinite(total_loss.item()):
                        total_weighted_loss += total_loss.item()
                        num_batches += 1
                except Exception as loss_error:
                    # Skip this batch if loss calculation fails
                    print(f"Loss calculation failed for batch {batch_idx}: {loss_error}")
                    continue
                
                # FIXED: Metrics calculation with proper error handling
                try:
                    inference_outputs = model(frames_batch)
                    metrics = calculate_detection_metrics(inference_outputs, targets, confidence_threshold=0.3)
                    
                    for k in ['precision', 'recall', 'f1']:
                        all_metrics[k].append(metrics[k])
                except Exception as metrics_error:
                    # If metrics calculation fails, use zeros
                    print(f"Metrics calculation failed for batch {batch_idx}: {metrics_error}")
                    for k in ['precision', 'recall', 'f1']:
                        all_metrics[k].append(0.0)
                
                # Update progress bar
                if num_batches > 0:
                    avg_loss = total_weighted_loss / num_batches
                    avg_f1 = np.mean(all_metrics['f1']) if all_metrics['f1'] else 0
                    progress_bar.set_postfix(loss=f"{avg_loss:.4f}", f1=f"{avg_f1:.3f}")
                
            except Exception as batch_error:
                print(f"Batch {batch_idx} failed completely: {batch_error}")
                continue
    
    # Calculate final metrics
    avg_loss = total_weighted_loss / num_batches if num_batches > 0 else float('inf')
    avg_metrics = {k: np.mean(v) if v else 0 for k, v in all_metrics.items()}
    
    return {
        'loss': avg_loss,
        'precision': avg_metrics['precision'],
        'recall': avg_metrics['recall'],
        'f1': avg_metrics['f1']
    }


def main():
    print("Starting UAV RT-DETR Training v4 - Enhanced Detection Metrics")
    
    # Configuration
    model_is_temporal = config.MODEL_IS_TEMPORAL
    num_epochs = config.NUM_EPOCHS
    batch_size = config.BATCH_SIZE
    learning_rate = config.LEARNING_RATE
    weight_decay = config.WEIGHT_DECAY
    grad_clip_max_norm = config.GRAD_CLIP_MAX_NORM
    validation_percentage = getattr(config, 'VALIDATION_PERCENTAGE', 0.2)  # Default to 20%
    
    # Early stopping
    early_stopping_patience = 20
    
    # Dataset Configuration
    IMG_HEIGHT = config.IMG_HEIGHT
    IMG_WIDTH = config.IMG_WIDTH
    temporal_seq_len = config.BASE_TEMPORAL_SEQ_LEN
    temporal_motion_in_channels = config.MOTION_CALC_IN_CHANNELS
    train_subset_percentage = config.TRAIN_SUBSET_PERCENTAGE
    val_subset_percentage = config.VAL_SUBSET_PERCENTAGE
    
    # DataLoader Configuration
    num_workers = config.NUM_WORKERS
    shuffle_train = config.SHUFFLE_TRAIN
    drop_last_train = config.DROP_LAST_TRAIN
    drop_last_val = config.DROP_LAST_VAL
    
    # Paths
    img_folder = config.IMG_FOLDER
    train_seq_file = config.TRAIN_SEQ_FILE
    val_seq_file = config.VAL_SEQ_FILE
    
    # Model configuration
    if model_is_temporal:
        model_type_str = "temporal"
        actual_seq_len = temporal_seq_len
        actual_motion_enabled = True
        config_basename = config.TEMPORAL_CONFIG_BASENAME
    else:
        model_type_str = "non_temporal"
        actual_seq_len = 1
        actual_motion_enabled = False
        config_basename = config.NON_TEMPORAL_CONFIG_BASENAME

    config_file_path = str(Path(script_dir) / 'configs' / 'rtdetr' / config_basename)
    output_dir = config.OUTPUT_DIR_FORMAT_STRING.format(model_type_str=model_type_str)
    
    resume_checkpoint_path = config.RESUME_CHECKPOINT_PATH
    
    monitor = EnhancedMonitor(output_dir, model_type_str)
    monitor.log("Starting UAV RT-DETR Training v4 - Enhanced Detection Metrics")
    monitor.log("=" * 80)
    monitor.log("CONFIGURATION:")
    monitor.log(f"  Model Type: {model_type_str.capitalize()}")
    monitor.log(f"  Epochs: {num_epochs}")
    monitor.log(f"  Batch size: {batch_size}")
    monitor.log(f"  Learning rate: {learning_rate}")
    monitor.log(f"  Weight decay: {weight_decay}")
    monitor.log(f"  Gradient clipping: {grad_clip_max_norm}")
    monitor.log(f"  Image dimensions: {IMG_HEIGHT}x{IMG_WIDTH}")
    if model_is_temporal:
        monitor.log(f"  Sequence length: {actual_seq_len}")
        monitor.log(f"  Motion enabled: {actual_motion_enabled}")
    monitor.log(f"  Train subset: {train_subset_percentage*100 if train_subset_percentage is not None else 'Full'}%")
    monitor.log(f"  Val subset: {val_subset_percentage*100 if val_subset_percentage is not None else 'Full'}%")
    monitor.log(f"  Config file: {config_file_path}")
    monitor.log(f"  Resume from: {resume_checkpoint_path if resume_checkpoint_path else 'Not resuming'}")
    monitor.log(f"  Early stopping patience: {early_stopping_patience} epochs")
    
    # Calculate validation interval from percentage
    validation_epoch_interval = max(1, int(num_epochs * validation_percentage))
    monitor.log(f"  Validation interval: Every {validation_epoch_interval} epochs ({validation_percentage*100:.0f}%)")
    monitor.log("=" * 80)
    
    try:
        monitor.log("Setting up model and datasets...")
        
        # Datasets
        train_dataset = UAVTemporalMotionDataset(
            img_folder=img_folder,
            seq_file=train_seq_file,
            seq_len=actual_seq_len,
            motion_enabled=actual_motion_enabled
        )
        
        val_dataset = UAVTemporalMotionDataset(
            img_folder=img_folder,
            seq_file=val_seq_file,
            seq_len=actual_seq_len,
            motion_enabled=actual_motion_enabled
        )
        
        monitor.log(f"Total Training samples: {len(train_dataset)}")
        monitor.log(f"Total Validation samples: {len(val_dataset)}")
        
        # Subsets
        if train_subset_percentage is not None and 0 < train_subset_percentage < 1.0:
            train_subset_size = int(len(train_dataset) * train_subset_percentage)
            train_subset = Subset(train_dataset, range(min(train_subset_size, len(train_dataset))))
        else:
            train_subset = train_dataset
        monitor.log(f"Using training subset: {len(train_subset)} samples")
            
        if val_subset_percentage is not None and 0 < val_subset_percentage < 1.0:
            val_subset_size = int(len(val_dataset) * val_subset_percentage)
            val_subset = Subset(val_dataset, range(min(val_subset_size, len(val_dataset))))
        else:
            val_subset = val_dataset
        monitor.log(f"Using validation subset: {len(val_subset)} samples")

        # Data loaders
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=shuffle_train,
            collate_fn=uav_temporal_motion_collate_fn, num_workers=num_workers,
            drop_last=drop_last_train, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            collate_fn=uav_temporal_motion_collate_fn, num_workers=num_workers,
            drop_last=drop_last_val, pin_memory=True
        )
        
        monitor.log(f"Training batches: {len(train_loader)}")
        monitor.log(f"Validation batches: {len(val_loader)}")
        
        # Model setup
        cfg = YAMLConfig(config_file_path, resume_from=None)
        if not model_is_temporal and 'DLANet' in cfg.yaml_cfg:
            cfg.yaml_cfg['DLANet']['pretrained'] = False
        if not model_is_temporal and 'HybridEncoder' in cfg.yaml_cfg:
            cfg.yaml_cfg['HybridEncoder']['eval_spatial_size'] = [IMG_HEIGHT, IMG_WIDTH]
        
        if model_is_temporal:
            monitor.log("Creating TemporalRTDETR model...")
            motion_module = MotionStrengthModule(t_window=actual_seq_len, in_channels=temporal_motion_in_channels)
            model = TemporalRTDETR(
                backbone=cfg.model.backbone,
                encoder=cfg.model.encoder,
                decoder=cfg.model.decoder,
                motion_module=motion_module,
                use_motion=actual_motion_enabled
            )
        else:
            monitor.log("Creating standard RTDETR model...")
            model = RTDETR(backbone=cfg.model.backbone, encoder=cfg.model.encoder, decoder=cfg.model.decoder)
            if hasattr(model, 'decoder') and hasattr(model.decoder, 'eval_spatial_size'):
                model.decoder.eval_spatial_size = None
        
        criterion = cfg.criterion
        monitor.log(f"Criterion losses: {criterion.losses}")
        monitor.log(f"Criterion weight_dict: {criterion.weight_dict}")
        
        # Optimizers and schedulers
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate*0.1)
        plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        criterion = criterion.to(device)
        monitor.log(f"Using device: {device}")
        monitor.log(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        
        # Resume training if specified
        start_epoch = 0
        if resume_checkpoint_path and Path(resume_checkpoint_path).exists():
            monitor.log(f"Resuming training from checkpoint: {resume_checkpoint_path}")
            checkpoint = torch.load(resume_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            
            if 'best_val_loss' in checkpoint:
                monitor.best_val_loss = checkpoint['best_val_loss']
                monitor.best_val_f1 = checkpoint.get('best_val_f1', 0)
                monitor.best_epoch = checkpoint.get('best_epoch', start_epoch - 1)
            
            monitor.log(f"Resumed from epoch {start_epoch - 1}")
        elif resume_checkpoint_path:
            monitor.log(f"WARNING: Checkpoint not found: {resume_checkpoint_path}")

        # Test forward pass
        monitor.log("Testing model forward pass...")
        model.eval()
        with torch.no_grad():
            for images_batch_test, _ in train_loader: 
                images_batch_test = images_batch_test.to(device)
                if not model_is_temporal and images_batch_test.dim() == 5 and images_batch_test.size(1) == 1:
                    images_batch_test = images_batch_test.squeeze(1)
                outputs_test = model(images_batch_test)
                monitor.log(f"Forward pass successful. Output keys: {list(outputs_test.keys())}")
                if 'pred_logits' in outputs_test:
                    monitor.log(f"  pred_logits shape: {outputs_test['pred_logits'].shape}")
                break 
        
        monitor.log("Starting training loop...")
        monitor.log("-" * 80)
        
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_metrics = train_epoch(model, criterion, optimizer, train_loader, device, monitor, model_is_temporal, grad_clip_max_norm)
            
            # Validation (only run at specified intervals)
            val_metrics = {'loss': float('inf'), 'precision': 0, 'recall': 0, 'f1': 0}
            run_validation = (
                (epoch + 1) % validation_epoch_interval == 0 or  # Regular intervals
                epoch == num_epochs - 1 or                       # Last epoch
                epoch == start_epoch                              # First epoch (for baseline)
            )
            
            if run_validation:
                val_metrics = validate_epoch(model, criterion, val_loader, device, model_is_temporal)
                plateau_scheduler.step(val_metrics['loss'])
            else:
                # Use last validation metrics for early stopping
                if monitor.metrics_history['val_loss']:
                    val_metrics['loss'] = monitor.metrics_history['val_loss'][-1]
                    val_metrics['f1'] = monitor.metrics_history['val_f1'][-1]
            
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update monitoring
            is_best = monitor.update_epoch(epoch, train_metrics, val_metrics, current_lr)
            epoch_time = time.time() - epoch_start_time
            
            # Log training loss components
            monitor.log("Training Loss Components:")
            for k, v in train_metrics['loss_components'].items():
                monitor.log(f"  {k}: {v:.4f}")
            monitor.log(f"Average gradient norm: {train_metrics['grad_norm']:.4f}")
            
            # Log epoch results
            monitor.log(f"\nEpoch {epoch+1}/{num_epochs}")
            monitor.log(f"  Train Loss: {train_metrics['loss']:.4f} | Train P: {train_metrics['precision']:.3f} | Train R: {train_metrics['recall']:.3f} | Train F1: {train_metrics['f1']:.3f}")
            
            if run_validation and val_metrics['f1'] > 0:
                monitor.log(f"  Val Loss:   {val_metrics['loss']:.4f} | Val P:   {val_metrics['precision']:.3f} | Val R:   {val_metrics['recall']:.3f} | Val F1:   {val_metrics['f1']:.3f}")
            elif run_validation:
                monitor.log(f"  Val Loss:   {val_metrics['loss']:.4f} | Val Metrics: N/A")
            else:
                monitor.log(f"  Validation skipped (next at epoch {((epoch // validation_epoch_interval) + 1) * validation_epoch_interval + 1})")
            
            monitor.log(f"  LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
            
            # Save checkpoint
            checkpoint_path = monitor.output_dir / f"epoch_{epoch+1}_{monitor.model_type_str}_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'best_val_loss': monitor.best_val_loss,
                'best_val_f1': monitor.best_val_f1,
                'best_epoch': monitor.best_epoch,
                'model_type_str': monitor.model_type_str
            }, checkpoint_path)
            
            if is_best:
                monitor.log(f"  🎉 New best model! Val F1: {val_metrics['f1']:.3f}, Val Loss: {val_metrics['loss']:.4f}")
                best_model_path = monitor.output_dir / f"best_{monitor.model_type_str}_model.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'best_val_loss': monitor.best_val_loss,
                    'best_val_f1': monitor.best_val_f1,
                    'best_epoch': monitor.best_epoch,
                    'model_type_str': monitor.model_type_str
                }, best_model_path)
                monitor.log(f"Saved best model to {best_model_path}")
            
            # Early stopping check
            if monitor.patience_counter >= early_stopping_patience:
                monitor.log(f"\nEarly stopping triggered after {early_stopping_patience} epochs without improvement")
                break
        
        monitor.log("\nTraining completed!")
        monitor.log(f"Best Epoch: {monitor.best_epoch+1}")
        monitor.log(f"Best Validation F1: {monitor.best_val_f1:.4f}")
        monitor.log(f"Best Validation Loss: {monitor.best_val_loss:.4f}")
        return True
        
    except Exception as e:
        monitor.log(f"Error in training: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("✅ Training completed successfully!")
    else:
        print("❌ Training failed. Check logs for details.")
