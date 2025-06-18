#!/usr/bin/env python3
"""
Proper evaluation script for RT-DETR models with correct metrics
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import config
from src.core import YAMLConfig
from src.data.uav_temporal import UAVTemporalMotionDataset, uav_temporal_motion_collate_fn
from src.zoo.rtdetr.temporal_rtdetr_fixed import MotionStrengthModule, TemporalRTDETR
from src.zoo.rtdetr.rtdetr import RTDETR


def bbox_iou(box1, box2):
    """Calculate IoU between two bounding boxes in cxcywh format"""
    # Convert cxcywh to xyxy
    def cxcywh_to_xyxy(boxes):
        x_c, y_c, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    box1_xyxy = cxcywh_to_xyxy(box1)
    box2_xyxy = cxcywh_to_xyxy(box2)
    
    # Calculate intersection
    inter_x1 = torch.max(box1_xyxy[:, None, 0], box2_xyxy[None, :, 0])
    inter_y1 = torch.max(box1_xyxy[:, None, 1], box2_xyxy[None, :, 1])
    inter_x2 = torch.min(box1_xyxy[:, None, 2], box2_xyxy[None, :, 2])
    inter_y2 = torch.min(box1_xyxy[:, None, 3], box2_xyxy[None, :, 3])
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Calculate union
    box1_area = (box1_xyxy[:, 2] - box1_xyxy[:, 0]) * (box1_xyxy[:, 3] - box1_xyxy[:, 1])
    box2_area = (box2_xyxy[:, 2] - box2_xyxy[:, 0]) * (box2_xyxy[:, 3] - box2_xyxy[:, 1])
    union_area = box1_area[:, None] + box2_area[None, :] - inter_area
    
    iou = inter_area / (union_area + 1e-6)
    return iou


def evaluate_model(model, data_loader, device, confidence_threshold=0.5, iou_threshold=0.5):
    """Evaluate model with proper object detection metrics"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    print("Running evaluation...")
    
    with torch.no_grad():
        for batch_idx, (frames_batch, targets) in enumerate(tqdm(data_loader)):
            frames_batch = frames_batch.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in target.items()} for target in targets]
            
            # Handle sequence dimension for non-temporal models
            if frames_batch.dim() == 5 and frames_batch.size(1) == 1:
                frames_batch = frames_batch.squeeze(1)
            
            # Forward pass (inference mode - no targets)
            outputs = model(frames_batch)
            
            # Process each sample in batch
            batch_size = frames_batch.shape[0]
            for i in range(batch_size):
                # Extract predictions for this sample
                pred_logits = outputs['pred_logits'][i]  # [num_queries, num_classes]
                pred_boxes = outputs['pred_boxes'][i]    # [num_queries, 4]
                
                # Apply sigmoid to get probabilities
                pred_probs = F.sigmoid(pred_logits)
                
                # Get confident predictions (for single class, take class 0)
                if pred_probs.shape[1] == 1:
                    confidence_scores = pred_probs[:, 0]
                else:
                    confidence_scores = pred_probs.max(dim=1)[0]
                
                confident_mask = confidence_scores > confidence_threshold
                
                pred_boxes_filtered = pred_boxes[confident_mask]
                pred_scores_filtered = confidence_scores[confident_mask]
                
                # Store predictions
                predictions = {
                    'boxes': pred_boxes_filtered.cpu(),
                    'scores': pred_scores_filtered.cpu(),
                    'image_id': targets[i].get('image_id', batch_idx * batch_size + i)
                }
                all_predictions.append(predictions)
                
                # Store ground truth (take first frame for temporal data)
                target = targets[i] if isinstance(targets[i], dict) else targets[i][0]
                gt_boxes = target.get('boxes', torch.tensor([]))
                gt_labels = target.get('labels', torch.tensor([]))
                
                ground_truth = {
                    'boxes': gt_boxes.cpu() if len(gt_boxes) > 0 else torch.tensor([]),
                    'labels': gt_labels.cpu() if len(gt_labels) > 0 else torch.tensor([]),
                    'image_id': target.get('image_id', batch_idx * batch_size + i)
                }
                all_targets.append(ground_truth)
    
    # Calculate metrics
    metrics = calculate_detection_metrics(all_predictions, all_targets, iou_threshold)
    return metrics


def calculate_detection_metrics(predictions, targets, iou_threshold=0.5):
    """Calculate precision, recall, F1, and AP for object detection"""
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    all_precisions = []
    all_recalls = []
    
    print(f"Calculating metrics for {len(predictions)} samples...")
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        gt_boxes = target['boxes']
        
        num_gt = len(gt_boxes)
        num_pred = len(pred_boxes)
        
        if num_pred == 0 and num_gt == 0:
            # True negative - no objects predicted or present
            continue
        elif num_pred == 0 and num_gt > 0:
            # False negatives - missed detections
            false_negatives += num_gt
            continue
        elif num_pred > 0 and num_gt == 0:
            # False positives - incorrect detections
            false_positives += num_pred
            continue
        
        # Calculate IoU matrix
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            iou_matrix = bbox_iou(pred_boxes, gt_boxes)
            
            # Find matches using greedy matching
            matched_gt = set()
            matched_pred = set()
            
            # Sort predictions by confidence
            sorted_indices = torch.argsort(pred_scores, descending=True)
            
            for pred_idx in sorted_indices:
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx in range(len(gt_boxes)):
                    if gt_idx in matched_gt:
                        continue
                    
                    iou = iou_matrix[pred_idx, gt_idx].item()
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_gt_idx >= 0:
                    # True positive
                    true_positives += 1
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(pred_idx.item())
                else:
                    # False positive
                    false_positives += 1
            
            # Unmatched ground truth are false negatives
            false_negatives += len(gt_boxes) - len(matched_gt)
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'total_predictions': sum(len(p['boxes']) for p in predictions),
        'total_targets': sum(len(t['boxes']) for t in targets)
    }
    
    return metrics


def main():
    print("RT-DETR Model Evaluation")
    print("=" * 40)
    
    # Configuration
    model_is_temporal = config.MODEL_IS_TEMPORAL
    actual_seq_len = config.BASE_TEMPORAL_SEQ_LEN if model_is_temporal else 1
    actual_motion_enabled = model_is_temporal
    
    print(f"Model type: {'Temporal' if model_is_temporal else 'Non-temporal'}")
    
    # Load best model checkpoint
    model_type_str = "temporal" if model_is_temporal else "non_temporal"
    checkpoint_path = f"output/uav_{model_type_str}_training_v2_train/best_{model_type_str}_model.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        output_dir = Path(f"output/uav_{model_type_str}_training_v2_train/")
        if output_dir.exists():
            for f in output_dir.glob("*.pth"):
                print(f"  {f}")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
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
        model = RTDETR(backbone=cfg.model.backbone, encoder=cfg.model.encoder, decoder=cfg.model.decoder)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint val_loss: {checkpoint.get('val_loss', 'unknown')}")
    
    # Setup validation dataset
    val_dataset = UAVTemporalMotionDataset(
        img_folder=config.IMG_FOLDER,
        seq_file=config.VAL_SEQ_FILE,
        seq_len=actual_seq_len,
        motion_enabled=actual_motion_enabled
    )
    
    # Use smaller subset for quick evaluation
    val_subset_size = min(1000, len(val_dataset))
    val_subset = torch.utils.data.Subset(val_dataset, range(val_subset_size))
    
    val_loader = DataLoader(
        val_subset, batch_size=8, shuffle=False,
        collate_fn=uav_temporal_motion_collate_fn, num_workers=2,
        drop_last=False
    )
    
    print(f"Evaluating on {val_subset_size} validation samples")
    
    # Evaluate at different confidence thresholds
    confidence_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    print("\nEvaluation Results:")
    print("-" * 80)
    print(f"{'Conf Thresh':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'TP':<6} {'FP':<6} {'FN':<6}")
    print("-" * 80)
    
    best_f1 = 0
    best_threshold = 0.5
    
    for conf_thresh in confidence_thresholds:
        metrics = evaluate_model(model, val_loader, device, confidence_threshold=conf_thresh)
        
        print(f"{conf_thresh:<12.1f} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
              f"{metrics['f1_score']:<10.3f} {metrics['true_positives']:<6} "
              f"{metrics['false_positives']:<6} {metrics['false_negatives']:<6}")
        
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_threshold = conf_thresh
    
    print("-" * 80)
    print(f"Best F1-Score: {best_f1:.3f} at confidence threshold {best_threshold}")
    
    # Detailed analysis at best threshold
    print(f"\nDetailed analysis at confidence threshold {best_threshold}:")
    best_metrics = evaluate_model(model, val_loader, device, confidence_threshold=best_threshold)
    
    print(f"Total predictions: {best_metrics['total_predictions']}")
    print(f"Total ground truth objects: {best_metrics['total_targets']}")
    print(f"True Positives: {best_metrics['true_positives']}")
    print(f"False Positives: {best_metrics['false_positives']}")
    print(f"False Negatives: {best_metrics['false_negatives']}")
    
    if best_metrics['f1_score'] < 0.1:
        print("\n⚠️  WARNING: Very low F1-score detected!")
        print("This suggests:")
        print("1. Model is not learning properly")
        print("2. Confidence threshold may be too high")
        print("3. Model may be predicting background everywhere")
        print("4. Training data or labels may have issues")


if __name__ == "__main__":
    main()
