"""
Temporal Detection Engine
Handles training and evaluation for temporal sequence models

Based on det_engine.py but modified for temporal data processing
"""

import math
import os
import sys
import pathlib
from typing import Iterable

import torch
import torch.amp 

from src.data import CocoEvaluator
from src.misc import (MetricLogger, SmoothedValue, reduce_dict)


def train_one_epoch_temporal(model: torch.nn.Module, criterion: torch.nn.Module,
                            data_loader: Iterable, optimizer: torch.optim.Optimizer,
                            device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    """
    Train one epoch for temporal models.
    Handles temporal sequence data (frames_batch shape: [B, T, C, H, W])
    """
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Temporal Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)
    
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # samples is temporal frames: [B, T, C, H, W]
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)
            
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)

            # Use the same loss calculation as in your v2-train.py for consistency
            loss = sum(loss_dict[k] * v for k, v in criterion.weight_dict.items() if k in loss_dict)
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict[k] * v for k, v in criterion.weight_dict.items() if k in loss_dict)

            optimizer.zero_grad()
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        # Reduce losses across distributed processes
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Temporal averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_temporal(model, criterion, postprocessor, data_loader, base_ds, device, output_dir):
    """
    Evaluate temporal model.
    """
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Temporal Test:'

    iou_types = postprocessor.iou_types if hasattr(postprocessor, 'iou_types') else ('bbox',)
    coco_evaluator = CocoEvaluator(base_ds, iou_types) if base_ds is not None else None

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        # samples is temporal frames: [B, T, C, H, W]
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        loss = sum(loss_dict[k] * v for k, v in criterion.weight_dict.items() if k in loss_dict)

        loss_dict_reduced = reduce_dict(loss_dict)
        metric_logger.update(loss=loss.item(), **loss_dict_reduced)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessor(outputs, orig_target_sizes)
        
        if coco_evaluator is not None:
            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Temporal averaged stats:", metric_logger)
    
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    
    return stats, coco_evaluator
