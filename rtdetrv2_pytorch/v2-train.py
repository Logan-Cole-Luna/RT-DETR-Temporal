#!/usr/bin/env python3
"""
Train script with option for Temporal Motion or Non-Temporal Base Model
Based on v1-train_simple_temporal.py
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from pathlib import Path
import json
import traceback
from tqdm import tqdm

import config  # Import the new configuration file

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_path = os.path.abspath(script_dir)

if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

from src.core import YAMLConfig
from src.data.uav_temporal import UAVTemporalMotionDataset, uav_temporal_motion_collate_fn
from src.zoo.rtdetr.temporal_rtdetr_fixed import MotionStrengthModule, TemporalRTDETR
from src.zoo.rtdetr.rtdetr import RTDETR


class SimpleMonitor:
    """Simple monitor without unicode characters"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = [] 
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        self.log_file = self.output_dir / "training_log.txt"
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("UAV Training Log (Temporal/Non-Temporal)\n")
            f.write("=" * 50 + "\n")
    
    def log(self, message):
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    
    def update_epoch(self, epoch, train_loss, val_loss, current_lr):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(current_lr)
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            return True
        return False


def train_epoch(model, criterion, optimizer, train_loader, device, monitor, model_is_temporal, grad_clip_max_norm):
    model.train()
    total_loss = 0
    num_batches = 0
    total_class_error = 0.0  # sum of classification error percentages
     
    # Use tqdm for progress bar, updating less frequently
    progress_bar = tqdm(train_loader, desc=f"Train Epoch", unit="batch", leave=False)
    for batch_idx, batch in enumerate(progress_bar):
        frames_batch, targets = batch
        
        frames_batch = frames_batch.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in target.items()} for target in targets]
        
        if not model_is_temporal:
            # For non-temporal, data loader might give [B, 1, C, H, W] if seq_len=1
            if frames_batch.dim() == 5 and frames_batch.size(1) == 1:
                frames_batch = frames_batch.squeeze(1)

        optimizer.zero_grad()
        outputs = model(frames_batch, targets)
        
        loss_dict = criterion(outputs, targets)
        # record classification error
        class_error = loss_dict.get('class_error', None)
        if class_error is not None:
            total_class_error += class_error.item()
        # Use the same loss calculation as v1
        losses = sum(loss_dict[k] * v for k, v in loss_dict.items() if k != 'class_error')

        losses.backward()
        if grad_clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
        optimizer.step()
        
        total_loss += losses.item()
        num_batches += 1
        # update progress bar with both loss and f1 for this batch
        if class_error is not None:
            batch_f1 = 100.0 - class_error.item()
            progress_bar.set_postfix(loss=f"{losses.item():.4f}", f1=f"{batch_f1:.2f}")
        else:
            progress_bar.set_postfix(loss=f"{losses.item():.4f}")
     
    # compute average metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_class_error = total_class_error / num_batches if num_batches > 0 else 0
    avg_f1 = 100.0 - avg_class_error
    return avg_loss, avg_f1


def validate_epoch(model, criterion, val_loader, device, model_is_temporal):
    model.eval()
    total_loss = 0
    num_batches = 0
    total_class_error = 0.0
    
    progress_bar = tqdm(val_loader, desc=f"Validate Epoch", leave=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            frames_batch, targets = batch
            
            frames_batch = frames_batch.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in target.items()} for target in targets]

            if not model_is_temporal:
                if frames_batch.dim() == 5 and frames_batch.size(1) == 1:
                    frames_batch = frames_batch.squeeze(1)
            
            outputs = model(frames_batch, targets)
            
            loss_dict = criterion(outputs, targets)
            # Use the same loss calculation as v1
            losses = sum(loss_dict[k] * v for k, v in loss_dict.items() if k != 'class_error')
            class_error = loss_dict.get('class_error', None)
            if class_error is not None:
                total_class_error += class_error.item()
            
            total_loss += losses.item()
            num_batches += 1
            if class_error is not None:
                batch_f1 = 100.0 - class_error.item()
                progress_bar.set_postfix(loss=f"{losses.item():.4f}", f1=f"{batch_f1:.2f}")
            else:
                progress_bar.set_postfix(loss=f"{losses.item():.4f}")
            
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_class_error = total_class_error / num_batches if num_batches > 0 else 0
    avg_f1 = 100.0 - avg_class_error
    return avg_loss, avg_f1


def main():
    print("Starting UAV Training (Temporal/Non-Temporal)")
    
    # ===== CONFIGURATION SECTION (from config.py) =====
    model_is_temporal = config.MODEL_IS_TEMPORAL

    # Training Configuration
    num_epochs = config.NUM_EPOCHS
    batch_size = config.BATCH_SIZE
    learning_rate = config.LEARNING_RATE
    weight_decay = config.WEIGHT_DECAY
    grad_clip_max_norm = config.GRAD_CLIP_MAX_NORM
    
    # Dataset Configuration
    IMG_HEIGHT = config.IMG_HEIGHT
    IMG_WIDTH = config.IMG_WIDTH

    # Temporal specific settings
    temporal_seq_len = config.BASE_TEMPORAL_SEQ_LEN
    temporal_motion_in_channels = config.MOTION_CALC_IN_CHANNELS

    # Dataset subset percentages
    train_subset_percentage = config.TRAIN_SUBSET_PERCENTAGE
    val_subset_percentage = config.VAL_SUBSET_PERCENTAGE
    
    # DataLoader Configuration
    num_workers = config.NUM_WORKERS
    shuffle_train = config.SHUFFLE_TRAIN
    drop_last_train = config.DROP_LAST_TRAIN
    drop_last_val = config.DROP_LAST_VAL
    
    # Paths Configuration
    img_folder = config.IMG_FOLDER
    train_seq_file = config.TRAIN_SEQ_FILE
    val_seq_file = config.VAL_SEQ_FILE
    
    # Conditional configuration based on model_is_temporal
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
        
    # Resume Configuration
    resume_checkpoint_path = config.RESUME_CHECKPOINT_PATH
    # ===== END CONFIGURATION SECTION =====
    
    monitor = SimpleMonitor(output_dir)
    monitor.log(f"Starting UAV {model_type_str.capitalize()} Training")
    monitor.log("=" * 50)
    monitor.log("CONFIGURATION:")
    monitor.log(f"  Model Type: {model_type_str.capitalize()}")
    monitor.log(f"  Epochs: {num_epochs}")
    monitor.log(f"  Batch size: {batch_size}")
    monitor.log(f"  Learning rate: {learning_rate}")
    monitor.log(f"  Image Dimensions: {IMG_HEIGHT}x{IMG_WIDTH}")
    if model_is_temporal:
        monitor.log(f"  Sequence length: {actual_seq_len}")
        monitor.log(f"  Motion enabled: {actual_motion_enabled}")
    monitor.log(f"  Train subset: {train_subset_percentage*100 if train_subset_percentage is not None else 'Full'}%")
    monitor.log(f"  Val subset: {val_subset_percentage*100 if val_subset_percentage is not None else 'Full'}%")
    monitor.log(f"  Config file: {config_file_path}")
    monitor.log(f"  Resume from: {resume_checkpoint_path if resume_checkpoint_path else 'Not resuming'}")
    
    validation_epoch_interval = max(1, int(num_epochs * 0.20)) 
    monitor.log(f"  Validation interval: Every {validation_epoch_interval} epochs")
    monitor.log("=" * 50)
    
    try:
        monitor.log(f"Setting up {model_type_str} model and datasets...")
        
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
        
        cfg = YAMLConfig(config_file_path, resume_from=None)
        # disable pretrained DLA for non-temporal
        if not model_is_temporal and 'DLANet' in cfg.yaml_cfg:
            cfg.yaml_cfg['DLANet']['pretrained'] = False
        # fix positional embedding size for non-temporal input
        if not model_is_temporal and 'HybridEncoder' in cfg.yaml_cfg:
            cfg.yaml_cfg['HybridEncoder']['eval_spatial_size'] = [IMG_HEIGHT, IMG_WIDTH]
        
        if model_is_temporal:
            monitor.log("Instantiating TemporalRTDETR model...")
            motion_module = MotionStrengthModule(t_window=actual_seq_len, in_channels=temporal_motion_in_channels)
            model = TemporalRTDETR(
                backbone=cfg.model.backbone,
                encoder=cfg.model.encoder,
                decoder=cfg.model.decoder,
                motion_module=motion_module,
                use_motion=actual_motion_enabled
            )
            criterion = cfg.criterion
            monitor.log("TemporalRTDETR model created.")
        else:
            monitor.log("Instantiating non-temporal RTDETR model...")
            model = RTDETR(backbone=cfg.model.backbone, encoder=cfg.model.encoder, decoder=cfg.model.decoder)
            # force decoder to always regenerate anchors to match current input size
            if hasattr(model, 'decoder') and hasattr(model.decoder, 'eval_spatial_size'):
                model.decoder.eval_spatial_size = None
            criterion = cfg.criterion
            monitor.log("Non-temporal RTDETR model created.")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        criterion = criterion.to(device)
        monitor.log(f"Using device: {device}")

        start_epoch = 0
        if resume_checkpoint_path and Path(resume_checkpoint_path).exists():
            monitor.log(f"Resuming training from checkpoint: {resume_checkpoint_path}")
            checkpoint = torch.load(resume_checkpoint_path, map_location=device)
            
            ckpt_model_type = checkpoint.get('model_type_str', model_type_str)
            if ckpt_model_type != model_type_str:
                monitor.log(f"WARNING: Checkpoint model type ('{ckpt_model_type}') differs from current script setting ('{model_type_str}'). Loading weights might fail or be partial.")

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            monitor.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            monitor.best_epoch = checkpoint.get('best_epoch', 0)
            if 'scheduler_state_dict' in checkpoint and hasattr(scheduler, 'load_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            monitor.log(f"Resumed from epoch {start_epoch -1}. Next epoch: {start_epoch}.")
            monitor.log(f"Resumed best_val_loss: {monitor.best_val_loss:.4f} at epoch {monitor.best_epoch +1 }")
        elif resume_checkpoint_path:
            monitor.log(f"WARNING: resume_checkpoint_path specified but file not found: {resume_checkpoint_path}. Starting from scratch.")

        monitor.log("Testing model forward pass (eval mode)...")
        model.eval()
        with torch.no_grad():
            for images_batch_test, _ in train_loader: 
                images_batch_test = images_batch_test.to(device)
                if not model_is_temporal:
                    if images_batch_test.dim() == 5 and images_batch_test.size(1) == 1:
                        images_batch_test = images_batch_test.squeeze(1)
                outputs_test = model(images_batch_test) 
                monitor.log(f"Forward pass successful. Output keys (example): {list(outputs_test.keys()) if isinstance(outputs_test, dict) else 'Output is not a dict'}")
                if isinstance(outputs_test, dict) and 'pred_logits' in outputs_test:
                    monitor.log(f"  pred_logits shape: {outputs_test['pred_logits'].shape}")
                break 
        
        monitor.log("Starting training loop...")
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            monitor.log(f"\nEpoch {epoch+1}/{num_epochs}")
            
            train_loss, train_f1 = train_epoch(model, criterion, optimizer, train_loader, device, monitor, model_is_temporal, grad_clip_max_norm)
            
            val_loss = float('inf')
            val_f1 = None
            if (epoch + 1) % validation_epoch_interval == 0 or epoch == num_epochs - 1:
                val_loss, val_f1 = validate_epoch(model, criterion, val_loader, device, model_is_temporal)
                scheduler.step(val_loss)
            else:
                val_loss = monitor.val_losses[-1] if monitor.val_losses else float('inf') 
            
            current_lr = optimizer.param_groups[0]['lr']
            is_best = monitor.update_epoch(epoch, train_loss, val_loss, current_lr)
            epoch_time = time.time() - epoch_start_time
            
            if val_f1 is not None:
                monitor.log(f"Epoch {epoch+1} Results: Train Loss={train_loss:.4f}, Train F1={train_f1:.2f}%, Val Loss={val_loss:.4f}, Val F1={val_f1:.2f}%, LR={current_lr:.1e}, Time={epoch_time:.1f}s")
            else:
                monitor.log(f"Epoch {epoch+1} Results: Train Loss={train_loss:.4f}, Train F1={train_f1:.2f}%, Val Loss={val_loss:.4f}, LR={current_lr:.1e}, Time={epoch_time:.1f}s")
            if is_best:
                monitor.log(f"  New best model found with Val Loss: {val_loss:.4f}")
            
            # Save checkpoint: if best, or on validation interval, or last epoch
            if is_best or (epoch + 1) % validation_epoch_interval == 0 or epoch == num_epochs - 1:
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': monitor.best_val_loss,
                    'best_epoch': monitor.best_epoch,
                    'config': cfg.yaml_cfg,
                    'model_type_str': model_type_str
                }
                base_name = f"epoch_{epoch+1}_{model_type_str}_model"
                pth_path = Path(output_dir) / f"{base_name}.pth"
                torch.save(checkpoint_data, str(pth_path))
                monitor.log(f"Saved checkpoint to {pth_path}")

                if is_best:
                     best_pth_path = Path(output_dir) / f"best_{model_type_str}_model.pth"
                     torch.save(checkpoint_data, str(best_pth_path))
                     monitor.log(f"Saved best model checkpoint to {best_pth_path}")

                # ONNX Export
                onnx_path = Path(output_dir) / f"{base_name}.onnx"
                try:
                    model.eval() 
                    if model_is_temporal:
                        dummy_input = torch.randn(1, actual_seq_len, 3, IMG_HEIGHT, IMG_WIDTH).to(device)
                        output_names_onnx = ['output'] 
                        dynamic_axes_onnx = {'input': {0: 'batch_size', 1: 'sequence_length'}, 'output': {0: 'batch_size'}}
                        input_names_onnx = ['input']

                        _test_output = model(dummy_input)
                        if isinstance(_test_output, dict) and 'pred_logits' in _test_output and 'pred_boxes' in _test_output:
                            output_names_onnx = ['pred_logits', 'pred_boxes']
                            dynamic_axes_onnx['pred_logits'] = {0: 'batch_size'}
                            dynamic_axes_onnx['pred_boxes'] = {0: 'batch_size'}
                        
                    else:
                        dummy_input = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).to(device)
                        input_names_onnx = ['images']
                        output_names_onnx = ['pred_logits', 'pred_boxes']
                        dynamic_axes_onnx = {
                            'images': {0: 'batch_size'},
                            'pred_logits': {0: 'batch_size'},
                            'pred_boxes': {0: 'batch_size'}
                        }

                        torch.onnx.export(model, dummy_input, onnx_path, verbose=False, input_names=input_names_onnx, output_names=output_names_onnx, dynamic_axes=dynamic_axes_onnx, opset_version=16) # Added opset_version
                        monitor.log(f"Model exported to {onnx_path}")

                except Exception as e_onnx:
                    monitor.log(f"Error during ONNX export for {base_name}.onnx: {e_onnx}")
                    # Optionally, print traceback for more details if needed
                    # import traceback
                    # traceback.print_exc()
        
        monitor.log("\nTraining completed!")
        monitor.log(f"Best Epoch: {monitor.best_epoch+1}")
        monitor.log(f"Best Validation Loss: {monitor.best_val_loss:.4f}")
        return True
        
    except Exception as e:
        monitor.log(f"Error in training: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if 'script_dir' not in globals():
        script_dir = os.path.dirname(os.path.abspath(__file__))

    success = main()
    if success:
        print(f"SUCCESS! Training completed!")
    else:
        print("Training failed. Check logs for details.")
