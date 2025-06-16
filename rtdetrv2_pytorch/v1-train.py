#!/usr/bin/env python3
"""
Simple Temporal Motion Training Script (no unicode issues)
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau # Added import
import time
from pathlib import Path
import json
import traceback
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core import YAMLConfig
from src.data.uav_temporal import UAVTemporalMotionDataset, uav_temporal_motion_collate_fn

class SimpleMonitor:
    """Simple monitor without unicode characters"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = [] # Added to track LR
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Initialize log file
        self.log_file = self.output_dir / "training_log.txt"
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("UAV Temporal Motion Training Log\n")
            f.write("=" * 50 + "\n")
    
    def log(self, message):
        """Log message to console and file"""
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    
    def update_epoch(self, epoch, train_loss, val_loss, current_lr): # Added current_lr
        """Update epoch metrics"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(current_lr) # Store LR
        
        # Check if this is the best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            return True
        return False

def train_epoch(model, criterion, optimizer, train_loader, device, monitor):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Wrap train_loader with tqdm for a progress bar
    progress_bar = tqdm(train_loader, desc=f"Train Epoch", leave=False)
    for batch_idx, batch in enumerate(progress_bar):
        frames_batch, targets = batch  # frames_batch: [B, T, C, H, W]
        
        # Move to device
        frames_batch = frames_batch.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in target.items()} for target in targets]
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(frames_batch, targets)
        
        # Compute loss
        loss_dict = criterion(outputs, targets)
        losses = sum(loss_dict[k] * v for k, v in loss_dict.items() if k != 'class_error')
        
        # Backward pass
        losses.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += losses.item()
        num_batches += 1
        
        # Update tqdm progress bar
        progress_bar.set_postfix(loss=losses.item())
        
        # Log progress (optional, as tqdm handles this)
        # if batch_idx % 100 == 0:
            # current_lr = optimizer.param_groups[0]['lr']
            # monitor.log(f"   Batch {batch_idx}/{len(train_loader)}: Loss = {losses.item():.4f}")
    
    return total_loss / num_batches if num_batches > 0 else 0

def validate_epoch(model, criterion, val_loader, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # Wrap val_loader with tqdm for a progress bar
    progress_bar = tqdm(val_loader, desc=f"Validate Epoch", leave=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            frames_batch, targets = batch
            
            # Move to device
            frames_batch = frames_batch.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in target.items()} for target in targets]
            
            # Forward pass
            outputs = model(frames_batch, targets)
            
            # Compute loss
            loss_dict = criterion(outputs, targets)
            losses = sum(loss_dict[k] * v for k, v in loss_dict.items() if k != 'class_error')
            
            total_loss += losses.item()
            num_batches += 1
            
            # Update tqdm progress bar
            progress_bar.set_postfix(loss=losses.item())
            
    return total_loss / num_batches if num_batches > 0 else 0

def main():
    """Main training function"""
    print("Starting UAV Temporal Motion Training")
    
    # ===== CONFIGURATION SECTION =====
    # Training Configuration
    num_epochs = 30
    batch_size = 16
    learning_rate = 1e-4
    weight_decay = 1e-4
    grad_clip_max_norm = 1.0
    early_stopping_patience = 5
    
    # Dataset Configuration
    seq_len = 5
    motion_enabled = True
    train_subset_percentage = 0.5  # Use % of training data, None for full
    val_subset_percentage = 0.3    # Use % of validation data, None for full
    
    # DataLoader Configuration
    num_workers = 4 # Changed from 0 to 4
    shuffle_train = True
    drop_last_train = True
    drop_last_val = False
    
    # Paths Configuration
    output_dir = "./output/uav_temporal_motion_training_v2"
    img_folder = "c:/data/processed_anti_uav_v2/images"
    train_seq_file = "c:/data/processed_anti_uav_v2/sequences/train.txt"
    val_seq_file = "c:/data/processed_anti_uav_v2/sequences/val.txt"
    config_file = './configs/rtdetr/rtdetr_dla34_6x_uav_temporal_fixed.yml'
    
    # Logging Configuration
    log_batch_interval = 100  # Log every N batches
    
    # Model Configuration
    use_motion = True
    motion_in_channels = 3
    
    # Resume Configuration
    resume_checkpoint_path = "output/uav_temporal_motion_training/epoch_18_temporal_model.pth" # None # Path to .pth file to resume from, e.g., "./output/uav_temporal_motion_training/epoch_5_temporal_model.pth"
    # ===== END CONFIGURATION SECTION =====
      # Initialize monitoring
    monitor = SimpleMonitor(output_dir)
    monitor.log("Starting UAV Temporal Motion Training")
    monitor.log("=" * 50)
    monitor.log("CONFIGURATION:")
    monitor.log(f"  Epochs: {num_epochs}")
    monitor.log(f"  Batch size: {batch_size}")
    monitor.log(f"  Learning rate: {learning_rate}")
    monitor.log(f"  Weight decay: {weight_decay}")
    monitor.log(f"  Gradient clipping: {grad_clip_max_norm}")
    monitor.log(f"  Early stopping patience: {early_stopping_patience}")
    monitor.log(f"  LR Scheduler: ReduceLROnPlateau (factor=0.1, patience=3)") # Added LR scheduler info
    monitor.log(f"  Sequence length: {seq_len}")
    monitor.log(f"  Motion enabled: {motion_enabled}")
    monitor.log(f"  Train subset percentage: {train_subset_percentage*100 if train_subset_percentage is not None else 'Full'}%")
    monitor.log(f"  Val subset percentage: {val_subset_percentage*100 if val_subset_percentage is not None else 'Full'}%")
    monitor.log(f"  Number of workers: {num_workers}")
    monitor.log(f"  Resume from checkpoint: {resume_checkpoint_path if resume_checkpoint_path else 'Not resuming'}")
    
    # Calculate and log validation interval
    validation_epoch_interval = max(1, int(num_epochs * 0.20))
    monitor.log(f"  Validation interval: Every {validation_epoch_interval} epochs")
    monitor.log("=" * 50)
    
    try:
        # Manual setup since config has issues
        monitor.log("Setting up temporal model manually...")
          # Create datasets
        train_dataset = UAVTemporalMotionDataset(
            img_folder=img_folder,
            seq_file=train_seq_file,
            seq_len=seq_len,
            motion_enabled=motion_enabled
        )
        
        val_dataset = UAVTemporalMotionDataset(
            img_folder=img_folder,
            seq_file=val_seq_file,
            seq_len=seq_len,
            motion_enabled=motion_enabled
        )
        
        monitor.log(f"Training samples: {len(train_dataset)}")
        monitor.log(f"Validation samples: {len(val_dataset)}")
        
        # Use subset for faster testing (if specified)
        if train_subset_percentage is not None:
            train_subset_size = int(len(train_dataset) * train_subset_percentage)
            train_subset = Subset(train_dataset, range(min(train_subset_size, len(train_dataset))))
            monitor.log(f"Using training subset: {len(train_subset)} samples ({train_subset_percentage*100}%)")
        else:
            train_subset = train_dataset
            monitor.log("Using full training dataset")
            
        if val_subset_percentage is not None:
            val_subset_size = int(len(val_dataset) * val_subset_percentage)
            val_subset = Subset(val_dataset, range(min(val_subset_size, len(val_dataset))))
            monitor.log(f"Using validation subset: {len(val_subset)} samples ({val_subset_percentage*100}%)")
        else:
            val_subset = val_dataset
            monitor.log("Using full validation dataset")
          # Create dataloaders
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            collate_fn=uav_temporal_motion_collate_fn,
            num_workers=num_workers,
            drop_last=drop_last_train,
            pin_memory=True # Added for potentially faster data transfer
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=uav_temporal_motion_collate_fn,
            num_workers=num_workers,
            drop_last=drop_last_val,
            pin_memory=True # Added for potentially faster data transfer
        )
        
        monitor.log(f"Training batches: {len(train_loader)}")
        monitor.log(f"Validation batches: {len(val_loader)}")
        
        # Load model from working config
        from src.zoo.rtdetr.temporal_rtdetr_fixed import MotionStrengthModule, TemporalRTDETR
          # Load base config
        base_cfg = YAMLConfig(config_file)
        backbone = base_cfg.model.backbone
        encoder = base_cfg.model.encoder
        decoder = base_cfg.model.decoder
        criterion = base_cfg.criterion
        
        # Create motion module
        motion_module = MotionStrengthModule(t_window=seq_len, in_channels=motion_in_channels)
        
        # Create temporal model
        model = TemporalRTDETR(
            backbone=backbone,
            encoder=encoder,
            decoder=decoder,
            motion_module=motion_module,
            use_motion=use_motion
        )
        
        monitor.log("Created temporal model successfully")
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning Rate Scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        
        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        monitor.log(f"Using device: {device}")

        # Load checkpoint if resuming
        start_epoch = 0
        if resume_checkpoint_path and Path(resume_checkpoint_path).exists():
            monitor.log(f"Resuming training from checkpoint: {resume_checkpoint_path}")
            checkpoint = torch.load(resume_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1 # Start from the next epoch
            
            # Restore monitor state if available in checkpoint (optional, but good for consistency)
            if 'val_loss' in checkpoint: # Assuming 'val_loss' was the metric for best_val_loss
                monitor.best_val_loss = checkpoint['val_loss'] 
            if 'best_val_loss' in checkpoint: # More explicit check
                 monitor.best_val_loss = checkpoint['best_val_loss']
            if 'best_epoch' in checkpoint:
                 monitor.best_epoch = checkpoint['best_epoch']
            else: # Fallback if only epoch and val_loss are there from older checkpoints
                 monitor.best_epoch = checkpoint['epoch']


            monitor.log(f"Resumed from epoch {checkpoint['epoch']}. Next epoch: {start_epoch}.")
            monitor.log(f"Resumed best_val_loss: {monitor.best_val_loss:.4f} at epoch {monitor.best_epoch +1 }")
        elif resume_checkpoint_path:
            monitor.log(f"WARNING: resume_checkpoint_path specified but file not found: {resume_checkpoint_path}. Starting from scratch.")

        # Test single forward pass
        monitor.log("Testing model forward pass...")
        model.eval()
        with torch.no_grad():
            for batch in train_loader:
                frames_batch, targets = batch
                frames_batch = frames_batch.to(device)
                outputs = model(frames_batch)
                monitor.log(f"Forward pass successful: {outputs['pred_logits'].shape}")
                break
        
        # Training loop
        monitor.log("Starting training loop...")
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            monitor.log(f"\\nEpoch {epoch+1}/{num_epochs}")
            monitor.log("-" * 40)
            
            # Training phase
            train_loss = train_epoch(model, criterion, optimizer, train_loader, device, monitor)
            
            # Validation phase (every N epochs)
            if (epoch + 1) % validation_epoch_interval == 0 or epoch == num_epochs - 1:
                val_loss = validate_epoch(model, criterion, val_loader, device)
            else:
                val_loss = monitor.val_losses[-1] if monitor.val_losses else float('inf') # Use last val_loss if not validating

            # Update monitoring
            current_lr = optimizer.param_groups[0]['lr']
            is_best = monitor.update_epoch(epoch, train_loss, val_loss, current_lr)
            
            # Step the scheduler
            if (epoch + 1) % validation_epoch_interval == 0 or epoch == num_epochs - 1:
                 scheduler.step(val_loss) # Step scheduler with validation loss
            
            # Time tracking
            epoch_time = time.time() - epoch_start_time
            
            # Log epoch results
            monitor.log(f"Epoch {epoch+1} Results:")
            monitor.log(f"  Train Loss: {train_loss:.4f}")
            monitor.log(f"  Val Loss: {val_loss:.4f}")
            monitor.log(f"  Current LR: {current_lr:.1e}") # Log current LR
            monitor.log(f"  Epoch Time: {epoch_time:.1f}s")
            monitor.log(f"  Best Model: {'Yes' if is_best else 'No'}")
            
            # Save checkpoint if best or after validation
            if is_best or (epoch + 1) % validation_epoch_interval == 0 or epoch == num_epochs - 1:
                checkpoint_path = Path(output_dir) / f"epoch_{epoch+1}_temporal_model.pth"
                onnx_path = Path(output_dir) / f"epoch_{epoch+1}_temporal_model.onnx"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': monitor.best_val_loss, # Save best_val_loss for resuming
                    'best_epoch': monitor.best_epoch, # Save best_epoch for resuming
                    'model_type': 'temporal_motion'
                }, checkpoint_path)
                monitor.log(f"  Checkpoint saved: {checkpoint_path}")
                
                # Export to ONNX
                try:
                    # Adjust dummy_input size to match expected model input (e.g., 512x640)
                    dummy_input = torch.randn(1, seq_len, 3, 512, 640).to(device) # H=512, W=640
                    torch.onnx.export(model, dummy_input, onnx_path, 
                                      input_names=['input'], output_names=['output'], 
                                      dynamic_axes={'input': {0: 'batch_size', 1: 'sequence_length'},
                                                    'output': {0: 'batch_size'}})
                    monitor.log(f"  ONNX model saved: {onnx_path}")
                except Exception as e:
                    monitor.log(f"  Error exporting ONNX model: {e}")
                    traceback.print_exc()

            # Early stopping
            if epoch - monitor.best_epoch > 5:
                monitor.log(f"Early stopping triggered. Best epoch: {monitor.best_epoch+1}")
                break
        
        # Final results
        monitor.log("\nTraining completed!")
        monitor.log(f"Best Epoch: {monitor.best_epoch+1}")
        monitor.log(f"Best Validation Loss: {monitor.best_val_loss:.4f}")
        
        return True
        
    except Exception as e:
        monitor.log(f"Error in training: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("SUCCESS! Temporal motion training completed!")
    else:
        print("Training failed. Check the logs for details.")
