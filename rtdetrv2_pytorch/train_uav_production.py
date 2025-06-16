#!/usr/bin/env python3
"""
🚁 UAV Temporal RT-DETR Production Training
Full training script with the fixed configuration and comprehensive monitoring
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from pathlib import Path
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core import YAMLConfig
from src.data.uav_temporal import UAVTemporalDataset, uav_single_frame_collate_fn

class TrainingMonitor:
    """Monitor and log training progress"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Initialize log file
        self.log_file = self.output_dir / "training_log.txt"
        with open(self.log_file, 'w') as f:
            f.write("UAV Temporal RT-DETR Training Log\n")
            f.write("=" * 50 + "\n")
    
    def log(self, message):
        """Log message to console and file"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    
    def update_epoch(self, epoch, train_loss, val_loss, lr):
        """Update epoch metrics"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(lr)
        
        # Check if this is the best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            return True  # Indicates best model
        return False
    
    def save_metrics(self):
        """Save training metrics to JSON"""
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }
        
        with open(self.output_dir / "training_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

def create_datasets(config):
    """Create training and validation datasets"""
    train_dataset = UAVTemporalDataset(
        img_folder="c:/data/processed_anti_uav_v2/images",
        seq_file="c:/data/processed_anti_uav_v2/sequences/train.txt",
        seq_len=5
    )
    
    val_dataset = UAVTemporalDataset(
        img_folder="c:/data/processed_anti_uav_v2/images",
        seq_file="c:/data/processed_anti_uav_v2/sequences/val.txt",
        seq_len=5
    )
    
    return train_dataset, val_dataset

def create_dataloaders(train_dataset, val_dataset, batch_size=4, num_workers=0):
    """Create training and validation dataloaders"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=uav_single_frame_collate_fn,
        num_workers=num_workers,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=uav_single_frame_collate_fn,
        num_workers=num_workers,
        drop_last=False
    )
    
    return train_loader, val_loader

def train_epoch(model, criterion, optimizer, train_loader, device, monitor):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        images, targets = batch
        images = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in target.items()} for target in targets]
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images, targets)
        
        # Compute loss
        loss_dict = criterion(outputs, targets)
        losses = sum(loss_dict[k] * v for k, v in loss_dict.items() if k != 'class_error')
        
        # Backward pass
        losses.backward()
        
        # Gradient clipping (optional, but helps stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += losses.item()
        num_batches += 1
        
        # Log progress every 100 batches
        if batch_idx % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            monitor.log(f"   Batch {batch_idx}/{len(train_loader)}: Loss = {losses.item():.4f}, LR = {current_lr:.6f}")
    
    return total_loss / num_batches if num_batches > 0 else 0

def validate_epoch(model, criterion, val_loader, device, monitor):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images, targets = batch
            images = images.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in target.items()} for target in targets]
            
            # Forward pass
            outputs = model(images, targets)
            
            # Compute loss
            loss_dict = criterion(outputs, targets)
            losses = sum(loss_dict[k] * v for k, v in loss_dict.items() if k != 'class_error')
            
            total_loss += losses.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, config, checkpoint_path, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config.yaml_cfg,
        'spatial_size': [512, 640]
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = checkpoint_path.parent / "best_model.pth"
        torch.save(checkpoint, best_path)

def run_production_training():
    """Run full production training"""
    print("🚁" * 20)
    print("UAV TEMPORAL RT-DETR PRODUCTION TRAINING")
    print("🚁" * 20)
    
    # Configuration
    config_path = './configs/rtdetr/rtdetr_dla34_6x_uav_temporal_fixed.yml'
    output_dir = "./output/uav_production_training"
    
    # Training hyperparameters
    num_epochs = 50
    batch_size = 4  # Adjust based on GPU memory
    learning_rate = 1e-4
    weight_decay = 1e-4
    
    # Initialize monitoring
    monitor = TrainingMonitor(output_dir)
    monitor.log(f"Starting UAV Temporal RT-DETR Production Training")
    monitor.log(f"Configuration: {config_path}")
    monitor.log(f"Output directory: {output_dir}")
    monitor.log(f"Epochs: {num_epochs}, Batch size: {batch_size}")
    monitor.log(f"Learning rate: {learning_rate}, Weight decay: {weight_decay}")
    
    try:
        # Load configuration
        cfg = YAMLConfig(config_path)
        model = cfg.model
        criterion = cfg.criterion
        
        # Create custom optimizer with better parameters
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        monitor.log(f"Using device: {device}")
        
        # Create datasets and dataloaders
        monitor.log("Creating datasets...")
        train_dataset, val_dataset = create_datasets(cfg)
        train_loader, val_loader = create_dataloaders(
            train_dataset, val_dataset, batch_size=batch_size
        )
        
        monitor.log(f"Training samples: {len(train_dataset)}")
        monitor.log(f"Validation samples: {len(val_dataset)}")
        monitor.log(f"Training batches: {len(train_loader)}")
        monitor.log(f"Validation batches: {len(val_loader)}")
        
        # Training loop
        monitor.log("Starting training loop...")
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            monitor.log(f"\nEpoch {epoch+1}/{num_epochs}")
            monitor.log("-" * 50)
            
            # Training phase
            train_loss = train_epoch(model, criterion, optimizer, train_loader, device, monitor)
            
            # Validation phase
            val_loss = validate_epoch(model, criterion, val_loader, device, monitor)
            
            # Update learning rate
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update monitoring
            is_best = monitor.update_epoch(epoch, train_loss, val_loss, current_lr)
            
            # Time tracking
            epoch_time = time.time() - epoch_start_time
            
            # Log epoch results
            monitor.log(f"Epoch {epoch+1} Results:")
            monitor.log(f"  Train Loss: {train_loss:.4f}")
            monitor.log(f"  Val Loss: {val_loss:.4f}")
            monitor.log(f"  Learning Rate: {current_lr:.6f}")
            monitor.log(f"  Epoch Time: {epoch_time:.1f}s")
            monitor.log(f"  Best Model: {'✅' if is_best else '❌'}")
            
            # Save checkpoints
            checkpoint_path = Path(output_dir) / f"checkpoint_epoch_{epoch+1}.pth"
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, cfg, checkpoint_path, is_best)
            
            if is_best:
                monitor.log(f"  🎉 New best model saved! Val Loss: {val_loss:.4f}")
            
            # Save metrics
            monitor.save_metrics()
            
            # Early stopping check (optional)
            if epoch - monitor.best_epoch > 10:  # Stop if no improvement for 10 epochs
                monitor.log(f"Early stopping triggered. Best epoch: {monitor.best_epoch+1}")
                break
        
        # Final results
        monitor.log("\n" + "🎯" * 20)
        monitor.log("TRAINING COMPLETED!")
        monitor.log("🎯" * 20)
        monitor.log(f"Best Epoch: {monitor.best_epoch+1}")
        monitor.log(f"Best Validation Loss: {monitor.best_val_loss:.4f}")
        monitor.log(f"Final Training Loss: {monitor.train_losses[-1]:.4f}")
        monitor.log(f"Total Epochs: {len(monitor.train_losses)}")
        
        # Test final inference
        monitor.log("\n🔍 Testing final model inference...")
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                images, targets = batch
                images = images.to(device)
                
                outputs = model(images)
                pred_logits = outputs['pred_logits']
                confidence_scores = torch.sigmoid(pred_logits)
                max_confidence = confidence_scores.max().item()
                
                monitor.log(f"✅ Final inference successful!")
                monitor.log(f"✅ Max confidence: {max_confidence:.4f}")
                break
        
        monitor.log(f"\n🎉 Production training completed successfully!")
        monitor.log(f"📁 All outputs saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        monitor.log(f"❌ Error in production training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_production_training()
    
    if success:
        print(f"""
🎉 SUCCESS! UAV Temporal RT-DETR training completed!

📊 Training Summary:
   ✅ Full dataset training completed
   ✅ Model checkpoints saved
   ✅ Training metrics logged
   ✅ Best model identified
   ✅ Inference verified

📁 Output Location:
   ./output/uav_production_training/

📈 Next Steps:
   1. Analyze training_metrics.json for performance insights
   2. Test best_model.pth on validation data
   3. Run inference on new UAV sequences
   4. Consider hyperparameter tuning for better performance
   5. Implement temporal extensions for future work

🎯 Model is ready for UAV detection tasks!
        """)
    else:
        print("❌ Training failed. Check the logs for details.")
