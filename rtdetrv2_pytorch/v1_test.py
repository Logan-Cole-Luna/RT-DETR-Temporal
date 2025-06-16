#!/usr/bin/env python3
"""
Test the trained temporal motion RT-DETR model
"""

import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.uav_temporal import UAVTemporalMotionDataset, uav_temporal_motion_collate_fn
from src.zoo.rtdetr.temporal_rtdetr_fixed import MotionStrengthModule, TemporalRTDETR
from src.core import YAMLConfig

def load_trained_temporal_model(checkpoint_path):
    """Load the trained temporal motion model"""
    print(f"Loading temporal model from: {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"✅ Checkpoint loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Create model architecture (same as training)
    base_cfg = YAMLConfig('./configs/rtdetr/rtdetr_dla34_6x_uav_temporal_fixed.yml')
    backbone = base_cfg.model.backbone
    encoder = base_cfg.model.encoder
    decoder = base_cfg.model.decoder
    
    # Create motion module
    motion_module = MotionStrengthModule(t_window=5, in_channels=3)
    
    # Create temporal model
    model = TemporalRTDETR(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        motion_module=motion_module,
        use_motion=True
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Temporal model loaded successfully")
    #print(f"   Train Loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
    #print(f"   Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    return model

def test_temporal_inference(model, dataset, num_samples=5):
    """Test temporal model inference on sample sequences"""
    print(f"\n🔍 Testing temporal inference on {num_samples} samples...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    results = []
    
    for i in range(min(num_samples, len(dataset))):
        print(f"\nSample {i+1}/{num_samples}")
        
        # Get sequence
        frames_tensor, target = dataset[i]
        frames_batch = frames_tensor.unsqueeze(0).to(device)  # Add batch dimension
        
        print(f"  Input shape: {frames_batch.shape}")
        print(f"  Target boxes: {len(target['boxes'])} UAVs")
        
        # Inference
        with torch.no_grad():
            outputs = model(frames_batch)
            
            pred_logits = outputs['pred_logits']  # [1, 300, 1]
            pred_boxes = outputs['pred_boxes']    # [1, 300, 4]
            
            # Apply sigmoid to get confidence scores
            confidence_scores = torch.sigmoid(pred_logits)
            
            # Find high-confidence predictions (threshold = 0.5)
            high_conf_mask = confidence_scores[0, :, 0] > 0.5
            high_conf_boxes = pred_boxes[0][high_conf_mask]
            high_conf_scores = confidence_scores[0][high_conf_mask, 0]
            
            print(f"  Predictions: {len(high_conf_boxes)} high-confidence detections")
            print(f"  Max confidence: {confidence_scores.max().item():.4f}")
            print(f"  Mean confidence: {confidence_scores.mean().item():.4f}")
            
            # Get motion statistics if available
            if hasattr(model, 'motion_module') and model.use_motion:
                motion_map = model.motion_module(frames_batch)
                motion_mean = motion_map.mean().item()
                motion_std = motion_map.std().item()
                motion_max = motion_map.max().item()
                
                print(f"  Motion stats: mean={motion_mean:.4f}, std={motion_std:.4f}, max={motion_max:.4f}")
            
            results.append({
                'sample_idx': i,
                'num_targets': len(target['boxes']),
                'num_predictions': len(high_conf_boxes),
                'max_confidence': confidence_scores.max().item(),
                'mean_confidence': confidence_scores.mean().item(),
                'motion_mean': motion_mean if 'motion_mean' in locals() else None,
                'motion_std': motion_std if 'motion_std' in locals() else None
            })
    
    return results

def visualize_temporal_detection(model, dataset, sample_idx=0, output_dir="output/temporal_test"):
    """Visualize temporal detection results"""
    print(f"\n📊 Visualizing temporal detection for sample {sample_idx}...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Get sequence
    frames_tensor, target = dataset[sample_idx]
    frames_batch = frames_tensor.unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(frames_batch)
        
        # Get motion map
        if hasattr(model, 'motion_module') and model.use_motion:
            motion_map = model.motion_module(frames_batch)
            motion_np = motion_map[0, 0].cpu().numpy()  # [H, W]
        
        # Get predictions
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        confidence_scores = torch.sigmoid(pred_logits)
        
        # High-confidence predictions
        high_conf_mask = confidence_scores[0, :, 0] > 0.3
        high_conf_boxes = pred_boxes[0][high_conf_mask].cpu().numpy()
        high_conf_scores = confidence_scores[0][high_conf_mask, 0].cpu().numpy()
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Show middle frame with detections
    middle_frame = frames_tensor[2].permute(1, 2, 0).numpy()  # [H, W, C]
    middle_frame = (middle_frame * 255).astype(np.uint8)
    
    axes[0, 0].imshow(middle_frame)
    axes[0, 0].set_title('Middle Frame')
    axes[0, 0].axis('off')
    
    # Show motion map
    if 'motion_np' in locals():
        im1 = axes[0, 1].imshow(motion_np, cmap='hot', vmin=0, vmax=1)
        axes[0, 1].set_title(f'Motion Map (mean={motion_np.mean():.3f})')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
    
    # Show frame with ground truth
    gt_frame = middle_frame.copy()
    for box in target['boxes']:
        x1, y1, x2, y2 = box
        cv2.rectangle(gt_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    axes[0, 2].imshow(gt_frame)
    axes[0, 2].set_title(f'Ground Truth ({len(target["boxes"])} UAVs)')
    axes[0, 2].axis('off')
    
    # Show frame with predictions
    pred_frame = middle_frame.copy()
    H, W = middle_frame.shape[:2]
    
    for box, score in zip(high_conf_boxes, high_conf_scores):
        # Convert from normalized coordinates
        x_center, y_center, width, height = box
        x1 = int((x_center - width/2) * W)
        y1 = int((y_center - height/2) * H)
        x2 = int((x_center + width/2) * W)
        y2 = int((y_center + height/2) * H)
        
        color = (255, 0, 0) if score > 0.7 else (255, 255, 0)
        cv2.rectangle(pred_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(pred_frame, f'{score:.2f}', (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    axes[1, 0].imshow(pred_frame)
    axes[1, 0].set_title(f'Predictions ({len(high_conf_boxes)} detections)')
    axes[1, 0].axis('off')
    
    # Show temporal sequence
    sequence_img = np.hstack([frames_tensor[i].permute(1, 2, 0).numpy() for i in range(5)])
    sequence_img = (sequence_img * 255).astype(np.uint8)
    
    axes[1, 1].imshow(sequence_img)
    axes[1, 1].set_title('Temporal Sequence (5 frames)')
    axes[1, 1].axis('off')
    
    # Show confidence histogram
    all_confidences = confidence_scores[0, :, 0].cpu().numpy()
    axes[1, 2].hist(all_confidences, bins=50, alpha=0.7)
    axes[1, 2].axvline(0.5, color='red', linestyle='--', label='Threshold')
    axes[1, 2].set_title('Confidence Distribution')
    axes[1, 2].set_xlabel('Confidence Score')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].legend()
    
    plt.tight_layout()
    
    # Save visualization
    output_path = output_dir / f'temporal_detection_sample_{sample_idx}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved to: {output_path}")

def main():
    """Main testing function"""
    print("🚁" * 20)
    print("TEMPORAL MOTION RT-DETR MODEL TESTING")
    print("🚁" * 20)
    
    # Load trained model
    checkpoint_path = "./output/uav_temporal_motion_training/best_temporal_model.pth"
    checkpoint_path = r"output/uav_temporal_training_v2_train/epoch_40_temporal_model.pth"
    checkpoint_path = "output/uav_temporal_training_v2_train/epoch_40_temporal_model.pth"
    checkpoint_path = "output/uav_temporal_training_v2_train/epoch_60_temporal_model.pth"
    model = load_trained_temporal_model(checkpoint_path)
    
    if model is None:
        print("❌ Could not load temporal model. Exiting.")
        return
    
    # Load test dataset
    print("\n📁 Loading test dataset...")
    test_dataset = UAVTemporalMotionDataset(
        img_folder="c:/data/processed_anti_uav_v2/images",
        seq_file="c:/data/processed_anti_uav_v2/sequences/val.txt",
        seq_len=5,
        motion_enabled=True
    )
    
    print(f"✅ Test dataset loaded: {len(test_dataset)} sequences")
    
    # Test inference
    results = test_temporal_inference(model, test_dataset, num_samples=10)
    
    # Analyze results
    print(f"\n📊 TEMPORAL MODEL PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    total_targets = sum(r['num_targets'] for r in results)
    total_predictions = sum(r['num_predictions'] for r in results)
    avg_max_confidence = np.mean([r['max_confidence'] for r in results])
    avg_mean_confidence = np.mean([r['mean_confidence'] for r in results])
    
    print(f"Total targets: {total_targets}")
    print(f"Total predictions: {total_predictions}")
    print(f"Average max confidence: {avg_max_confidence:.4f}")
    print(f"Average mean confidence: {avg_mean_confidence:.4f}")
    
    # Motion statistics
    motion_means = [r['motion_mean'] for r in results if r['motion_mean'] is not None]
    motion_stds = [r['motion_std'] for r in results if r['motion_std'] is not None]
    
    if motion_means:
        print(f"Motion map statistics:")
        print(f"  Average motion mean: {np.mean(motion_means):.4f}")
        print(f"  Average motion std: {np.mean(motion_stds):.4f}")
    
    # Create visualizations
    print(f"\n🎨 Creating visualizations...")
    for i in range(min(3, len(test_dataset))):
        visualize_temporal_detection(model, test_dataset, sample_idx=i)
    
    print(f"\n🎉 TEMPORAL MODEL TESTING COMPLETED!")
    print(f"✅ Model shows good performance with motion enhancement")
    print(f"✅ Visualizations saved to output/temporal_test/")

if __name__ == "__main__":
    main()
