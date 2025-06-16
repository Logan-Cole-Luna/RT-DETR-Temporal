#!/usr/bin/env python3
"""
🚁 UAV RT-DETR Inference Script
Test trained model on UAV images with visualization
"""

import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core import YAMLConfig
from src.data.uav_temporal import UAVTemporalDataset, uav_single_frame_collate_fn

def load_trained_model(checkpoint_path, config_path):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load configuration
    cfg = YAMLConfig(config_path)
    model = cfg.model
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully on {device}")
    print(f"✅ Trained for {checkpoint.get('epoch', 'unknown')} epochs")
    print(f"✅ Best validation loss: {checkpoint.get('val_loss', 'unknown')}")
    
    return model, device

def preprocess_image(image_path):
    """Preprocess single image for inference"""
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor and normalize
    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    
    # Add batch dimension
    image = image.unsqueeze(0)  # [1, 3, H, W]
    
    return image

def postprocess_predictions(outputs, confidence_threshold=0.3):
    """Postprocess model outputs to get detections"""
    pred_logits = outputs['pred_logits']  # [1, num_queries, num_classes]
    pred_boxes = outputs['pred_boxes']    # [1, num_queries, 4]
    
    # Apply sigmoid to get confidence scores
    confidence_scores = torch.sigmoid(pred_logits)
    
    # Get predictions above threshold
    max_confidence = confidence_scores.max(dim=-1)[0]  # [1, num_queries]
    valid_predictions = max_confidence > confidence_threshold
    
    detections = []
    
    for i in range(pred_logits.shape[0]):  # Batch size (should be 1)
        valid_mask = valid_predictions[i]
        
        if valid_mask.sum() > 0:
            valid_boxes = pred_boxes[i][valid_mask]  # [num_valid, 4]
            valid_scores = max_confidence[i][valid_mask]  # [num_valid]
            
            # Convert boxes from [cx, cy, w, h] (normalized) to [x1, y1, x2, y2] (pixel)
            # Note: boxes are in normalized coordinates (0-1)
            
            for box, score in zip(valid_boxes, valid_scores):
                detections.append({
                    'box': box.cpu().numpy(),  # [cx, cy, w, h] normalized
                    'confidence': score.cpu().item()
                })
    
    return detections

def draw_detections(image, detections, output_path=None):
    """Draw detection boxes on image"""
    # Convert tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:  # Remove batch dimension
            image = image.squeeze(0)
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
    
    # Convert RGB back to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    h, w = image.shape[:2]
    
    # Draw each detection
    for detection in detections:
        box = detection['box']  # [cx, cy, w, h] normalized
        confidence = detection['confidence']
        
        # Convert normalized coordinates to pixel coordinates
        cx, cy, box_w, box_h = box
        cx, cy, box_w, box_h = cx * w, cy * h, box_w * w, box_h * h
        
        # Convert to [x1, y1, x2, y2]
        x1 = int(cx - box_w / 2)
        y1 = int(cy - box_h / 2)
        x2 = int(cx + box_w / 2)
        y2 = int(cy + box_h / 2)
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw confidence score
        label = f"UAV: {confidence:.3f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Save output image
    if output_path:
        cv2.imwrite(str(output_path), image)
        print(f"✅ Output saved to: {output_path}")
    
    return image

def run_inference_on_image(model, device, image_path, output_path=None, confidence_threshold=0.3):
    """Run inference on a single image"""
    print(f"\n🔍 Running inference on: {image_path}")
    
    try:
        # Preprocess image
        image_tensor = preprocess_image(image_path)
        image_tensor = image_tensor.to(device)
        
        print(f"   Image shape: {image_tensor.shape}")
        
        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)
        
        # Postprocess predictions
        detections = postprocess_predictions(outputs, confidence_threshold)
        
        print(f"   Found {len(detections)} detections above threshold {confidence_threshold}")
        
        # Print detection details
        for i, detection in enumerate(detections):
            print(f"   Detection {i+1}: confidence = {detection['confidence']:.3f}")
        
        # Visualize results
        if output_path or len(detections) > 0:
            result_image = draw_detections(image_tensor, detections, output_path)
        
        return detections
        
    except Exception as e:
        print(f"❌ Error in inference: {e}")
        import traceback
        traceback.print_exc()
        return []

def run_inference_on_dataset(model, device, confidence_threshold=0.3, num_samples=10):
    """Run inference on sample images from validation dataset"""
    print(f"\n🔍 Running inference on validation dataset...")
    
    try:
        # Create validation dataset
        val_dataset = UAVTemporalDataset(
            img_folder="c:/data/processed_anti_uav_v2/images",
            seq_file="c:/data/processed_anti_uav_v2/sequences/val.txt",
            seq_len=5
        )
        
        print(f"   Validation dataset: {len(val_dataset)} sequences")
        
        # Create output directory
        output_dir = Path("./output/inference_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        total_detections = 0
        
        # Run inference on first few samples
        for i in range(min(num_samples, len(val_dataset))):
            # Get sample
            sample = val_dataset[i]
            image_tensor, target = sample
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0).to(device)
            
            # Run inference
            with torch.no_grad():
                outputs = model(image_tensor)
            
            # Postprocess
            detections = postprocess_predictions(outputs, confidence_threshold)
            total_detections += len(detections)
            
            # Save visualization
            output_path = output_dir / f"inference_sample_{i:03d}.jpg"
            draw_detections(image_tensor, detections, output_path)
            
            print(f"   Sample {i+1}: {len(detections)} detections")
        
        print(f"\n✅ Inference completed!")
        print(f"   Total samples: {num_samples}")
        print(f"   Total detections: {total_detections}")
        print(f"   Average detections per image: {total_detections/num_samples:.2f}")
        print(f"   Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error in dataset inference: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='UAV RT-DETR Inference')
    parser.add_argument('--checkpoint', type=str, 
                       default='./output/uav_fixed_checkpoints/uav_fixed_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str,
                       default='./configs/rtdetr/rtdetr_dla34_6x_uav_temporal_fixed.yml',
                       help='Path to model configuration')
    parser.add_argument('--image', type=str, help='Path to input image for inference')
    parser.add_argument('--output', type=str, help='Path to save output image')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Confidence threshold for detections')
    parser.add_argument('--dataset', action='store_true',
                       help='Run inference on validation dataset samples')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of dataset samples to process')
    
    args = parser.parse_args()
    
    print("🚁" * 20)
    print("UAV RT-DETR INFERENCE")
    print("🚁" * 20)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Confidence threshold: {args.confidence}")
    
    # Load model
    model, device = load_trained_model(args.checkpoint, args.config)
    
    if args.image:
        # Single image inference
        detections = run_inference_on_image(
            model, device, args.image, args.output, args.confidence
        )
        
        if len(detections) > 0:
            print(f"\n🎯 Found {len(detections)} UAV detections!")
        else:
            print(f"\n❌ No UAVs detected above confidence threshold {args.confidence}")
    
    elif args.dataset:
        # Dataset inference
        run_inference_on_dataset(model, device, args.confidence, args.num_samples)
    
    else:
        print("\n❌ Please specify either --image or --dataset")
        print("Examples:")
        print("  python inference_uav.py --image path/to/image.jpg --output result.jpg")
        print("  python inference_uav.py --dataset --num_samples 20")

if __name__ == "__main__":
    main()
