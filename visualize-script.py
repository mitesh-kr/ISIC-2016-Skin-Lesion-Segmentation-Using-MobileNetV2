import os
import argparse
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from models.segmentation_model import SegmentationModel
from utils.data_loader import load_images

def visualize_predictions(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test data
    test_path = os.path.join(args.data_dir, "test")
    test_masks_path = os.path.join(args.data_dir, "test_masks")
    test_image_names, test_images, test_masks = load_images(test_path, test_masks_path)
    
    # Move test data to device
    test_images = test_images.to(device)
    test_masks = test_masks.to(device)
    
    # Load model
    model = SegmentationModel().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Generate predictions
    with torch.no_grad():
        pred_masks = model(test_images)
    
    # Select indices to visualize (either specific indices or random)
    if args.indices:
        indices = [int(idx) for idx in args.indices.split(',')]
    else:
        import numpy as np
        indices = list(np.random.choice(len(test_images), args.num_samples, replace=False))
    
    # Visualize each selected sample
    for i, idx in enumerate(indices):
        if idx >= len(test_images):
            print(f"Warning: Index {idx} out of range (max: {len(test_images)-1}). Skipping.")
            continue
            
        # Get image, ground truth mask, and prediction
        image = test_images[idx].cpu()
        true_mask = test_masks[idx].cpu()
        pred_mask = pred_masks[idx].cpu()
        pred_mask_thresholded = (pred_mask > args.threshold).float()
        
        # Create figure for visualization
        plt.figure(figsize=(15, 3))
        
        # Original image
        plt.subplot(1, 4, 1)
        plt.imshow(transforms.ToPILImage()(image))
        plt.title("Original Image")
        plt.axis('off')
        
        # Ground truth mask
        plt.subplot(1, 4, 2)
        plt.imshow(transforms.ToPILImage()(true_mask), cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis('off')
        
        # Predicted mask (raw)
        plt.subplot(1, 4, 3)
        plt.imshow(transforms.ToPILImage()(pred_mask), cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')
        
        # Predicted mask (thresholded)
        plt.subplot(1, 4, 4)
        plt.imshow(transforms.ToPILImage()(pred_mask_thresholded), cmap='gray')
        plt.title(f"Thresholded (>{args.threshold})")
        plt.axis('off')
        
        # Add overall title with image name
        plt.suptitle(f"Image: {test_image_names[idx]}")
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(args.output_dir, f"visualization_{i}_{test_image_names[idx]}.png")
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
        plt.close()
    
    # Create comparison visualization if comparing two models
    if args.compare_model:
        compare_model = SegmentationModel().to(device)
        compare_model.load_state_dict(torch.load(args.compare_model, map_location=device))
        compare_model.eval()
        
        with torch.no_grad():
            compare_pred_masks = compare_model(test_images)
        
        for i, idx in enumerate(indices):
            if idx >= len(test_images):
                continue
                
            # Get image, ground truth mask, and predictions from both models
            image = test_images[idx].cpu()
            true_mask = test_masks[idx].cpu()
            pred_mask1 = pred_masks[idx].cpu()
            pred_mask1_thresholded = (pred_mask1 > args.threshold).float()
            pred_mask2 = compare_pred_masks[idx].cpu()
            pred_mask2_thresholded = (pred_mask2 > args.threshold).float()
            
            # Create figure for comparison
            plt.figure(figsize=(15, 6))
            
            # Original image
            plt.subplot(2, 3, 1)
            plt.imshow(transforms.ToPILImage()(image))
            plt.title("Original Image")
            plt.axis('off')
            
            # Ground truth mask
            plt.subplot(2, 3, 2)
            plt.imshow(transforms.ToPILImage()(true_mask), cmap='gray')
            plt.title("Ground Truth Mask")
            plt.axis('off')
            
            # Model 1 prediction (raw)
            plt.subplot(2, 3, 3)
            plt.imshow(transforms.ToPILImage()(pred_mask1), cmap='gray')
            plt.title("Model 1 (Raw)")
            plt.axis('off')
            
            # Model 1 prediction (thresholded)
            plt.subplot(2, 3, 4)
            plt.imshow(transforms.ToPILImage()(pred_mask1_thresholded), cmap='gray')
            plt.title("Model 1 (Thresholded)")
            plt.axis('off')
            
            # Model 2 prediction (raw)
            plt.subplot(2, 3, 5)
            plt.imshow(transforms.ToPILImage()(pred_mask2), cmap='gray')
            plt.title("Model 2 (Raw)")
            plt.axis('off')
            
            # Model 2 prediction (thresholded)
            plt.subplot(2, 3, 6)
            plt.imshow(transforms.ToPILImage()(pred_mask2_thresholded), cmap='gray')
            plt.title("Model 2 (Thresholded)")
            plt.axis('off')
            
            # Add overall title with image name
            plt.suptitle(f"Comparison - Image: {test_image_names[idx]}")
            plt.tight_layout()
            
            # Save figure
            save_path = os.path.join(args.output_dir, f"comparison_{i}_{test_image_names[idx]}.png")
            plt.savefig(save_path)
            print(f"Saved comparison to {save_path}")
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize segmentation predictions")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the trained model")
    parser.add_argument("--compare_model", type=str, 
                        help="Optional path to a second model for comparison")
    parser.add_argument("--data_dir", type=str, default="./data", 
                        help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./visualizations", 
                        help="Directory to save visualizations")
    parser.add_argument("--num_samples", type=int, default=10, 
                        help="Number of random samples to visualize")
    parser.add_argument("--indices", type=str, 
                        help="Comma-separated list of specific indices to visualize")
    parser.add_argument("--threshold", type=float, default=0.5, 
                        help="Threshold for binary segmentation")
    
    args = parser.parse_args()
    
    # Visualize predictions
    visualize_predictions(args)
