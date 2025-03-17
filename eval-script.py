import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from models.segmentation_model import SegmentationModel
from utils.data_loader import load_images
from utils.metrics import IoU, dice

def evaluate(args):
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
    
    # Define loss function
    loss_function = nn.BCELoss()
    
    # Evaluate model
    with torch.no_grad():
        test_pred = model(test_images)
        test_loss = loss_function(test_pred, test_masks).item()
        test_iou = IoU(test_pred, test_masks)
        test_dice_score = dice(test_pred, test_masks)
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test IoU: {test_iou:.4f}")
        print(f"Test Dice Score: {test_dice_score:.4f}")
    
    # Save results
    with open(f"{args.output_dir}/evaluation_results.txt", "w") as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test IoU: {test_iou:.4f}\n")
        f.write(f"Test Dice Score: {test_dice_score:.4f}\n")
    
    # Visualize some predictions if requested
    if args.visualize:
        visualize_samples(test_images, test_masks, test_pred, test_image_names, 
                         args.num_samples, args.output_dir)
    
    return test_loss, test_iou, test_dice_score

def visualize_samples(images, masks, predictions, image_names, num_samples, output_dir):
    """
    Visualize a number of sample predictions from the test set.
    
    Args:
        images (torch.Tensor): Input images
        masks (torch.Tensor): Ground truth masks
        predictions (torch.Tensor): Predicted masks
        image_names (list): List of image filenames
        num_samples (int): Number of samples to visualize
        output_dir (str): Directory to save visualizations
    """
    # Create visualization directory if it doesn't exist
    vis_dir = os.path.join(output_dir, "visualizations")
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # Select sample indices
    if num_samples > len(images):
        num_samples = len(images)
    
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        # Get image, mask, and prediction
        img = images[idx].cpu()
        mask = masks[idx].cpu()
        pred = predictions[idx].cpu()
        pred_thresholded = (pred > 0.5).float()
        
        # Create figure
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 4, 1)
        plt.imshow(transforms.ToPILImage()(img))
        plt.title("Original Image")
        plt.axis("off")
        
        # Ground truth mask
        plt.subplot(1, 4, 2)
        plt.imshow(transforms.ToPILImage()(mask), cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis("off")
        
        # Predicted mask (raw)
        plt.subplot(1, 4, 3)
        plt.imshow(transforms.ToPILImage()(pred), cmap='gray')
        plt.title("Predicted Mask")
        plt.axis("off")
        
        # Predicted mask (thresholded)
        plt.subplot(1, 4, 4)
        plt.imshow(transforms.ToPILImage()(pred_thresholded), cmap='gray')
        plt.title("Thresholded Mask")
        plt.axis("off")
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"sample_{i}_{image_names[idx]}"))
        plt.close()
    
    print(f"Saved {num_samples} visualizations to {vis_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation model on ISIC dataset")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the trained model")
    parser.add_argument("--data_dir", type=str, default="./data", 
                        help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", 
                        help="Directory to save results")
    parser.add_argument("--visualize", action="store_true", 
                        help="Visualize predictions on test samples")
    parser.add_argument("--num_samples", type=int, default=10, 
                        help="Number of test samples to visualize")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Evaluate model
    evaluate(args)