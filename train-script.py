import os
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from models.segmentation_model import SegmentationModel
from utils.data_loader import create_dataloaders
from utils.metrics import IoU, dice, plot, count_parameters

def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset paths
    train_path = os.path.join(args.data_dir, "train")
    train_masks_path = os.path.join(args.data_dir, "train_masks")
    val_path = os.path.join(args.data_dir, "test")
    val_masks_path = os.path.join(args.data_dir, "test_masks")
    
    # Create dataloaders
    train_dataloader, val_images, val_masks, _ = create_dataloaders(
        train_path, train_masks_path, val_path, val_masks_path, 
        batch_size=args.batch_size
    )
    
    # Move validation data to device
    val_images = val_images.to(device)
    val_masks = val_masks.to(device)
    
    # Create model and send to device
    model = SegmentationModel().to(device)
    
    # Set up loss function
    loss_function = nn.BCELoss()
    
    # Set learning rates based on training mode
    if args.mode == "feature_extraction":
        print("Running in feature extraction mode (frozen encoder)")
        encoder_lr = 0
        # Freeze encoder weights
        for param in model.encoder.parameters():
            param.requires_grad = False
    else:  # finetuning
        print("Running in finetuning mode (trainable encoder)")
        encoder_lr = args.encoder_lr
    
    # Define optimizer with separate learning rates
    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters(), 'lr': encoder_lr},
        {'params': model.decoder.parameters(), 'lr': args.decoder_lr}
    ])
    
    # Lists to store metrics
    train_loss_list = []
    val_loss_list = []
    val_iou_list = []
    val_dice_score_list = []
    
    # Training loop
    for epoch in range(args.epochs):
        train_loss = 0
        
        # Training
        model.train()
        for X_batch, y_batch in train_dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            y_pred = model(X_batch)
            loss = loss_function(y_pred, y_batch)
            train_loss += loss.item()
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_dataloader)
        train_loss_list.append(avg_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            y_pred_val = model(val_images)
            val_loss = loss_function(y_pred_val, val_masks).item()
            val_iou = IoU(y_pred_val, val_masks)
            val_dice_score = dice(y_pred_val, val_masks)
            
            print(f'Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | '
                  f'Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | '
                  f'Val Dice Score: {val_dice_score:.4f}')
        
        val_loss_list.append(val_loss)
        val_iou_list.append(val_iou)
        val_dice_score_list.append(val_dice_score)
        
        # Save model at specified intervals
        if (epoch + 1) % args.save_interval == 0:
            save_path = f"{args.output_dir}/{args.mode}_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
    
    # Save final model
    final_save_path = f"{args.output_dir}/{args.mode}_final.pth"
    torch.save(model.state_dict(), final_save_path)
    print(f"Final model saved to {final_save_path}")
    
    # Plot training and validation loss
    plot(
        epoch_values1=train_loss_list, 
        epoch_values2=val_loss_list, 
        ylabel1='Train Loss', 
        ylabel2='Val Loss', 
        title=f'Train Loss vs Val Loss ({args.mode})'
    )
    
    # Count and print model parameters
    trainable_params, non_trainable_params = count_parameters(model)
    print(f"Number of trainable parameters: {trainable_params}")
    print(f"Number of non-trainable parameters: {non_trainable_params}")
    
    # Save metrics to file
    with open(f"{args.output_dir}/{args.mode}_metrics.txt", "w") as f:
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Final Train Loss: {train_loss_list[-1]:.4f}\n")
        f.write(f"Final Val Loss: {val_loss_list[-1]:.4f}\n")
        f.write(f"Final IoU: {val_iou_list[-1]:.4f}\n")
        f.write(f"Final Dice Score: {val_dice_score_list[-1]:.4f}\n")
        f.write(f"Number of trainable parameters: {trainable_params}\n")
        f.write(f"Number of non-trainable parameters: {non_trainable_params}\n")
    
    return model, train_loss_list, val_loss_list, val_iou_list, val_dice_score_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model on ISIC dataset")
    parser.add_argument("--mode", type=str, choices=["feature_extraction", "finetuning"], 
                        default="feature_extraction", 
                        help="Training mode: feature_extraction or finetuning")
    parser.add_argument("--data_dir", type=str, default="./data", 
                        help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", 
                        help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=45, 
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, 
                        help="Number of training epochs")
    parser.add_argument("--encoder_lr", type=float, default=0.0001, 
                        help="Learning rate for encoder (only used in finetuning mode)")
    parser.add_argument("--decoder_lr", type=float, default=0.01, 
                        help="Learning rate for decoder")
    parser.add_argument("--save_interval", type=int, default=10, 
                        help="Save model every N epochs")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Train model
    train(args)
