import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

def load_images(image_folder_path, mask_folder_path):
    """
    Load images and masks from the specified folders, resize to 128x128, and convert to tensors.
    
    Args:
        image_folder_path (str): Path to the folder containing images
        mask_folder_path (str): Path to the folder containing mask images
        
    Returns:
        tuple: Tuple containing (image_file_names, image_tensors, mask_tensors)
    """
    image_file_names = sorted(os.listdir(image_folder_path))
    mask_file_names = sorted(os.listdir(mask_folder_path))

    images = torch.tensor([])
    masks = torch.tensor([])
    
    for image_name, mask_name in zip(image_file_names, mask_file_names):
        # Check if image and mask names match
        if image_name.split('.')[0] == mask_name.split('.')[0]:
            image_path = os.path.join(image_folder_path, image_name)
            mask_path = os.path.join(mask_folder_path, mask_name)

            # Load and preprocess image
            image = Image.open(image_path)
            image = transforms.Resize((128, 128))(image)
            image = transforms.ToTensor()(image)
            images = torch.cat((images, image.unsqueeze(0)), dim=0)

            # Load and preprocess mask
            mask = Image.open(mask_path)
            mask = transforms.Resize((128, 128))(mask)
            mask = transforms.ToTensor()(mask)
            masks = torch.cat((masks, mask.unsqueeze(0)), dim=0)

    return image_file_names, images, masks

def create_dataloaders(train_path, train_masks_path, val_path, val_masks_path, batch_size=45):
    """
    Create DataLoader for training and validation datasets.
    
    Args:
        train_path (str): Path to training images
        train_masks_path (str): Path to training masks
        val_path (str): Path to validation images
        val_masks_path (str): Path to validation masks
        batch_size (int): Batch size for training
        
    Returns:
        tuple: Tuple containing (train_dataloader, val_images, val_masks, val_image_names)
    """
    # Load training data
    _, train_images, train_masks = load_images(train_path, train_masks_path)
    train_dataset = TensorDataset(train_images, train_masks)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )
    
    # Load validation data
    val_image_names, val_images, val_masks = load_images(val_path, val_masks_path)
    
    return train_dataloader, val_images, val_masks, val_image_names
