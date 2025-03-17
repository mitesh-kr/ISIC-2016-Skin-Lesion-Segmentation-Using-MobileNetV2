import torch
import matplotlib.pyplot as plt

def IoU(pred_mask, true_mask):
    """
    Calculate the mean Intersection over Union (IoU) between predicted and true masks.
    
    Args:
        pred_mask (torch.Tensor): Predicted mask tensor
        true_mask (torch.Tensor): Ground truth mask tensor
        
    Returns:
        float: Mean IoU score
    """
    # Flatten the masks along the batch dimension
    true_mask_flat = true_mask.view(true_mask.size(0), -1)
    pred_mask_flat = pred_mask.view(pred_mask.size(0), -1)
    
    # Calculate intersection and union
    intersect = torch.sum(pred_mask_flat * true_mask_flat, dim=1)
    union = torch.sum(pred_mask_flat, dim=1) + torch.sum(true_mask_flat, dim=1) - intersect
    
    # Calculate IoU for each sample and mean
    iou = intersect / union
    mean_iou = torch.mean(iou)
    
    return mean_iou

def dice(pred_mask, true_mask):
    """
    Calculate the mean Dice coefficient between predicted and true masks.
    
    Args:
        pred_mask (torch.Tensor): Predicted mask tensor
        true_mask (torch.Tensor): Ground truth mask tensor
        
    Returns:
        float: Mean Dice coefficient
    """
    # Flatten the masks along the batch dimension
    pred_mask_flat = pred_mask.view(pred_mask.size(0), -1)
    true_mask_flat = true_mask.view(true_mask.size(0), -1)
    
    # Calculate intersection and sums
    intersect = torch.sum(pred_mask_flat * true_mask_flat, dim=1)
    fsum = torch.sum(pred_mask_flat, dim=1)
    ssum = torch.sum(true_mask_flat, dim=1)
    
    # Calculate dice for each sample and mean
    dice = (2 * intersect) / (fsum + ssum + 1e-7)
    mean_dice = torch.mean(dice)
    
    return mean_dice.item()

def plot(epoch_values1, epoch_values2, ylabel1, ylabel2, title):
    """
    Plot two metrics over epochs for comparison.
    
    Args:
        epoch_values1 (list): First set of values to plot
        epoch_values2 (list): Second set of values to plot
        ylabel1 (str): Label for the first set of values
        ylabel2 (str): Label for the second set of values
        title (str): Plot title
    """
    epochs = range(1, len(epoch_values1) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, epoch_values1, marker='o', linestyle='-', color='r', label=ylabel1)
    plt.plot(epochs, epoch_values2, marker='o', linestyle='-', color='g', label=ylabel2)
    plt.xlabel('Epoch')
    plt.ylabel('Values')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def count_parameters(model):
    """
    Count the number of trainable and non-trainable parameters in a model.
    
    Args:
        model (torch.nn.Module): The model to analyze
        
    Returns:
        tuple: (trainable_parameters, non_trainable_parameters)
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable_params, non_trainable_params
