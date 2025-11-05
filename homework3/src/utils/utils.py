import torch
import numpy as np
import random
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path, map_location=None):
    return torch.load(path, map_location=map_location)

def get_device(device_str=None):
    if device_str is not None:
        return torch.device(device_str)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_dice_iou(preds, targets):
    eps = 1e-7
    intersection = (preds * targets).sum((1, 2, 3)) # Sum over C, H, W
    sum_preds_targets = preds.sum((1, 2, 3)) + targets.sum((1, 2, 3))
    dice = (2. * intersection) / (sum_preds_targets + eps)
    iou = intersection / (sum_preds_targets - intersection + eps)
    return dice.mean().item(), iou.mean().item()

def compute_dice_iou_sample(pred, target):
    eps = 1e-7
    intersection = (pred * target).sum()
    sum_preds_targets = pred.sum() + target.sum()
    dice = (2. * intersection) / (sum_preds_targets + eps)
    iou = intersection / (sum_preds_targets - intersection + eps)
    return dice.item(), iou.item()

def convert_to_binary(logits, threshold=0.5, is_gt=False):
    # If ground truth mask, combine class 1 (thick vessels) and 2 (thin vessels) 
    if is_gt:
        logits = logits.unsqueeze(1) if logits.dim() == 3 else logits
        return (logits > 0).float()

    # If prediction mask has 3 channels (background, thick vessels, thin vessels)
    if logits.shape[1] == 3:
        probs = torch.softmax(logits, dim=1)
        # thick_vessel_mask = (probs[:, 1, ...] > threshold).float()
        # thin_vessel_mask = (probs[:, 2, ...] > threshold).float()
        # vessel_mask = (thick_vessel_mask + thin_vessel_mask) > 0
        vessel_probs = probs[:, 1, ...] + probs[:, 2, ...]
        vessel_mask = (vessel_probs > threshold).float()
        return vessel_mask.unsqueeze(1)
    
    raise ValueError("Error in convert_to_binary")