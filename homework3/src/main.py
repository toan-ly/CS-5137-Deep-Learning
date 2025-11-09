import os
import glob
import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm

from .data.dataset import *
from .utils import compute_dice_iou, convert_to_binary, compute_dice_iou_sample, postprocess_mask, plot_loss
from .models.unet import UNet
from monai.inferers import sliding_window_inference
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / 'data'
CKPT_DIR = ROOT / 'checkpoints'
RESULTS_DIR = ROOT / 'figures' / 'eval'
METRIC_PATH = RESULTS_DIR / 'metrics.csv'

os.makedirs(RESULTS_DIR, exist_ok=True)


USE_GREEN = False
BATCH_SIZE = 2
IM_SIZE = 512
THRESHOLD = 0.6

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _overlay(img, gt_mask, pred_mask):
    """
    Create an overlay image showing GT and Pred masks on the original image
    Blue: GT only (False Negative)
    Green: Pred only (False Positive)
    Red: Overlap (True Positive)
    """
    overlay_img = img.copy()
    gt = gt_mask == 1
    pred = pred_mask == 1

    overlay_img[pred] = [0, 1, 0] # Green for Pred
    overlay_img[gt] = [0, 0, 1] # Blue for GT
    overlay_img[gt & pred] = [1, 0, 0] # Red for overlap
    return overlay_img

def visualize_test(imgs, masks, preds, model_name=None, save_path=None, num_samples=2):
    """
    Plot the original images, GT and Pred masks for test set

    Args:
        imgs: Original images 
        masks: Ground truth masks (in binary)
        preds: Predicted masks (in binary)
        save_path: Path to save the figure
        num_samples: Number of samples to visualize (number of rows)
    """
    fig, axes = plt.subplots(num_samples, 4, figsize=(12, 3.5 * num_samples))

    if num_samples == 1:
        axes = axes[np.newaxis, :]
    for i in range(num_samples):
        img = imgs[i].permute(1, 2, 0).numpy()
        # if img.max() > 1:
        #     img = img / 255.0
        mask = masks[i].squeeze().numpy()
        pred = preds[i].squeeze().numpy()

        if img.shape[2] > 3:
            img = img[:, :, :3]
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('GT')
        axes[i, 1].axis('off')

        dice, iou = compute_dice_iou_sample(pred, mask)
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title(f'Pred (Dice: {dice:.2f}, IoU: {iou:.2f})')
        axes[i, 2].axis('off')

        overlay = _overlay(img, mask, pred)
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay')
        axes[i, 3].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if model_name:
        plt.suptitle(f'{model_name}', fontsize=14)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.close(fig)

def postprocessing(preds):
    """
    Post-process predicted masks

    Args:
        preds: Predicted masks (batch_size, 1, H, W)
    """
    batch_size = preds.shape[0]
    processed_preds = []
    for i in range(batch_size):
        mask = preds[i, 0].cpu().numpy()
        mask = postprocess_mask(mask, radius=1)
        processed_preds.append(torch.tensor(mask).unsqueeze(0))
    processed_preds = torch.stack(processed_preds, dim=0)
    return processed_preds

@torch.no_grad()
def test(model, test_loader, model_name, threshold=0.5):
    """
    Test the model and compute Dice and IoU

    Args:
        model: Trained model
        test_loader: DataLoader for the test set
        model_name: Name of the model (for saving results)
        threshold: Threshold for converting logits to binary masks
    """
    model.eval()

    pred_path = RESULTS_DIR / model_name
    os.makedirs(pred_path, exist_ok=True)

    dices, ious = [], []
    i = 0
    for batch in tqdm(test_loader, desc="Testing", leave=False):
        i += 1
        imgs = batch['image'].to(DEVICE)
        masks = batch['mask'].to(DEVICE)
        logits = sliding_window_inference(
            inputs=imgs,
            roi_size=(128, 128),
            sw_batch_size=4,
            overlap=0.5,
            predictor=model,
            mode='gaussian'
        )
        preds = convert_to_binary(logits, threshold=threshold)
        masks = convert_to_binary(masks, threshold=threshold, is_gt=True)
        # preds = postprocessing(preds)

        dice, iou = compute_dice_iou(preds, masks)
        dices.append(dice)
        ious.append(iou)

        visualize_test(
            imgs, masks, preds, 
            save_path=pred_path / f'{i}.png',
            num_samples=imgs.size(0),
            model_name=model_name
        )

    return np.mean(dices), np.mean(ious)

def load_model(weight_path):
    """
    Load a trained model from a checkpoint file

    Args:
        weight_path: Path to the model weights file
    """
    ckpt = torch.load(weight_path, map_location=DEVICE)
    model = UNet(**ckpt['model_config'])
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(DEVICE).eval()
    return model, ckpt

def evaluate_model(checkpoint_dirs, loss_dirs, test_loader):
    """
    Evaluate all trained models on the test set

    Args:
        checkpoint_dirs: List of model weights paths
        loss_dirs: List of training history csv paths
        test_loader: DataLoader for the test set
    """
    metrics = []
    for ckpt_path, loss_path in zip(checkpoint_dirs, loss_dirs):
        model, ckpt = load_model(ckpt_path)
        model_config = ckpt['model_config']
        n_params = ckpt['n_params']
        features = model_config['features']
        depth = len(features) - 1
        use_norm = 'norm' if model_config.get('norm_type', None) else 'nonorm'
        upsampling_mode = model_config['up_mode']

        if model_config['block_type'] == 'base':
            model_name = 'UNET'
        elif model_config['block_type'] == 'residual':
            model_name = 'ResUNET'

        model_name += f'[{features[0]}-{depth}]_{use_norm}_{upsampling_mode}'
        print(f'Model: {model_name}')
        print({ckpt_path})

        if (RESULTS_DIR / model_name).exists():
            print(f"Results for {model_name} already exist. Skipping evaluation.\n")
            continue

        plot_loss(pd.read_csv(loss_path), save_dir=RESULTS_DIR / model_name)
        dice, iou = test(model, test_loader, model_name, threshold=THRESHOLD)
        print(f"Test Dice: {dice:.4f}, Test IoU: {iou:.4f}\n")

        metrics.append({
            'Model': model_name,
            'Parameters': f'{n_params / 1e6:.2f}M',
            'Dice': dice,
            'IoU': iou
        })
    metrics_df = pd.DataFrame(metrics)
    if METRIC_PATH.exists():
        existing_df = pd.read_csv(METRIC_PATH)
        metrics_df = pd.concat([existing_df, metrics_df], ignore_index=True)
    metrics_df = metrics_df.sort_values(by='Dice', ascending=False).reset_index(drop=True)
    metrics_df.to_csv(METRIC_PATH, index=False)
    print(f"Saved evaluation metrics to {METRIC_PATH}")

def rename_checkpoints_dirs(ckpt_dirs):
    """
    Rename checkpoint directories based on model configuration
    (Only for better file organization)
    """
    for ckpt_path in ckpt_dirs:
        _, ckpt = load_model(ckpt_path)
        model_config = ckpt['model_config']
        features = model_config['features']
        depth = len(features) - 1
        upsampling_mode = model_config['up_mode']

        if model_config['block_type'] == 'base':
            model_name = 'UNET'
        elif model_config['block_type'] == 'residual':
            model_name = 'ResUNET'
        model_name += f'[{features[0]}-{depth}]_{upsampling_mode}'
        
        orig_path = CKPT_DIR / ckpt_path.split('/')[-3] 
        new_path = CKPT_DIR / model_name
        if new_path.exists():
            print(f"Directory {new_path} already exists")
            continue

        os.rename(orig_path, new_path)


def main():
    print(f'=' * 20 + ' Evaluation ' + '=' * 20)
    test_loader = make_test_loader(
        data_root=DATA_DIR,
        im_size=IM_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=0,
        use_green_channel=USE_GREEN
    )
    
    loss_dirs = glob.glob(str(CKPT_DIR / '*/*/training_history.csv'))
    ckpt_dirs = glob.glob(str(CKPT_DIR / '*/*/best_model.pth'))
    evaluate_model(ckpt_dirs, loss_dirs, test_loader)

    # rename_checkpoints_dirs(ckpt_dirs)


if __name__ == "__main__":
    main()
