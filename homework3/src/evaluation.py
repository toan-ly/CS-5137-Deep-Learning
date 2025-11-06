import os
import glob
import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm
from .data.dataset import *
from .utils import plot_loss, compute_dice_iou, convert_to_binary, compute_dice_iou_sample
from .models.unet import UNet
from monai.inferers import sliding_window_inference
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt


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

def visualize_test(imgs, masks, preds, save_path=None, num_samples=2):
    """
    Plot the original images, GT and Pred masks for test set

    Args:
        imgs: Original images 
        masks: Ground truth masks (in binary)
        preds: Predicted masks (in binary)
        save_path: Path to save the figure
        num_samples: Number of samples to visualize (number of rows)
    """
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 4 * num_samples))
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
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        dice, iou = compute_dice_iou_sample(pred, mask)
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title(f'Prediction\nDice: {dice:.2f}, IoU: {iou:.2f}')
        axes[i, 2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)

    plt.close(fig)


@torch.no_grad()
def test(model, test_loader, model_name, threshold=0.5):
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

        dice, iou = compute_dice_iou(preds, masks)
        dices.append(dice)
        ious.append(iou)

        visualize_test(
            imgs, masks, preds, 
            save_path=pred_path / f'{i}.png',
            num_samples=imgs.size(0)
        )

    return np.mean(dices), np.mean(ious)

def load_model(weight_path):
    ckpt = torch.load(weight_path, map_location=DEVICE)
    model = UNet(**ckpt['model_config'])
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(DEVICE).eval()
    return model, ckpt

def main():
    print(f'=' * 20 + ' Evaluation ' + '=' * 20)
    test_loader = make_test_loader(
        data_root=DATA_DIR,
        im_size=IM_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=0,
        use_green_channel=USE_GREEN
    )

    metrics = []
    ckpt_dirs = glob.glob(str(CKPT_DIR / '*/*/best_model.pth'))
    for ckpt_path in ckpt_dirs:
        model, ckpt = load_model(ckpt_path)
        model_config = ckpt['model_config']
        n_params = ckpt['n_params']
        features = model_config['features']
        depth = len(features) - 1
        loss = ckpt['train_config']['loss']
        use_norm = 'norm' if model_config.get('norm_type', None) else 'no-norm'

        if model_config['block_type'] == 'base':
            model_name = 'UNET'
        elif model_config['block_type'] == 'residual':
            model_name = 'ResUNET'
        model_name += f'[{features[0]}-{depth}]-{use_norm}-{loss}'
        print(f'Model: {model_name}')
        print({ckpt_path})

        dice, iou = test(model, test_loader, model_name, threshold=THRESHOLD)
        print(f"Test Dice: {dice:.4f}, Test IoU: {iou:.4f}\n")

        n_params = f'{n_params / 1e6:.2f}M'  # convert to million
        metrics.append({
            'Model': model_name,
            'Parameters': n_params,
            'Dice': dice,
            'IoU': iou
        })

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(METRIC_PATH, index=False)
    print(f"Saved evaluation metrics to {METRIC_PATH}")




if __name__ == "__main__":
    main()
