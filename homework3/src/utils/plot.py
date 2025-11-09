import matplotlib.pyplot as plt
import os
import torch
from .utils import compute_dice_iou_sample, convert_to_binary

def plot_loss(loss_df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].plot(loss_df['epoch'], loss_df['train_loss'], label='train_loss')
    axs[0].plot(loss_df['epoch'], loss_df['val_loss'], label='val_loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training and Validation Loss')
    axs[0].legend(loc='upper right')

    axs[1].plot(loss_df['epoch'], loss_df['train_dice'], label='train_dice')
    axs[1].plot(loss_df['epoch'], loss_df['val_dice'], label='val_dice')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Dice')
    axs[1].set_title('Training and Validation Dice')
    axs[1].legend(loc='lower right')

    axs[2].plot(loss_df['epoch'], loss_df['train_iou'], label='train_iou')
    axs[2].plot(loss_df['epoch'], loss_df['val_iou'], label='val_iou')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('IoU')
    axs[2].set_title('Training and Validation IoU')
    axs[2].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'train_curve.png'))

@torch.no_grad()
def visualize(model, loader, epoch, save_path=None, num_samples=3, device='cpu'):
    model.eval()
    imgs, masks, preds = [], [], []
    for batch in loader:
        img = batch['image'].to(device)
        mask = batch['mask'].to(device)

        with torch.no_grad():
            logits = model(img)
            pred_bin = convert_to_binary(logits)
            mask_bin = convert_to_binary(mask, is_gt=True)

        imgs.append(img.cpu())
        masks.append(mask_bin.cpu())
        preds.append(pred_bin.cpu())

        if len(imgs) * img.size(0) >= num_samples:
            break

    imgs = torch.cat(imgs, dim=0)[:num_samples]
    masks = torch.cat(masks, dim=0)[:num_samples]
    preds = torch.cat(preds, dim=0)[:num_samples]

    plt.rcParams['figure.figsize'] = [10, 4 * num_samples]
    fig, axes = plt.subplots(num_samples, 3)
    for i in range(num_samples):
        img = imgs[i].permute(1, 2, 0).numpy()
        if img.max() > 1:
            img = img / 255.0  # Normalize for visualization

        pred = preds[i].squeeze().numpy()
        mask = masks[i].squeeze().numpy()
        dice, iou = compute_dice_iou_sample(pred, mask)

        # If image has 4 channels after appending clahe, only display rgb
        if img.shape[2] > 3:
            img = img[:, :, :3]
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title(f'Prediction\nDice: {dice:.2f}, IoU: {iou:.2f}')
        axes[i, 2].axis('off')

    plt.suptitle(f'Epoch {epoch+1}')
    plt.tight_layout()
    # plt.show()
    if save_path:
        save_path = os.path.join(save_path, f'epoch_{epoch+1}.png')
        plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)