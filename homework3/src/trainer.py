import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm
from monai.losses import DiceLoss
import matplotlib.pyplot as plt
from PIL import Image
import os

def compute_dice_iou(preds, targets):
    eps = 1e-7
    intersection = (preds * targets).sum((1, 2, 3)) # Sum over C, H, W
    sum_preds_targets = preds.sum((1, 2, 3)) + targets.sum((1, 2, 3))
    dice = (2. * intersection) / (sum_preds_targets + eps)
    iou = intersection / (sum_preds_targets - intersection + eps)
    return dice.mean().item(), iou.mean().item()

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        loss='bce_dice',
        lr=0.001,
        early_stopping=True,
        early_stopping_patience=3,
        scheduler=None,
    ):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = self._get_loss(loss)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = None
        if scheduler:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        # Early stopping
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        # self.best_metric = -float('inf')
        self.best_metric = float('inf')
        self.epochs_no_improve = 0
        self.best_weights = None

        self.checkpoint = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': [],
        }

    def train_one_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        dices, ious = [], []
        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            imgs = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * imgs.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            dice, iou = compute_dice_iou(preds, masks)
            dices.append(dice)
            ious.append(iou)

        epoch_loss = epoch_loss / len(self.train_loader.dataset)
        return epoch_loss, np.mean(dices), np.mean(ious)
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        epoch_loss = 0.0
        dices, ious = [], []
        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            imgs = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            outputs = self.model(imgs)
            loss = self.criterion(outputs, masks)
            epoch_loss += loss.item() * imgs.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            dice, iou = compute_dice_iou(preds, masks)
            dices.append(dice)
            ious.append(iou)

        epoch_loss = epoch_loss / len(self.val_loader.dataset)
        return epoch_loss, np.mean(dices), np.mean(ious)

    def fit(self, epochs=10, verbose=True, save_model_path=None, save_plots_path=None):
        start_time = time.time()
        for epoch in range(epochs):
            train_loss, train_dice, train_iou = self.train_one_epoch()
            val_loss, val_dice, val_iou = self.validate()

            self.checkpoint['train_loss'].append(train_loss)
            self.checkpoint['val_loss'].append(val_loss)
            self.checkpoint['val_dice'].append(val_dice)
            self.checkpoint['val_iou'].append(val_iou)

            if verbose and (epoch + 1) % (epochs // 10) == 0:
                print(f'Epoch {epoch+1}/{epochs} - '
                      f'Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Train IoU: {train_iou:.4f} | '
                      f'Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}')
                self._visualize(num_samples=3, save_path=save_plots_path, epoch=epoch)

            # Early stopping
            if self.early_stopping:
                # if val_dice > self.best_metric:
                if val_loss < self.best_metric:
                    # self.best_metric = val_dice
                    self.best_metric = val_loss
                    self.epochs_no_improve = 0
                    self.best_weights = self.model.state_dict()
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= self.early_stopping_patience:
                        print("=> Early stopping at epoch", epoch+1)
                        if self.best_weights is not None:
                            self.model.load_state_dict(self.best_weights)
                        break

            if self.scheduler:
                self.scheduler.step()

        # Load best weights
        if self.best_weights is not None:
            self.model.load_state_dict(self.best_weights)

        total_time = time.time() - start_time
        print(f'Training time: {total_time:.2f}s | (best val dice = {self.best_metric:.4f})')

        # Save model
        if save_model_path:
            torch.save(self.model.state_dict(), save_model_path)
            print(f'Model saved to {save_model_path}')

        return self.checkpoint
        
    @torch.no_grad()
    def test(self):
        self.model.eval()
        dices, ious = [], []
        for batch in tqdm(self.test_loader, desc="Testing", leave=False):
            imgs = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            outputs = self.model(imgs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            dice, iou = compute_dice_iou(preds, masks)
            dices.append(dice)
            ious.append(iou)

        return np.mean(dices), np.mean(ious)

    def _get_loss(self, loss_name):
        if loss_name == 'bce':
            return nn.BCEWithLogitsLoss()
        if loss_name == 'dice':
            return DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
        if loss_name == 'bce_dice':
            dice = DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
            bce = nn.BCEWithLogitsLoss()
            return lambda pred, target: 0.5 * bce(pred, target) + 0.5 * dice(pred, target)
        raise ValueError(f'Unknown loss function: {loss_name}')

    @torch.no_grad()
    def _visualize(self, epoch, save_path=None, num_samples=3):
        self.model.eval()
        imgs, masks, preds = [], [], []
        for batch in self.val_loader:
            img = batch['image'].to(self.device)
            mask = batch['mask'].to(self.device)

            with torch.no_grad():
                output = self.model(img)
                prob = torch.sigmoid(output)
                pred = (prob > 0.5).float()

            imgs.append(img.cpu())
            masks.append(mask.cpu())
            preds.append(pred.cpu())

            if len(imgs) * img.size(0) >= num_samples:
                break
    
        imgs = torch.cat(imgs, dim=0)[:num_samples]
        masks = torch.cat(masks, dim=0)[:num_samples]
        preds = torch.cat(preds, dim=0)[:num_samples]

        plt.rcParams['figure.figsize'] = [12, 4 * num_samples]
        fig, axes = plt.subplots(num_samples, 3)
        for i in range(num_samples):
            img = imgs[i].permute(1, 2, 0).numpy()
            if img.max() > 1:
                img = img / 255.0  # Normalize for visualization

            pred = preds[i].squeeze().numpy()
            mask = masks[i].squeeze().numpy()
            dice, iou = compute_dice_iou(pred, mask)

            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(masks[i].squeeze(), cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(preds[i].squeeze(), cmap='gray')
            axes[i, 2].set_title(f'Prediction\nDice: {dice:.2f}, IoU: {iou:.2f}')
            axes[i, 2].axis('off')

        plt.suptitle(f'Epoch {epoch+1}')
        plt.tight_layout()
        if save_path:
            save_path = os.path.join(save_path, f'epoch_{epoch+1}.png')
            plt.savefig(save_path)
        plt.show()
        plt.close(fig)
      