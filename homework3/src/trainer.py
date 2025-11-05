import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm
from monai.losses import (
    DiceLoss, FocalLoss, TverskyLoss, 
    HausdorffDTLoss, DiceCELoss
)
from monai.inferers import sliding_window_inference
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
        loss='dice',
        optimizer_name='adam',
        lr=0.001,
        early_stopping=True,
        patience=3,
        scheduler=None,
    ):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_name = loss
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.scheduler_name = scheduler
        self.criterion = self._get_loss(loss)
        self.optimizer = self._get_optimizer(optimizer_name, lr)
        self.scheduler = None
        if scheduler:
            self.scheduler = self._get_scheduler(scheduler)

        # Early stopping
        self.early_stopping = early_stopping
        self.patience = patience
        # self.best_metric = -float('inf')
        self.best_metric = float('inf')
        self.epochs_no_improve = 0
        self.best_weights = None

        self.checkpoint = {
            'train_loss': [],
            'train_dice': [],
            'train_iou': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': [],
        }

    def train_one_epoch(self, epoch, epochs):
        self.model.train()
        epoch_loss = 0.0
        dices, ious = [], []
        desc = f"Epoch [{epoch+1}/{epochs}] Training"
        for batch in tqdm(self.train_loader, desc=desc, leave=False):
            imgs = batch['image'].to(self.device) # [B, C, H, W]
            masks = batch['mask'].to(self.device) # [B, 1, H, W]

            self.optimizer.zero_grad()

            logits = self.model(imgs) # [B, 3, H, W] for 3 classes
            loss = self.criterion(logits, masks)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * imgs.size(0)

            with torch.no_grad():
                dice, iou = compute_dice_iou(self._convert_to_binary(logits), self._convert_to_binary(masks, is_mask=True))
                dices.append(dice)
                ious.append(iou)

        epoch_loss = epoch_loss / len(self.train_loader.dataset)
        return epoch_loss, np.mean(dices), np.mean(ious)
    
    @torch.no_grad()
    def validate(self, epoch, epochs):
        self.model.eval()
        epoch_loss = 0.0
        dices, ious = [], []
        desc = f"Epoch [{epoch+1}/{epochs}] Validation"
        for batch in tqdm(self.val_loader, desc=desc, leave=False):
            imgs = batch['image'].to(self.device) # [B, C, H, W]
            masks = batch['mask'].to(self.device) # [B, 1, H, W]

            logits = self.model(imgs) 
            loss = self.criterion(logits, masks)
            epoch_loss += loss.item() * imgs.size(0)

            dice, iou = compute_dice_iou(self._convert_to_binary(logits), self._convert_to_binary(masks, is_mask=True))
            dices.append(dice)
            ious.append(iou)

        epoch_loss = epoch_loss / len(self.val_loader.dataset)
        return epoch_loss, np.mean(dices), np.mean(ious)

    def fit(self, epochs=10, verbose=True, save_model_path=None, save_plots_path=None):
        start_time = time.time()
        for epoch in range(epochs):
            train_loss, train_dice, train_iou = self.train_one_epoch(epoch, epochs)
            val_loss, val_dice, val_iou = self.validate(epoch, epochs)

            self.checkpoint['train_loss'].append(train_loss)
            self.checkpoint['train_dice'].append(train_dice)
            self.checkpoint['train_iou'].append(train_iou)
            self.checkpoint['val_loss'].append(val_loss)
            self.checkpoint['val_dice'].append(val_dice)
            self.checkpoint['val_iou'].append(val_iou)

            if verbose and (epoch + 1) % 5 == 0:
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
                    if self.epochs_no_improve >= self.patience:
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
        print(f'Training time: {total_time:.2f}s | (best val metrics = {self.best_metric:.4f})')

        # Save model
        if save_model_path:
            self._save_checkpoint(save_model_path)
            print(f'Model saved to {save_model_path}')

        return self.checkpoint
        
    @torch.no_grad()
    def test(self):
        self.model.eval()
        dices, ious = [], []
        for batch in tqdm(self.test_loader, desc="Testing", leave=False):
            imgs = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            logits = sliding_window_inference(
                inputs=imgs,
                roi_size=(128, 128),
                sw_batch_size=4,
                overlap=0.5,
                predictor=self.model,
            )

            dice, iou = compute_dice_iou(self._convert_to_binary(logits), self._convert_to_binary(masks, is_mask=True))
            dices.append(dice)
            ious.append(iou)

        return np.mean(dices), np.mean(ious)

    @torch.no_grad()
    def _visualize(self, epoch, save_path=None, num_samples=3):
        self.model.eval()
        imgs, masks, preds = [], [], []
        for batch in self.val_loader:
            img = batch['image'].to(self.device)
            mask = batch['mask'].to(self.device)

            with torch.no_grad():
                logits = self.model(img)
                pred_bin = self._convert_to_binary(logits)
                mask_bin = self._convert_to_binary(mask, is_mask=True)

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
            dice, iou = self._dice_iou_sample(pred, mask)

            # If image has 4 channels after appending clahe, only display rgb
            if img.shape[2] == 4:
                img = img[:, :, :3]
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
        # plt.show()
        if save_path:
            save_path = os.path.join(save_path, f'epoch_{epoch+1}.png')
            plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    
    def _dice_iou_sample(self, pred, target):
        eps = 1e-7
        intersection = (pred * target).sum()
        sum_preds_targets = pred.sum() + target.sum()
        dice = (2. * intersection) / (sum_preds_targets + eps)
        iou = intersection / (sum_preds_targets - intersection + eps)
        return dice.item(), iou.item()
      
    def _get_optimizer(self, optim_name, lr):
        if optim_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        if optim_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        if optim_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        raise ValueError(f'Unknown optimizer: {optim_name}')    

    def _get_scheduler(self, scheduler_name):
        scheduler_map = {
            'step': optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5),
            'plateau': optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5),
            'cosine': optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-6),
        }
        if scheduler_name not in scheduler_map:
            raise ValueError(f'Unknown scheduler: {scheduler_name}')
        return scheduler_map[scheduler_name]

    def _get_loss(self, name):
        """
        Examples of loss_name:
            'bce' 
            'dice'
            'bce+dice'
            'bce+dice@0.3,0.7'
        """
        criterions = {
            'bce': nn.BCEWithLogitsLoss,
            'dice': lambda: DiceLoss(sigmoid=True, squared_pred=True),
            'focal': lambda: FocalLoss(gamma=2.0),
            'tversky': lambda: TverskyLoss(sigmoid=True, alpha=0.3, beta=0.7),
            'hausdorff': lambda: HausdorffDTLoss(sigmoid=True, include_background=True),
            'dicece': lambda: DiceCELoss(
                include_background=True,
                to_onehot_y=True,
                softmax=True,
                weight=torch.tensor([1.0, 1.0, 3.0], device=self.device)
            )
        }

        name = name.strip().lower()

        if '+' in name:
            if '@' in name:
                loss_part, weight_part = name.split('@')
                weights = [float(w) for w in weight_part.split(',')]
            else:
                loss_part = name
                weights = None
            loss_names = loss_part.split('+')
            if loss_names[0] not in criterions or loss_names[1] not in criterions:
                raise ValueError(f'Unknown loss(es): {loss_names[0]}, {loss_names[1]}')
            
            w1, w2 = (0.5, 0.5) if weights is None else (weights[0], weights[1])
            criterion1 = criterions[loss_names[0]]()
            criterion2 = criterions[loss_names[1]]()

            return lambda preds, targets: w1 * criterion1(preds, targets) + w2 * criterion2(preds, targets)
        
        if name not in criterions:
            raise ValueError(f'Unknown loss: {name}')
        return criterions[name]()
        
    def _save_checkpoint(self, path):
        ckpt = {
            'state_dict': self.model.state_dict(),
            'model_config': self.model.model_config,
            'train_config': {
                'loss': self.loss_name,
                'optimizer': self.optimizer_name,
                'lr': self.lr,
                'scheduler': self.scheduler_name,
            }
        }
        torch.save(ckpt, path)

    def _convert_to_binary(self, logits, threshold=0.5, is_mask=False):
        if is_mask:
            logits = logits.unsqueeze(1) if logits.dim() == 3 else logits
            return (logits > 0).float()
        if logits.shape[1] == 3:
            probs = torch.softmax(logits, dim=1)
            vessel_probs = probs[:, 1, ...] + probs[:, 2, ...]
            vessel_probs = vessel_probs.unsqueeze(1) if vessel_probs.dim() == 3 else vessel_probs
            return (vessel_probs > threshold).float()
        else:
            probs = torch.sigmoid(logits)
            probs = probs.unsqueeze(1) if probs.dim() == 3 else probs
            return (probs > threshold).float()
