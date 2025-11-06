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
from .utils import *

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
        self.n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
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
        self.best_dice = -float('inf')
        self.best_loss = float('inf')
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
                dice, iou = compute_dice_iou(convert_to_binary(logits), convert_to_binary(masks, is_gt=True))
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

            logits = self.model(imgs) # [B, 3, H, W] for 3 classes
            loss = self.criterion(logits, masks)
            epoch_loss += loss.item() * imgs.size(0)

            dice, iou = compute_dice_iou(convert_to_binary(logits), convert_to_binary(masks, is_gt=True))
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
                visualize(self.model, self.val_loader, epoch, save_path=save_plots_path, num_samples=3, device=self.device)

            self.best_dice = max(self.best_dice, val_dice)

            # Early stopping
            if self.early_stopping:
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
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
        print(f'Training time: {total_time:.2f}s | (best val dice = {self.best_dice:.4f})')

        # Save model
        if save_model_path:
            self._save_checkpoint(save_model_path)
            print(f'Model saved to {save_model_path}')

        return self.checkpoint
      
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
        cls_weights = torch.tensor([1.0, 1.0, 3.0], device=self.device)
        criterions = {
            'bce': nn.BCEWithLogitsLoss,
            'dice': lambda: DiceLoss(softmax=True, squared_pred=True, to_onehot_y=True),
            'focal': lambda: FocalLoss(
                gamma=2.0,
                weight=cls_weights,
                use_softmax=True,
                to_onehot_y=True
            ),
            'tversky': lambda: TverskyLoss(
                softmax=True, 
                to_onehot_y=True,
                alpha=0.3, beta=0.7
            ),
            'hausdorff': lambda: HausdorffDTLoss(
                softmax=True, 
                include_background=True, 
                to_onehot_y=True
            ),
            'dicece': lambda: DiceCELoss(
                include_background=True,
                to_onehot_y=True,
                softmax=True,
                weight=cls_weights
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
            'n_params': self.n_params,
            'train_config': {
                'loss': self.loss_name,
                'optimizer': self.optimizer_name,
                'lr': self.lr,
                'scheduler': self.scheduler_name,
            }
        }
        torch.save(ckpt, path)


