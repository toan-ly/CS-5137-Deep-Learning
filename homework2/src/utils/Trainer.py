import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        lr=0.001,
        early_stopping=True,
        early_stopping_patience=3,
    ):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Early stopping
        self.early_stopping = early_stopping
        if early_stopping:
            self.early_stopping_patience = early_stopping_patience
            self.best_val_loss = float('inf')
            self.epochs_no_improve = 0
            self.best_weights = None

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': [],
        }

    def train_one_epoch(self):
        self.model.train()
        epoch_loss, correct, total = [], 0, 0

        all_preds, all_gts = [], []
        for X_train, y_train in tqdm.tqdm(self.train_loader, desc="Training", leave=False):
            X_train, y_train = X_train.to(self.device), y_train.to(self.device)

            self.optimizer.zero_grad()
            y_preds = self.model(X_train)
            loss = self.criterion(y_preds, y_train)
            loss.backward()
            self.optimizer.step()

            epoch_loss.append(loss.item())

            # _, preds = torch.max(y_preds.data, 1)
            _, preds = y_preds.detach().max(1)
            total += y_train.size(0)
            correct += (preds == y_train).sum().item()

            all_preds.extend(preds.detach().cpu().numpy())
            all_gts.extend(y_train.detach().cpu().numpy())
        epoch_loss = np.mean(epoch_loss)
        epoch_acc = correct / total

        # Compute F1 macro since all classes are equally important
        epoch_f1 = f1_score(all_gts, all_preds, average='macro')

        return epoch_loss, epoch_acc, epoch_f1

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        epoch_loss, correct, total = 0.0, 0, 0
        all_preds, all_gts = [], []
        with torch.no_grad():
            for X_val, y_val in tqdm.tqdm(self.val_loader, desc="Validation", leave=False):
                X_val, y_val = X_val.to(self.device), y_val.to(self.device)

                y_preds = self.model(X_val)
                loss = self.criterion(y_preds, y_val)

                epoch_loss += loss.item() * X_val.size(0)
                _, preds = torch.max(y_preds, 1)
                total += y_val.size(0)
                correct += (preds == y_val).sum().item()

                all_preds.extend(preds.detach().cpu().numpy())
                all_gts.extend(y_val.detach().cpu().numpy())

        epoch_loss = epoch_loss / len(self.val_loader.dataset)
        epoch_acc = correct / total 
        epoch_f1 = f1_score(all_gts, all_preds, average='macro')

        return epoch_loss, epoch_acc, epoch_f1

    @torch.no_grad()
    def test(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_test, y_test in tqdm.tqdm(self.test_loader, desc='Testing', leave=False):
                X_test, y_test = X_test.to(self.device), y_test.to(self.device)
                y_preds = self.model(X_test)
                _, preds = torch.max(y_preds, 1)
                correct += (preds == y_test).sum().item()
                total += y_test.size(0)

        return correct / total

    def fit(self, epochs=10, verbose=True, saved_model_path=None):
        start_time = time.time()
        for epoch in range(epochs):
            train_loss, train_acc, train_f1 = self.train_one_epoch()
            val_loss, val_acc, val_f1 = self.validate()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)

            if verbose and (epoch + 1) % (epochs // 10) == 0:
                print(f'EPOCH [{epoch+1}/{epochs}]'
                      f' Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}'
                      f' | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

            if self.early_stopping:
                if val_loss < self.best_val_loss: # Improvement
                    self.best_val_loss = val_loss
                    self.epochs_no_improve = 0
                    self.best_weights = self.model.state_dict()
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= self.early_stopping_patience:
                        print("=> Early stopping at epoch", epoch+1)
                        if self.best_weights is not None:
                            self.model.load_state_dict(self.best_weights)
                        break
        # Load best weights after training
        if self.best_weights is not None:
            self.model.load_state_dict(self.best_weights)

        if saved_model_path:
            torch.save(self.model.state_dict(), saved_model_path)

        print(f'Training time: {time.time() - start_time:.2f}s')

        return self.history