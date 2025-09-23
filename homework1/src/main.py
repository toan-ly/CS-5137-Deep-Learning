from unicodedata import name
from utils.evaluation import *
from utils.load_data import load_processed_data
from models.dnn import DNN
from models.linear_regression import LinearRegressionSklearn 
import torch
from torch.utils.data import DataLoader, Dataset
import time
from pathlib import Path
import numpy as np
import pandas as pd
import argparse

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def pipeline(
        model, 
        epochs=50, 
        verbose=True, 
        early_stopping=True, 
        patience=40, 
        grad_clip=True, 
        title=None,
        save_path=None,
):
    """
    Train, evaluate model and plot/save loss curve

    Args:
        model: model instance (DNN or Linear Regression)
        epochs: number of training epochs
        verbose: whether to print training progress
        early_stopping: whether to use early stopping
        patience: number of epochs to wait for improvement before stopping
        grad_clip: whether to use gradient clipping to prevent exploding gradients
        title: title for the loss plot
        save_path: path to save the loss plot
    Returns:
        r2: R² score on the test set
    """
    train_losses, val_losses = model.fit(
        train_loader, 
        val_loader, 
        epochs=epochs, 
        verbose=verbose, 
        early_stopping=early_stopping, 
        patience=patience, 
        grad_clip=grad_clip
    )
    _, _, r2 = test_model(model, X_test, y_test)
    if title is None:
        title = f'{model.__class__.__name__}'
    plot_loss(train_losses, val_losses, log=True, title=title, save_path=save_path)

    return r2

# --------------------- Paths -----------------------
time_stamp = time.strftime("%Y%m%d")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data' / 'processed'
MODEL_ROOT = ROOT / 'weights'
PLOT_ROOT = ROOT / 'figures'

MODEL_ROOT.mkdir(parents=True, exist_ok=True)
PLOT_ROOT.mkdir(parents=True, exist_ok=True)

# --------------------- Load Data -----------------------
data_path = str(DATA_DIR)
X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data(data_path)

# Create DataLoaders
batch_size = 128
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ---------------------- Hyperparameters ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
args = parser.parse_args()
learning_rate = args.lr

epochs = 1000
patience = 30
early_stopping = True
grad_clip = True

activation = 'leaky_relu'
batch_norm = True
dropout = 0.0
l2_lambda = 0.0
weight_decay = 0.0
momentum = 0.9

architectures = {
    'DNN-16': [16],
    'DNN-30-8': [30, 8],
    'DNN-30-16-8': [30, 16, 8],
    'DNN-30-16-8-4': [30, 16, 8, 4],
    # 'DNN-Custom': [64, 32, 16, 8]
}


# ---------------------- Linear Regression ----------------------
ln_reg = LinearRegressionSklearn()
ln_reg.fit(X_train, y_train)
_, _, r2_lr = test_model(ln_reg, X_test, y_test)
save_linear_regression(ln_reg, path=MODEL_ROOT / f'{time_stamp}_linear_r2_{r2_lr:.4f}.joblib')


# ---------------------- Deep Neural Network ----------------------
results = {}
models = {}

for model_name, hidden_layers in architectures.items():
    dnn = DNN(
        input_dim=X_train.shape[1],
        hidden_layers=hidden_layers,
        output_dim=1,
        lr=learning_rate,
        dropout=dropout,
        batch_norm=batch_norm,
        l2_lambda=l2_lambda,
        activation=activation, 
        weight_decay=weight_decay,
        momentum=momentum,
    )
    models[model_name] = dnn
    print(f'\nTraining {model_name}...')
    save_path = PLOT_ROOT / f'{time_stamp}_{learning_rate}_{model_name}_loss.png'
    loss_plot_title = f'{model_name} (Layers: {hidden_layers}, LR: {learning_rate})'
    results[model_name] = pipeline(dnn, epochs=epochs, early_stopping=early_stopping, patience=patience, title=loss_plot_title, save_path=save_path)

# ---------------------- Save Best Model -----------------
best_model_name = max(results, key=results.get)
best_r2 = results[best_model_name]
best_model = models[best_model_name]
print(f'Best model: {best_model_name} with r2: {best_r2:.4f}')
save_path = MODEL_ROOT / f'{time_stamp}_{learning_rate}_{best_model_name}_r2_{best_r2:.4f}.pt'
save_dnn(
    best_model, 
    path=save_path, 
    input_dim=X_train.shape[1], 
    hidden_layers=architectures[best_model_name], 
    output_dim=1,
    extra_config={
        'activation': activation,
        'batch_norm': batch_norm,
        'dropout': dropout,
        'l2_lambda': l2_lambda,
        'weight_decay': weight_decay,
        'momentum': momentum,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'patience': patience,
        'early_stopping': early_stopping,
        'grad_clip': grad_clip,
    }
)

# ---------------------- Summary ----------------------
summary = pd.DataFrame(
    [{"model": "Linear Regression", "learning rate": learning_rate, "r2": r2_lr}] + 
    [{"model": name, "learning rate": learning_rate, "r2": r2} for name, r2 in results.items()]
).sort_values(by='r2', ascending=False)

csv_path = PLOT_ROOT / 'model_performance_summary.csv'
if csv_path.exists():
    existing_summary = pd.read_csv(csv_path)
    summary = pd.concat([existing_summary, summary]).sort_values(by='r2', ascending=False)
else:
    summary = summary.sort_values(by='r2', ascending=False)
summary.to_csv(csv_path, index=False)
print(f'Model performance summary saved to {csv_path}')