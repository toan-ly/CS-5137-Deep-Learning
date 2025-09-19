from sklearn.preprocessing import StandardScaler
from utils.evaluation import *
from utils.load_data import load_processed_data
from models.dnn import DNN
from models.linear_regression import LinearRegressionSklearn 
import torch
from torch.utils.data import DataLoader, Dataset
import time
from pathlib import Path

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
):
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
    plot_loss(train_losses, val_losses, log=True, title=title)

    return r2

time_stamp = time.strftime("%Y%m%d")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data' / 'processed'
MODEL_ROOT = ROOT / 'weights'
MODEL_ROOT.mkdir(parents=True, exist_ok=True)
data_path = str(DATA_DIR)
X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data(data_path)

batch_size = 64
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

epochs = 500
patience = 30
early_stopping = True
grad_clip = True

activation = 'leaky_relu'
learning_rate = 0.01
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
    'DNN-Custom': [64, 32, 16, 8]
}

ln_reg = LinearRegressionSklearn()
ln_reg.fit(X_train, y_train)
_, _, r2 = test_model(ln_reg, X_test, y_test)
save_linear_regression(ln_reg, path=MODEL_ROOT / f'{time_stamp}_linear_r2_{r2:.4f}.joblib')


results = {}
models = {}

dnn_16 = DNN(
    input_dim=X_train.shape[1],
    hidden_layers=architectures['DNN-16'],
    output_dim=1,
    lr=learning_rate,
    dropout=dropout,
    batch_norm=batch_norm,
    l2_lambda=l2_lambda,
    activation=activation, 
    weight_decay=weight_decay,
    momentum=momentum,
)
models['DNN-16'] = dnn_16
results['DNN-16'] = pipeline(dnn_16, epochs=epochs, early_stopping=early_stopping, patience=patience, title='DNN-16')


dnn_30_8 = DNN(
    input_dim=X_train.shape[1],
    hidden_layers=architectures['DNN-30-8'],
    output_dim=1,
    lr=learning_rate,
    dropout=dropout,
    batch_norm=batch_norm,
    l2_lambda=l2_lambda,
    activation=activation,
    weight_decay=weight_decay,
    momentum=momentum,
)
models['DNN-30-8'] = dnn_30_8
results['DNN-30-8'] = pipeline(dnn_30_8, epochs=epochs, early_stopping=early_stopping, patience=patience, title='DNN-30-8')

dnn_30_16_8 = DNN(
    input_dim=X_train.shape[1],
    hidden_layers=architectures['DNN-30-16-8'],
    output_dim=1,
    lr=learning_rate,
    dropout=dropout,
    batch_norm=batch_norm,
    l2_lambda=l2_lambda,
    activation=activation,
    weight_decay=weight_decay,
    momentum=momentum,
)
models['DNN-30-16-8'] = dnn_30_16_8
results['DNN-30-16-8'] = pipeline(dnn_30_16_8, epochs=epochs, early_stopping=early_stopping, patience=patience, title='DNN-30-16-8')

dnn_30_16_8_4 = DNN(
    input_dim=X_train.shape[1],
    hidden_layers=architectures['DNN-30-16-8-4'],
    output_dim=1,
    lr=learning_rate,
    dropout=dropout,
    batch_norm=batch_norm,
    l2_lambda=l2_lambda,
    activation=activation,
    weight_decay=weight_decay,
    momentum=momentum,
)
models['DNN-30-16-8-4'] = dnn_30_16_8_4
results['DNN-30-16-8-4'] = pipeline(dnn_30_16_8_4, epochs=epochs, early_stopping=early_stopping, patience=patience, title='DNN-30-16-8-4')

custom_dnn = DNN(
    input_dim=X_train.shape[1],
    hidden_layers=architectures['DNN-Custom'],
    output_dim=1,
    lr=learning_rate,
    dropout=dropout,
    batch_norm=batch_norm,
    l2_lambda=l2_lambda,
    activation=activation,
    weight_decay=weight_decay,
    momentum=momentum,
)
models['Custom DNN'] = custom_dnn
results['Custom DNN'] = pipeline(custom_dnn, epochs=epochs, early_stopping=early_stopping, patience=patience, title='Custom DNN')

best_model_name = max(results, key=results.get)
best_r2 = results[best_model_name]
best_model = models[best_model_name]
print(f'Best model: {best_model_name} with R²: {best_r2:.4f}')
save_path = MODEL_ROOT / f'{time_stamp}_{best_model_name}_r2_{best_r2:.4f}.pt'
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

