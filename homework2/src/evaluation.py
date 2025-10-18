import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from utils import load_model, load_test_data, plot_loss_by_lr
import tqdm
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc
import glob 

from models import DNN, CNN, ResNet, VGG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim = 28*28
input_channels = 1
output_dim = 10
epochs = 10
activation = 'relu'
dropout = 0.0
batch_norm = True

MODELS = {
    'DNN': DNN(
        input_dim=input_dim,
        hidden_layers=[256, 128],
        output_dim=output_dim,
        dropout=dropout,
        batch_norm=batch_norm,
        activation=activation,
    ),
    'CNN': CNN(
        input_channels=input_channels,
        conv_layers=[16, 32],
        fc_layers=[256],
        output_dim=output_dim,
        dropout=dropout,
        batch_norm=batch_norm,
        activation=activation,
    ),
    'ResNet19': ResNet(
        input_channels=input_channels,
        n_blocks=[2, 2, 2, 2],
        output_dim=output_dim,
    ),
    'VGG16': VGG(
        input_channels=input_channels,
        output_dim=output_dim,
        batch_norm=batch_norm,
        fc_layers=[128],
        dropout=dropout,
    ),
}

@torch.no_grad()
def test(model, test_loader):
    """
    Evaluate the model on the test set
    """
    model.eval()
    correct, total = 0, 0
    all_preds, all_gts = [], []
    with torch.no_grad():
        for X_test, y_test in tqdm.tqdm(test_loader, desc='Testing', leave=False):
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_preds = model(X_test)
            _, preds = torch.max(y_preds, 1)
            correct += (preds == y_test).sum().item()
            total += y_test.size(0)

            all_preds.extend(preds.detach().cpu().numpy())
            all_gts.extend(y_test.detach().cpu().numpy())
    
    test_acc = correct / total
    test_f1_weighted = f1_score(all_gts, all_preds, average='weighted', zero_division=0)
    test_f1_macro = f1_score(all_gts, all_preds, average='macro', zero_division=0)
    test_roc_auc = 0

    return test_acc, test_f1_weighted, test_f1_macro, test_roc_auc, all_preds, all_gts

if __name__ == "__main__":
    ROOT = Path(__file__).parent.parent
    DATA_DIR = ROOT / 'data'
    RESULTS_DIR = ROOT / 'results'
    MODEL_DIR = ROOT / 'weights'

    history = pd.read_csv(RESULTS_DIR / 'histories.csv')
    plot_loss_by_lr(history, save_dir=RESULTS_DIR)

    test_loader = load_test_data(data_dir=str(DATA_DIR), batch_size=512, use_cached=True)
    model_files = glob.glob(str(MODEL_DIR / '*.pth'))
    for model_file in model_files:
        model_name, lr = Path(model_file).stem.split('_') 
        if model_name in MODELS:
            model = MODELS[model_name] 
            model = load_model(model, model_file, device)
            test_acc, test_f1_weighted, test_f1_macro, test_roc_auc, all_preds, all_gts = test(model, test_loader)
            print(f'Model: {model_name} | Learning Rate: {lr}')
            print(f'Test Accuracy: {test_acc:.4f}')
            print(f'Test F1 Weighted: {test_f1_weighted:.4f}')
            print(f'Test F1 Macro: {test_f1_macro:.4f}')
            print(f'Test ROC AUC: {test_roc_auc:.4f}')
            print('-----------------------------------')




