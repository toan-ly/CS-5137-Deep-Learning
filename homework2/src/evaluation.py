import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from utils import *
import tqdm
from sklearn.metrics import f1_score, roc_auc_score
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
        hidden_layers=[128, 64],
        output_dim=output_dim,
        dropout=dropout,
        batch_norm=batch_norm,
        activation=activation,
    ),
    'CNN': CNN(
        input_channels=input_channels,
        conv_layers=[16, 32],
        fc_layers=[64],
        output_dim=output_dim,
        dropout=dropout,
        batch_norm=batch_norm,
        activation=activation,
    ),
    'ResNet18': ResNet(
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

def softmax(logits):
    # To make softmax more numerically stable
    z = logits - np.max(logits, axis=1, keepdims=True)
    e_z = np.exp(z)
    return e_z / np.sum(e_z, axis=1, keepdims=True)

@torch.no_grad()
def test_model(model, test_loader):
    """
    Evaluate the model on the test set
    """
    model.eval()
    correct, total = 0, 0
    all_preds, all_gts = [], []
    all_logits = []
    with torch.no_grad():
        for X_test, y_test in tqdm.tqdm(test_loader, desc='Testing', leave=False):
            X_test, y_test = X_test.to(device), y_test.to(device)
            logits = model(X_test) # (batch_size, num_classes)
            all_logits.append(logits.detach().cpu().numpy())
            
            preds = logits.argmax(dim=1)
            correct += (preds == y_test).sum().item()
            total += y_test.size(0)

            all_preds.extend(preds.detach().cpu().numpy())
            all_gts.extend(y_test.detach().cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0) # Combine logits from all batches (num_samples, num_classes)
    probs = softmax(all_logits)
    y_true = np.array(all_gts)
    y_true = np.eye(output_dim)[y_true]  # One-hot encode true labels

    test_acc = correct / total
    test_f1_weighted = f1_score(all_gts, all_preds, average='weighted', zero_division=0)
    test_f1_macro = f1_score(all_gts, all_preds, average='macro', zero_division=0)

    # Compute ROC AUC for multi-class using one-vs-rest
    test_roc_auc_macro = roc_auc_score(y_true, probs, average='macro', multi_class='ovr')
    test_roc_auc_micro = roc_auc_score(y_true, probs, average='micro', multi_class='ovr')

    return {
        'test_acc': test_acc,
        'test_f1_weighted': test_f1_weighted,
        'test_f1_macro': test_f1_macro,
        'test_roc_auc_macro': test_roc_auc_macro,
        'test_roc_auc_micro': test_roc_auc_micro,
        'y_true': y_true,
        'probs': probs,
    }

if __name__ == "__main__":
    ROOT = Path(__file__).parent.parent
    DATA_DIR = ROOT / 'data'
    RESULTS_DIR = ROOT / 'results'
    MODEL_DIR = ROOT / 'weights'

    history = pd.read_csv(RESULTS_DIR / 'histories.csv')
    plot_loss_by_lr(history, save_dir=RESULTS_DIR)

    test_loader = load_test_data(data_dir=str(DATA_DIR), batch_size=512, use_cached=True)
    model_files = glob.glob(str(MODEL_DIR / '*.pth'))
    performance_records = []
    for model_file in model_files:
        model_name, lr = Path(model_file).stem.split('_') 
        if model_name in MODELS:
            model = MODELS[model_name] 
            model = load_model(model, model_file, device)
            results = test_model(model, test_loader)
            print(f'Model: {model_name} | Learning Rate: {lr}')
            print(f'    Test Accuracy: {results["test_acc"]:.4f}')
            print(f'    Test F1 Weighted: {results["test_f1_weighted"]:.4f}')
            print(f'    Test F1 Macro: {results["test_f1_macro"]:.4f}')
            print(f'    Test AUC Macro: {results["test_roc_auc_macro"]:.4f}')
            print(f'    Test AUC Micro: {results["test_roc_auc_micro"]:.4f}')
            print('-----------------------------------')

            plot_roc_curve(
                results['y_true'], 
                results['probs'], 
                save_dir=RESULTS_DIR, 
                model_name=model_name, 
                lr=lr
            )

            performance_records.append({
                'model': model_name,
                'lr': lr,
                'accuracy': round(results['test_acc'], 4),
                'f1_weighted': round(results['test_f1_weighted'], 4),
                'f1_macro': round(results['test_f1_macro'], 4),
                'auc_macro_ovr': round(results['test_roc_auc_macro'], 4),
                'auc_micro_ovr': round(results['test_roc_auc_micro'], 4),
            })

    performance_df = pd.DataFrame(performance_records).sort_values(by=['model', 'lr']).reset_index(drop=True)
    performance_df.to_csv(RESULTS_DIR / 'performance_summary.csv', index=False)







