import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import pandas as pd
import matplotlib.pyplot as plt

def load_test_data(data_dir='../data', batch_size=512, use_cached=True):
    if use_cached:
        X = torch.load(f'{data_dir}/X_test.pt', map_location='cpu')
        y = torch.load(f'{data_dir}/y_test.pt', map_location='cpu')
        test_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)
    else:
        T = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        test_dataset = datasets.MNIST(root=data_dir, train=False, transform=T, download=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def plot_loss_by_lr(loss_df, save_dir):
    save_dir = save_dir / 'loss_plots'
    save_dir.mkdir(parents=True, exist_ok=True)

    for (model, lr), group in loss_df.groupby(['model', 'lr']):
        group = group.sort_values(by='epoch')
        epochs = group['epoch'].values

        fig, axes = plt.subplots(1, 3, figsize=(6*3, 5))

        axes[0].plot(epochs, group['train_loss'], label='Train Loss')
        axes[0].plot(epochs, group['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend(loc='upper right')

        axes[1].plot(epochs, group['train_acc'], label='Train Accuracy')
        axes[1].plot(epochs, group['val_acc'], label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend(loc='upper right')

        axes[2].plot(epochs, group['train_f1'], label='Train F1')
        axes[2].plot(epochs, group['val_f1'], label='Val F1')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1')
        axes[2].legend(loc='upper right')


        plt.suptitle(f'{model} (lr={lr})', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_dir / f'{model}_lr{lr}.png')
        plt.close(fig)

