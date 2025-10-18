import pandas as pd
import matplotlib.pyplot as plt


def plot_loss_by_lr(loss_df, save_dir):
    save_dir = save_dir / 'loss_plots'
    save_dir.mkdir(parents=True, exist_ok=True)

    for (model, lr), group in loss_df.groupby(['model', 'lr']):
        group = group.sort_values(by='epoch')
        epochs = group['epoch'].values

        fig, axes = plt.subplots(1, 2, figsize=(6*2, 5))

        axes[0].plot(epochs, group['train_loss'], label='Train Loss')
        axes[0].plot(epochs, group['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend(loc='upper right')

        axes[1].plot(epochs, group['train_acc'], label='Train Accuracy')
        axes[1].plot(epochs, group['val_acc'], label='Val Accuracy')
        axes[1].plot(epochs, group['train_f1'], label='Train F1', linestyle='--')
        axes[1].plot(epochs, group['val_f1'], label='Val F1', linestyle='--')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend(loc='upper right')


        plt.suptitle(f'{model} (lr={lr})', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_dir / f'{model}_{lr}.png')
        plt.close(fig)

