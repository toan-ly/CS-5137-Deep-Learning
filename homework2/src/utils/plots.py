import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_loss_by_lr(loss_df, save_dir):
    """
    Plots training and validation loss, accuracy, and F1 score 
    over epochs for different models and learning rates.
    """
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
        axes[2].set_ylabel('F1 (Macro)')
        axes[2].legend(loc='upper right')


        plt.suptitle(f'{model} (lr={lr})', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_dir / f'{model}_{lr}.png')
        plt.close(fig)

def plot_roc_curve(y_true, probs, save_dir, model_name=None, lr=None):
    """
    Plots the ROC curve for multi-class classification.

    Args:
        y_true: true labels in ONE-HOT ENCODED format, shape (n_samples, n_classes)
        probs: predicted probabilities, shape (n_samples, n_classes)
        save_path: path to save the ROC curve plot
    """
    n_classes = probs.shape[1]

    save_path = save_dir / 'roc_curve' 
    save_path.mkdir(parents=True, exist_ok=True)

    tpr, fpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0, 1, 1000)
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])

    mean_tpr /= n_classes
    mean_auc = auc(fpr_grid, mean_tpr)

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=1, label=f'Class {i} (auc = {roc_auc[i]:.4f})')
    plt.plot(fpr_grid, mean_tpr, color='b', lw=2, linestyle='--',
             label=f'Mean ROC (auc = {mean_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC Curve ({model_name} - {lr})')
    plt.legend(loc='lower right')

    # Add zoom in since all classes are very close at top left corner
    ax_zoom = inset_axes(plt.gca(), width="32%", height="32%", loc='upper left',
                        bbox_to_anchor=(0.1, -0.05, 1, 1),
                        bbox_transform=plt.gca().transAxes, borderpad=1.5)
    for i in range(n_classes):
        ax_zoom.plot(fpr[i], tpr[i], lw=1)
    ax_zoom.plot(fpr_grid, mean_tpr, color='b', lw=2, linestyle='--')
    ax_zoom.set_xlim(0, 0.08)
    ax_zoom.set_ylim(0.96, 1.0)

    plt.tight_layout()
    plt.savefig(save_path / f'roc_curve_{model_name}_{lr}.png')
    plt.close()