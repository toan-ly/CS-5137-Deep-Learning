import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from .utils import *
from .gcn import *
from .preprocess import *

from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def run_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    train=True,
):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_logits = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)

        if train:
            optimizer.zero_grad()

        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, batch.y)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * batch.y.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(batch.y.detach().cpu())

    total_graphs = len(loader.dataset)
    avg_loss = total_loss / total_graphs

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    acc, f1, auc_score = compute_metrics(all_targets, all_logits)

    return avg_loss, acc, f1, auc_score


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    save_path: str | None = None,
    patience: int = 30,
    plot_path: str | None = None,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
    )

    best_val_f1 = -1.0
    best_epoch = 0

    train_losses, val_losses = [], []
    epochs_list = []
    
    pbar = tqdm(range(1, epochs + 1) , desc="Training", leave=False, ncols=150)
    for epoch in pbar:
        train_loss, train_acc, train_f1, train_auc = run_one_epoch(
            model, train_loader, optimizer, criterion, device, train=True
        )

        with torch.no_grad():
            val_loss, val_acc, val_f1, val_auc = run_one_epoch(
                model, val_loader, None, criterion, device, train=False
            )

        scheduler.step(val_f1)

        pbar.set_postfix({
            "train_loss": f"{train_loss:.2f}",
            "train_acc": f"{train_acc:.2f}",
            "train_f1": f"{train_f1:.2f}",
            "val_loss": f"{val_loss:.2f}",
            "val_acc": f"{val_acc:.2f}",
            "val_f1": f"{val_f1:.2f}",
        })

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epochs_list.append(epoch)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            if epoch - best_epoch >= patience:
                print(f"\n=> Early stopping at epoch {epoch}")
                break

    print(f"\nBest validation F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # Plot loss
    if plot_path:
        plt.figure(figsize=(8, 6))
        plt.plot(epochs_list, train_losses, label='Train Loss')
        plt.plot(epochs_list, val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss over Epochs')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()



def test_model(
    model,
    test_loader,
    device=torch.device('cpu'),
    plot_path: str | None = None,
):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y)

            total_loss += loss.item() * batch.y.size(0)
            all_logits.append(logits.detach().cpu())
            all_targets.append(batch.y.detach().cpu())

    total_graphs = len(test_loader.dataset)
    avg_loss = total_loss / total_graphs

    all_logits = torch.cat(all_logits, dim=0)   # [N, C]
    all_targets = torch.cat(all_targets, dim=0) # [N]

    acc, f1, auc_score = compute_metrics(all_targets, all_logits)

    print("=== Test performance ===")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"AUC: {auc_score:.4f}")


    if plot_path is not None:
        probs = F.softmax(all_logits, dim=1).numpy()
        y_true = all_targets.numpy()
        n_classes = probs.shape[1]

        # Labels in one-hot encoded format
        y_true_onehot = label_binarize(y_true, classes=list(range(n_classes)))

        tpr, fpr, roc_auc = {}, {}, {}

        # Per-class ROC
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Micro-average ROC
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_true_onehot.ravel(), probs.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            linestyle='--',
            label=f"micro-average ROC (AUC = {roc_auc['micro']:.3f})",
        )

        # Plot per-class ROC
        for i in range(n_classes):
            plt.plot(
                fpr[i],
                tpr[i],
                label=f"Class {i} (AUC = {roc_auc[i]:.3f})",
            )

        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title("ROC curves")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    return avg_loss, acc, f1, auc_score
    


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ROOT = Path(__file__).parent.parent 
    DATA_DIR = ROOT / "data" / "ENZYMES"
    data_list, num_node_features = preprocess(DATA_DIR)

    weights_path = ROOT / "weights" 
    os.makedirs(weights_path, exist_ok=True)

    loss_path = ROOT / "results"
    os.makedirs(loss_path, exist_ok=True)

    print(f"# graphs: {len(data_list)}")
    print(f"# node features: {num_node_features}")

    # Hyperparameters
    epochs = 300
    lr = 5e-4
    weight_decay = 1e-4
    patience = 30
    hidden_channels = 256
    out_channels = 6
    dropout = 0.3
    batch_size = 32
    gcn_pooling = 'concat'  

    train_loader, val_loader, test_loader = get_dataloaders(
        data_list, 
        batch_size=batch_size, 
        train_size=0.8,
    )
    
    models = {
        'gcn[1]': GCN(
            in_dim=num_node_features,
            hidden_dim=hidden_channels,
            out_dim=out_channels,
            num_layers=1,   
            dropout=dropout,
            pooling=gcn_pooling,
        ),
        'gcn[2]': GCN(
            in_dim=num_node_features,
            hidden_dim=hidden_channels,
            out_dim=out_channels,
            num_layers=2,   
            dropout=dropout,
            pooling=gcn_pooling,
        ),
        'gcn[3]': GCN(
            in_dim=num_node_features,
            hidden_dim=hidden_channels,
            out_dim=out_channels,
            num_layers=3,   
            dropout=dropout,
            pooling=gcn_pooling,
        ),
    }

    results_summary = []
    for model_name, model in models.items():
        print(f"\n=== Training model: {model_name} ===")
        model = model.to(device)

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"# trainable parameters: {params}")

        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            save_path=os.path.join(weights_path, f"{model_name}.pt"),
            plot_path=os.path.join(loss_path, f"{model_name}_loss.png"),
        )

        # Evaluate best checkpoint on test set + plot ROC curve
        model.load_state_dict(torch.load(os.path.join(weights_path, f"{model_name}.pt"), map_location=device))

        test_loss, test_acc, test_f1, test_auc = test_model(
            model,
            test_loader,
            device=device,
            plot_path=os.path.join(loss_path, f"{model_name}_roc.png"),
        )
        results_summary.append({
            'model': model_name,
            'params': params,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'test_auc': test_auc,
        })

    # Print summary of all models
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<10} {'Params':<12} {'Acc':<8} {'F1':<8} {'AUC':<8}")
    print(f"{'-'*60}")
    for result in results_summary:
        print(f"{result['model']:<10} {result['params']:<12,} {result['test_acc']:<8.4f} {result['test_f1']:<8.4f} {result['test_auc']:<8.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()