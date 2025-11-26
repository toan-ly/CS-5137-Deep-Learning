import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def compute_metrics(
    y_true: torch.Tensor,
    logits: torch.Tensor,
):
    probs = F.softmax(logits, dim=1)
    y_true = y_true.cpu().numpy()
    y_prob = probs.detach().cpu().numpy()
    y_pred = probs.argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    return acc, f1, auc

