"""Shared utilities: seeding, data loading, metrics, inference."""

from __future__ import annotations

import random

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def encode_labels(y_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Encode raw labels into 0..K-1 integers.

    Returns
    -------
    encoded : ndarray of int64
    classes : ndarray of the original class values
    class_to_idx : dict mapping original → encoded
    """
    classes = np.array(sorted(np.unique(y_raw)))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    encoded = np.array([class_to_idx[c] for c in y_raw], dtype=np.int64)
    return encoded, classes, class_to_idx


def make_loader(
    X: np.ndarray,
    y: np.ndarray | None = None,
    batch_size: int = 32,
    shuffle: bool = False,
) -> DataLoader:
    X_t = torch.tensor(X, dtype=torch.float32)
    if y is None:
        return DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=shuffle)
    return DataLoader(
        TensorDataset(X_t, torch.tensor(y, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on a DataLoader.

    Returns
    -------
    preds : ndarray of int  — argmax predictions
    confs : ndarray of float — max probability per sample
    probs : ndarray of float  — full probability matrix (N, C)
    """
    model.eval()
    all_logits = []
    for batch in loader:
        all_logits.append(model(batch[0].to(device)).cpu())
    logits = torch.cat(all_logits, dim=0)
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)
    confs = probs.max(axis=1)
    return preds, confs, probs
