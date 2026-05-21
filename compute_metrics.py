#!/usr/bin/env python
"""Compute and save f1/macro_f1 metrics for train_final.py experiments."""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score

# Load data
X_labeled = pd.read_csv("data/train_labeled_features.csv").values
y_labeled = pd.read_csv("data/train_labeled_labels.csv").values.ravel()
X_unlabeled = pd.read_csv("data/train_unlabeled_features.csv").values
X_test_df = pd.read_csv("data/test_features.csv")
test_ids = X_test_df['Id'].values if 'Id' in X_test_df.columns else np.arange(len(X_test_df))
X_test = X_test_df.drop(columns=['Id']).values if 'Id' in X_test_df.columns else X_test_df.values

print(f"Data: Labeled={X_labeled.shape}, Unlabeled={X_unlabeled.shape}, Test={X_test.shape}")


class MLP(nn.Module):
    def __init__(self, in_dim=512, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x.float())


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, f1


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def run_training(x_label, y_label, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    indices = np.random.permutation(len(x_label))
    val_size = int(0.2 * len(x_label))
    train_idx, val_idx = indices[val_size:], indices[:val_size]

    x_train, y_train = x_label[train_idx], y_label[train_idx]
    x_val, y_val = x_label[val_idx], y_label[val_idx]

    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    model = MLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(30):
        train_epoch(model, train_loader, optimizer, criterion)

    model.eval()
    val_preds = []
    with torch.no_grad():
        for i in range(0, len(x_val), 32):
            batch = x_val[i:i+32]
            logits = model(batch)
            preds = torch.argmax(logits, dim=1).numpy()
            val_preds.extend(preds)
    val_preds = np.array(val_preds)
    val_acc, val_f1 = compute_metrics(y_val.numpy(), val_preds)

    return val_acc, val_f1


def run_self_training(x_label, y_label, x_unlabel, seed, threshold=0.90, max_rounds=5):
    np.random.seed(seed)
    torch.manual_seed(seed)

    indices = np.random.permutation(len(x_label))
    val_size = int(0.2 * len(x_label))
    train_idx, val_idx = indices[val_size:], indices[:val_size]

    x_train_label = x_label[train_idx]
    y_train_label = y_label[train_idx]
    x_val = x_label[val_idx]
    y_val = y_label[val_idx]

    current_x = x_train_label.clone()
    current_y = y_train_label.clone()
    current_unlabel = x_unlabel.clone()

    for round_idx in range(max_rounds):
        train_ds = TensorDataset(current_x, current_y)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

        model = MLP()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(25):
            train_epoch(model, train_loader, optimizer, criterion)

        if round_idx < max_rounds - 1:
            model.eval()
            with torch.no_grad():
                probs = torch.softmax(model(current_unlabel), dim=1)
                max_probs, pseudo_labels = torch.max(probs, dim=1)

            mask = max_probs >= threshold
            selected_x = current_unlabel[mask]
            selected_y = pseudo_labels[mask]

            if selected_x.shape[0] == 0:
                break

            current_x = torch.cat([current_x, selected_x])
            current_y = torch.cat([current_y, selected_y])
            current_unlabel = current_unlabel[~mask]

    model.eval()
    val_preds = []
    with torch.no_grad():
        for i in range(0, len(x_val), 32):
            batch = x_val[i:i+32]
            logits = model(batch)
            preds = torch.argmax(logits, dim=1).numpy()
            val_preds.extend(preds)
    val_preds = np.array(val_preds)
    val_acc, val_f1 = compute_metrics(y_val.numpy(), val_preds)

    return val_acc, val_f1


# Main
x_label = torch.tensor(X_labeled, dtype=torch.float32)
y_label = torch.tensor(y_labeled, dtype=torch.long)
x_unlabel = torch.tensor(X_unlabeled, dtype=torch.float32)

np.random.seed(12345)
SEEDS = np.random.randint(0, 99999, 30).tolist()
threshold = 0.90

results = []

# Version 1
print("=== Version 1: Pure 100-sample training ===")
for seed in SEEDS:
    val_acc, val_f1 = run_training(x_label, y_label, seed)
    results.append({'version': 'v1', 'seed': seed, 'acc': val_acc, 'macro_f1': val_f1})
    print(f"Seed {seed}: acc={val_acc:.4f} f1={val_f1:.4f}")

# Version 2
print("\n=== Version 2: Self-Training ===")
for seed in SEEDS:
    val_acc, val_f1 = run_self_training(x_label, y_label, x_unlabel, seed, threshold=threshold)
    results.append({'version': 'v2', 'seed': seed, 'acc': val_acc, 'macro_f1': val_f1})
    print(f"Seed {seed}: acc={val_acc:.4f} f1={val_f1:.4f}")

# Save to CSV
df = pd.DataFrame(results)
os.makedirs("ML_Assignment2/outputs/github_final", exist_ok=True)
df.to_csv("ML_Assignment2/outputs/github_final/metrics.csv", index=False)

# Summary
print("\n=== Summary ===")
for ver in ['v1', 'v2']:
    subset = df[df['version'] == ver]
    print(f"{ver}: mean_acc={subset['acc'].mean():.4f} std={subset['acc'].std():.4f} mean_f1={subset['macro_f1'].mean():.4f}")

print(f"\nSaved: ML_Assignment2/outputs/github_final/metrics.csv")