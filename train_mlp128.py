#!/usr/bin/env python
"""架构: 512→128→10, 多seed自训练."""
"""此版本稳定在0.47625"""

import os
os.environ['ASSIGNMENT2_DATA_DIR'] = 'data'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats

# 加载数据
X_labeled = pd.read_csv("data/train_labeled_features.csv").values
y_labeled = pd.read_csv("data/train_labeled_labels.csv").values.ravel()
X_unlabeled = pd.read_csv("data/train_unlabeled_features.csv").values
X_test_df = pd.read_csv("data/test_features.csv")
test_ids = X_test_df['Id'].values if 'Id' in X_test_df.columns else np.arange(len(X_test_df))
X_test = X_test_df.drop(columns=['Id']).values if 'Id' in X_test_df.columns else X_test_df.values

print(f"Data: Labeled={X_labeled.shape}, Unlabeled={X_unlabeled.shape}, Test={X_test.shape}")

# 高分团队MLP架构: 512→128→10
class MLP(nn.Module):
    def __init__(self, in_dim=512, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x.float())

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
    """纯100样本训练"""
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_ds = TensorDataset(x_label, y_label)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    model = MLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(30):
        train_epoch(model, train_loader, optimizer, criterion)

    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test, dtype=torch.float32)
        probs = torch.softmax(model(test_tensor), dim=1)
        predictions = torch.argmax(probs, dim=1).numpy()

    return predictions

def run_self_training(x_label, y_label, x_unlabel, seed, threshold=0.90, max_rounds=5):
    """标准self-training"""
    np.random.seed(seed)
    torch.manual_seed(seed)

    current_x = x_label.clone()
    current_y = y_label.clone()
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
                print(f"  Seed {seed}: round {round_idx+1} no pseudo-labels (threshold={threshold})")
                break

            print(f"  Seed {seed}: round {round_idx+1} added {selected_x.shape[0]} pseudo-labels")
            current_x = torch.cat([current_x, selected_x])
            current_y = torch.cat([current_y, selected_y])
            current_unlabel = current_unlabel[~mask]

    # 最终refit - 用全部100个原始labeled data
    train_ds = TensorDataset(x_label, y_label)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    model = MLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(30):
        train_epoch(model, train_loader, optimizer, criterion)

    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test, dtype=torch.float32)
        probs = torch.softmax(model(test_tensor), dim=1)
        predictions = torch.argmax(probs, dim=1).numpy()

    return predictions

# 主程序
x_label = torch.tensor(X_labeled, dtype=torch.float32)
y_label = torch.tensor(y_labeled, dtype=torch.long)
x_unlabel = torch.tensor(X_unlabeled, dtype=torch.float32)

# 30个随机seeds测试
np.random.seed(12345)
SEEDS = np.random.randint(0, 99999, 10).tolist()
threshold = 0.97  # 用之前更好的0.88

print("=== 版本1: 纯100样本训练 (MLP128) ===")
all_preds_no_st = []
for seed in SEEDS:
    preds = run_training(x_label, y_label, seed)
    all_preds_no_st.append(preds)
print(f"  Done: {len(SEEDS)} seeds")

all_preds_no_st = np.array(all_preds_no_st)
final_preds_no_st = stats.mode(all_preds_no_st, axis=0, keepdims=False)[0]

print("\n=== 版本2: Self-Training (MLP128, threshold=0.88) ===")
all_preds_st = []
for seed in SEEDS:
    preds = run_self_training(x_label, y_label, x_unlabel, seed, threshold=threshold)
    all_preds_st.append(preds)
print(f"  Done: {len(SEEDS)} seeds")

all_preds_st = np.array(all_preds_st)
final_preds_st = stats.mode(all_preds_st, axis=0, keepdims=False)[0]

# 保存
os.makedirs("outputs/mlp128_final", exist_ok=True)
submission1 = pd.DataFrame({'Id': test_ids, 'Category': final_preds_no_st})
submission1.to_csv("outputs/mlp128_final/submission_v1_pure100.csv", index=False)
submission2 = pd.DataFrame({'Id': test_ids, 'Category': final_preds_st})
submission2.to_csv("outputs/mlp128_final/submission_v2_st.csv", index=False)

diff = np.sum(final_preds_no_st != final_preds_st)
print(f"\n差异: {diff} ({diff/8000*100:.1f}%)")

# 投票融合v1+v2
all_preds = np.vstack([all_preds_no_st, all_preds_st])
final_preds = stats.mode(all_preds, axis=0, keepdims=False)[0]

submission = pd.DataFrame({'Id': test_ids, 'Category': final_preds})
submission.to_csv("outputs/mlp128_final/submission.csv", index=False)

print("\n=== Final Distribution ===")
unique, counts = np.unique(final_preds, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Class {u}: {c} ({c/len(final_preds)*100:.1f}%)")

print("\nSaved: outputs/mlp128_final/submission.csv")