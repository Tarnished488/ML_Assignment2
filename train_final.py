#!/usr/bin/env python
"""基于GitHub最新代码的训练脚本 - 多seed自训练."""
"""此版本稳定在0.475-0.47800，配置不同则不同."""
"""0.47800配置：阈值0.9 seed50个"""

import os
os.environ['ASSIGNMENT2_DATA_DIR'] = 'ML_Assignment2/data'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from sklearn.metrics import f1_score

# 加载数据
X_labeled = pd.read_csv("data/train_labeled_features.csv").values
y_labeled = pd.read_csv("data/train_labeled_labels.csv").values.ravel()
X_unlabeled = pd.read_csv("data/train_unlabeled_features.csv").values
X_test_df = pd.read_csv("data/test_features.csv")
test_ids = X_test_df['Id'].values if 'Id' in X_test_df.columns else np.arange(len(X_test_df))
X_test = X_test_df.drop(columns=['Id']).values if 'Id' in X_test_df.columns else X_test_df.values

print(f"Data: Labeled={X_labeled.shape}, Unlabeled={X_unlabeled.shape}, Test={X_test.shape}")

# 恢复原MLP 256
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

from sklearn.metrics import f1_score, accuracy_score

def compute_metrics(y_true, y_pred):
    """计算accuracy和macro_f1"""
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

def run_training(x_label, y_label, seed, use_pseudo=False):
    """直接用100个样本训练，不用self-training"""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 划分训练/验证集用于计算指标
    indices = np.random.permutation(len(x_label))
    val_size = int(0.2 * len(x_label))
    train_idx, val_idx = indices[val_size:], indices[:val_size]

    x_train, y_train = x_label[train_idx], y_label[train_idx]
    x_val, y_val = x_label[val_idx], y_label[val_idx]

    # 训练
    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    model = MLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(30):
        train_epoch(model, train_loader, optimizer, criterion)

    # 用验证集评估
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

    # 最终refit - 用全部100个原始labeled data
    train_ds = TensorDataset(x_label, y_label)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    model = MLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(30):
        train_epoch(model, train_loader, optimizer, criterion)

    # 预测
    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test, dtype=torch.float32)
        probs = torch.softmax(model(test_tensor), dim=1)
        predictions = torch.argmax(probs, dim=1).numpy()

    return predictions, val_acc, val_f1

def run_self_training(x_label, y_label, x_unlabel, seed, threshold=0.90, max_rounds=5):
    """标准self-training，返回预测结果和验证指标"""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 划分训练/验证集
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

    # 验证集评估
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

    return predictions, val_acc, val_f1

# 主程序 - 直接用最佳配置
x_label = torch.tensor(X_labeled, dtype=torch.float32)
y_label = torch.tensor(y_labeled, dtype=torch.long)
x_unlabel = torch.tensor(X_unlabeled, dtype=torch.float32)

# 10个seeds
np.random.seed(12345)
SEEDS = np.random.randint(0, 99999, 30).tolist()  # 10个随机seeds
threshold = 0.9

print("=== 版本1: 纯100样本训练 ===")
all_preds_no_st = []
all_val_acc_no_st = []
all_val_f1_no_st = []
for seed in SEEDS:
    preds, val_acc, val_f1 = run_training(x_label, y_label, seed)
    all_preds_no_st.append(preds)
    all_val_acc_no_st.append(val_acc)
    all_val_f1_no_st.append(val_f1)
    print(f"Seed {seed}: acc={val_acc:.4f} f1={val_f1:.4f}", end=" ")
print()
print(f"V1 平均: acc={np.mean(all_val_acc_no_st):.4f} f1={np.mean(all_val_f1_no_st):.4f} (std acc={np.std(all_val_acc_no_st):.4f})")

all_preds_no_st = np.array(all_preds_no_st)
final_preds_no_st = stats.mode(all_preds_no_st, axis=0, keepdims=False)[0]

print("\n=== 版本2: Self-Training ===")
all_preds_st = []
all_val_acc_st = []
all_val_f1_st = []
for seed in SEEDS:
    preds, val_acc, val_f1 = run_self_training(x_label, y_label, x_unlabel, seed, threshold=threshold)
    all_preds_st.append(preds)
    all_val_acc_st.append(val_acc)
    all_val_f1_st.append(val_f1)
    print(f"Seed {seed}: acc={val_acc:.4f} f1={val_f1:.4f}", end=" ")
print()
print(f"V2 平均: acc={np.mean(all_val_acc_st):.4f} f1={np.mean(all_val_f1_st):.4f} (std acc={np.std(all_val_acc_st):.4f})")

all_preds_st = np.array(all_preds_st)
final_preds_st = stats.mode(all_preds_st, axis=0, keepdims=False)[0]

# 保存
os.makedirs("ML_Assignment2/outputs/github_final", exist_ok=True)
submission1 = pd.DataFrame({'Id': test_ids, 'Category': final_preds_no_st})
submission1.to_csv("ML_Assignment2/outputs/github_final/submission_v1_pure100.csv", index=False)
submission2 = pd.DataFrame({'Id': test_ids, 'Category': final_preds_st})
submission2.to_csv("ML_Assignment2/outputs/github_final/submission_v2_st.csv", index=False)

diff = np.sum(final_preds_no_st != final_preds_st)
print(f"\n差异: {diff} ({diff/8000*100:.1f}%)")

# 投票融合v1+v2
all_preds = np.vstack([all_preds_no_st, all_preds_st])
final_preds = stats.mode(all_preds, axis=0, keepdims=False)[0]

submission = pd.DataFrame({'Id': test_ids, 'Category': final_preds})
submission.to_csv("ML_Assignment2/outputs/github_final/submission.csv", index=False)

print("\n=== Final Distribution ===")
unique, counts = np.unique(final_preds, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Class {u}: {c} ({c/len(final_preds)*100:.1f}%)")

print("\nSaved: submission.csv (fusion)")