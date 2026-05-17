import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp import build_mlp
from src.preprocessing.data_loader import get_train_val_split, load_test_data, load_unlabeled_data


def parse_hidden_dims(value):
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def encode_labels(y):
    classes = np.array(sorted(np.unique(y)))
    class_to_index = {label: idx for idx, label in enumerate(classes)}
    encoded = np.array([class_to_index[label] for label in y], dtype=np.int64)
    return encoded, classes


def make_loader(X, y=None, batch_size=32, shuffle=False):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    if y is None:
        dataset = TensorDataset(X_tensor)
    else:
        y_tensor = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def accuracy_and_macro_f1(y_true, y_pred, num_classes):
    accuracy = float(np.mean(y_true == y_pred))
    f1_scores = []
    for cls in range(num_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        f1_scores.append(f1)
    return accuracy, float(np.mean(f1_scores))


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_count = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        total_count += X_batch.size(0)

    return total_loss / total_count


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_logits = []
    for batch in loader:
        X_batch = batch[0].to(device)
        all_logits.append(model(X_batch).cpu())
    logits = torch.cat(all_logits, dim=0)
    probabilities = torch.softmax(logits, dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    return predictions, confidences, probabilities


def train_model(args, X_train, y_train, X_val, y_val, num_classes, device):
    model = build_mlp(
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        dropout=args.dropout,
        activation=args.activation,
    ).to(device)

    train_loader = make_loader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, args.batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
    )

    best_state = None
    best_val_acc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_pred, _, _ = predict(model, val_loader, device)
        val_acc, val_macro_f1 = accuracy_and_macro_f1(y_val, val_pred, num_classes)
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d} | loss={train_loss:.4f} "
                f"| val_acc={val_acc:.4f} | val_macro_f1={val_macro_f1:.4f} "
                f"| lr={optimizer.param_groups[0]['lr']:.6f}"
            )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    model.load_state_dict(best_state)
    return model, best_val_acc, best_epoch


def train_fixed_epochs(args, X_train, y_train, num_classes, device, epochs):
    model = build_mlp(
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        dropout=args.dropout,
        activation=args.activation,
    ).to(device)

    train_loader = make_loader(X_train, y_train, args.batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == epochs:
            print(f"Final refit epoch {epoch:03d} | loss={train_loss:.4f}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train an assignment-compliant PyTorch MLP with Iterative SSL.")
    parser.add_argument("--data-dir", default=None, help="Directory containing Kaggle CSV files.")
    parser.add_argument("--output", default="outputs/mlp_submission.csv", help="Submission CSV path.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--hidden-dims", default="256,128,64")
    parser.add_argument("--activation", choices=["relu", "gelu"], default="gelu")
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=12)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--skip-final-refit",
        action="store_true",
        help="Skip retraining the selected model on all labeled data before submission.",
    )
    
    # ======= 成员 C 的半监督调参战场 =======
    parser.add_argument("--use-pseudo-labels", action="store_true", default=True, help="是否启用半监督")
    parser.add_argument("--pseudo-threshold", type=float, default=0.90, help="伪标签置信度阈值")
    parser.add_argument("--max-rounds", type=int, default=5, help="自训练滚动迭代最大轮数")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 导入 A 预处理好的数据
    X_train_clean, X_val, y_train_raw, y_val_raw, scaler = get_train_val_split(
        val_size=args.val_size,
        random_state=args.seed,
        data_dir=args.data_dir,
    )
    y_train_clean, classes = encode_labels(y_train_raw)
    class_to_index = {label: idx for idx, label in enumerate(classes)}
    y_val = np.array([class_to_index[label] for label in y_val_raw], dtype=np.int64)
    num_classes = len(classes)

    # 加载无标签大水池
    X_unlabeled_orig = scaler.transform(load_unlabeled_data(args.data_dir))

    # 初始化滚动变量
    X_labeled_current = X_train_clean.copy()
    y_labeled_current = y_train_clean.copy()
    X_unlabeled_current = X_unlabeled_orig.copy()

    print(f"\n[初始状态] 干净有标签数据: {X_labeled_current.shape[0]}条 | 无标签池: {X_unlabeled_current.shape[0]}条")

    # 2. 开启成员 C 的多轮滚动迭代自训练（Iterative Self-training）
    best_overall_model = None
    best_overall_val_acc = -1.0
    best_overall_epoch = args.epochs
    best_refit_X = None
    best_refit_y = None

    max_rounds = args.max_rounds if args.use_pseudo_labels else 1

    for round_idx in range(1, max_rounds + 1):
        print(f"\n=======================================================")
        print(f"🔄 半监督滚动训练 - 第 {round_idx} 轮 (Round {round_idx})")
        print(f"=======================================================")
        print(f"当前训练集样本总数（含已并入的伪标签）: {X_labeled_current.shape[0]}")

        # 训练当前轮次的模型（调用 B 的复杂模型和训练流）
        model, current_val_acc, current_best_epoch = train_model(
            args, X_labeled_current, y_labeled_current, X_val, y_val, num_classes, device
        )
        print(f"✨ Round {round_idx} 完成！验证集 Accuracy = {current_val_acc:.4f}")

        # 记录全剧最优模型，用来做最终提交
        if current_val_acc > best_overall_val_acc:
            best_overall_val_acc = current_val_acc
            best_overall_model = model
            best_overall_epoch = current_best_epoch
            best_refit_X = np.vstack([X_labeled_current, X_val])
            best_refit_y = np.concatenate([y_labeled_current, y_val])

        # 如果到了最后一轮，或者不开启半监督，直接退出循环
        if round_idx == max_rounds or not args.use_pseudo_labels:
            break

        # 如果无标签水池被抽干了，也退出
        if X_unlabeled_current.shape[0] == 0:
            print("❌ 无标签数据池已完全耗尽，停止滚动。")
            break

        # 3. 核心：用当前轮次表现最好的模型，去预测剩下的无标签数据
        unlabeled_loader = make_loader(X_unlabeled_current, batch_size=args.batch_size, shuffle=False)
        pseudo_labels, pseudo_confidences, _ = predict(model, unlabeled_loader, device)

        # 4. 过滤：抓出满足置信度阈值的样本
        keep_mask = pseudo_confidences >= args.pseudo_threshold
        num_accepted = int(keep_mask.sum())
        
        print(f"🎯 过滤报告: 在剩余的 {X_unlabeled_current.shape[0]} 条无标签数据中，有 {num_accepted} 条预测置信度 >= {args.pseudo_threshold}")

        # 如果没有任何样本达标，雪球滚不动了，提前结束
        if num_accepted == 0:
            print("⚠️ 没有新的无标签数据符合高置信度要求，滚动自训练提前终止。")
            break

        # 5. 滚雪球：把合规的伪标签样本合并到训练集里
        X_labeled_current = np.vstack([X_labeled_current, X_unlabeled_current[keep_mask]])
        y_labeled_current = np.concatenate([y_labeled_current, pseudo_labels[keep_mask]])

        # 6. 瘦身：把这批被选走的样本从无标签池子里删掉，防止下一轮重复选择
        X_unlabeled_current = X_unlabeled_current[~keep_mask]
        print(f"👉 成功合并！下一轮(Round {round_idx + 1}) 的训练集规模将扩大至: {X_labeled_current.shape[0]}条")

    print(f"\n🏁 半监督滚动全流程结束！历史最高验证集准确率: {best_overall_val_acc:.4f}")

    if not args.skip_final_refit:
        print(
            "\nRefitting final model on all labeled data "
            f"(plus accepted pseudo-labels from the best round): {best_refit_X.shape[0]} samples, "
            f"{best_overall_epoch} epochs."
        )
        best_overall_model = train_fixed_epochs(
            args,
            best_refit_X,
            best_refit_y,
            num_classes,
            device,
            epochs=best_overall_epoch,
        )

    # 7. 导出预测用于 Kaggle 提交（自动使用历史表现最好的模型权重）
    print("\n正在读取测试集并用最优半监督模型生成 Kaggle 提交文件...")
    test_ids, X_test_raw = load_test_data(args.data_dir)
    X_test = scaler.transform(X_test_raw)
    test_loader = make_loader(X_test, batch_size=args.batch_size, shuffle=False)
    test_pred, _, _ = predict(best_overall_model, test_loader, device)
    test_labels = classes[test_pred]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Id": test_ids, "Category": test_labels}).to_csv(output_path, index=False)
    print(f"🎉 任务完成！提交文件已成功保存在: {output_path}")


if __name__ == "__main__":
    main()
