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
    return model, best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Train an assignment-compliant PyTorch MLP.")
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
    parser.add_argument("--use-pseudo-labels", action="store_true")
    parser.add_argument("--pseudo-threshold", type=float, default=0.9)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train, X_val, y_train_raw, y_val_raw, scaler = get_train_val_split(
        val_size=args.val_size,
        random_state=args.seed,
        data_dir=args.data_dir,
    )
    y_train, classes = encode_labels(y_train_raw)
    class_to_index = {label: idx for idx, label in enumerate(classes)}
    y_val = np.array([class_to_index[label] for label in y_val_raw], dtype=np.int64)
    num_classes = len(classes)

    model, best_val_acc = train_model(args, X_train, y_train, X_val, y_val, num_classes, device)
    print(f"Best validation accuracy before pseudo-labeling: {best_val_acc:.4f}")

    X_labeled = X_train
    y_labeled = y_train
    X_unlabeled = scaler.transform(load_unlabeled_data(args.data_dir))

    if args.use_pseudo_labels:
        unlabeled_loader = make_loader(X_unlabeled, batch_size=args.batch_size, shuffle=False)
        pseudo_labels, pseudo_confidences, _ = predict(model, unlabeled_loader, device)
        keep_mask = pseudo_confidences >= args.pseudo_threshold
        print(
            f"Pseudo-labels kept: {int(keep_mask.sum())}/{len(keep_mask)} "
            f"(threshold={args.pseudo_threshold})"
        )
        if keep_mask.any():
            X_labeled = np.vstack([X_labeled, X_unlabeled[keep_mask]])
            y_labeled = np.concatenate([y_labeled, pseudo_labels[keep_mask]])

    model, _ = train_model(args, X_labeled, y_labeled, X_val, y_val, num_classes, device)

    test_ids, X_test_raw = load_test_data(args.data_dir)
    X_test = scaler.transform(X_test_raw)
    test_loader = make_loader(X_test, batch_size=args.batch_size, shuffle=False)
    test_pred, _, _ = predict(model, test_loader, device)
    test_labels = classes[test_pred]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Id": test_ids, "Category": test_labels}).to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")


if __name__ == "__main__":
    main()
