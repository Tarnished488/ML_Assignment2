"""Train CNN model with Label Propagation + Self-Training for ensemble diversity."""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.cnn import build_cnn_32x16
from src.preprocessing.data_loader import (
    get_train_val_split,
    load_test_data,
    load_unlabeled_data,
)
from src.ssl.label_propagation import LabelPropagationSSL


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(X, y=None, batch_size=64, shuffle=False, drop_last=False):
    if y is None:
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    else:
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last and y is not None)


def compute_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_logits = []
    for batch in loader:
        X = batch[0].to(device)
        all_logits.append(model(X).cpu())
    logits = torch.cat(all_logits, dim=0)
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)
    return preds, probs


def self_training_augment(model, X_labeled, y_labeled, X_unlabeled, device, threshold=0.9):
    """Self-training: use model to pseudo-label unlabeled data, keep high-confidence predictions."""
    loader = DataLoader(
        TensorDataset(torch.tensor(X_unlabeled, dtype=torch.float32)),
        batch_size=256, shuffle=False
    )
    model.eval()
    all_logits, all_X = [], []
    with torch.no_grad():
        for (X_b,) in loader:
            all_logits.append(model(X_b.to(device)).cpu())
            all_X.append(X_b)

    logits = torch.cat(all_logits, dim=0)
    probs = torch.softmax(logits, dim=1).numpy()
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)

    # Keep high-confidence pseudo-labels
    keep = confs >= threshold
    X_aug = np.vstack([X_labeled, X_unlabeled[keep]])
    y_aug = np.concatenate([y_labeled, preds[keep]])
    return X_aug, y_aug, keep.sum(), confs[keep].mean() if keep.sum() > 0 else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="cnn_st")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lp-k", type=int, default=20)
    parser.add_argument("--self-train-rounds", type=int, default=3)
    parser.add_argument("--st-threshold", type=float, default=0.9)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Self-training: {args.self_train_rounds} rounds, threshold={args.st_threshold}")

    # Load data
    X_train, X_val, y_train, y_val, scaler = get_train_val_split(data_dir=args.data_dir)
    X_unlabeled = load_unlabeled_data(data_dir=args.data_dir)
    X_unlabeled_scaled = scaler.transform(X_unlabeled)
    ids_test, X_test_raw = load_test_data(data_dir=args.data_dir)
    X_test = scaler.transform(X_test_raw)

    print(f"Labeled: {len(X_train)}, Val: {len(X_val)}, Unlabeled: {len(X_unlabeled)}")

    # Build model
    model = build_cnn_32x16(
        input_dim=512, num_classes=10,
        conv1_channels=16, conv2_channels=32, hidden_dim=256,
    ).to(device)

    val_loader = make_loader(X_val, y_val, batch_size=args.batch_size, drop_last=False)
    test_loader = make_loader(X_test, batch_size=args.batch_size, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
    ce_fn = nn.CrossEntropyLoss()

    best_state, best_acc, best_f1, best_epoch = None, 0, 0, 0
    patience_counter = 0

    # Self-training rounds
    for st_round in range(1, args.self_train_rounds + 1):
        print(f"\n=== Self-Training Round {st_round} ===")

        # Label Propagation for initial round
        if st_round == 1:
            print("Running Label Propagation...")
            lp = LabelPropagationSSL(n_neighbors=args.lp_k, confidence_threshold=0.5)
            X_all = np.vstack([X_train, X_unlabeled_scaled])
            pseudo_labels, confidences = lp.propagate(X_all, y_train, len(X_train))
            keep_mask = lp.select_pseudo_labels(pseudo_labels, confidences, 10)
            X_current = np.vstack([X_train, X_unlabeled_scaled[keep_mask]])
            y_current = np.concatenate([y_train, pseudo_labels[keep_mask]])
            print(f"LP: kept {keep_mask.sum()} pseudo-labels")
        else:
            # Self-training: use model to pseudo-label
            X_current, y_current, n_pseudo, avg_conf = self_training_augment(
                model, X_train, y_train, X_unlabeled_scaled, device, args.st_threshold
            )
            print(f"Self-train: kept {n_pseudo} pseudo-labels (avg conf={avg_conf:.3f})")
            if n_pseudo == 0:
                print("No confident pseudo-labels, skipping round")
                break

        train_loader = make_loader(X_current, y_current, batch_size=args.batch_size, shuffle=True, drop_last=True)

        # Training
        for epoch in range(1, args.epochs // args.self_train_rounds + 1):
            model.train()
            total_loss, total_count = 0, 0
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                loss = ce_fn(model(X_b), y_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * X_b.size(0)
                total_count += X_b.size(0)

            train_loss = total_loss / total_count
            val_pred, _ = predict(model, val_loader, device)
            metrics = compute_metrics(y_val, val_pred)
            scheduler.step(metrics["accuracy"])

            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                best_f1 = metrics["macro_f1"]
                best_epoch = (st_round - 1) * (args.epochs // args.self_train_rounds) + epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            global_epoch = (st_round - 1) * (args.epochs // args.self_train_rounds) + epoch
            if epoch % 10 == 0:
                print(f"Ep {global_epoch:03d} | loss={train_loss:.4f} | val_acc={metrics['accuracy']:.4f} | best={best_acc:.4f}")

            if patience_counter >= 15:
                print(f"Early stopping at epoch {global_epoch}")
                break

    print(f"\nBest val acc: {best_acc:.4f} at epoch {best_epoch}")

    # Load best model
    model.load_state_dict(best_state)

    # Predict test
    test_pred, test_probs = predict(model, test_loader, device)

    # Save outputs
    out_dir = Path(args.output_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    pd.DataFrame({"Id": ids_test, "Category": test_pred}).to_csv(out_dir / "submission.csv", index=False)
    np.save(out_dir / "test_probs.npy", test_probs)
    np.save(out_dir / "test_ids.npy", ids_test)
    torch.save(model.state_dict(), out_dir / "model.pt")

    unique, counts = np.unique(test_pred, return_counts=True)
    print(f"Test distribution: {dict(zip(unique, counts))}")
    print(f"Model saved to {out_dir}")


if __name__ == "__main__":
    main()