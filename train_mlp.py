"""End-to-end training script with semi-supervised learning.

Usage examples:
    # Supervised only (baseline)
    python train_mlp.py --name mlp_supervised

    # Label Propagation → MLP (strongest for few labels)
    python train_mlp.py --name mlp_lp --use-ssl --ssl-method label_propagation

    # Label Propagation → MLP → iterative re-labeling
    python train_mlp.py --name mlp_selftrain --use-ssl --ssl-method self_training --ssl-rounds 3

    # LP + VAT consistency regularisation
    python train_mlp.py --name mlp_lp_vat --use-ssl --ssl-method label_propagation --use-vat
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp import build_mlp
from src.preprocessing.data_loader import (
    get_train_val_split,
    load_labeled_data,
    load_test_data,
    load_unlabeled_data,
)
from src.ssl.consistency import CombinedSSLLoss
from src.ssl.label_propagation import LabelPropagationSSL
from src.ssl.self_training import SelfTrainingSSL
from src.visualization.visualizer import (
    plot_confusion_matrix,
    plot_decision_boundary,
    plot_loss_landscape,
    plot_pca,
    plot_training_curves,
    plot_tsne,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def encode_labels(y_raw):
    classes = np.array(sorted(np.unique(y_raw)))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    encoded = np.array([class_to_idx[c] for c in y_raw], dtype=np.int64)
    return encoded, classes, class_to_idx


def make_loader(X, y=None, batch_size=32, shuffle=False):
    X_t = torch.tensor(X, dtype=torch.float32)
    if y is None:
        return DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=shuffle)
    return DataLoader(
        TensorDataset(X_t, torch.tensor(y, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def compute_metrics(y_true, y_pred):
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_logits = []
    for batch in loader:
        all_logits.append(model(batch[0].to(device)).cpu())
    logits = torch.cat(all_logits, dim=0)
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)
    confs = probs.max(axis=1)
    return preds, confs, probs


# ---------------------------------------------------------------------------
# Training with optional VAT consistency regularisation
# ---------------------------------------------------------------------------


def train_model(
    model, X_train, y_train, X_val, y_val, num_classes, device, args,
    stage_name="train", X_unlabeled=None, epochs=None,
):
    """Train model with CE + optional VAT loss. Returns model + history."""
    max_epochs = epochs if epochs is not None else args.epochs
    """Train model with CE + optional VAT loss. Returns model + history."""
    criterion = CombinedSSLLoss(
        vat_weight=args.vat_weight if args.use_vat else 0.0,
        pi_weight=args.pi_weight if args.use_vat else 0.0,
        vat_epsilon=args.vat_epsilon,
        ramp_up_epochs=args.vat_rampup,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience,
    )

    train_loader = make_loader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, args.batch_size, shuffle=False)
    unlabeled_loader = None
    if X_unlabeled is not None and args.use_vat:
        unlabeled_loader = make_loader(X_unlabeled, batch_size=args.batch_size * 4, shuffle=True)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": [], "lr": []}
    best_state, best_val_acc, best_val_f1, best_epoch = None, -1.0, -1.0, 0
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        criterion.set_epoch(epoch)
        total_loss, total_count = 0.0, 0

        unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader else None

        for batch in train_loader:
            X_b, y_b = batch[0].to(device), batch[1].to(device)

            # Get unlabeled batch if available
            x_unl = None
            if unlabeled_iter is not None:
                try:
                    x_unl = next(unlabeled_iter)[0].to(device)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    x_unl = next(unlabeled_iter)[0].to(device)

            loss = criterion(model, X_b, y_b, x_unl)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * X_b.size(0)
            total_count += X_b.size(0)

        train_loss = total_loss / total_count

        # Validation
        val_pred, _, _ = predict(model, val_loader, device)
        val_metrics = compute_metrics(y_val, val_pred)
        scheduler.step(val_metrics["accuracy"])

        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # Compute val loss
        ce_fn = nn.CrossEntropyLoss()
        val_ce = 0.0
        model.eval()
        with torch.no_grad():
            for X_b, y_b in val_loader:
                val_ce += ce_fn(model(X_b.to(device)), y_b.to(device)).item() * X_b.size(0)
        history["val_loss"].append(val_ce / len(y_val))

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"  [{stage_name}] Epoch {epoch:03d} | loss={train_loss:.4f} "
                f"| val_acc={val_metrics['accuracy']:.4f} "
                f"| val_macro_f1={val_metrics['macro_f1']:.4f} "
                f"| lr={optimizer.param_groups[0]['lr']:.6f}"
            )

        if epochs_no_improve >= args.patience:
            print(f"  Early stopping at epoch {epoch}; best was {best_epoch}.")
            break

    model.load_state_dict(best_state)
    return model, history, best_epoch, best_val_acc, best_val_f1


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def generate_visualizations(model, history, X_train, y_train, X_val, y_val,
                            X_unlabeled, classes, device, args):
    plot_dir = Path(args.output_dir) / args.name / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    class_names = [str(c) for c in classes]
    num_classes = len(classes)

    print(f"\nGenerating visualisations → {plot_dir}/")

    # 1. PCA: labeled data
    X_lab = np.vstack([X_train, X_val])
    y_lab = np.concatenate([y_train, y_val])
    plot_pca(X_lab, y_lab, "PCA — Labeled Data", class_names,
             save_path=str(plot_dir / "01_pca_labeled.png"))

    # 2. PCA: labeled + unlabeled
    y_unl_placeholder = np.full(len(X_unlabeled), -1, dtype=np.int64)
    X_all = np.vstack([X_lab, X_unlabeled])
    y_all = np.concatenate([y_lab, y_unl_placeholder])
    plot_pca(X_all, y_all, "PCA — Labeled + Unlabeled", class_names,
             save_path=str(plot_dir / "02_pca_all.png"))

    # 3. t-SNE (sampled)
    sample_n = min(2000, len(X_all))
    idx = np.random.RandomState(args.seed).choice(len(X_all), sample_n, replace=False)
    plot_tsne(X_all[idx], y_all[idx], "t-SNE — Sampled", class_names,
              save_path=str(plot_dir / "03_tsne.png"))

    # 4. Decision boundary
    plot_decision_boundary(model, X_lab, y_lab, device, "Decision Boundary",
                           class_names, save_path=str(plot_dir / "04_decision_boundary.png"))

    # 5. Loss landscape
    plot_loss_landscape(model, X_val, y_val, nn.CrossEntropyLoss(), device,
                        "Loss Landscape", save_path=str(plot_dir / "05_loss_landscape.png"))

    # 6. Confusion matrix
    val_loader = make_loader(X_val, y_val, args.batch_size)
    val_pred, _, _ = predict(model, val_loader, device)
    plot_confusion_matrix(y_val, val_pred, class_names, "Confusion Matrix (Val)",
                          save_path=str(plot_dir / "06_confusion_matrix.png"))

    # 7. Training curves
    plot_training_curves(history, "Training Curves",
                         save_path=str(plot_dir / "07_training_curves.png"))

    print("Visualisations done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train MLP with SSL + visualisations.")

    # Experiment
    parser.add_argument("--name", default="mlp_ssl")
    parser.add_argument("--group", default="Group_XX")
    parser.add_argument("--output-dir", default="outputs")

    # Data
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--val-size", type=float, default=0.2)

    # Model
    parser.add_argument("--hidden-dims", default="512,256,128,64")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--activation", choices=["relu", "gelu"], default="gelu")

    # Training
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=15)
    parser.add_argument("--patience", type=int, default=50)

    # SSL
    parser.add_argument("--use-ssl", action="store_true")
    parser.add_argument("--ssl-method", choices=["label_propagation", "self_training"],
                        default="label_propagation")
    parser.add_argument("--lp-k", type=int, default=10, help="k-NN neighbors for LP")
    parser.add_argument("--lp-alpha", type=float, default=0.99, help="LP clamping factor")
    parser.add_argument("--lp-top-k", type=int, default=300,
                        help="Top-k pseudo-labels per class")
    parser.add_argument("--lp-conf", type=float, default=0.6,
                        help="LP min confidence threshold")
    parser.add_argument("--ssl-rounds", type=int, default=2,
                        help="Self-training rounds (1=LP only)")
    parser.add_argument("--ssl-mlp-threshold", type=float, default=0.85,
                        help="MLP confidence threshold for re-labeling")

    # VAT consistency regularisation
    parser.add_argument("--use-vat", action="store_true")
    parser.add_argument("--vat-weight", type=float, default=0.3)
    parser.add_argument("--pi-weight", type=float, default=0.1)
    parser.add_argument("--vat-epsilon", type=float, default=2.0)
    parser.add_argument("--vat-rampup", type=int, default=50)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--no-viz", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    X_train_raw, X_val_raw, y_train_raw, y_val_raw, scaler = get_train_val_split(
        val_size=args.val_size, random_state=args.seed, data_dir=args.data_dir,
    )
    y_train, classes, class_to_idx = encode_labels(y_train_raw)
    y_val = np.array([class_to_idx[c] for c in y_val_raw], dtype=np.int64)
    num_classes = len(classes)
    X_unlabeled = scaler.transform(load_unlabeled_data(args.data_dir))

    print(f"Labeled: {len(y_train)} train / {len(y_val)} val | "
          f"Unlabeled: {len(X_unlabeled)} | Classes: {num_classes}")

    # ------------------------------------------------------------------
    # 2. SSL: generate pseudo-labels
    # ------------------------------------------------------------------
    X_train_ssl = X_train_raw
    y_train_ssl = y_train
    X_unlabeled_remaining = X_unlabeled

    if args.use_ssl:
        if args.ssl_method == "label_propagation":
            ssl_fn = LabelPropagationSSL(
                n_neighbors=args.lp_k,
                alpha=args.lp_alpha,
                class_balanced=True,
                top_k_per_class=args.lp_top_k,
                confidence_threshold=args.lp_conf,
            )
            X_aug, y_aug = ssl_fn(
                model=None, X_labeled=X_train_raw, y_labeled=y_train,
                X_unlabeled=X_unlabeled, device=device, num_classes=num_classes,
            )
        elif args.ssl_method == "self_training":
            ssl_fn = SelfTrainingSSL(
                lp_kwargs=dict(
                    n_neighbors=args.lp_k,
                    alpha=args.lp_alpha,
                    top_k_per_class=args.lp_top_k,
                    confidence_threshold=args.lp_conf,
                ),
                mlp_relabel_threshold=args.ssl_mlp_threshold,
                max_rounds=args.ssl_rounds,
            )
            # SelfTrainingSSL needs a model for rounds 2+, so we do a
            # quick supervised pre-train first
            hidden_dims = tuple(int(x.strip()) for x in args.hidden_dims.split(","))
            pre_model = build_mlp(
                input_dim=X_train_raw.shape[1], num_classes=num_classes,
                hidden_dims=hidden_dims, dropout=args.dropout,
                activation=args.activation,
            ).to(device)
            pre_model, _, _, _, _ = train_model(
                pre_model, X_train_raw, y_train, X_val_raw, y_val,
                num_classes, device, args, stage_name="pretrain",
                epochs=min(args.epochs, 50),
            )
            X_aug, y_aug = ssl_fn(
                model=pre_model, X_labeled=X_train_raw, y_labeled=y_train,
                X_unlabeled=X_unlabeled, device=device, num_classes=num_classes,
            )

        X_train_ssl = X_aug
        y_train_ssl = y_aug

    print(f"\nTraining set: {len(y_train_ssl)} samples "
          f"(original {len(y_train)} + pseudo {len(y_train_ssl) - len(y_train)})")

    # ------------------------------------------------------------------
    # 3. Train final model on augmented data
    # ------------------------------------------------------------------
    hidden_dims = tuple(int(x.strip()) for x in args.hidden_dims.split(","))
    model = build_mlp(
        input_dim=X_train_ssl.shape[1], num_classes=num_classes,
        hidden_dims=hidden_dims, dropout=args.dropout,
        activation=args.activation,
    ).to(device)

    # Pass unlabeled data for VAT (only if use_vat)
    unlabeled_for_vat = X_unlabeled if args.use_vat else None

    model, history, best_epoch, best_val_acc, best_val_f1 = train_model(
        model, X_train_ssl, y_train_ssl, X_val_raw, y_val,
        num_classes, device, args, stage_name="final",
        X_unlabeled=unlabeled_for_vat,
    )

    print(f"\n{'='*50}")
    print(f"Best val_acc={best_val_acc:.4f}  macro_f1={best_val_f1:.4f}  "
          f"(epoch {best_epoch})")

    # ------------------------------------------------------------------
    # 4. Generate submission
    # ------------------------------------------------------------------
    out = Path(args.output_dir) / args.name
    out.mkdir(parents=True, exist_ok=True)

    test_ids, X_test_raw = load_test_data(args.data_dir)
    X_test = scaler.transform(X_test_raw)
    test_loader = make_loader(X_test, batch_size=args.batch_size)
    test_pred, _, _ = predict(model, test_loader, device)
    test_labels = classes[test_pred]

    sub_path = out / "submission.csv"
    import pandas as pd
    pd.DataFrame({"Id": test_ids, "Category": test_labels}).to_csv(sub_path, index=False)
    print(f"Submission saved → {sub_path}")

    # ------------------------------------------------------------------
    # 5. Visualisations
    # ------------------------------------------------------------------
    if not args.no_viz:
        generate_visualizations(
            model, history, X_train_ssl, y_train_ssl,
            X_val_raw, y_val, X_unlabeled, classes, device, args,
        )

    print(f"\nDone. Outputs in {out}/")


if __name__ == "__main__":
    main()
