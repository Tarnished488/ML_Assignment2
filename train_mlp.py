"""End-to-end training with LP + Knowledge Distillation for semi-supervised learning.

Usage:
    # LP → Knowledge Distillation (best approach, ~0.5+ accuracy)
    python train_mlp.py --name mlp_distill --use-ssl --ssl-method distill

    # LP → Hard pseudo-labels (baseline)
    python train_mlp.py --name mlp_pseudo --use-ssl --ssl-method pseudo

    # LP → Distill + VAT consistency
    python train_mlp.py --name mlp_distill_vat --use-ssl --ssl-method distill --use-vat

    # Supervised only (100 labeled samples)
    python train_mlp.py --name mlp_supervised
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.cnn import build_cnn_32x16, build_cnn_8x64
from src.models.mlp import build_mlp
from src.preprocessing.data_loader import (
    get_train_val_split,
    load_test_data,
    load_unlabeled_data,
    _resolve_data_dir,
)
from src.ssl.consistency import CombinedSSLLoss
from src.ssl.label_propagation import LabelPropagationSSL
from src.ssl.self_training import SelfTrainingSSL
from src.utils import compute_metrics, encode_labels, make_loader, predict, set_seed
from src.visualization.visualizer import (
    plot_confusion_matrix,
    plot_decision_boundary,
    plot_loss_landscape,
    plot_pca,
    plot_training_curves,
    plot_tsne,
)


# ---------------------------------------------------------------------------
# Knowledge Distillation Training
# ---------------------------------------------------------------------------


def train_distill(
    model, X_labeled, y_labeled, lp_soft, X_unlabeled,
    X_val, y_val, device, args,
):
    """Train MLP via knowledge distillation from LP soft probability distributions.

    Loss = CE(labeled) + alpha * T² * KL(LP_soft || MLP_soft_T) + VAT

    LP soft probabilities preserve class-ambiguity information that hard
    pseudo-labels discard, preventing class collapse.
    """
    T = args.distill_T
    alpha = args.distill_alpha

    labeled_dataset = TensorDataset(
        torch.tensor(X_labeled, dtype=torch.float32),
        torch.tensor(y_labeled, dtype=torch.long),
    )
    labeled_loader = DataLoader(labeled_dataset, batch_size=args.batch_size,
                                shuffle=True, drop_last=True)

    unlabeled_dataset = TensorDataset(
        torch.tensor(X_unlabeled, dtype=torch.float32),
        torch.tensor(lp_soft, dtype=torch.float32),
    )
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size * 2, shuffle=True)

    # Separate VAT loader with all unlabeled data (no LP targets needed for VAT)
    vat_loader = None
    if args.use_vat:
        vat_dataset = TensorDataset(torch.tensor(X_unlabeled, dtype=torch.float32))
        vat_loader = DataLoader(vat_dataset, batch_size=args.batch_size * 4, shuffle=True)

    val_loader = make_loader(X_val, y_val, args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience,
    )
    vat_criterion = CombinedSSLLoss(
        vat_weight=args.vat_weight if args.use_vat else 0.0,
        pi_weight=0.0,
        vat_epsilon=args.vat_epsilon,
        ramp_up_epochs=args.vat_rampup,
    )

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": [], "lr": []}
    best_state, best_val_acc, best_val_f1, best_epoch = None, -1.0, -1.0, 0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        vat_criterion.set_epoch(epoch)
        total_loss, total_ce, total_kd, total_vat, total_count = 0.0, 0.0, 0.0, 0.0, 0

        unlabeled_iter = iter(unlabeled_loader)
        vat_iter = iter(vat_loader) if vat_loader else None

        for X_l, y_l in labeled_loader:
            X_l, y_l = X_l.to(device), y_l.to(device)

            try:
                X_u, lp_u = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                X_u, lp_u = next(unlabeled_iter)
            X_u, lp_u = X_u.to(device), lp_u.to(device)

            # Supervised CE on labeled data
            logits_l = model(X_l)
            ce_loss = F.cross_entropy(logits_l, y_l)

            # KD: KL(LP_soft || MLP_soft_T) * T²
            logits_u = model(X_u)
            log_probs_T = F.log_softmax(logits_u / T, dim=1)
            log_lp = torch.log(lp_u + 1e-12)
            kd_loss = (
                F.kl_div(log_probs_T, log_lp, reduction="batchmean", log_target=True)
                * (T ** 2)
            )

            # VAT on unlabeled data
            vat_loss_val = torch.tensor(0.0, device=device)
            if vat_iter is not None:
                try:
                    X_vat = next(vat_iter)[0].to(device)
                except StopIteration:
                    vat_iter = iter(vat_loader)
                    X_vat = next(vat_iter)[0].to(device)
                from src.ssl.consistency import vat_loss as vat_fn
                vat_loss_val = vat_fn(model, X_vat, epsilon=args.vat_epsilon)

            ramp = min(1.0, epoch / args.vat_rampup) if args.use_vat else 0.0
            loss = ce_loss + alpha * kd_loss + ramp * args.vat_weight * vat_loss_val

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * X_l.size(0)
            total_ce += ce_loss.item() * X_l.size(0)
            total_kd += kd_loss.item() * X_l.size(0)
            total_vat += vat_loss_val.item() * X_l.size(0)
            total_count += X_l.size(0)

        train_loss = total_loss / total_count

        # Validation
        val_pred, _, _ = predict(model, val_loader, device)
        val_metrics = compute_metrics(y_val, val_pred)
        scheduler.step(val_metrics["accuracy"])

        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # Val CE loss (for plotting)
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

        if epoch == 1 or epoch % args.log_every == 0:
            print(
                f"  [distill] Epoch {epoch:03d} | loss={train_loss:.4f} "
                f"(CE={total_ce/total_count:.4f} KD={total_kd/total_count:.4f} "
                f"VAT={total_vat/total_count:.4f}) "
                f"| val_acc={val_metrics['accuracy']:.4f} "
                f"| val_f1={val_metrics['macro_f1']:.4f} "
                f"| lr={optimizer.param_groups[0]['lr']:.6f}"
            )

        if epochs_no_improve >= args.patience:
            print(f"  Early stopping at epoch {epoch}; best was {best_epoch}.")
            break

    model.load_state_dict(best_state)
    return model, history, best_epoch, best_val_acc, best_val_f1


# ---------------------------------------------------------------------------
# Standard CE training (for pseudo-label baseline / supervised)
# ---------------------------------------------------------------------------


def train_standard(
    model, X_train, y_train, X_val, y_val, device, args,
    stage_name="train", X_unlabeled=None,
):
    """Train with CE loss + optional VAT consistency."""
    criterion = CombinedSSLLoss(
        vat_weight=args.vat_weight if args.use_vat else 0.0,
        pi_weight=args.pi_weight if args.use_vat else 0.0,
        vat_epsilon=args.vat_epsilon,
        ramp_up_epochs=args.vat_rampup,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience,
    )

    # Use drop_last=True to avoid BatchNorm1d error on last incomplete batch
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = make_loader(X_val, y_val, args.batch_size, shuffle=False)
    unlabeled_loader = None
    if X_unlabeled is not None and args.use_vat:
        unlabeled_loader = make_loader(X_unlabeled, batch_size=args.batch_size * 4, shuffle=True)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": [], "lr": []}
    best_state, best_val_acc, best_val_f1, best_epoch = None, -1.0, -1.0, 0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        criterion.set_epoch(epoch)
        total_loss, total_count = 0.0, 0

        unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader else None

        for batch in train_loader:
            X_b, y_b = batch[0].to(device), batch[1].to(device)

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

        val_pred, _, _ = predict(model, val_loader, device)
        val_metrics = compute_metrics(y_val, val_pred)
        scheduler.step(val_metrics["accuracy"])

        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

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

        if epoch == 1 or epoch % args.log_every == 0:
            print(
                f"  [{stage_name}] Epoch {epoch:03d} | loss={train_loss:.4f} "
                f"| val_acc={val_metrics['accuracy']:.4f} "
                f"| val_f1={val_metrics['macro_f1']:.4f} "
                f"| lr={optimizer.param_groups[0]['lr']:.6f}"
            )

        if epochs_no_improve >= args.patience:
            print(f"  Early stopping at epoch {epoch}; best was {best_epoch}.")
            break

    model.load_state_dict(best_state)
    return model, history, best_epoch, best_val_acc, best_val_f1


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def generate_visualizations(model, history, X_train, y_train, X_val, y_val,
                            X_unlabeled, classes, device, args):
    plot_dir = Path(args.output_dir) / args.name / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    class_names = [str(c) for c in classes]

    print(f"\nGenerating visualizations -> {plot_dir}/")

    X_lab = np.vstack([X_train, X_val])
    y_lab = np.concatenate([y_train, y_val])

    # 1. PCA: labeled data
    plot_pca(X_lab, y_lab, title="PCA - Labeled Data",
             class_names=class_names,
             save_path=str(plot_dir / "01_pca_labeled.png"))

    # 2. PCA: labeled + unlabeled
    y_unl_placeholder = np.full(len(X_unlabeled), -1, dtype=np.int64)
    X_all = np.vstack([X_lab, X_unlabeled])
    y_all = np.concatenate([y_lab, y_unl_placeholder])
    plot_pca(X_all, y_all, title="PCA - Labeled + Unlabeled",
             class_names=class_names,
             save_path=str(plot_dir / "02_pca_all.png"))

    # 3. t-SNE (sampled)
    sample_n = min(2000, len(X_all))
    idx = np.random.RandomState(args.seed).choice(len(X_all), sample_n, replace=False)
    plot_tsne(X_all[idx], y_all[idx], title="t-SNE - Sampled",
              class_names=class_names,
              save_path=str(plot_dir / "03_tsne.png"))

    # 4. Decision boundary
    plot_decision_boundary(model, X_lab, y_lab, device,
                           title="Decision Boundary",
                           class_names=class_names,
                           save_path=str(plot_dir / "04_decision_boundary.png"))

    # 5. Loss landscape
    plot_loss_landscape(model, X_val, y_val, nn.CrossEntropyLoss(), device,
                        title="Loss Landscape",
                        save_path=str(plot_dir / "05_loss_landscape.png"))

    # 6. Confusion matrix
    val_loader = make_loader(X_val, y_val, args.batch_size)
    val_pred, _, _ = predict(model, val_loader, device)
    plot_confusion_matrix(y_val, val_pred, class_names=class_names,
                          title="Confusion Matrix (Val)",
                          save_path=str(plot_dir / "06_confusion_matrix.png"))

    # 7. Training curves
    plot_training_curves(history, title="Training Curves",
                         save_path=str(plot_dir / "07_training_curves.png"))

    print("Visualizations done.")


def build_model(input_dim, num_classes, args, device):
    if args.model_type == "mlp":
        hidden_dims = tuple(int(x.strip()) for x in args.hidden_dims.split(","))
        model = build_mlp(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=args.dropout,
            activation=args.activation,
            use_residual=True,
            norm=args.norm,
        ).to(device)
        model_desc = (
            f"MLP{hidden_dims} | residual=True | norm={args.norm} | "
            f"activation={args.activation}"
        )
    elif args.model_type == "cnn":
        cnn_builders = {
            "32x16": build_cnn_32x16,
            "8x64": build_cnn_8x64,
        }
        model = cnn_builders[args.cnn_layout](
            input_dim=input_dim,
            num_classes=num_classes,
            conv1_channels=args.cnn_conv1_channels,
            conv2_channels=args.cnn_conv2_channels,
            hidden_dim=args.cnn_hidden_dim,
        ).to(device)
        model_desc = (
            f"CNN[{args.cnn_layout}] | conv=({args.cnn_conv1_channels},{args.cnn_conv2_channels}) | "
            f"hidden={args.cnn_hidden_dim}"
        )
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")

    print(f"\nModel: {model_desc} | params={sum(p.numel() for p in model.parameters()):,}")
    return model


def train_self_training(
    X_train_raw,
    y_train,
    X_val_raw,
    y_val,
    X_unlabeled,
    num_classes,
    device,
    args,
):
    """Run iterative self-training and return the final trained model."""
    ssl = SelfTrainingSSL(
        lp_kwargs=dict(
            n_neighbors=args.lp_k,
            alpha=args.lp_alpha,
            class_balanced=True,
            top_k_per_class=args.lp_top_k,
            confidence_threshold=args.lp_conf,
        ),
        mlp_relabel_threshold=args.self_train_threshold,
        max_rounds=args.self_train_rounds,
        batch_size=args.batch_size,
    )

    round_summaries = []
    X_current, y_current, X_remaining, first_round = ssl.bootstrap_with_label_propagation(
        X_labeled=X_train_raw,
        y_labeled=y_train,
        X_unlabeled=X_unlabeled,
        num_classes=num_classes,
    )
    round_summaries.append(first_round)
    print(
        f"  [SelfTraining] Round 1/{args.self_train_rounds}: "
        f"+{first_round['added']} pseudo-labeled, "
        f"remaining_unlabeled={first_round['remaining_unlabeled']}"
    )

    model = None
    history = None
    best_epoch = 0
    best_val_acc = -1.0
    best_val_f1 = -1.0

    for round_idx in range(1, args.self_train_rounds + 1):
        model = build_model(X_current.shape[1], num_classes, args, device)
        stage_name = f"self_train_r{round_idx}"
        unlabeled_for_vat = X_remaining if args.use_vat and len(X_remaining) > 0 else None
        model, history, best_epoch, best_val_acc, best_val_f1 = train_standard(
            model,
            X_current,
            y_current,
            X_val_raw,
            y_val,
            device,
            args,
            stage_name=stage_name,
            X_unlabeled=unlabeled_for_vat,
        )
        print(
            f"  [SelfTraining] Round {round_idx} val_acc={best_val_acc:.4f} "
            f"macro_f1={best_val_f1:.4f}"
        )
        round_summaries[-1]["train_size"] = int(len(y_current))
        round_summaries[-1]["best_epoch"] = int(best_epoch)
        round_summaries[-1]["best_val_acc"] = float(best_val_acc)
        round_summaries[-1]["best_val_macro_f1"] = float(best_val_f1)

        if round_idx >= args.self_train_rounds or len(X_remaining) == 0:
            break

        X_current, y_current, X_remaining, record = ssl.relabel_with_model(
            model=model,
            X_labeled=X_current,
            y_labeled=y_current,
            X_unlabeled=X_remaining,
            device=device,
            round_idx=round_idx + 1,
        )
        round_summaries.append(record)
        print(
            f"  [SelfTraining] Round {round_idx + 1}/{args.self_train_rounds}: "
            f"+{record['added']} pseudo-labeled, "
            f"remaining_unlabeled={record['remaining_unlabeled']}"
        )
        if record["added"] == 0:
            break

    return model, history, best_epoch, best_val_acc, best_val_f1, X_current, y_current, round_summaries


def make_experiment_name(args):
    base = args.base_name if getattr(args, "base_name", None) else args.name
    model_part = args.model_type if args.model_type == "mlp" else f"cnn_{args.cnn_layout}"
    ssl_part = args.ssl_method if args.use_ssl else "supervised"
    return f"{base}_{model_part}_{ssl_part}"


def run_single_experiment(args, device):
    set_seed(args.seed)
    args.name = make_experiment_name(args)
    print(f"\n{'#' * 72}")
    print(f"Running: {args.name}")
    print(f"{'#' * 72}")

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
    # 2. SSL: Label Propagation → pseudo-labels or soft targets
    # ------------------------------------------------------------------
    X_train_ssl = X_train_raw
    y_train_ssl = y_train
    lp_soft = None  # soft probabilities for distillation
    self_training_rounds = None

    if args.use_ssl and args.ssl_method in {"distill", "pseudo"}:
        lp = LabelPropagationSSL(
            n_neighbors=args.lp_k,
            alpha=args.lp_alpha,
            class_balanced=True,
            top_k_per_class=args.lp_top_k,
            confidence_threshold=args.lp_conf,
        )

        X_all = np.vstack([X_train_raw, X_unlabeled])
        n_labeled = len(y_train)

        if args.ssl_method == "distill":
            # Get LP soft probability distributions for distillation
            print(f"  [LP] Building k-NN graph (k={args.lp_k}, n={X_all.shape[0]}) ...")
            lp_soft = lp.propagate_soft(X_all, y_train, n_labeled)
            print(f"  [LP] Soft probabilities ready: {lp_soft.shape}")

            # Also select a class-balanced subset of hard pseudo-labels
            # to augment the tiny labeled set for CE loss
            pseudo_labels, confidences = lp.propagate(X_all, y_train, n_labeled)
            keep = lp.select_pseudo_labels(pseudo_labels, confidences, num_classes)
            print(f"  [LP] Hard pseudo-labels for CE augmentation: {keep.sum()}/{len(keep)}")

            if keep.any():
                X_train_ssl = np.vstack([X_train_raw, X_unlabeled[keep]])
                y_train_ssl = np.concatenate([y_train, pseudo_labels[keep]])
        else:
            # Hard pseudo-labels only (baseline)
            X_train_ssl, y_train_ssl = lp(
                model=None, X_labeled=X_train_raw, y_labeled=y_train,
                X_unlabeled=X_unlabeled, device=device, num_classes=num_classes,
            )

        print(f"Training set: {len(y_train_ssl)} samples "
              f"(original {len(y_train)} + pseudo {len(y_train_ssl) - len(y_train)})")

    # ------------------------------------------------------------------
    # 3. Build model and train
    # ------------------------------------------------------------------
    if args.use_ssl and args.ssl_method == "self_training":
        print(
            "Training: Self-Training | "
            f"rounds={args.self_train_rounds} | threshold={args.self_train_threshold}"
        )
        model, history, best_epoch, best_val_acc, best_val_f1, X_train_ssl, y_train_ssl, self_training_rounds = train_self_training(
            X_train_raw,
            y_train,
            X_val_raw,
            y_val,
            X_unlabeled,
            num_classes,
            device,
            args,
        )
    else:
        model = build_model(X_train_ssl.shape[1], num_classes, args, device)

    if args.use_ssl and args.ssl_method == "distill" and lp_soft is not None:
        print(f"Training: Knowledge Distillation | T={args.distill_T} | alpha={args.distill_alpha}")
        model, history, best_epoch, best_val_acc, best_val_f1 = train_distill(
            model, X_train_ssl, y_train_ssl, lp_soft, X_unlabeled,
            X_val_raw, y_val, device, args,
        )
    elif not (args.use_ssl and args.ssl_method == "self_training"):
        unlabeled_for_vat = X_unlabeled if args.use_vat else None
        model, history, best_epoch, best_val_acc, best_val_f1 = train_standard(
            model, X_train_ssl, y_train_ssl, X_val_raw, y_val, device, args,
            stage_name="train", X_unlabeled=unlabeled_for_vat,
        )

    print(f"\n{'='*50}")
    print(f"Best val_acc={best_val_acc:.4f}  macro_f1={best_val_f1:.4f}  (epoch {best_epoch})")

    # ------------------------------------------------------------------
    # 4. Generate submission
    # ------------------------------------------------------------------
    out = Path(args.output_dir) / args.name
    out.mkdir(parents=True, exist_ok=True)

    test_ids, X_test_raw = load_test_data(args.data_dir)
    X_test = scaler.transform(X_test_raw)
    test_loader = make_loader(X_test, batch_size=args.batch_size)
    test_pred, test_confs, test_probs = predict(model, test_loader, device)
    test_labels = classes[test_pred]

    sub_path = out / "submission.csv"
    pd.DataFrame({"Id": test_ids, "Category": test_labels}).to_csv(sub_path, index=False)
    print(f"Submission saved -> {sub_path}")

    # Save probabilities for ensemble
    np.save(out / "test_probs.npy", test_probs)
    np.save(out / "test_ids.npy", test_ids)

    # Save model
    torch.save(model.state_dict(), out / "model.pt")

    val_loader = make_loader(X_val_raw, y_val, args.batch_size, shuffle=False)
    val_pred, _, _ = predict(model, val_loader, device)
    final_val_metrics = compute_metrics(y_val, val_pred)

    metrics = {
        "name": args.name,
        "model_type": args.model_type,
        "cnn_layout": args.cnn_layout if args.model_type == "cnn" else None,
        "ssl_enabled": bool(args.use_ssl),
        "ssl_method": args.ssl_method if args.use_ssl else "supervised",
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "best_val_macro_f1": float(best_val_f1),
        "final_val_metrics": final_val_metrics,
        "train_size_final": int(len(y_train_ssl)),
        "pseudo_labels_added": int(len(y_train_ssl) - len(y_train)),
    }
    if self_training_rounds is not None:
        metrics["self_training_rounds"] = self_training_rounds

    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics saved -> {metrics_path}")

    # Class distribution summary
    unique, counts = np.unique(test_labels, return_counts=True)
    print(f"Prediction distribution: {dict(zip(unique, counts))}")

    # ------------------------------------------------------------------
    # 5. Visualizations
    # ------------------------------------------------------------------
    if not args.no_viz:
        generate_visualizations(
            model, history, X_train_ssl, y_train_ssl,
            X_val_raw, y_val, X_unlabeled, classes, device, args,
        )

    print(f"\nDone. Outputs in {out}/")
    return metrics


def planned_experiments(args):
    configs = []
    model_variants = [
        ("mlp", None),
        ("cnn", "32x16"),
        ("cnn", "8x64"),
    ]
    ssl_variants = [
        (False, "supervised"),
        (True, "pseudo"),
        (True, "distill"),
        (True, "self_training"),
    ]

    for model_type, cnn_layout in model_variants:
        for use_ssl, ssl_method in ssl_variants:
            run_args = SimpleNamespace(**vars(args))
            run_args.model_type = model_type
            run_args.cnn_layout = cnn_layout or args.cnn_layout
            run_args.use_ssl = use_ssl
            run_args.ssl_method = ssl_method if use_ssl else args.ssl_method
            configs.append(run_args)
    return configs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train all model/SSL combinations.")

    # Experiment
    parser.add_argument("--name", default="all_models")
    parser.add_argument("--group", default="Group_XX")
    parser.add_argument("--output-dir", default="outputs")

    # Data
    parser.add_argument(
        "--data-dir",
        default=_resolve_data_dir(),
    )
    parser.add_argument("--val-size", type=float, default=0.2)

    # Model
    parser.add_argument("--model-type", choices=["mlp", "cnn"], default="mlp")
    parser.add_argument("--hidden-dims", default="256,128,64")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--activation", choices=["relu", "gelu", "silu"], default="gelu")
    parser.add_argument("--norm", choices=["batch", "layer"], default="batch")
    parser.add_argument("--cnn-layout", choices=["32x16", "8x64"], default="32x16")
    parser.add_argument("--cnn-conv1-channels", type=int, default=8)
    parser.add_argument("--cnn-conv2-channels", type=int, default=16)
    parser.add_argument("--cnn-hidden-dim", type=int, default=128)

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
    parser.add_argument("--ssl-method", choices=["distill", "pseudo", "self_training"],
                        default="distill")
    parser.add_argument("--lp-k", type=int, default=10)
    parser.add_argument("--lp-alpha", type=float, default=0.99)
    parser.add_argument("--lp-top-k", type=int, default=300)
    parser.add_argument("--lp-conf", type=float, default=0.6)
    parser.add_argument("--self-train-rounds", type=int, default=3)
    parser.add_argument("--self-train-threshold", type=float, default=0.85)

    # Distillation
    parser.add_argument("--distill-T", type=float, default=2.0,
                        help="Temperature for softening logits in KD")
    parser.add_argument("--distill-alpha", type=float, default=5.0,
                        help="Weight for KD loss relative to CE")

    # VAT
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
    args.base_name = args.name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_metrics = []
    for run_args in planned_experiments(args):
        metrics = run_single_experiment(run_args, device)
        all_metrics.append(metrics)

    summary_dir = Path(args.output_dir) / args.base_name
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "summary.csv"
    pd.DataFrame(all_metrics).sort_values(
        by=["best_val_acc", "best_val_macro_f1"], ascending=False
    ).to_csv(summary_path, index=False)
    print(f"\nSummary saved -> {summary_path}")


if __name__ == "__main__":
    main()
