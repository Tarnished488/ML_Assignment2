"""Experiment pipeline for semi-supervised learning.

Provides a config-driven, model-agnostic, SSL-method-agnostic pipeline that
handles data loading, training, evaluation, metric tracking, and submission
generation.
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.preprocessing.data_loader import get_train_val_split, load_test_data, load_unlabeled_data

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ModelFactory = Callable[..., nn.Module]
SSLMethod = Callable[..., tuple[np.ndarray, np.ndarray]]
# SSLMethod: (model, X_labeled, y_labeled, X_unlabeled, device, **kwargs)
#   -> (X_augmented, y_augmented)


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------


@dataclass
class TrainingHistory:
    """Records per-epoch metrics during training."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    val_macro_f1: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)

    def log(
        self,
        train_loss: float,
        val_loss: float,
        val_acc: float,
        val_macro_f1: float,
        lr: float,
    ) -> None:
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        self.val_macro_f1.append(val_macro_f1)
        self.lr.append(lr)

    def to_dict(self) -> dict:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
            "val_macro_f1": self.val_macro_f1,
            "lr": self.lr,
        }


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """All parameters for a single experiment."""

    # --- Identification ---
    name: str = "experiment"
    group: str = "Group_XX"
    output_dir: str = "outputs"

    # --- Data ---
    data_dir: Optional[str] = None
    val_size: float = 0.2

    # --- Model ---
    model_factory: Optional[ModelFactory] = None
    model_kwargs: dict = field(default_factory=dict)
    model_type: str = "mlp"  # "mlp", "cnn", "logistic", "random_forest", "decision_tree"

    # --- Training ---
    epochs: int = 200
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_factor: float = 0.5
    lr_patience: int = 12
    early_stop_patience: int = 40

    # --- SSL ---
    ssl_method: Optional[SSLMethod] = None
    ssl_kwargs: dict = field(default_factory=dict)
    use_ssl: bool = False

    # --- Reproducibility ---
    seed: int = 42

    # --- Logging ---
    log_every: int = 10

    # --- Device override ---
    device: Optional[str] = None  # "cpu", "cuda", or None (auto-detect)


# ---------------------------------------------------------------------------
# Experiment result
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    """Stores all outputs from a completed experiment."""

    config: ExperimentConfig
    history: TrainingHistory
    best_val_acc: float
    best_val_macro_f1: float
    best_epoch: int
    best_state: dict

    # Test predictions
    test_ids: Optional[np.ndarray] = None
    test_predictions: Optional[np.ndarray] = None
    test_probabilities: Optional[np.ndarray] = None

    # Per-class metrics on validation set
    val_report: dict = field(default_factory=dict)

    # Timing
    train_time_seconds: float = 0.0
    ssl_time_seconds: float = 0.0

    # Extra metadata
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.config.name,
            "model_type": self.config.model_type,
            "use_ssl": self.config.use_ssl,
            "best_val_acc": self.best_val_acc,
            "best_val_macro_f1": self.best_val_macro_f1,
            "best_epoch": self.best_epoch,
            "train_time_s": self.train_time_seconds,
            "ssl_time_s": self.ssl_time_seconds,
            "val_report": self.val_report,
            "history": self.history.to_dict(),
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ExperimentPipeline:
    """Orchestrates a single experiment: data → supervised training → SSL → eval → submit."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(
            config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._set_seed(config.seed)

        # These are populated during run()
        self.scaler = None
        self.classes: Optional[np.ndarray] = None
        self.class_to_index: dict = {}
        self.num_classes: int = 0
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_val: Optional[np.ndarray] = None
        self.y_val: Optional[np.ndarray] = None
        self.X_unlabeled: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @staticmethod
    def _set_seed(seed: int) -> None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _load_data(self) -> None:
        cfg = self.config
        X_train, X_val, y_train_raw, y_val_raw, scaler = get_train_val_split(
            val_size=cfg.val_size,
            random_state=cfg.seed,
            data_dir=cfg.data_dir,
        )
        self.scaler = scaler

        self.classes = np.array(sorted(np.unique(y_train_raw)))
        self.num_classes = len(self.classes)
        self.class_to_index = {label: idx for idx, label in enumerate(self.classes)}

        self.y_train = np.array([self.class_to_index[l] for l in y_train_raw], dtype=np.int64)
        self.y_val = np.array([self.class_to_index[l] for l in y_val_raw], dtype=np.int64)

        self.X_train = X_train
        self.X_val = X_val
        self.X_unlabeled = scaler.transform(load_unlabeled_data(cfg.data_dir))

    # ------------------------------------------------------------------
    # DataLoader helpers
    # ------------------------------------------------------------------

    def _make_loader(self, X, y=None, shuffle=False):
        X_t = torch.tensor(X, dtype=torch.float32)
        if y is None:
            ds = TensorDataset(X_t)
        else:
            ds = TensorDataset(X_t, torch.tensor(y, dtype=torch.long))
        return DataLoader(ds, batch_size=self.config.batch_size, shuffle=shuffle)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                         num_classes: int) -> dict:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict(self, model: nn.Module, loader: DataLoader):
        model.eval()
        all_logits = []
        for batch in loader:
            X_b = batch[0].to(self.device)
            all_logits.append(model(X_b).cpu())
        logits = torch.cat(all_logits, dim=0)
        probs = torch.softmax(logits, dim=1).numpy()
        preds = probs.argmax(axis=1)
        confs = probs.max(axis=1)
        return preds, confs, probs

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train_epoch(self, model, loader, criterion, optimizer):
        model.train()
        total_loss, total_count = 0.0, 0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(self.device), y_b.to(self.device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_b.size(0)
            total_count += X_b.size(0)
        return total_loss / total_count

    @torch.no_grad()
    def _eval_loss(self, model, loader, criterion):
        model.eval()
        total_loss, total_count = 0.0, 0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(self.device), y_b.to(self.device)
            total_loss += criterion(model(X_b), y_b).item() * X_b.size(0)
            total_count += X_b.size(0)
        return total_loss / total_count

    def _train_model(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        stage_name: str = "supervised",
    ) -> tuple[nn.Module, TrainingHistory, int, float, float]:
        """Train *model* (in-place) on the given data. Returns best metrics."""
        cfg = self.config
        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(self.X_val, self.y_val, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                       weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=cfg.lr_factor, patience=cfg.lr_patience,
        )

        history = TrainingHistory()
        best_state = None
        best_val_acc = -1.0
        best_val_macro_f1 = -1.0
        best_epoch = 0
        epochs_no_improve = 0

        for epoch in range(1, cfg.epochs + 1):
            train_loss = self._train_epoch(model, train_loader, criterion, optimizer)
            val_loss = self._eval_loss(model, val_loader, criterion)
            val_pred, _, _ = self._predict(model, val_loader)
            metrics = self._compute_metrics(self.y_val, val_pred, self.num_classes)
            scheduler.step(metrics["accuracy"])

            history.log(train_loss, val_loss, metrics["accuracy"],
                        metrics["macro_f1"], optimizer.param_groups[0]["lr"])

            if metrics["accuracy"] > best_val_acc:
                best_val_acc = metrics["accuracy"]
                best_val_macro_f1 = metrics["macro_f1"]
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epoch == 1 or epoch % cfg.log_every == 0 or epoch == cfg.epochs:
                print(
                    f"  [{stage_name}] Epoch {epoch:03d} | loss={train_loss:.4f} "
                    f"| val_acc={metrics['accuracy']:.4f} "
                    f"| val_macro_f1={metrics['macro_f1']:.4f} "
                    f"| lr={optimizer.param_groups[0]['lr']:.6f}"
                )

            if epochs_no_improve >= cfg.early_stop_patience:
                print(f"  Early stopping at epoch {epoch}; best was {best_epoch}.")
                break

        model.load_state_dict(best_state)
        return model, history, best_epoch, best_val_acc, best_val_macro_f1

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> ExperimentResult:
        """Execute the full experiment pipeline."""
        cfg = self.config
        print(f"\n{'='*60}")
        print(f"Experiment: {cfg.name}")
        print(f"Model: {cfg.model_type} | SSL: {cfg.use_ssl} | Device: {self.device}")
        print(f"{'='*60}")

        # 1. Load data -------------------------------------------------------
        self._load_data()
        print(f"Labeled: {len(self.y_train)} train / {len(self.y_val)} val  "
              f"| Unlabeled: {len(self.X_unlabeled)}  "
              f"| Classes: {self.num_classes}")

        # 2. Build model -----------------------------------------------------
        if cfg.model_factory is not None:
            model = cfg.model_factory(
                input_dim=self.X_train.shape[1],
                num_classes=self.num_classes,
                **cfg.model_kwargs,
            ).to(self.device)
        else:
            raise ValueError("model_factory must be provided in ExperimentConfig.")

        # 3. Supervised training ---------------------------------------------
        t0 = time.perf_counter()
        model, history, best_epoch, best_val_acc, best_val_macro_f1 = self._train_model(
            model, self.X_train, self.y_train, stage_name="supervised"
        )
        train_time = time.perf_counter() - t0
        print(f"Supervised best val_acc={best_val_acc:.4f}  "
              f"macro_f1={best_val_macro_f1:.4f}  time={train_time:.1f}s")

        ssl_time = 0.0
        X_augmented, y_augmented = self.X_train, self.y_train

        # 4. SSL -------------------------------------------------------------
        if cfg.use_ssl and cfg.ssl_method is not None:
            t0 = time.perf_counter()
            X_augmented, y_augmented = cfg.ssl_method(
                model=model,
                X_labeled=self.X_train,
                y_labeled=self.y_train,
                X_unlabeled=self.X_unlabeled,
                device=self.device,
                num_classes=self.num_classes,
                **cfg.ssl_kwargs,
            )
            ssl_time = time.perf_counter() - t0
            print(f"SSL augmented data: {len(y_augmented)} samples  "
                  f"(+{len(y_augmented) - len(self.y_train)} pseudo-labeled)  "
                  f"time={ssl_time:.1f}s")

            # Re-train on augmented data
            model, history, best_epoch, best_val_acc, best_val_macro_f1 = self._train_model(
                model, X_augmented, y_augmented, stage_name="ssl"
            )
            print(f"SSL best val_acc={best_val_acc:.4f}  macro_f1={best_val_macro_f1:.4f}")

        # 5. Validation report -----------------------------------------------
        val_loader = self._make_loader(self.X_val, self.y_val, shuffle=False)
        val_pred, _, _ = self._predict(model, val_loader)
        val_report = classification_report(
            self.y_val, val_pred,
            labels=list(range(self.num_classes)),
            target_names=[str(c) for c in self.classes],
            output_dict=True,
            zero_division=0,
        )

        # 6. Test predictions ------------------------------------------------
        test_ids, X_test_raw = load_test_data(cfg.data_dir)
        X_test = self.scaler.transform(X_test_raw)
        test_loader = self._make_loader(X_test, shuffle=False)
        test_pred, _, test_probs = self._predict(model, test_loader)

        result = ExperimentResult(
            config=cfg,
            history=history,
            best_val_acc=best_val_acc,
            best_val_macro_f1=best_val_macro_f1,
            best_epoch=best_epoch,
            best_state={k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()},
            test_ids=test_ids,
            test_predictions=test_pred,
            test_probabilities=test_probs,
            val_report=val_report,
            train_time_seconds=train_time,
            ssl_time_seconds=ssl_time,
            extra={"X_augmented_shape": (X_augmented.shape, y_augmented.shape)},
        )

        # 7. Save ------------------------------------------------------------
        self._save_result(result)
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_result(self, result: ExperimentResult) -> None:
        cfg = self.config
        out = Path(cfg.output_dir) / cfg.name
        out.mkdir(parents=True, exist_ok=True)

        # Submission CSV
        test_labels = self.classes[result.test_predictions]
        sub = pd.DataFrame({"Id": result.test_ids, "Category": test_labels})
        sub.to_csv(out / "submission.csv", index=False)

        # Metrics JSON
        metrics_path = out / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        # Best model weights
        torch.save(result.best_state, out / "best_model.pt")

        print(f"\nSaved to {out}/  (submission.csv, metrics.json, best_model.pt)")


# ---------------------------------------------------------------------------
# Built-in SSL methods
# ---------------------------------------------------------------------------


def pseudo_label_ssl(
    model: nn.Module,
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_unlabeled: np.ndarray,
    device: torch.device,
    num_classes: int,
    threshold: float = 0.9,
    batch_size: int = 256,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Standard pseudo-labeling: add high-confidence predictions as labeled data."""
    loader = DataLoader(
        TensorDataset(torch.tensor(X_unlabeled, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False,
    )
    model.eval()
    all_logits = []
    with torch.no_grad():
        for (X_b,) in loader:
            all_logits.append(model(X_b.to(device)).cpu())
    logits = torch.cat(all_logits, dim=0)
    probs = torch.softmax(logits, dim=1)
    confs, preds = probs.max(dim=1)
    keep = confs >= threshold
    keep_np = keep.numpy()
    print(f"  Pseudo-label: kept {keep_np.sum()}/{len(keep_np)} "
          f"(threshold={threshold})")
    if keep_np.any():
        return (
            np.vstack([X_labeled, X_unlabeled[keep_np]]),
            np.concatenate([y_labeled, preds[keep_np].numpy()]),
        )
    return X_labeled, y_labeled


def iterative_pseudo_label_ssl(
    model: nn.Module,
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_unlabeled: np.ndarray,
    device: torch.device,
    num_classes: int,
    threshold: float = 0.9,
    batch_size: int = 256,
    max_iterations: int = 5,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterative self-training: repeatedly pseudo-label, re-train, repeat.

    Note: this method retrains internally; the pipeline will do a final
    re-train on the returned augmented data.
    """
    X_current, y_current = X_labeled.copy(), y_labeled.copy()
    remaining = X_unlabeled.copy()
    remaining_indices = np.arange(len(remaining))

    for it in range(max_iterations):
        if len(remaining) == 0:
            break
        loader = DataLoader(
            TensorDataset(torch.tensor(remaining, dtype=torch.float32)),
            batch_size=batch_size, shuffle=False,
        )
        model.eval()
        all_logits = []
        with torch.no_grad():
            for (X_b,) in loader:
                all_logits.append(model(X_b.to(device)).cpu())
        logits = torch.cat(all_logits, dim=0)
        probs = torch.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)
        keep = confs >= threshold
        keep_np = keep.numpy()
        if keep_np.sum() == 0:
            print(f"  Iter {it+1}: no pseudo-labels above threshold, stopping.")
            break
        print(f"  Iter {it+1}: kept {keep_np.sum()}/{len(keep_np)} "
              f"(threshold={threshold})")
        X_current = np.vstack([X_current, remaining[keep_np]])
        y_current = np.concatenate([y_current, preds[keep_np].numpy()])
        remaining = remaining[~keep_np]

    return X_current, y_current
