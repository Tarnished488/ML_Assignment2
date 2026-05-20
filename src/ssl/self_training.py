"""Self-training helpers for iterative semi-supervised learning.

Improvements over the baseline:
  - Per-class dynamic thresholds calibrated from validation confidence.
  - Progressive threshold scheduling: start high (0.95), decay each round.
  - Top-k per round to cap pseudo-label additions and avoid noise accumulation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.ssl.label_propagation import LabelPropagationSSL


class SelfTrainingSSL:
    """Multi-round self-training with label propagation bootstrapping.

    Parameters
    ----------
    lp_kwargs : dict or None
        Keyword arguments forwarded to ``LabelPropagationSSL`` for the
        initial bootstrap round.
    mlp_relabel_threshold : float
        Fallback global confidence threshold for model-based relabeling
        (used when dynamic thresholds are disabled).
    max_rounds : int
        Maximum number of self-training rounds.
    batch_size : int
        Batch size for model inference on unlabeled data.
    dynamic_threshold : bool
        When True, calibrate per-class thresholds from val performance
        and apply progressive decay across rounds.
    initial_threshold : float
        Starting threshold for the first model-relabeling round (Round 1).
    threshold_decay : float
        Multiplicative decay factor applied to the base threshold each round.
    min_threshold : float
        Floor for the decaying base threshold.
    top_k_per_round : int or None
        Maximum total pseudo-labels to add per round.  ``None`` disables.
    per_class_adjustment : float
        Strength of per-class adjustment relative to mean class accuracy.
    """

    def __init__(
        self,
        lp_kwargs: dict | None = None,
        mlp_relabel_threshold: float = 0.85,
        max_rounds: int = 2,
        batch_size: int = 256,
        dynamic_threshold: bool = True,
        initial_threshold: float = 0.95,
        threshold_decay: float = 0.85,
        min_threshold: float = 0.70,
        top_k_per_round: int | None = 500,
        per_class_adjustment: float = 0.15,
    ):
        lp_defaults = dict(
            n_neighbors=10,
            alpha=0.99,
            class_balanced=True,
            top_k_per_class=300,
            confidence_threshold=0.6,
        )
        if lp_kwargs:
            lp_defaults.update(lp_kwargs)
        self.lp = LabelPropagationSSL(**lp_defaults)
        self.mlp_relabel_threshold = mlp_relabel_threshold
        self.max_rounds = max_rounds
        self.batch_size = batch_size

        self.dynamic_threshold = dynamic_threshold
        self.initial_threshold = initial_threshold
        self.threshold_decay = threshold_decay
        self.min_threshold = min_threshold
        self.top_k_per_round = top_k_per_round
        self.per_class_adjustment = per_class_adjustment

    # ------------------------------------------------------------------
    # Per-class threshold calibration
    # ------------------------------------------------------------------

    def calibrate_per_class_thresholds(
        self,
        model: nn.Module,
        X_val: np.ndarray,
        y_val: np.ndarray,
        device: torch.device,
        round_idx: int,
        num_classes: int,
    ) -> np.ndarray:
        """Return per-class confidence thresholds for the given round.

        Steps
        ----
        1. Compute a **progressive base threshold** that starts high and
           decays with ``threshold_decay`` each round, floored by
           ``min_threshold``.
        2. Evaluate the model on the validation set to obtain per-class
           accuracy.
        3. Adjust each class threshold **down** for high-accuracy classes
           and **up** for low-accuracy (confusable) classes.
        """
        base = max(
            self.initial_threshold * (self.threshold_decay ** max(round_idx - 1, 0)),
            self.min_threshold,
        )

        if X_val is None or y_val is None or len(y_val) == 0:
            return np.full(num_classes, base, dtype=np.float64)

        val_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long),
            ),
            batch_size=self.batch_size,
            shuffle=False,
        )
        model.eval()
        all_preds, all_confs = [], []
        with torch.no_grad():
            for X_b, _ in val_loader:
                logits = model(X_b.to(device))
                probs = torch.softmax(logits, dim=1)
                all_preds.append(probs.argmax(dim=1).cpu())
                all_confs.append(probs.max(dim=1).values.cpu())
        val_preds = torch.cat(all_preds).numpy()
        val_confs = torch.cat(all_confs).numpy()

        per_class_acc = np.zeros(num_classes, dtype=np.float64)
        for c in range(num_classes):
            c_mask = y_val == c
            if c_mask.sum() > 0:
                per_class_acc[c] = (val_preds[c_mask] == c).mean()
            else:
                per_class_acc[c] = 0.5  # neutral prior for unseen classes

        mean_acc = per_class_acc.mean()
        thresholds = base * (
            1.0 + self.per_class_adjustment * (mean_acc - per_class_acc)
        )
        thresholds = np.clip(thresholds, base * 0.70, 0.99)
        return thresholds.astype(np.float64)

    # ------------------------------------------------------------------
    # Pseudo-label selection
    # ------------------------------------------------------------------

    def _select_model_pseudo_labels(
        self,
        preds: np.ndarray,
        confs: np.ndarray,
        num_classes: int,
        threshold: float | None = None,
        top_k_per_class: int | None = None,
        per_class_thresholds: np.ndarray | None = None,
    ) -> np.ndarray:
        """Select model pseudo-labels with confidence filtering + class balancing.

        When ``per_class_thresholds`` is provided each class uses its own
        confidence floor instead of a single global threshold.
        """
        keep = np.zeros(len(preds), dtype=bool)

        for cls in range(num_classes):
            cls_idx = np.where(preds == cls)[0]
            if len(cls_idx) == 0:
                continue

            thresh = (
                per_class_thresholds[cls]
                if per_class_thresholds is not None
                else (threshold if threshold is not None else self.mlp_relabel_threshold)
            )
            cls_idx = cls_idx[confs[cls_idx] >= thresh]
            if len(cls_idx) == 0:
                continue

            if top_k_per_class is not None and len(cls_idx) > top_k_per_class:
                order = np.argsort(-confs[cls_idx])[:top_k_per_class]
                cls_idx = cls_idx[order]
            keep[cls_idx] = True

        return keep

    def _apply_top_k_overall(
        self,
        keep: np.ndarray,
        confs: np.ndarray,
    ) -> np.ndarray:
        """Limit the total number of selected pseudo-labels to ``top_k_per_round``."""
        if self.top_k_per_round is None or keep.sum() <= self.top_k_per_round:
            return keep

        kept_indices = np.where(keep)[0]
        order = np.argsort(-confs[kept_indices])[:self.top_k_per_round]
        new_keep = np.zeros(len(keep), dtype=bool)
        new_keep[kept_indices[order]] = True
        return new_keep

    def _select_model_pseudo_labels_balanced(
        self,
        preds: np.ndarray,
        confs: np.ndarray,
        num_classes: int,
        per_class_thresholds: np.ndarray,
        top_k_per_class: int | None = None,
    ) -> np.ndarray:
        """Class-balanced selection with per-class thresholds + overall top-k cap."""
        keep = self._select_model_pseudo_labels(
            preds=preds,
            confs=confs,
            num_classes=num_classes,
            per_class_thresholds=per_class_thresholds,
            top_k_per_class=top_k_per_class,
        )
        keep = self._apply_top_k_overall(keep, confs)
        return keep

    # ------------------------------------------------------------------
    # Model inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _mlp_pseudo_label(
        self,
        model: nn.Module,
        X_unlabeled: np.ndarray,
        device: torch.device,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Use the trained model to predict pseudo-labels on unlabeled data."""
        loader = DataLoader(
            TensorDataset(torch.tensor(X_unlabeled, dtype=torch.float32)),
            batch_size=self.batch_size,
            shuffle=False,
        )
        model.eval()
        all_logits = []
        for (X_b,) in loader:
            all_logits.append(model(X_b.to(device)).cpu())
        logits = torch.cat(all_logits, dim=0)
        probs = torch.softmax(logits, dim=1).numpy()
        preds = probs.argmax(axis=1)
        confs = probs.max(axis=1)
        return preds, confs

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def bootstrap_with_label_propagation(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        num_classes: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Build the initial augmented set from label propagation."""
        n_labeled = len(y_labeled)
        X_all = np.vstack([X_labeled, X_unlabeled])
        pseudo_labels, confidences = self.lp.propagate(X_all, y_labeled, n_labeled)
        keep = self.lp.select_pseudo_labels(pseudo_labels, confidences, num_classes)

        if keep.any():
            X_aug = np.vstack([X_labeled, X_unlabeled[keep]])
            y_aug = np.concatenate([y_labeled, pseudo_labels[keep]])
            X_remaining = X_unlabeled[~keep]
        else:
            X_aug = X_labeled
            y_aug = y_labeled
            X_remaining = X_unlabeled

        record = {
            "round": 1,
            "method": "label_propagation",
            "added": int(keep.sum()),
            "remaining_unlabeled": int(len(X_remaining)),
        }
        return X_aug, y_aug, X_remaining, record

    def bootstrap_with_model(
        self,
        model: nn.Module,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        device: torch.device,
        num_classes: int,
        threshold: float | None = None,
        top_k_per_class: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Build the initial augmented set directly from a pretrained model."""
        if len(X_unlabeled) == 0:
            record = {
                "round": 1,
                "method": "pretrained_model",
                "added": 0,
                "remaining_unlabeled": 0,
                "threshold": float(
                    self.mlp_relabel_threshold if threshold is None else threshold
                ),
            }
            return X_labeled, y_labeled, X_unlabeled, record

        preds, confs = self._mlp_pseudo_label(model, X_unlabeled, device)
        keep = self._select_model_pseudo_labels(
            preds=preds,
            confs=confs,
            num_classes=num_classes,
            threshold=threshold,
            top_k_per_class=top_k_per_class,
        )
        if self.top_k_per_round is not None:
            keep = self._apply_top_k_overall(keep, confs)

        if keep.any():
            X_aug = np.vstack([X_labeled, X_unlabeled[keep]])
            y_aug = np.concatenate([y_labeled, preds[keep]])
            X_remaining = X_unlabeled[~keep]
        else:
            X_aug = X_labeled
            y_aug = y_labeled
            X_remaining = X_unlabeled

        threshold_used = self.mlp_relabel_threshold if threshold is None else threshold
        record = {
            "round": 1,
            "method": "pretrained_model",
            "added": int(keep.sum()),
            "remaining_unlabeled": int(len(X_remaining)),
            "threshold": float(threshold_used),
        }
        return X_aug, y_aug, X_remaining, record

    # ------------------------------------------------------------------
    # Model-based relabeling  (with dynamic thresholds)
    # ------------------------------------------------------------------

    def relabel_with_model(
        self,
        model: nn.Module,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        device: torch.device,
        round_idx: int,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Relabel the remaining unlabeled pool with the current model.

        When ``dynamic_threshold`` is enabled this method:
        1. Computes a progressive base threshold for the round.
        2. Calibrates per-class thresholds from validation performance.
        3. Caps the total pseudo-labels added via ``top_k_per_round``.
        """
        if len(X_unlabeled) == 0:
            record = {
                "round": round_idx,
                "method": "model_relabel",
                "added": 0,
                "remaining_unlabeled": 0,
            }
            return X_labeled, y_labeled, X_unlabeled, record

        preds, confs = self._mlp_pseudo_label(model, X_unlabeled, device)
        num_classes = len(np.unique(y_labeled))

        if self.dynamic_threshold and X_val is not None and y_val is not None:
            per_class_thresholds = self.calibrate_per_class_thresholds(
                model=model,
                X_val=X_val,
                y_val=y_val,
                device=device,
                round_idx=round_idx,
                num_classes=num_classes,
            )
            effective_threshold = float(per_class_thresholds.mean())
            keep = self._select_model_pseudo_labels_balanced(
                preds=preds,
                confs=confs,
                num_classes=num_classes,
                per_class_thresholds=per_class_thresholds,
            )
        else:
            base = max(
                self.initial_threshold * (self.threshold_decay ** max(round_idx - 1, 0)),
                self.min_threshold,
            ) if self.dynamic_threshold else self.mlp_relabel_threshold
            per_class_thresholds = None
            effective_threshold = base
            keep = self._select_model_pseudo_labels(
                preds=preds,
                confs=confs,
                num_classes=num_classes,
                threshold=base,
                top_k_per_class=None,
            )
            if self.top_k_per_round is not None:
                keep = self._apply_top_k_overall(keep, confs)

        if keep.any():
            X_aug = np.vstack([X_labeled, X_unlabeled[keep]])
            y_aug = np.concatenate([y_labeled, preds[keep]])
            X_remaining = X_unlabeled[~keep]
        else:
            X_aug = X_labeled
            y_aug = y_labeled
            X_remaining = X_unlabeled

        record = {
            "round": round_idx,
            "method": "model_relabel",
            "added": int(keep.sum()),
            "remaining_unlabeled": int(len(X_remaining)),
            "threshold": float(effective_threshold),
        }
        if per_class_thresholds is not None:
            record["per_class_thresholds"] = per_class_thresholds.tolist()
        return X_aug, y_aug, X_remaining, record

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def __call__(
        self,
        model: nn.Module,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        device,
        num_classes: int,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compatibility wrapper for a simple self-training pass."""
        X_aug, y_aug, X_remaining, record = self.bootstrap_with_label_propagation(
            X_labeled=X_labeled,
            y_labeled=y_labeled,
            X_unlabeled=X_unlabeled,
            num_classes=num_classes,
        )
        print(
            f"  [SelfTraining] Round 1 done: {len(y_aug)} total samples "
            f"(+{record['added']} pseudo-labeled)"
        )

        for rnd in range(2, self.max_rounds + 1):
            X_aug, y_aug, X_remaining, record = self.relabel_with_model(
                model=model,
                X_labeled=X_aug,
                y_labeled=y_aug,
                X_unlabeled=X_remaining,
                device=device,
                round_idx=rnd,
            )
            print(
                f"  [SelfTraining] Round {rnd} done: {len(y_aug)} total "
                f"({record['added']} pseudo-labeled at conf>={record.get('threshold', 'N/A')})"
            )
            if record["added"] == 0:
                break

        return X_aug, y_aug
