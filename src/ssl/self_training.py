"""Self-training helpers for iterative semi-supervised learning."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.ssl.label_propagation import LabelPropagationSSL


class SelfTrainingSSL:
    """Multi-round self-training with label propagation bootstrapping."""

    def __init__(
        self,
        lp_kwargs: dict | None = None,
        mlp_relabel_threshold: float = 0.85,
        max_rounds: int = 2,
        batch_size: int = 256,
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

    def relabel_with_model(
        self,
        model: nn.Module,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        device: torch.device,
        round_idx: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Relabel the remaining unlabeled pool with the current model."""
        if len(X_unlabeled) == 0:
            record = {
                "round": round_idx,
                "method": "model_relabel",
                "added": 0,
                "remaining_unlabeled": 0,
            }
            return X_labeled, y_labeled, X_unlabeled, record

        preds, confs = self._mlp_pseudo_label(model, X_unlabeled, device)
        keep = confs >= self.mlp_relabel_threshold

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
        }
        return X_aug, y_aug, X_remaining, record

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
                f"({record['added']} pseudo-labeled at conf>={self.mlp_relabel_threshold})"
            )
            if record["added"] == 0:
                break

        return X_aug, y_aug
