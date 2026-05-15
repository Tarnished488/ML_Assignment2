"""Self-training pipeline: Label Propagation → MLP → iterative re-labeling.

Combines graph-based pseudo-label generation with model-based refinement.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.ssl.label_propagation import LabelPropagationSSL


class SelfTrainingSSL:
    """Multi-round self-training with label propagation bootstrapping.

    Round 1: Label Propagation → initial pseudo-labels → train MLP
    Round 2+: MLP pseudo-labels on unlabeled → retrain MLP

    Parameters
    ----------
    lp_kwargs : dict
        Keyword arguments passed to LabelPropagationSSL.
    mlp_relabel_threshold : float
        Confidence threshold when the MLP re-labels unlabeled data in
        subsequent rounds.
    max_rounds : int
        Total training rounds (1 = LP only, 2+ = iterative re-labeling).
    batch_size : int
        Batch size for MLP inference.
    """

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
        """Use the trained MLP to predict on unlabeled data."""
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
        """Run multi-round self-training. Returns augmented training data."""
        n_labeled = len(y_labeled)

        # ---- Round 1: Label Propagation ----
        print("  [SelfTraining] Round 1: Label Propagation ...")
        X_all = np.vstack([X_labeled, X_unlabeled])
        pseudo_labels, confidences = self.lp.propagate(X_all, y_labeled, n_labeled)
        keep = self.lp.select_pseudo_labels(pseudo_labels, confidences, num_classes)

        X_aug = X_labeled
        y_aug = y_labeled
        if keep.any():
            X_aug = np.vstack([X_labeled, X_unlabeled[keep]])
            y_aug = np.concatenate([y_labeled, pseudo_labels[keep]])
        print(f"  [SelfTraining] Round 1 done: {len(y_aug)} total samples "
              f"(+{len(y_aug) - n_labeled} pseudo-labeled)")

        # ---- Rounds 2+: MLP-based re-labeling ----
        for rnd in range(2, self.max_rounds + 1):
            print(f"  [SelfTraining] Round {rnd}: MLP pseudo-labeling ...")
            mlp_preds, mlp_confs = self._mlp_pseudo_label(model, X_unlabeled, device)
            mlp_keep = mlp_confs >= self.mlp_relabel_threshold

            if mlp_keep.any():
                X_aug = np.vstack([X_labeled, X_unlabeled[mlp_keep]])
                y_aug = np.concatenate([y_labeled, mlp_preds[mlp_keep]])
            print(f"  [SelfTraining] Round {rnd} done: {len(y_aug)} total "
                  f"({int(mlp_keep.sum())} pseudo-labeled at conf≥{self.mlp_relabel_threshold})")

        return X_aug, y_aug
