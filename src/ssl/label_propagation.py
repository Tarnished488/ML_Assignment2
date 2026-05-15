"""Label Propagation / Label Spreading for semi-supervised learning.

Graph-based SSL that propagates labels through a k-NN similarity graph.
Does NOT require an initial trained model — works directly on feature space.
This makes it far more effective than pseudo-labeling when labeled data is
extremely scarce (e.g. 100 samples for 10 classes).

Reference: Zhu & Ghahramani, "Learning from Labeled and Unlabeled Data
with Graph Laplacians" (ICML 2002).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


class LabelPropagationSSL:
    """k-NN graph label propagation with RBF kernel.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors for graph construction.
    alpha : float
        Clamping factor in (0, 1).  Closer to 1 → trust the graph more;
        closer to 0 → trust initial labels more.  Typical range 0.8–0.99.
    sigma_mode : str
        How to set RBF bandwidth sigma.
        "median"  → median distance to k-th neighbor (default, robust).
        "local"   → per-pair sigma based on local distances.
    max_iter : int
        Maximum propagation iterations.
    tol : float
        Convergence tolerance (max absolute change in label distributions).
    class_balanced : bool
        If True, select top-k confident pseudo-labels *per class* instead of
        a global threshold.  Prevents majority-class bias.
    top_k_per_class : int or None
        Number of pseudo-labels to keep per class (when class_balanced=True).
        None → keep all above confidence threshold.
    confidence_threshold : float
        Minimum confidence to accept a pseudo-label (used when
        class_balanced=False or as a hard floor).
    """

    def __init__(
        self,
        n_neighbors: int = 10,
        alpha: float = 0.99,
        sigma_mode: str = "median",
        max_iter: int = 500,
        tol: float = 1e-6,
        class_balanced: bool = True,
        top_k_per_class: int | None = 300,
        confidence_threshold: float = 0.6,
    ):
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.sigma_mode = sigma_mode
        self.max_iter = max_iter
        self.tol = tol
        self.class_balanced = class_balanced
        self.top_k_per_class = top_k_per_class
        self.confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_knn_graph(self, X: np.ndarray) -> sparse.csr_matrix:
        """Build a symmetric k-NN graph with RBF-weighted edges."""
        k = min(self.n_neighbors + 1, X.shape[0])  # +1 to exclude self
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        # Drop self-loops (first column)
        distances = distances[:, 1:]
        indices = indices[:, 1:]

        n = X.shape[0]
        actual_k = distances.shape[1]

        if self.sigma_mode == "median":
            sigma = np.median(distances[:, -1])
            sigma = max(sigma, 1e-8)
            weights = np.exp(-distances ** 2 / (2 * sigma ** 2))
        elif self.sigma_mode == "local":
            # Per-row sigma: distance to k-th neighbor
            sigmas = distances[:, -1:]
            sigmas = np.maximum(sigmas, 1e-8)
            weights = np.exp(-distances ** 2 / (2 * sigmas ** 2))
        else:
            raise ValueError(f"Unknown sigma_mode: {self.sigma_mode}")

        # Build sparse adjacency
        row = np.repeat(np.arange(n), actual_k)
        col = indices.ravel()
        data = weights.ravel()

        W = sparse.csr_matrix((data, (row, col)), shape=(n, n))
        # Symmetrize: W = (W + W^T) / 2 to handle non-reciprocal edges
        W = (W + W.T) / 2.0
        return W

    @staticmethod
    def _normalize_graph(W: sparse.csr_matrix) -> sparse.csr_matrix:
        """Compute D^{-1/2} W D^{-1/2} (symmetric normalisation)."""
        d = np.array(W.sum(axis=1)).ravel()
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        D_inv_sqrt = sparse.diags(d_inv_sqrt)
        return D_inv_sqrt @ W @ D_inv_sqrt

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def propagate(
        self,
        X_all: np.ndarray,
        y_labeled: np.ndarray,
        n_labeled: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run label propagation and return pseudo-labels + confidences.

        Parameters
        ----------
        X_all : ndarray of shape (n_labeled + n_unlabeled, D)
            Standardised features for all data.
        y_labeled : ndarray of shape (n_labeled,)
            Integer class labels for the first *n_labeled* rows.
        n_labeled : int
            Number of labeled samples (first n_labeled rows of X_all).

        Returns
        -------
        pseudo_labels : ndarray of shape (n_unlabeled,)
        confidences : ndarray of shape (n_unlabeled,)
        """
        n_total = X_all.shape[0]
        classes = np.unique(y_labeled)
        n_classes = len(classes)
        class_to_idx = {c: i for i, c in enumerate(classes)}

        # One-hot label matrix — only labeled rows are non-zero
        Y0 = np.zeros((n_total, n_classes), dtype=np.float64)
        for i in range(n_labeled):
            Y0[i, class_to_idx[y_labeled[i]]] = 1.0

        # Build graph + normalise
        W = self._build_knn_graph(X_all)
        S = self._normalize_graph(W)

        # Iterative propagation:  Y^{t+1} = alpha * S @ Y^t + (1-alpha) * Y0
        Y = Y0.copy()
        unlabeled_slice = slice(n_labeled, n_total)

        for _ in range(self.max_iter):
            Y_new = self.alpha * (S @ Y) + (1.0 - self.alpha) * Y0
            # Clamp labeled rows to ground truth
            Y_new[:n_labeled] = Y0[:n_labeled]

            delta = np.abs(Y_new[unlabeled_slice] - Y[unlabeled_slice]).max()
            Y = Y_new
            if delta < self.tol:
                break

        # Row-normalise to get "probabilities"
        row_sums = Y.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        Y /= row_sums

        pseudo_probs = Y[n_labeled:]  # (n_unlabeled, n_classes)
        pseudo_labels = pseudo_probs.argmax(axis=1)
        confidences = pseudo_probs.max(axis=1)

        return pseudo_labels, confidences

    def propagate_soft(
        self,
        X_all: np.ndarray,
        y_labeled: np.ndarray,
        n_labeled: int,
    ) -> np.ndarray:
        """Run label propagation and return soft probability distributions.

        Returns
        -------
        probs : ndarray of shape (n_unlabeled, n_classes)
            Row-normalised probability distribution over classes for each
            unlabeled sample.  Useful for knowledge distillation.
        """
        _, _ = self.propagate(X_all, y_labeled, n_labeled)
        # Re-run is wasteful — extract the logic to a shared helper if needed.
        # For now, re-run with the same logic inline.
        n_total = X_all.shape[0]
        classes = np.unique(y_labeled)
        n_classes = len(classes)
        class_to_idx = {c: i for i, c in enumerate(classes)}

        Y0 = np.zeros((n_total, n_classes), dtype=np.float64)
        for i in range(n_labeled):
            Y0[i, class_to_idx[y_labeled[i]]] = 1.0

        W = self._build_knn_graph(X_all)
        S = self._normalize_graph(W)

        Y = Y0.copy()
        unlabeled_slice = slice(n_labeled, n_total)

        for _ in range(self.max_iter):
            Y_new = self.alpha * (S @ Y) + (1.0 - self.alpha) * Y0
            Y_new[:n_labeled] = Y0[:n_labeled]
            delta = np.abs(Y_new[unlabeled_slice] - Y[unlabeled_slice]).max()
            Y = Y_new
            if delta < self.tol:
                break

        row_sums = Y.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        Y /= row_sums

        return Y[n_labeled:]  # (n_unlabeled, n_classes)

    # ------------------------------------------------------------------
    # Pseudo-label selection
    # ------------------------------------------------------------------

    def select_pseudo_labels(
        self,
        pseudo_labels: np.ndarray,
        confidences: np.ndarray,
        n_classes: int,
    ) -> np.ndarray:
        """Return a boolean mask of which pseudo-labels to keep.

        Uses class-balanced selection (top-k per class) or global threshold.
        """
        keep = np.zeros(len(pseudo_labels), dtype=bool)

        if self.class_balanced and self.top_k_per_class is not None:
            for cls in range(n_classes):
                cls_mask = pseudo_labels == cls
                cls_indices = np.where(cls_mask)[0]
                if len(cls_indices) == 0:
                    continue
                # Sort by confidence, descending
                sorted_idx = cls_indices[np.argsort(-confidences[cls_indices])]
                top = sorted_idx[: self.top_k_per_class]
                # Apply confidence floor
                top = top[confidences[top] >= self.confidence_threshold]
                keep[top] = True
        else:
            keep = confidences >= self.confidence_threshold

        return keep

    # ------------------------------------------------------------------
    # High-level API  (matches the SSLMethod interface)
    # ------------------------------------------------------------------

    def __call__(
        self,
        model,  # unused — LP doesn't need a model
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        device,
        num_classes: int,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run LP + pseudo-label selection → augmented training set."""
        n_labeled = len(y_labeled)
        X_all = np.vstack([X_labeled, X_unlabeled])

        print(f"  [LP] Building k-NN graph (k={self.n_neighbors}, "
              f"n_total={X_all.shape[0]}) ...")
        pseudo_labels, confidences = self.propagate(X_all, y_labeled, n_labeled)

        keep = self.select_pseudo_labels(pseudo_labels, confidences, num_classes)
        kept = int(keep.sum())
        print(f"  [LP] Pseudo-labels kept: {kept}/{len(keep)} "
              f"(conf≥{self.confidence_threshold:.2f})")

        if kept == 0:
            return X_labeled, y_labeled

        return (
            np.vstack([X_labeled, X_unlabeled[keep]]),
            np.concatenate([y_labeled, pseudo_labels[keep]]),
        )
