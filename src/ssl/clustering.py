"""Clustering-based semi-supervised learning via KMeans.

Clusters all data (labeled + unlabeled) using KMeans, then assigns
pseudo-labels to unlabeled samples within each cluster via majority voting
of labeled neighbours.  Confidence is derived from cluster purity — higher
purity → more trustworthy pseudo-labels.

Reference: Zhu & Goldberg, "Introduction to Semi-Supervised Learning" (2009),
Chapter 5: "Semi-supervised clustering."
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class ClusteringSSL:
    """KMeans-based clustering pseudo-labeling.

    Parameters
    ----------
    n_clusters : int
        Number of KMeans clusters (should be >= num_classes).
    confidence_threshold : float
        Minimum cluster purity / label ratio to accept a pseudo-label.
    class_balanced : bool
        If True, select top-k confident pseudo-labels *per class*.
    top_k_per_class : int or None
        Number of pseudo-labels to keep per class when class_balanced=True.
    """

    def __init__(
        self,
        n_clusters: int = 50,
        confidence_threshold: float = 0.5,
        class_balanced: bool = True,
        top_k_per_class: int | None = 300,
    ):
        self.n_clusters = n_clusters
        self.confidence_threshold = confidence_threshold
        self.class_balanced = class_balanced
        self.top_k_per_class = top_k_per_class

    def _cluster_and_vote(
        self,
        X_all: np.ndarray,
        y_labeled: np.ndarray,
        n_labeled: int,
        num_classes: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run KMeans, then vote within each cluster to produce pseudo-labels.

        Returns
        -------
        pseudo_labels : ndarray of shape (n_unlabeled,)
        confidences : ndarray of shape (n_unlabeled,)
        """
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
        )
        cluster_ids = kmeans.fit_predict(X_all)  # (n_total,)

        # Nearest labeled centroid per cluster → distance-based fallback
        centroids = kmeans.cluster_centers_

        # Build per-cluster label histogram from labeled data
        cluster_labeled_hist = {}
        for idx in range(n_labeled):
            cid = cluster_ids[idx]
            lbl = y_labeled[idx]
            if cid not in cluster_labeled_hist:
                cluster_labeled_hist[cid] = {}
            cluster_labeled_hist[cid][lbl] = cluster_labeled_hist[cid].get(lbl, 0) + 1

        unlabeled_ids = cluster_ids[n_labeled:]
        pseudo_labels = np.zeros(len(unlabeled_ids), dtype=np.int64)
        confidences = np.zeros(len(unlabeled_ids), dtype=np.float64)

        for i, global_idx in enumerate(range(n_labeled, len(X_all))):
            cid = cluster_ids[global_idx]

            if cid not in cluster_labeled_hist:
                # No labeled sample in this cluster — assign by nearest centroid
                # among centroids that have at least one labeled sample
                confidences[i] = 0.0
                pseudo_labels[i] = 0
                continue

            hist = cluster_labeled_hist[cid]
            total_labeled_in_cluster = sum(hist.values())
            # Majority label and its purity
            majority_label = max(hist, key=hist.get)
            purity = hist[majority_label] / total_labeled_in_cluster

            # Distance to cluster centroid as additional confidence signal
            dist_to_centroid = np.linalg.norm(
                X_all[global_idx] - centroids[cid]
            )
            # Normalise distance: smaller → higher confidence
            dist_factor = 1.0 / (1.0 + dist_to_centroid)

            pseudo_labels[i] = majority_label
            # Blend cluster purity with distance proximity
            confidences[i] = 0.7 * purity + 0.3 * dist_factor * purity

        return pseudo_labels, confidences

    def _select_pseudo_labels(
        self,
        pseudo_labels: np.ndarray,
        confidences: np.ndarray,
        num_classes: int,
    ) -> np.ndarray:
        """Return boolean mask of which pseudo-labels to keep."""
        keep = np.zeros(len(pseudo_labels), dtype=bool)

        if self.class_balanced and self.top_k_per_class is not None:
            for cls in range(num_classes):
                cls_mask = pseudo_labels == cls
                cls_indices = np.where(cls_mask)[0]
                if len(cls_indices) == 0:
                    continue
                sorted_idx = cls_indices[np.argsort(-confidences[cls_indices])]
                top = sorted_idx[: self.top_k_per_class]
                top = top[confidences[top] >= self.confidence_threshold]
                keep[top] = True
        else:
            keep = confidences >= self.confidence_threshold

        return keep

    def __call__(
        self,
        model,  # unused — clustering doesn't need a model
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        device,
        num_classes: int,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run KMeans clustering → pseudo-label → augmented training set."""
        n_labeled = len(y_labeled)
        X_all = np.vstack([X_labeled, X_unlabeled])

        print(
            f"  [ClusteringSSL] KMeans (k={self.n_clusters}, "
            f"n_total={X_all.shape[0]}) ..."
        )
        pseudo_labels, confidences = self._cluster_and_vote(
            X_all, y_labeled, n_labeled, num_classes,
        )

        keep = self._select_pseudo_labels(pseudo_labels, confidences, num_classes)
        kept = int(keep.sum())
        print(
            f"  [ClusteringSSL] Pseudo-labels kept: {kept}/{len(keep)} "
            f"(conf≥{self.confidence_threshold:.2f})"
        )

        if kept == 0:
            return X_labeled, y_labeled

        return (
            np.vstack([X_labeled, X_unlabeled[keep]]),
            np.concatenate([y_labeled, pseudo_labels[keep]]),
        )
