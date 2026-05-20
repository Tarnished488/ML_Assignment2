"""Clustering-based semi-supervised learning via label broadcasting.

All samples are clustered together. If a cluster contains labeled samples, the
majority labeled class is broadcast to unlabeled samples in that cluster. The
cluster purity is used as the pseudo-label confidence.
"""

from __future__ import annotations

import numpy as np


class ClusteringSSL:
    """KMeans clustering + majority-label broadcasting.

    The KMeans implementation is intentionally local numpy code rather than a
    ready-made sklearn model, which keeps the method easy to explain for the
    assignment report and interview.
    """

    def __init__(
        self,
        n_clusters: int = 50,
        confidence_threshold: float = 0.5,
        class_balanced: bool = True,
        top_k_per_class: int | None = 300,
        min_labeled_per_cluster: int = 1,
        max_iter: int = 100,
        n_init: int = 5,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.confidence_threshold = confidence_threshold
        self.class_balanced = class_balanced
        self.top_k_per_class = top_k_per_class
        self.min_labeled_per_cluster = min_labeled_per_cluster
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

    @staticmethod
    def _squared_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        x_norm = np.sum(X * X, axis=1, keepdims=True)
        c_norm = np.sum(centers * centers, axis=1)
        distances = x_norm + c_norm[None, :] - 2.0 * (X @ centers.T)
        return np.maximum(distances, 0.0)

    @staticmethod
    def _replace_empty_center(
        X: np.ndarray,
        centers: np.ndarray,
        labels: np.ndarray,
        distances: np.ndarray,
        cid: int,
    ) -> None:
        assigned_dist = distances[np.arange(X.shape[0]), labels]
        farthest_idx = int(np.argmax(assigned_dist))
        centers[cid] = X[farthest_idx]

    def _init_centers(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """KMeans++ center initialization."""
        n_samples = X.shape[0]
        n_clusters = min(self.n_clusters, n_samples)
        centers = np.empty((n_clusters, X.shape[1]), dtype=np.float64)
        centers[0] = X[rng.integers(n_samples)]

        closest_dist_sq = self._squared_distances(X, centers[:1]).ravel()
        for center_idx in range(1, n_clusters):
            total = closest_dist_sq.sum()
            if total <= 0:
                chosen = rng.integers(n_samples)
            else:
                chosen = rng.choice(n_samples, p=closest_dist_sq / total)
            centers[center_idx] = X[chosen]
            new_dist_sq = self._squared_distances(
                X, centers[center_idx:center_idx + 1]
            ).ravel()
            closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)
        return centers

    def _fit_predict_kmeans(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run KMeans and return cluster ids plus centers."""
        X = X.astype(np.float64, copy=False)
        base_rng = np.random.default_rng(self.random_state)
        best_labels = None
        best_centers = None
        best_inertia = np.inf

        for _ in range(self.n_init):
            rng = np.random.default_rng(base_rng.integers(0, 2**32 - 1))
            centers = self._init_centers(X, rng)
            labels = np.full(X.shape[0], -1, dtype=np.int64)

            for _ in range(self.max_iter):
                distances = self._squared_distances(X, centers)
                new_labels = distances.argmin(axis=1)
                if np.array_equal(new_labels, labels):
                    break
                labels = new_labels

                for cid in range(centers.shape[0]):
                    members = X[labels == cid]
                    if len(members) == 0:
                        self._replace_empty_center(X, centers, labels, distances, cid)
                    else:
                        centers[cid] = members.mean(axis=0)

            distances = self._squared_distances(X, centers)
            inertia = float(distances[np.arange(X.shape[0]), labels].sum())
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
                best_centers = centers.copy()

        return best_labels, best_centers

    def _cluster_and_vote(
        self,
        X_all: np.ndarray,
        y_labeled: np.ndarray,
        n_labeled: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cluster all data, then broadcast cluster majority labels."""
        cluster_ids, _ = self._fit_predict_kmeans(X_all)

        cluster_labeled_hist: dict[int, dict[int, int]] = {}
        for idx in range(n_labeled):
            cid = int(cluster_ids[idx])
            label = int(y_labeled[idx])
            cluster_labeled_hist.setdefault(cid, {})
            cluster_labeled_hist[cid][label] = cluster_labeled_hist[cid].get(label, 0) + 1

        pseudo_labels = np.zeros(X_all.shape[0] - n_labeled, dtype=np.int64)
        confidences = np.zeros(X_all.shape[0] - n_labeled, dtype=np.float64)

        for out_idx, global_idx in enumerate(range(n_labeled, X_all.shape[0])):
            cid = int(cluster_ids[global_idx])
            hist = cluster_labeled_hist.get(cid)
            if not hist:
                continue

            labeled_count = sum(hist.values())
            if labeled_count < self.min_labeled_per_cluster:
                continue

            majority_label = max(hist, key=hist.get)
            purity = hist[majority_label] / labeled_count
            pseudo_labels[out_idx] = majority_label
            confidences[out_idx] = purity

        return pseudo_labels, confidences

    def _select_pseudo_labels(
        self,
        pseudo_labels: np.ndarray,
        confidences: np.ndarray,
        num_classes: int,
    ) -> np.ndarray:
        keep = np.zeros(len(pseudo_labels), dtype=bool)

        if self.class_balanced and self.top_k_per_class is not None:
            for cls in range(num_classes):
                cls_indices = np.where(pseudo_labels == cls)[0]
                if len(cls_indices) == 0:
                    continue
                sorted_idx = cls_indices[np.argsort(-confidences[cls_indices])]
                top = sorted_idx[: self.top_k_per_class]
                top = top[confidences[top] >= self.confidence_threshold]
                keep[top] = True
        else:
            keep = confidences >= self.confidence_threshold

        return keep

    def generate_pseudo_labels(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate cluster-broadcast labels and purity confidences."""
        n_labeled = len(y_labeled)
        X_all = np.vstack([X_labeled, X_unlabeled])
        return self._cluster_and_vote(
            X_all=X_all,
            y_labeled=y_labeled,
            n_labeled=n_labeled,
        )

    def select_pseudo_labels(
        self,
        pseudo_labels: np.ndarray,
        confidences: np.ndarray,
        num_classes: int,
    ) -> np.ndarray:
        """Public selection API for callers that add extra filters."""
        return self._select_pseudo_labels(pseudo_labels, confidences, num_classes)

    def __call__(
        self,
        model,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        device,
        num_classes: int,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return labeled data augmented with cluster-broadcast pseudo-labels."""
        del model, device
        n_labeled = len(y_labeled)
        X_all = np.vstack([X_labeled, X_unlabeled])

        print(
            f"  [ClusteringSSL] KMeans label broadcasting "
            f"(k={self.n_clusters}, n_total={X_all.shape[0]}) ..."
        )
        pseudo_labels, confidences = self._cluster_and_vote(
            X_all=X_all,
            y_labeled=y_labeled,
            n_labeled=n_labeled,
        )

        keep = self._select_pseudo_labels(pseudo_labels, confidences, num_classes)
        kept = int(keep.sum())
        print(
            f"  [ClusteringSSL] Broadcast labels kept: {kept}/{len(keep)} "
            f"(purity>={self.confidence_threshold:.2f})"
        )

        if kept == 0:
            return X_labeled, y_labeled

        return (
            np.vstack([X_labeled, X_unlabeled[keep]]),
            np.concatenate([y_labeled, pseudo_labels[keep]]),
        )
