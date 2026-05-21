"""Clustering-based semi-supervised learning via label broadcasting.

Two strategies are provided:

1. **ClusteringSSL** – Standard KMeans++ clustering + majority-label broadcasting.
2. **ConstrainedClusteringSSL** – Seed-based constrained KMeans where labeled
   samples anchor cluster centers and are clamped to their true class during
   assignment.  Optionally integrates with Label Propagation for a
   "cluster-then-propagate" pipeline.
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


class ConstrainedClusteringSSL(ClusteringSSL):
    """Seed-based constrained KMeans + optional Label Propagation pipeline.

    Instead of random KMeans++ init, cluster centers are **seeded** from the
    labeled samples (one per class or the mean per class).  During assignment,
    labeled samples are clamped to their ground-truth cluster so they cannot
    drift into neighbouring clusters.

    Optionally, after broadcasting labels within clusters, a Label Propagation
    pass can further propagate the expanded labels through a k-NN graph
    ("cluster-then-propagate").

    Reference
    ---------
    Basu, Banerjee & Mooney, "Semi-supervised Clustering by Seeding"
    (ICML 2002).
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
        seed_mode: str = "centroid",
        max_seeds_per_class: int = 5,
        propagate_after_cluster: bool = True,
        lp_kwargs: dict | None = None,
    ):
        super().__init__(
            n_clusters=n_clusters,
            confidence_threshold=confidence_threshold,
            class_balanced=class_balanced,
            top_k_per_class=top_k_per_class,
            min_labeled_per_cluster=min_labeled_per_cluster,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
        )
        self.seed_mode = seed_mode
        self.max_seeds_per_class = max_seeds_per_class
        self.propagate_after_cluster = propagate_after_cluster

        lp_defaults = dict(
            n_neighbors=10,
            alpha=0.99,
            class_balanced=True,
            top_k_per_class=300,
            confidence_threshold=0.6,
        )
        if lp_kwargs:
            lp_defaults.update(lp_kwargs)
        self.lp_kwargs = lp_defaults

    # ------------------------------------------------------------------
    # Seed-based center initialisation
    # ------------------------------------------------------------------

    def _init_centers_seeded(
        self,
        X: np.ndarray,
        y_labeled: np.ndarray,
        n_labeled: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[int, int]]:
        """Initialise cluster centers using labeled samples as seeds.

        Returns
        -------
        centers : ndarray (n_clusters, D)
        seed_cluster_map : dict  {class_id → cluster_id}
            Mapping from true class to the cluster that class seeds.
        """
        n_clusters = min(self.n_clusters, X.shape[0])
        n_features = X.shape[1]
        classes = np.unique(y_labeled)
        n_classes = len(classes)
        n_seed_clusters = min(n_classes, n_clusters)
        centers = np.empty((n_clusters, n_features), dtype=np.float64)
        seed_cluster_map: dict[int, int] = {}

        if self.seed_mode == "centroid":
            for i, cls in enumerate(classes):
                cid = i
                cls_samples = X[:n_labeled][y_labeled == cls]
                centers[cid] = cls_samples.mean(axis=0)
                seed_cluster_map[int(cls)] = cid
        elif self.seed_mode == "multi":
            cid = 0
            for cls in classes:
                cls_samples = X[:n_labeled][y_labeled == cls]
                n_seeds = min(self.max_seeds_per_class, len(cls_samples))
                if n_seeds == 1 or len(cls_samples) <= self.max_seeds_per_class:
                    chosen = cls_samples
                else:
                    indices = rng.choice(len(cls_samples), n_seeds, replace=False)
                    chosen = cls_samples[indices]
                for j in range(len(chosen)):
                    if cid >= n_clusters:
                        break
                    centers[cid] = chosen[j]
                    if j == 0:
                        seed_cluster_map[int(cls)] = cid
                    cid += 1
                if cid >= n_clusters:
                    break
            n_seed_clusters = cid
        else:
            raise ValueError(f"Unknown seed_mode: {self.seed_mode}")

        # Fill remaining centers with KMeans++ on unlabeled data
        if n_seed_clusters < n_clusters:
            X_unlabeled = X[n_labeled:]
            remaining = n_clusters - n_seed_clusters

            # KMeans++ for the remainder
            closest_dist_sq = self._squared_distances(X_unlabeled, centers[:n_seed_clusters]).min(axis=1)
            for cid in range(n_seed_clusters, n_clusters):
                total = closest_dist_sq.sum()
                if total <= 0:
                    chosen = rng.integers(len(X_unlabeled))
                else:
                    chosen = rng.choice(len(X_unlabeled), p=closest_dist_sq / total)
                centers[cid] = X_unlabeled[chosen]
                new_dist_sq = self._squared_distances(
                    X_unlabeled, centers[cid:cid + 1]
                ).ravel()
                closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)

        return centers, seed_cluster_map

    # ------------------------------------------------------------------
    # Constrained KMeans  (labeled samples clamped)
    # ------------------------------------------------------------------

    def _fit_predict_constrained_kmeans(
        self,
        X: np.ndarray,
        y_labeled: np.ndarray,
        n_labeled: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run constrained KMeans with labeled samples fixed to seed clusters."""
        X = X.astype(np.float64, copy=False)
        n_clusters = min(self.n_clusters, X.shape[0])
        classes = np.unique(y_labeled)
        class_to_seed: dict[int, int] = {int(c): c % n_clusters for c in classes}

        base_rng = np.random.default_rng(self.random_state)
        best_labels = None
        best_centers = None
        best_inertia = np.inf

        for _ in range(self.n_init):
            rng = np.random.default_rng(base_rng.integers(0, 2**32 - 1))
            centers, seed_map = self._init_centers_seeded(
                X, y_labeled, n_labeled, rng
            )
            class_to_seed = seed_map

            labels = np.full(X.shape[0], -1, dtype=np.int64)
            # Clamp labeled samples
            for i in range(n_labeled):
                cls = int(y_labeled[i])
                labels[i] = class_to_seed.get(cls, cls % n_clusters)

            for _ in range(self.max_iter):
                distances = self._squared_distances(X, centers)

                # Unlabeled: assign to nearest center
                new_labels = labels.copy()
                unl_dists = distances[n_labeled:]
                new_labels[n_labeled:] = unl_dists.argmin(axis=1)

                # Labeled: keep clamped to seed cluster
                for i in range(n_labeled):
                    cls = int(y_labeled[i])
                    new_labels[i] = class_to_seed.get(cls, cls % n_clusters)

                if np.array_equal(new_labels, labels):
                    break
                labels = new_labels

                for cid in range(n_clusters):
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

    # ------------------------------------------------------------------
    # Cluster + propagate pipeline
    # ------------------------------------------------------------------

    def generate_pseudo_labels(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run constrained KMeans, then broadcast labels within clusters."""
        n_labeled = len(y_labeled)
        X_all = np.vstack([X_labeled, X_unlabeled])

        cluster_ids, _ = self._fit_predict_constrained_kmeans(
            X_all, y_labeled, n_labeled
        )

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

    def cluster_then_propagate(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        num_classes: int,
        lp_confidence_threshold: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cluster-then-propagate pipeline.

        1. Run constrained KMeans → broadcast cluster labels.
        2. Select high-confidence cluster pseudo-labels.
        3. Use the expanded labeled set as seeds for Label Propagation.
        4. LP propagates labels through a k-NN graph for final predictions.

        Returns
        -------
        pseudo_labels : ndarray (n_unlabeled,)
        confidences : ndarray (n_unlabeled,)
            Final pseudo-labels and confidences from LP.
        """
        from src.ssl.label_propagation import LabelPropagationSSL

        print(
            f"  [ConstrainedClustering] Seed-based KMeans "
            f"(k={self.n_clusters}, seed_mode={self.seed_mode}) ..."
        )
        cluster_labels, cluster_confs = self.generate_pseudo_labels(
            X_labeled=X_labeled,
            y_labeled=y_labeled,
            X_unlabeled=X_unlabeled,
        )
        keep = self._select_pseudo_labels(cluster_labels, cluster_confs, num_classes)
        print(
            f"  [ConstrainedClustering] Broadcast labels kept: "
            f"{int(keep.sum())}/{len(keep)} (purity>={self.confidence_threshold:.2f})"
        )

        if not keep.any():
            # Fall back to LP on original labeled data only
            n_labeled = len(y_labeled)
            X_all = np.vstack([X_labeled, X_unlabeled])
            lp = LabelPropagationSSL(**self.lp_kwargs)
            return lp.propagate(X_all, y_labeled, n_labeled)

        # Expand labeled set with high-confidence cluster pseudo-labels
        X_expanded = np.vstack([X_labeled, X_unlabeled[keep]])
        y_expanded = np.concatenate([y_labeled, cluster_labels[keep]])
        X_remaining = X_unlabeled[~keep]

        print(
            f"  [ConstrainedClustering] Expanded labeled set: {len(y_expanded)} "
            f"(+{int(keep.sum())} from clusters). Running LP ..."
        )

        lp = LabelPropagationSSL(**self.lp_kwargs)
        if lp_confidence_threshold is not None:
            lp.confidence_threshold = lp_confidence_threshold

        n_expanded = len(y_expanded)
        X_all = np.vstack([X_expanded, X_remaining])
        pseudo_labels, confidences = lp.propagate(X_all, y_expanded, n_expanded)
        keep_lp = lp.select_pseudo_labels(pseudo_labels, confidences, num_classes)

        final_labels = np.full(len(X_unlabeled), -1, dtype=np.int64)
        final_confs = np.zeros(len(X_unlabeled), dtype=np.float64)

        # Cluster-kept indices get their cluster labels
        kept_indices = np.where(keep)[0]
        final_labels[kept_indices] = cluster_labels[keep]
        final_confs[kept_indices] = cluster_confs[keep]

        # LP-kept indices on remaining get LP labels
        remaining_indices = np.where(~keep)[0]
        final_labels[remaining_indices[keep_lp]] = pseudo_labels[keep_lp]
        final_confs[remaining_indices[keep_lp]] = confidences[keep_lp]

        print(
            f"  [ConstrainedClustering] Final: {int(final_labels >= 0)} pseudo-labels "
            f"({int(keep.sum())} cluster + {int(keep_lp.sum())} LP)"
        )
        return final_labels, final_confs

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
        """High-level API: constrained clustering → label broadcast → augment."""
        del model, device

        if self.propagate_after_cluster:
            pseudo_labels, confidences = self.cluster_then_propagate(
                X_labeled=X_labeled,
                y_labeled=y_labeled,
                X_unlabeled=X_unlabeled,
                num_classes=num_classes,
            )
            keep = confidences >= self.confidence_threshold
            kept = int(keep.sum())
        else:
            pseudo_labels, confidences = self.generate_pseudo_labels(
                X_labeled=X_labeled,
                y_labeled=y_labeled,
                X_unlabeled=X_unlabeled,
            )
            keep = self._select_pseudo_labels(pseudo_labels, confidences, num_classes)
            kept = int(keep.sum())

        print(
            f"  [ConstrainedClustering] Final pseudo-labels kept: {kept}/{len(keep)}"
        )

        if kept == 0:
            return X_labeled, y_labeled

        return (
            np.vstack([X_labeled, X_unlabeled[keep]]),
            np.concatenate([y_labeled, pseudo_labels[keep]]),
        )
