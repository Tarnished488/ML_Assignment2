"""Ensemble multiple model predictions with multiple strategies."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

def hard_voting(probs_list):
    """Hard voting: majority vote"""
    preds_list = [p.argmax(axis=1) for p in probs_list]
    stacked = np.stack(preds_list, axis=1)
    final_preds = []
    for row in stacked:
        unique, counts = np.unique(row, return_counts=True)
        final_preds.append(unique[counts.argmax()])
    return np.array(final_preds)

def soft_voting(probs_list, weights=None):
    """Soft voting: average probabilities"""
    if weights is None:
        weights = [1.0] * len(probs_list)
    weights = np.array(weights) / sum(weights)
    avg_probs = sum(w * p for w, p in zip(weights, probs_list))
    return avg_probs.argmax(axis=1)

def weighted_soft_voting(probs_list):
    """Weighted soft voting based on model performance"""
    # Simple weighting: more weight to models with better diversity
    # In practice, you'd use validation scores
    weights = [1.0] * len(probs_list)
    return soft_voting(probs_list, weights)

def rank_averaging(probs_list):
    """Rank averaging: average of predicted ranks"""
    ranked_preds = []
    for probs in probs_list:
        ranks = np.argsort(-probs, axis=1)  # descending order
        ranked_preds.append(ranks)
    stacked_ranks = np.stack(ranked_preds, axis=1)
    avg_ranks = np.mean(stacked_ranks, axis=1)
    final_ranks = np.argsort(-avg_ranks, axis=1)
    return final_ranks[:, 0]

def median_voting(probs_list):
    """Median voting: use median probability"""
    stacked_probs = np.stack(probs_list, axis=1)
    median_probs = np.median(stacked_probs, axis=1)
    return median_probs.argmax(axis=1)


def load_run_artifacts(run_dir):
    run_path = Path(run_dir)
    probs_path = run_path / "test_probs.npy"
    ids_path = run_path / "test_ids.npy"
    metrics_path = run_path / "metrics.json"

    if not probs_path.exists() or not ids_path.exists():
        raise FileNotFoundError(f"Missing test_probs.npy/test_ids.npy under {run_path}")

    probs = np.load(probs_path)
    ids = np.load(ids_path)
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return probs, ids, metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True,
                        help="List of run directories containing test_probs.npy")
    parser.add_argument("--output", default="outputs/ensemble/submission.csv")
    parser.add_argument("--method", default="soft", choices=["soft", "hard", "weighted", "rank", "median"],
                        help="Ensemble method (default: soft)")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Per-model weights (default: equal)")
    parser.add_argument("--auto-weights", default="none",
                        choices=["none", "best_val_acc", "best_val_macro_f1"],
                        help="Derive weights from each run's metrics.json.")
    args = parser.parse_args()

    all_probs = []
    all_ids = None
    derived_weights = []
    for run_dir in args.runs:
        try:
            probs, ids, metrics = load_run_artifacts(run_dir)
        except FileNotFoundError as exc:
            print(f"WARNING: {exc}, skipping.")
            continue

        all_probs.append(probs)
        if all_ids is None:
            all_ids = ids
        elif not np.array_equal(all_ids, ids):
            raise ValueError(f"Test Id mismatch detected in {run_dir}")

        if args.auto_weights == "none":
            derived_weights.append(1.0)
        else:
            metric_val = metrics.get(args.auto_weights)
            if metric_val is None:
                metric_val = 1.0
            derived_weights.append(max(float(metric_val), 1e-6))

        # Print per-model distribution
        preds = probs.argmax(axis=1)
        unique, counts = np.unique(preds, return_counts=True)
        metric_str = ""
        if args.auto_weights != "none":
            metric_str = f" | weight_source={args.auto_weights}:{derived_weights[-1]:.4f}"
        print(f"{run_dir}: {dict(zip(unique, counts))}{metric_str}")

    if not all_probs:
        print("No valid runs found.")
        return

    weights = args.weights if args.weights is not None else derived_weights

    # Choose ensemble method
    if args.method == "soft":
        final_preds = soft_voting(all_probs, weights)
    elif args.method == "hard":
        final_preds = hard_voting(all_probs)
    elif args.method == "weighted":
        final_preds = soft_voting(all_probs, weights)
    elif args.method == "rank":
        final_preds = rank_averaging(all_probs)
    elif args.method == "median":
        final_preds = median_voting(all_probs)
    else:
        final_preds = soft_voting(all_probs, weights)

    # final_preds are already 0..K-1 integers from argmax
    unique, counts = np.unique(final_preds, return_counts=True)
    print(f"\nEnsemble ({args.method}): {dict(zip(unique, counts))}")
    if weights is not None:
        print(f"Weights: {[round(float(w), 4) for w in weights]}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Id": all_ids, "Category": final_preds}).to_csv(out, index=False)
    print(f"Ensemble submission saved -> {out}")


if __name__ == "__main__":
    main()
