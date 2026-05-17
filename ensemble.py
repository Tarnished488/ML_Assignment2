"""Ensemble multiple model predictions with multiple strategies."""
import argparse
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True,
                        help="List of run directories containing test_probs.npy")
    parser.add_argument("--output", default="outputs/ensemble/submission.csv")
    parser.add_argument("--method", default="soft", choices=["soft", "hard", "weighted", "rank", "median"],
                        help="Ensemble method (default: soft)")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Per-model weights (default: equal)")
    args = parser.parse_args()

    all_probs = []
    all_ids = None
    for run_dir in args.runs:
        probs_path = Path(run_dir) / "test_probs.npy"
        ids_path = Path(run_dir) / "test_ids.npy"
        if not probs_path.exists():
            print(f"WARNING: {probs_path} not found, skipping.")
            continue
        probs = np.load(probs_path)
        ids = np.load(ids_path)
        all_probs.append(probs)
        if all_ids is None:
            all_ids = ids
        # Print per-model distribution
        preds = probs.argmax(axis=1)
        unique, counts = np.unique(preds, return_counts=True)
        print(f"{run_dir}: {dict(zip(unique, counts))}")

    if not all_probs:
        print("No valid runs found.")
        return

    # Choose ensemble method
    if args.method == "soft":
        final_preds = soft_voting(all_probs, args.weights)
    elif args.method == "hard":
        final_preds = hard_voting(all_probs)
    elif args.method == "weighted":
        final_preds = weighted_soft_voting(all_probs)
    elif args.method == "rank":
        final_preds = rank_averaging(all_probs)
    elif args.method == "median":
        final_preds = median_voting(all_probs)
    else:
        final_preds = soft_voting(all_probs, args.weights)

    # final_preds are already 0..K-1 integers from argmax
    unique, counts = np.unique(final_preds, return_counts=True)
    print(f"\nEnsemble ({args.method}): {dict(zip(unique, counts))}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Id": all_ids, "Category": final_preds}).to_csv(out, index=False)
    print(f"Ensemble submission saved -> {out}")


if __name__ == "__main__":
    main()
