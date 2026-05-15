"""Ensemble multiple model predictions via soft voting (averaged probabilities)."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True,
                        help="List of run directories containing test_probs.npy")
    parser.add_argument("--output", default="outputs/ensemble/submission.csv")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Per-model weights (default: equal)")
    args = parser.parse_args()

    all_probs = []
    for run_dir in args.runs:
        probs_path = Path(run_dir) / "test_probs.npy"
        ids_path = Path(run_dir) / "test_ids.npy"
        if not probs_path.exists():
            print(f"WARNING: {probs_path} not found, skipping.")
            continue
        probs = np.load(probs_path)
        ids = np.load(ids_path)
        all_probs.append(probs)
        # Print per-model distribution
        preds = probs.argmax(axis=1)
        unique, counts = np.unique(preds, return_counts=True)
        print(f"{run_dir}: {dict(zip(unique, counts))}")

    if not all_probs:
        print("No valid runs found.")
        return

    weights = args.weights or [1.0] * len(all_probs)
    weights = np.array(weights) / sum(weights)

    avg_probs = sum(w * p for w, p in zip(weights, all_probs))
    final_preds = avg_probs.argmax(axis=1)

    unique, counts = np.unique(final_preds, return_counts=True)
    print(f"\nEnsemble: {dict(zip(unique, counts))}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Id": ids, "Category": final_preds}).to_csv(out, index=False)
    print(f"Ensemble submission saved -> {out}")


if __name__ == "__main__":
    main()
