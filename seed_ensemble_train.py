"""Train one semi-supervised run per seed, then ensemble all test probabilities."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble import soft_voting
from src.preprocessing.data_loader import _resolve_data_dir


DEFAULT_SEEDS = [42, 123, 456, 789, 1024]


def resolve_run_weight(metrics: dict) -> float:
    """Choose a safe ensemble weight even when no validation split is used."""
    for key in ("best_val_macro_f1", "best_val_acc"):
        value = metrics.get(key)
        if value is not None:
            return max(float(value), 1e-6)
    return 1.0


def build_train_command(args, seed: int) -> list[str]:
    cmd = [
        sys.executable,
        "train_mlp.py",
        "--name", f"{args.name}_s{seed}",
        "--output-dir", args.output_dir,
        "--data-dir", args.data_dir,
        "--models", "mlp",
        "--use-ssl",
        "--ssl-method", args.ssl_method,
        "--hidden-dims", args.hidden_dims,
        "--dropout", str(args.dropout),
        "--activation", args.activation,
        "--norm", args.norm,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--patience", str(args.patience),
        "--lp-k", str(args.lp_k),
        "--lp-alpha", str(args.lp_alpha),
        "--lp-top-k", str(args.lp_top_k),
        "--lp-conf", str(args.lp_conf),
        "--self-train-rounds", str(args.self_train_rounds),
        "--self-train-threshold", str(args.self_train_threshold),
        "--pretrain-epochs", str(args.pretrain_epochs),
        "--pretrain-patience", str(args.pretrain_patience),
        "--pretrain-threshold-quantile", str(args.pretrain_threshold_quantile),
        "--pretrain-threshold-min", str(args.pretrain_threshold_min),
        "--pretrain-threshold-max", str(args.pretrain_threshold_max),
        "--pretrain-top-k", str(args.pretrain_top_k),
        "--seed", str(seed),
        "--no-baseline",
        "--no-viz",
    ]

    if not args.no_pretrain_first:
        cmd.append("--pretrain-first")
    if args.use_vat:
        cmd.append("--use-vat")
    if args.use_all_labeled:
        cmd.append("--use-all-labeled")
    if args.pretrain_lr is not None:
        cmd.extend(["--pretrain-lr", str(args.pretrain_lr)])
    return cmd


def run_dir_for(args, seed: int) -> Path:
    return Path(args.output_dir) / f"{args.name}_s{seed}_mlp_{args.ssl_method}"


def main():
    parser = argparse.ArgumentParser(
        description="Run the same SSL setup across multiple seeds and ensemble them."
    )
    parser.add_argument("--name", default="seed_ensemble")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--data-dir", default=_resolve_data_dir())
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--ssl-method", choices=["pseudo", "distill", "self_training"],
                        default="self_training")
    parser.add_argument("--ensemble-output", default="outputs/ensemble/submission.csv")
    parser.add_argument("--hidden-dims", default="128,64")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--activation", choices=["relu", "gelu", "silu"], default="gelu")
    parser.add_argument("--norm", choices=["batch", "layer"], default="batch")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--lp-k", type=int, default=10)
    parser.add_argument("--lp-alpha", type=float, default=0.99)
    parser.add_argument("--lp-top-k", type=int, default=300)
    parser.add_argument("--lp-conf", type=float, default=0.6)
    parser.add_argument("--self-train-rounds", type=int, default=3)
    parser.add_argument("--self-train-threshold", type=float, default=0.85)
    parser.add_argument("--no-pretrain-first", action="store_true",
                        help="Disable the default pretrain-then-self-train pipeline.")
    parser.add_argument("--pretrain-epochs", type=int, default=120)
    parser.add_argument("--pretrain-patience", type=int, default=25)
    parser.add_argument("--pretrain-lr", type=float, default=None)
    parser.add_argument("--pretrain-threshold-quantile", type=float, default=0.75)
    parser.add_argument("--pretrain-threshold-min", type=float, default=0.70)
    parser.add_argument("--pretrain-threshold-max", type=float, default=0.95)
    parser.add_argument("--pretrain-top-k", type=int, default=128)
    parser.add_argument("--use-vat", action="store_true")
    parser.add_argument("--use-all-labeled", action="store_true")
    args = parser.parse_args()

    run_summaries = []
    probs_list = []
    ids_ref = None
    weights = []

    for seed in args.seeds:
        cmd = build_train_command(args, seed)
        print(f"\n[{seed}] Running {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parent)
        if proc.returncode != 0:
            raise RuntimeError(f"Seed {seed} failed with return code {proc.returncode}")

        run_dir = run_dir_for(args, seed)
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        probs = np.load(run_dir / "test_probs.npy")
        ids = np.load(run_dir / "test_ids.npy")

        if ids_ref is None:
            ids_ref = ids
        elif not np.array_equal(ids_ref, ids):
            raise ValueError(f"Test Id mismatch detected for seed {seed}")

        probs_list.append(probs)
        weights.append(resolve_run_weight(metrics))

        run_summaries.append({
            "seed": seed,
            "run_dir": str(run_dir),
            "best_val_acc": metrics.get("best_val_acc"),
            "best_val_macro_f1": metrics.get("best_val_macro_f1"),
            "best_epoch": metrics.get("best_epoch"),
            "train_size_final": metrics.get("train_size_final"),
        })

    final_preds = soft_voting(probs_list, weights)
    out_path = Path(args.ensemble_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Id": ids_ref, "Category": final_preds}).to_csv(out_path, index=False)

    summary_path = out_path.with_name(f"{out_path.stem}_summary.json")
    summary_path.write_text(json.dumps({
        "name": args.name,
        "ssl_method": args.ssl_method,
        "seeds": args.seeds,
        "weights": weights,
        "runs": run_summaries,
    }, indent=2), encoding="utf-8")

    print(f"\nEnsemble submission saved -> {out_path}")
    print(f"Summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
