"""Hyperparameter tuning for the ML_Assignment2 SSL pipeline.

Supports grid search and random search.  Each trial runs train_mlp.py
as a subprocess so that GPU memory / random state is fully isolated.

Usage:
    # Grid search over a small set of parameters
    python tune_hyperparams.py --method grid --ssl-method distill --trials 0

    # Random search with N trials
    python tune_hyperparams.py --method random --ssl-method distill --trials 30

    # Tune LP parameters specifically
    python tune_hyperparams.py --method grid --ssl-method pseudo --preset lp

    # Resume a partial run (skips trials that already have results)
    python tune_hyperparams.py --method random --ssl-method distill --trials 50

Output:
    outputs/tuning/<ssl_method>/tuning_results.csv  — per-trial metrics
    outputs/tuning/<ssl_method>/<trial_name>/        — full outputs per trial
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing.data_loader import _resolve_data_dir

# ---------------------------------------------------------------------------
# Parameter search spaces
# ---------------------------------------------------------------------------

# Each preset defines which params to sweep and their candidate values.
# "fixed" values won't be searched — they stay constant.

PRESETS: dict[str, dict[str, Any]] = {
    # ── Model architecture ──────────────────────────────────────────
    "model": {
        "hidden_dims": ["128", "128,64", "128,64,32", "256,128,64"],
        "dropout": [0.2, 0.3, 0.4, 0.5, 0.6],
        "activation": ["relu", "gelu", "silu"],
        "norm": ["batch", "layer"],
    },

    # ── Training dynamics ───────────────────────────────────────────
    "train": {
        "lr": [1e-4, 3e-4, 5e-4, 1e-3, 3e-3],
        "weight_decay": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        "batch_size": [16, 32, 64],
    },

    # ── Label Propagation ───────────────────────────────────────────
    "lp": {
        "lp_k": [5, 10, 15, 20, 30],
        "lp_alpha": [0.90, 0.95, 0.99, 0.999],
        "lp_conf": [0.4, 0.5, 0.6, 0.7],
        "lp_top_k": [100, 200, 300, 500],
    },

    # ── Knowledge Distillation ──────────────────────────────────────
    "distill": {
        "distill_T": [1.5, 2.0, 3.0, 4.0, 5.0],
        "distill_alpha": [1.0, 3.0, 5.0, 10.0, 20.0, 30.0],
    },

    # ── VAT consistency ─────────────────────────────────────────────
    "vat": {
        "vat_weight": [0.1, 0.2, 0.3, 0.5, 1.0],
        "vat_epsilon": [1.0, 2.0, 4.0, 6.0, 8.0],
    },

    # ── Self-training ───────────────────────────────────────────────
    "self_train": {
        "self_train_rounds": [2, 3, 4, 5],
        "self_train_threshold": [0.75, 0.80, 0.85, 0.90, 0.95],
    },

    "clustering": {
        "cluster_n_clusters": [30, 50, 80, 120],
        "cluster_conf": [0.4, 0.5, 0.6, 0.7],
        "cluster_top_k": [200, 300, 500],
        "cluster_model_conf": [0.6, 0.7, 0.8],
    },

    "cluster_propagation": {
        "cluster_n_clusters": [30, 50, 80, 120],
        "cluster_conf": [0.4, 0.5, 0.6],
        "cluster_top_k": [300, 500],
        "cluster_model_conf": [0.6, 0.7, 0.8],
        "lp_k": [10, 15, 20],
        "lp_conf": [0.5, 0.6, 0.7],
        "lp_top_k": [300, 500],
    },

    # ── ST fine-tuning (threshold + LP interaction) ─────────────────
    # Self-adaptive: per-class thresholds + curriculum decay per round.
    # Literature (FreeMatch/PabLO 2024): fixed 0.95 is suboptimal;
    # per-class adaptive + curriculum → 5-11% improvement.
    "st_finetune": {
        "self_train_rounds": [3, 4, 5],
        "self_train_threshold": [0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        "lp_conf": [0.4, 0.5, 0.6, 0.7],
        "lp_top_k": [200, 300, 400, 500],
    },

    # ── ST dynamic threshold tuning ────────────────────────────────
    "st_dynamic": {
        "st_initial_threshold": [0.90, 0.95, 0.98],
        "st_threshold_decay": [0.75, 0.80, 0.85, 0.90],
        "st_min_threshold": [0.60, 0.65, 0.70, 0.75],
        "st_top_k_per_round": [300, 400, 500, 600, 800],
        "st_per_class_adjustment": [0.05, 0.10, 0.15, 0.20],
        "self_train_rounds": [3, 4, 5],
        "self_train_threshold": [0.80, 0.85, 0.90],
    },

    # ── Combined — top-12 most impactful params (for random search) ──
    "combined": {
        "lr": [1e-4, 3e-4, 5e-4, 1e-3, 3e-3],
        "dropout": [0.2, 0.3, 0.4, 0.5, 0.6],
        "weight_decay": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        "lp_k": [5, 10, 15, 20, 30],
        "lp_alpha": [0.90, 0.95, 0.99],
        "lp_conf": [0.4, 0.5, 0.6, 0.7],
        "distill_T": [1.5, 2.0, 3.0, 4.0, 5.0],
        "distill_alpha": [1.0, 3.0, 5.0, 10.0, 20.0],
        "vat_weight": [0.1, 0.2, 0.3, 0.5],
        "vat_epsilon": [1.0, 2.0, 4.0, 6.0],
        "self_train_rounds": [3, 4, 5],
        "self_train_threshold": [0.75, 0.80, 0.85, 0.90, 0.95],
    },
}

# Default fixed values used when a parameter is NOT in the search space
DEFAULT_FIXED = {
    "hidden_dims": "128,64",
    "dropout": 0.3,
    "activation": "gelu",
    "norm": "batch",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "lp_k": 10,
    "lp_alpha": 0.99,
    "lp_conf": 0.6,
    "lp_top_k": 300,
    "distill_T": 2.0,
    "distill_alpha": 5.0,
    "vat_weight": 0.3,
    "vat_epsilon": 2.0,
    "self_train_rounds": 3,
    "self_train_threshold": 0.85,
    "cluster_n_clusters": 50,
    "cluster_conf": 0.5,
    "cluster_top_k": 300,
    "cluster_min_labeled": 1,
    "cluster_max_iter": 80,
    "cluster_n_init": 3,
    "cluster_model_conf": 0.7,
    "st_initial_threshold": 0.95,
    "st_threshold_decay": 0.85,
    "st_min_threshold": 0.70,
    "st_top_k_per_round": 500,
    "st_per_class_adjustment": 0.15,
    "cluster_seed_mode": "centroid",
    "cluster_max_seeds": 5,
    "epochs": 300,
    "patience": 50,
    "lr_factor": 0.5,
    "lr_patience": 15,
}


# ---------------------------------------------------------------------------
# Search strategy helpers
# ---------------------------------------------------------------------------


def build_grid(param_space: dict[str, list]) -> list[dict[str, Any]]:
    """Cartesian product of all parameter values → list of config dicts."""
    keys = list(param_space.keys())
    values = list(param_space.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def sample_random(param_space: dict[str, list], n: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Sample n random configs from the parameter space (with replacement)."""
    keys = list(param_space.keys())
    configs = []
    for _ in range(n):
        cfg = {}
        for k in keys:
            cfg[k] = rng.choice(param_space[k])
        configs.append(cfg)
    return configs


# ---------------------------------------------------------------------------
# Trial execution
# ---------------------------------------------------------------------------


def build_command(trial_cfg: dict[str, Any], ssl_method: str, data_dir: str) -> list[str]:
    """Convert a trial config dict into a train_mlp.py CLI argument list."""
    cmd = [
        sys.executable, "train_mlp.py",
        "--data-dir", data_dir,
        "--models", "mlp",
        "--use-ssl",
        "--ssl-method", ssl_method,
        "--no-viz",
    ]

    # Map trial_cfg keys → CLI flags
    key_to_flag = {
        "hidden_dims": "--hidden-dims",
        "dropout": "--dropout",
        "activation": "--activation",
        "norm": "--norm",
        "lr": "--lr",
        "weight_decay": "--weight-decay",
        "batch_size": "--batch-size",
        "lp_k": "--lp-k",
        "lp_alpha": "--lp-alpha",
        "lp_conf": "--lp-conf",
        "lp_top_k": "--lp-top-k",
        "distill_T": "--distill-T",
        "distill_alpha": "--distill-alpha",
        "vat_weight": "--vat-weight",
        "vat_epsilon": "--vat-epsilon",
        "self_train_rounds": "--self-train-rounds",
        "self_train_threshold": "--self-train-threshold",
        "cluster_n_clusters": "--cluster-n-clusters",
        "cluster_conf": "--cluster-conf",
        "cluster_top_k": "--cluster-top-k",
        "cluster_min_labeled": "--cluster-min-labeled",
        "cluster_max_iter": "--cluster-max-iter",
        "cluster_n_init": "--cluster-n-init",
        "cluster_model_conf": "--cluster-model-conf",
        "st_initial_threshold": "--st-initial-threshold",
        "st_threshold_decay": "--st-threshold-decay",
        "st_min_threshold": "--st-min-threshold",
        "st_top_k_per_round": "--st-top-k-per-round",
        "st_per_class_adjustment": "--st-per-class-adjustment",
        "cluster_seed_mode": "--cluster-seed-mode",
        "cluster_max_seeds": "--cluster-max-seeds",
        "epochs": "--epochs",
        "patience": "--patience",
        "lr_factor": "--lr-factor",
        "lr_patience": "--lr-patience",
        "seed": "--seed",
    }

    for key, value in trial_cfg.items():
        flag = key_to_flag.get(key)
        if flag:
            cmd.extend([flag, str(value)])

    # VAT is enabled implicitly for distill; explicitly for others via --use-vat
    if ssl_method == "distill":
        cmd.append("--use-vat")

    if trial_cfg.get("pretrain_first"):
        cmd.append("--pretrain-first")
    if trial_cfg.get("cluster_require_agreement") is False:
        cmd.append("--cluster-no-agreement")

    return cmd


def compact_trial_name(
    idx: int,
    trial_cfg: dict[str, Any],
    search_keys: set[str] | None = None,
) -> str:
    """Compact readable directory name that stays under Windows path limits."""
    if search_keys is None:
        search_keys = set(trial_cfg)

    aliases = {
        "activation": "act",
        "batch_size": "bs",
        "distill_T": "T",
        "distill_alpha": "da",
        "dropout": "do",
        "epochs": "ep",
        "hidden_dims": "hd",
        "lp_alpha": "lpa",
        "lp_conf": "lpc",
        "lp_k": "lpk",
        "lp_top_k": "lpt",
        "lr": "lr",
        "lr_factor": "lrf",
        "lr_patience": "lrp",
        "norm": "nm",
        "patience": "pat",
        "seed": "sd",
        "self_train_rounds": "str",
        "self_train_threshold": "stt",
        "cluster_n_clusters": "cn",
        "cluster_conf": "cc",
        "cluster_top_k": "ct",
        "cluster_min_labeled": "cml",
        "cluster_max_iter": "cmi",
        "cluster_n_init": "cni",
        "cluster_model_conf": "cmc",
        "st_initial_threshold": "sit",
        "st_threshold_decay": "std",
        "st_min_threshold": "smt",
        "st_top_k_per_round": "stk",
        "st_per_class_adjustment": "spa",
        "cluster_seed_mode": "csm",
        "cluster_max_seeds": "cms",
        "vat_epsilon": "ve",
        "vat_weight": "vw",
        "weight_decay": "wd",
    }

    def format_value(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4g}"
        if isinstance(value, str):
            return value.replace(",", "x")
        return str(value)

    compact_parts = []
    for key in sorted(search_keys):
        if key not in trial_cfg:
            continue
        compact_parts.append(f"{aliases.get(key, key)}{format_value(trial_cfg[key])}")

    signature = "|".join(
        f"{key}={format_value(trial_cfg[key])}"
        for key in sorted(search_keys)
        if key in trial_cfg
    )
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:8]
    return f"trial{idx:03d}_{'_'.join(compact_parts)}_{digest}"


def experiment_dir(
    output_dir: Path,
    idx: int,
    trial_cfg: dict[str, Any],
    ssl_method: str,
    search_keys: set[str] | None = None,
) -> Path:
    """Resolve the train_mlp output directory for a trial."""
    return output_dir / f"{compact_trial_name(idx, trial_cfg, search_keys)}_mlp_{ssl_method}"


def trial_name(idx: int, trial_cfg: dict[str, Any], search_keys: set[str] | None = None) -> str:
    """Short readable directory name — only includes searched-over params."""
    if search_keys is None:
        search_keys = set(trial_cfg)  # fallback: all keys
    parts = [f"trial{idx:03d}"]
    for k in sorted(search_keys):
        if k not in trial_cfg:
            continue
        v = trial_cfg[k]
        if isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        else:
            parts.append(f"{k}={v}")
    return "_".join(parts)


def run_trial(
    idx: int,
    trial_cfg: dict[str, Any],
    output_dir: Path,
    ssl_method: str,
    data_dir: str,
    search_keys: set[str] | None = None,
    timeout: int = 1800,
) -> dict[str, Any] | None:
    """Run a single trial.  Returns metrics dict or None on failure."""
    tdir = output_dir / compact_trial_name(idx, trial_cfg, search_keys)
    exp_dir = experiment_dir(output_dir, idx, trial_cfg, ssl_method, search_keys)
    metrics_path = exp_dir / "metrics.json"
    legacy_exp_dir = output_dir / f"{trial_name(idx, trial_cfg, search_keys)}_mlp_{ssl_method}"
    if not metrics_path.exists() and legacy_exp_dir != exp_dir:
        legacy_metrics_path = legacy_exp_dir / "metrics.json"
        if legacy_metrics_path.exists():
            exp_dir = legacy_exp_dir
            metrics_path = legacy_metrics_path
    if metrics_path.exists():
        # Already completed — load and return
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["_trial_idx"] = idx
            metrics["_trial_dir"] = str(exp_dir)
            for k, v in trial_cfg.items():
                metrics[f"_param_{k}"] = v
            print(f"  [{idx}] SKIP (already done): {metrics.get('best_val_acc', '?')}")
            return metrics
        except Exception:
            pass  # corrupted — re-run

    cmd = build_command(trial_cfg, ssl_method, data_dir) + ["--name", str(tdir.name)]
    # Redirect output_dir so the trial lands under our tuning directory
    cmd[cmd.index("--name") + 1] = str(tdir.name)
    # Override --output-dir
    cmd.extend(["--output-dir", str(tdir.parent)])

    # Also set --log-every high to reduce log spam
    cmd.extend(["--log-every", "50"])

    cfg_summary = ", ".join(f"{k}={v}" for k, v in trial_cfg.items())
    print(f"  [{idx}] Running: {cfg_summary}")

    try:
        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path(__file__).resolve().parent),
        )
        elapsed = time.perf_counter() - t0

        if proc.returncode != 0:
            # Print last 20 lines of stderr for debugging
            stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
            print(f"  [{idx}] FAILED (rc={proc.returncode}): {stderr_tail}")
            return None

        # The actual output dir path depends on how train_mlp constructs the name.
        # train_mlp uses `make_experiment_name` → "<base_name>_<model>_<ssl_method>"
        # Since we set --name, the experiment name becomes: tdir.name + "_mlp_" + ssl_method
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["_trial_idx"] = idx
            metrics["_elapsed_s"] = round(elapsed, 1)
            metrics["_trial_dir"] = str(exp_dir)
            for k, v in trial_cfg.items():
                metrics[f"_param_{k}"] = v

            best_acc = metrics.get("best_val_acc", "?")
            print(f"  [{idx}] DONE  acc={best_acc}  f1={metrics.get('best_val_macro_f1','?')}  "
                  f"time={elapsed:.0f}s")
            return metrics
        else:
            print(f"  [{idx}] WARNING: no metrics.json found at {metrics_path}")
            return None

    except subprocess.TimeoutExpired:
        print(f"  [{idx}] TIMEOUT after {timeout}s (subprocess already killed)")
        return None


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for ML_Assignment2 SSL pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets (--preset):
  model       Model architecture (hidden_dims, dropout, activation, norm)
  train       Training dynamics (lr, weight_decay, batch_size)
  lp          Label Propagation (lp_k, lp_alpha, lp_conf, lp_top_k)
  distill     Knowledge Distillation (distill_T, distill_alpha)
  vat         VAT consistency (vat_weight, vat_epsilon)
  self_train  Self-training (rounds, threshold)
  st_finetune Self-training fine-tuning (threshold + LP interaction)
  st_dynamic  Self-training dynamic threshold (per-class adaptive, curriculum decay)
  clustering   Cluster label broadcasting (clusters, purity, model filter)
  cluster_propagation  Cluster broadcasting + label propagation agreement
  combined    Top-10 most impactful params (for random search)

Examples:
  py tune_hyperparams.py --method grid --ssl-method pseudo --preset lp
  py tune_hyperparams.py --method random --ssl-method self_training --preset st_dynamic --trials 30
""",
    )

    # Search config
    parser.add_argument("--method", choices=["grid", "random"], default="grid",
                        help="Search strategy")
    parser.add_argument("--preset", default="combined",
                        help="Comma-separated preset names to use for search space. "
                             "Available: model, train, lp, distill, vat, self_train, st_finetune, "
                             "st_dynamic, clustering, cluster_propagation, combined")
    parser.add_argument("--trials", type=int, default=0,
                        help="Number of random trials (0 = use all grid combinations)")
    parser.add_argument("--ssl-method", choices=["pseudo", "distill", "self_training", "clustering", "cluster_propagation", "constrained_clustering"],
                        default="distill")

    # Experiment
    parser.add_argument("--output-dir", default="outputs/tuning")
    parser.add_argument("--data-dir", default=_resolve_data_dir())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Max seconds per trial")
    parser.add_argument("--pretrain-first", action="store_true",
                        help="Use labeled pretraining before retraining trials.")

    args = parser.parse_args()

    # ── Build search space from presets ─────────────────────────────
    preset_names = [p.strip() for p in args.preset.split(",")]
    param_space: dict[str, list] = {}
    for pname in preset_names:
        if pname not in PRESETS:
            print(f"ERROR: unknown preset '{pname}'. Available: {list(PRESETS)}")
            sys.exit(1)
        param_space.update(PRESETS[pname])

    # ── Generate trial configs ──────────────────────────────────────
    rng = np.random.default_rng(args.seed)

    if args.method == "grid":
        configs = build_grid(param_space)
        print(f"Grid search: {len(configs)} combinations from {list(param_space.keys())}")
    else:
        if args.trials <= 0:
            print("ERROR: --trials is required for random search")
            sys.exit(1)
        configs = sample_random(param_space, args.trials, rng)
        print(f"Random search: {len(configs)} trials from {list(param_space.keys())}")

    if len(configs) == 0:
        print("No trials to run.")
        return

    # ── Fill defaults for params not in search space ────────────────
    for cfg in configs:
        for key, default_val in DEFAULT_FIXED.items():
            if key not in cfg:
                cfg[key] = default_val
        cfg["seed"] = args.seed
        cfg["pretrain_first"] = args.pretrain_first
        cfg["cluster_require_agreement"] = True

    # ── Output directory ────────────────────────────────────────────
    output_dir = Path(args.output_dir) / args.ssl_method
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "tuning_results.csv"

    print(f"\nSSL method: {args.ssl_method}")
    print(f"Output:    {output_dir}")
    print()

    # ── Run trials ──────────────────────────────────────────────────
    all_metrics = []
    n_total = len(configs)
    n_fail = 0

    t_start = time.perf_counter()
    for i, cfg in enumerate(configs):
        metrics = run_trial(
            idx=i + 1,
            trial_cfg=cfg,
            output_dir=output_dir,
            ssl_method=args.ssl_method,
            data_dir=args.data_dir,
            search_keys=set(param_space.keys()),
            timeout=args.timeout,
        )
        if metrics is None:
            n_fail += 1
        else:
            all_metrics.append(metrics)

        # Save incrementally (resilient to interruption)
        if all_metrics:
            df = pd.DataFrame(all_metrics)
            # Move _param_* columns next to metrics
            param_cols = [c for c in df.columns if c.startswith("_param_")]
            metric_cols = [c for c in df.columns if not c.startswith("_")]
            df = df[param_cols + metric_cols]
            df.to_csv(results_path, index=False)

    # ── Summary ─────────────────────────────────────────────────────
    elapsed_total = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"Done.  {len(all_metrics)}/{n_total} completed, {n_fail} failed.  "
          f"Total time: {elapsed_total/60:.1f} min")

    if all_metrics:
        df = pd.DataFrame(all_metrics)
        best_idx = df["best_val_acc"].idxmax()
        best = df.iloc[best_idx]
        print(f"\nBest trial (by val_acc):")
        print(f"  val_acc   = {best['best_val_acc']:.4f}")
        print(f"  macro_f1  = {best['best_val_macro_f1']:.4f}")
        print(f"  epoch     = {int(best['best_epoch'])}")
        param_cols = [c for c in df.columns if c.startswith("_param_")]
        for pc in param_cols:
            print(f"  {pc[7:]:20s} = {best[pc]}")

        # Top-5 summary
        top5 = df.nlargest(5, "best_val_acc")[
            param_cols + ["best_val_acc", "best_val_macro_f1", "best_epoch"]
        ]
        print(f"\nTop-5 trials:")
        print(top5.to_string(index=False))

        print(f"\nFull results: {results_path}")


if __name__ == "__main__":
    main()
