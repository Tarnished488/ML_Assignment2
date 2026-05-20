"""使用 sklearn.model_selection 对 ST 自训练参数进行网格搜索调参。

该脚本放在 src/ 外部，已加入 .gitignore，不会提交到项目仓库。

用法：
    # 基础用法：搜索 self_train_threshold 和 self_train_rounds
    py grid_search_tune.py

    # 自定义搜索空间
    py grid_search_tune.py --param-grid '{"self_train_threshold": [0.7, 0.85, 0.95], "self_train_rounds": [3, 4], "lp_conf": [0.5, 0.6, 0.7]}'

工作原理：
    - 使用 sklearn.model_selection.ParameterGrid 生成参数组合
    - 每个组合调用 train_mlp.py（80-20 划分）进行训练
    - 读取 metrics.json 获取验证集 macro_f1 / accuracy
    - 输出最佳参数组合
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import ParameterGrid

# ---------------------------------------------------------------------------
# 默认搜索空间（控制组合数：4×3×3×3=108 → 约 54 小时，可自行缩小）
# ---------------------------------------------------------------------------
DEFAULT_PARAM_GRID = {
    "self_train_threshold": [0.7, 0.85, 0.95],
    "self_train_rounds": [3, 4, 5],
    "lp_conf": [0.5, 0.6, 0.7],
    "lp_top_k": [200, 300, 500],
}

# 固定参数（不参与搜索）
FIXED_PARAMS = {
    "models": "mlp",
    "use-ssl": "",
    "ssl-method": "self_training",
    "pretrain-first": "",
    "no-baseline": "",
    "hidden-dims": "128,64",
    "dropout": 0.3,
    "activation": "gelu",
    "epochs": 200,
    "patience": 40,
}

# 多折验证的种子列表（每个种子产生一个 80-20 划分）
CV_SEEDS = [42, 123, 456]


def run_single_trial(params: dict, seed: int, data_dir: str, dry_run: bool = False) -> dict | None:
    """运行单次 train_mlp.py 并返回验证指标。"""
    name = (
        f"gs_trial_thr={params['self_train_threshold']}"
        f"_r={params['self_train_rounds']}"
        f"_conf={params['lp_conf']}"
        f"_topk={params['lp_top_k']}"
        f"_s{seed}"
    )

    cmd = [
        sys.executable, "train_mlp.py",
        "--name", name,
        "--seed", str(seed),
        "--data-dir", data_dir,
    ]
    for k, v in FIXED_PARAMS.items():
        if v == "":
            cmd.append(f"--{k}")
        else:
            cmd.extend([f"--{k}", str(v)])
    cmd.extend(["--self-train-threshold", str(params["self_train_threshold"])])
    cmd.extend(["--self-train-rounds", str(params["self_train_rounds"])])
    cmd.extend(["--lp-conf", str(params["lp_conf"])])
    cmd.extend(["--lp-top-k", str(params["lp_top_k"])])

    if dry_run:
        print(f"  [DRY-RUN] {' '.join(cmd)}")
        return {"accuracy": 0.0, "macro_f1": 0.0}

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            print(f"  [FAIL] 返回码={result.returncode}")
            print(f"  STDERR: {result.stderr[-500:]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] 超过 1 小时")
        return None

    # 读取 metrics.json（输出目录会被 train_mlp.py 自动加上 _mlp_self_training）
    metrics_path = Path("outputs") / (name + "_mlp_self_training") / "metrics.json"
    if not metrics_path.exists():
        metric_files = sorted(Path("outputs").glob(f"{name}*/metrics.json"))
        if metric_files:
            metrics_path = metric_files[0]
        else:
            print(f"  [FAIL] 找不到 metrics.json (looking for {name}*/metrics.json)")
            return None

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "accuracy": metrics["best_val_acc"],
        "macro_f1": metrics["best_val_macro_f1"],
        "epoch": metrics.get("best_epoch"),
        "train_size": metrics.get("train_size_final"),
        "name": metrics["name"],
    }


def main():
    parser = argparse.ArgumentParser(description="ST 自训练网格搜索调参 (sklearn.model_selection)")
    parser.add_argument("--param-grid", default=None,
                        help='JSON 格式的搜索空间，例如：\'{"self_train_threshold":[0.7,0.9]}\'')
    parser.add_argument("--data-dir", default="data",
                        help="数据目录，默认 data/")
    parser.add_argument("--cv-seeds", nargs="+", type=int, default=CV_SEEDS,
                        help=f"多折验证种子列表（默认 {CV_SEEDS}）")
    parser.add_argument("--scoring", choices=["accuracy", "macro_f1"], default="macro_f1",
                        help="优化指标（默认 macro_f1）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印命令，不真正执行")
    parser.add_argument("--top-n", type=int, default=5,
                        help="显示前 N 个最佳结果（默认 5）")
    args = parser.parse_args()

    # 解析搜索空间
    if args.param_grid:
        param_grid = json.loads(args.param_grid)
    else:
        param_grid = DEFAULT_PARAM_GRID

    grid = list(ParameterGrid(param_grid))
    total = len(grid)
    cv = len(args.cv_seeds)
    print(f"{'='*60}")
    print(f"Grid Search 调参 (sklearn.model_selection.ParameterGrid)")
    print(f"{'='*60}")
    print(f"搜索参数: {list(param_grid.keys())}")
    print(f"组合数: {total} × {cv} 折 = {total * cv} 次训练")
    print(f"优化指标: {args.scoring}")
    print(f"验证种子: {args.cv_seeds}")
    print()

    all_results = []
    for idx, params in enumerate(grid):
        print(f"\n[{idx+1}/{total}] {params}")
        trial_scores = []
        for seed in args.cv_seeds:
            print(f"  种子={seed} ...", end=" ", flush=True)
            t0 = time.time()
            result = run_single_trial(params, seed, args.data_dir, args.dry_run)
            elapsed = time.time() - t0
            if result is not None:
                score = result[args.scoring]
                print(f"{args.scoring}={score:.4f}  ({elapsed:.0f}s)")
                trial_scores.append(score)
            else:
                print(f"失败 ({elapsed:.0f}s)")
                trial_scores.append(None)

        valid_scores = [s for s in trial_scores if s is not None]
        if valid_scores:
            mean_score = float(np.mean(valid_scores))
            std_score = float(np.std(valid_scores)) if len(valid_scores) > 1 else 0.0
        else:
            mean_score, std_score = -1.0, 0.0

        all_results.append({
            "params": params,
            "cv_mean": mean_score,
            "cv_std": std_score,
            "seeds": args.cv_seeds,
            "per_seed": trial_scores,
        })
        print(f"  → CV mean {args.scoring}={mean_score:.4f} ± {std_score:.4f}")

    # 按 CV 均值降序排列
    all_results.sort(key=lambda r: r["cv_mean"], reverse=True)

    print(f"\n{'='*60}")
    print(f"Top {min(args.top_n, len(all_results))} 结果（按 {args.scoring}）")
    print(f"{'='*60}")
    for rank, r in enumerate(all_results[:args.top_n], 1):
        params_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        print(f"  #{rank}  {args.scoring}={r['cv_mean']:.4f} ± {r['cv_std']:.4f}  |  {params_str}")

    # 保存完整结果
    out_path = Path("outputs/grid_search_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({
            "scoring": args.scoring,
            "param_grid": param_grid,
            "cv_seeds": args.cv_seeds,
            "results": all_results,
            "best": all_results[0] if all_results else None,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n完整结果已保存到 {out_path}")

    if all_results:
        best = all_results[0]
        print(f"\n最佳参数: {best['params']}")
        print(f"最佳 CV {args.scoring}: {best['cv_mean']:.4f} ± {best['cv_std']:.4f}")


if __name__ == "__main__":
    main()
