#!/usr/bin/env python3
"""
Paired significance analysis for kcat prediction baselines.

This script compares PocketGNN / CatPred / CataPro on shared samples using:
1) Paired bootstrap confidence intervals for metric advantages
2) Paired permutation tests for p-values

Positive "advantage" means method_a is better than method_b.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


HIGHER_BETTER = {"pearson", "r2"}
LOWER_BETTER = {"mae", "rmse"}
METRICS = ("pearson", "r2", "mae", "rmse")


@dataclass
class MethodData:
    name: str
    df: pd.DataFrame  # columns: sample_id, y_true, y_pred


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_true - y_pred
    return {
        "pearson": safe_pearson(y_true, y_pred),
        "r2": safe_r2(y_true, y_pred),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
    }


def metric_advantage(metric: str, a_val: float, b_val: float) -> float:
    if metric in HIGHER_BETTER:
        return a_val - b_val
    if metric in LOWER_BETTER:
        return b_val - a_val
    raise ValueError(f"Unknown metric: {metric}")


def bootstrap_advantage(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric: str,
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    n = y_true.size
    obs_a = compute_metrics(y_true, y_pred_a)[metric]
    obs_b = compute_metrics(y_true, y_pred_b)[metric]
    obs_adv = metric_advantage(metric, obs_a, obs_b)

    boot = np.zeros(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m_a = compute_metrics(y_true[idx], y_pred_a[idx])[metric]
        m_b = compute_metrics(y_true[idx], y_pred_b[idx])[metric]
        boot[i] = metric_advantage(metric, m_a, m_b)

    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(obs_adv), float(lo), float(hi)


def permutation_pvalue(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric: str,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    obs_a = compute_metrics(y_true, y_pred_a)[metric]
    obs_b = compute_metrics(y_true, y_pred_b)[metric]
    obs_adv = metric_advantage(metric, obs_a, obs_b)

    n = y_true.size
    extreme = 0
    for _ in range(n_perm):
        mask = rng.random(n) < 0.5
        perm_a = y_pred_a.copy()
        perm_b = y_pred_b.copy()
        perm_a[mask], perm_b[mask] = y_pred_b[mask], y_pred_a[mask]
        p_a = compute_metrics(y_true, perm_a)[metric]
        p_b = compute_metrics(y_true, perm_b)[metric]
        adv = metric_advantage(metric, p_a, p_b)
        if abs(adv) >= abs(obs_adv):
            extreme += 1
    return float((extreme + 1) / (n_perm + 1))


def load_methods(
    pocketgnn_csv: Path,
    catpred_csv: Path,
    catapro_csv: Path,
) -> Dict[str, MethodData]:
    pocketgnn = pd.read_csv(pocketgnn_csv).rename(
        columns={
            "experimental value[log10]": "y_true",
            "predicted value[log10]": "y_pred",
        }
    )[["sample_id", "y_true", "y_pred"]]

    catpred = pd.read_csv(catpred_csv).rename(
        columns={
            "experimental_value_log10": "y_true",
            "catpred_prediction_log10": "y_pred",
        }
    )[["sample_id", "y_true", "y_pred"]]

    catapro = pd.read_csv(catapro_csv).rename(columns={"catapro_pred": "y_pred"})[
        ["sample_id", "y_true", "y_pred"]
    ]

    for df in (pocketgnn, catpred, catapro):
        df.dropna(subset=["sample_id", "y_true", "y_pred"], inplace=True)
        df["sample_id"] = df["sample_id"].astype(str)

    return {
        "PocketGNN": MethodData(name="PocketGNN", df=pocketgnn),
        "CatPred": MethodData(name="CatPred", df=catpred),
        "CataPro": MethodData(name="CataPro", df=catapro),
    }


def compare_two_methods(
    a_name: str,
    b_name: str,
    a_df: pd.DataFrame,
    b_df: pd.DataFrame,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> Dict:
    merged = a_df.merge(b_df, on="sample_id", suffixes=("_a", "_b"))
    merged["y_true_diff"] = merged["y_true_a"] - merged["y_true_b"]
    merged = merged[np.abs(merged["y_true_diff"]) < 1e-8].copy()

    y_true = merged["y_true_a"].to_numpy(dtype=np.float64)
    y_pred_a = merged["y_pred_a"].to_numpy(dtype=np.float64)
    y_pred_b = merged["y_pred_b"].to_numpy(dtype=np.float64)

    rng = np.random.default_rng(seed)
    out = {
        "method_a": a_name,
        "method_b": b_name,
        "n_samples": int(len(merged)),
        "method_a_metrics": compute_metrics(y_true, y_pred_a),
        "method_b_metrics": compute_metrics(y_true, y_pred_b),
        "advantage_definition": "positive => method_a better",
        "metrics": {},
    }

    for metric in METRICS:
        obs_adv, ci_lo, ci_hi = bootstrap_advantage(
            y_true, y_pred_a, y_pred_b, metric, n_boot=n_boot, rng=rng
        )
        pval = permutation_pvalue(
            y_true, y_pred_a, y_pred_b, metric, n_perm=n_perm, rng=rng
        )
        out["metrics"][metric] = {
            "advantage": obs_adv,
            "bootstrap_ci_95": [ci_lo, ci_hi],
            "permutation_pvalue": pval,
            "significant_alpha_0_05": bool(pval < 0.05),
        }
    return out


def make_markdown_report(results: Dict, output_path: Path) -> None:
    lines = ["# Significance Analysis", ""]
    lines.append("## Pairwise Tests")
    lines.append("")
    for comp_key, comp in results["comparisons"].items():
        lines.append(
            f"### {comp['method_a']} vs {comp['method_b']} (n={comp['n_samples']})"
        )
        lines.append("")
        lines.append("| Metric | Advantage (A better > 0) | 95% CI | p-value | Significant |")
        lines.append("|---|---:|---|---:|---|")
        for metric in METRICS:
            m = comp["metrics"][metric]
            ci = m["bootstrap_ci_95"]
            sig = "Yes" if m["significant_alpha_0_05"] else "No"
            lines.append(
                f"| {metric} | {m['advantage']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {m['permutation_pvalue']:.4g} | {sig} |"
            )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired significance analysis for kcat benchmarks.")
    parser.add_argument(
        "--pocketgnn_csv",
        type=Path,
        default=Path("data/raw/kcat_test_results.csv"),
    )
    parser.add_argument(
        "--catpred_csv",
        type=Path,
        default=Path("results/baseline_comparison/catpred_predictions_with_truth.csv"),
    )
    parser.add_argument(
        "--catapro_csv",
        type=Path,
        default=Path("results/baseline_comparison/catapro_results.csv"),
    )
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--n_perm", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260315)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/significance"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    methods = load_methods(args.pocketgnn_csv, args.catpred_csv, args.catapro_csv)
    pairs = [("PocketGNN", "CatPred"), ("PocketGNN", "CataPro"), ("CatPred", "CataPro")]

    results = {
        "config": {
            "n_boot": args.n_boot,
            "n_perm": args.n_perm,
            "seed": args.seed,
        },
        "method_sizes": {name: int(len(m.df)) for name, m in methods.items()},
        "comparisons": {},
    }

    for i, (a, b) in enumerate(pairs):
        comp = compare_two_methods(
            a_name=a,
            b_name=b,
            a_df=methods[a].df,
            b_df=methods[b].df,
            n_boot=args.n_boot,
            n_perm=args.n_perm,
            seed=args.seed + i * 1000,
        )
        results["comparisons"][f"{a}_vs_{b}"] = comp

    json_path = args.output_dir / "significance_results.json"
    md_path = args.output_dir / "SIGNIFICANCE_REPORT.md"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    make_markdown_report(results, md_path)

    print(f"[OK] Saved JSON: {json_path}")
    print(f"[OK] Saved report: {md_path}")


if __name__ == "__main__":
    main()
