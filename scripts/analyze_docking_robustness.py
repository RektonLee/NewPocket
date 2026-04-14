#!/usr/bin/env python3
"""
Analyze whether DiffDock quality proxies correlate with downstream prediction error.

This script uses an already-built DiffDock subset and tests whether the current
PocketGNN predictions degrade on low-confidence or larger pockets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def compute_metrics(df: pd.DataFrame) -> dict:
    y_true = df["experimental value[log10]"].to_numpy(dtype=np.float64)
    y_pred = df["predicted value[log10]"].to_numpy(dtype=np.float64)
    return {
        "n_samples": int(len(df)),
        "pearson": float(pearsonr(y_true, y_pred)[0]),
        "r2": safe_r2(y_true, y_pred),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
    }


def build_confidence_bins(df: pd.DataFrame) -> pd.DataFrame:
    q1, q2 = df["confidence"].quantile([1 / 3, 2 / 3]).tolist()
    out = df.copy()
    out["confidence_bin"] = pd.cut(
        out["confidence"],
        bins=[-np.inf, q1, q2, np.inf],
        labels=["low", "mid", "high"],
        include_lowest=True,
    )
    out["confidence_range"] = pd.cut(
        out["confidence"],
        bins=[-np.inf, q1, q2, np.inf],
        labels=[f"<= {q1:.2f}", f"{q1:.2f}-{q2:.2f}", f"> {q2:.2f}"],
        include_lowest=True,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze docking robustness.")
    parser.add_argument(
        "--docking_quality_csv",
        type=Path,
        default=Path("results/docking_quality/docking_quality_full.csv"),
    )
    parser.add_argument(
        "--prediction_csv",
        type=Path,
        default=Path("data/processed/kcat_test_new_diffdock.csv"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/docking_robustness"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    quality = pd.read_csv(args.docking_quality_csv)
    pred = pd.read_csv(args.prediction_csv)
    df = quality.merge(
        pred[["sample_id", "experimental value[log10]", "predicted value[log10]"]],
        on="sample_id",
        how="inner",
    )
    df["abs_error"] = (
        df["experimental value[log10]"] - df["predicted value[log10]"]
    ).abs()

    correlations = {}
    for col in ("confidence", "n_nodes", "n_edges", "n_ligand_atoms"):
        pearson_r, pearson_p = pearsonr(df[col], df["abs_error"])
        spearman_r, spearman_p = spearmanr(df[col], df["abs_error"])
        correlations[col] = {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
        }

    binned = build_confidence_bins(df)
    by_bin = []
    for bin_name, group in binned.groupby("confidence_bin", observed=False):
        metrics = compute_metrics(group)
        metrics["bin"] = str(bin_name)
        metrics["range"] = str(group["confidence_range"].iloc[0])
        by_bin.append(metrics)

    results = {
        "n_samples": int(len(df)),
        "overall_metrics": compute_metrics(df),
        "correlations_with_abs_error": correlations,
        "by_confidence_bin": by_bin,
    }

    (args.output_dir / "docking_robustness.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(by_bin).to_csv(args.output_dir / "docking_robustness_by_bin.csv", index=False)

    report_lines = [
        "# Docking Robustness Analysis",
        "",
        f"- Samples: {len(df)}",
        f"- Overall Pearson: {results['overall_metrics']['pearson']:.3f}",
        f"- Overall R2: {results['overall_metrics']['r2']:.3f}",
        f"- Overall MAE: {results['overall_metrics']['mae']:.3f}",
        "",
        "## Correlation With Absolute Error",
        "",
        "| Variable | Pearson r | Pearson p | Spearman rho | Spearman p |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, value in correlations.items():
        report_lines.append(
            f"| {key} | {value['pearson_r']:.4f} | {value['pearson_p']:.4g} | {value['spearman_r']:.4f} | {value['spearman_p']:.4g} |"
        )

    report_lines.extend(
        [
            "",
            "## Metrics By DiffDock Confidence Tertile",
            "",
            "| Bin | Range | N | Pearson r | R2 | MAE |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in by_bin:
        report_lines.append(
            f"| {row['bin']} | {row['range']} | {row['n_samples']} | {row['pearson']:.3f} | {row['r2']:.3f} | {row['mae']:.3f} |"
        )
    (args.output_dir / "DOCKING_ROBUSTNESS_REPORT.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=200)

    axes[0, 0].hist(df["confidence"], bins=40, color="#4c78a8", edgecolor="black", alpha=0.8)
    axes[0, 0].set_title("DiffDock confidence distribution")
    axes[0, 0].set_xlabel("confidence")
    axes[0, 0].set_ylabel("count")

    axes[0, 1].scatter(df["confidence"], df["abs_error"], alpha=0.45, s=18, color="#dd8452")
    axes[0, 1].set_title("Confidence vs absolute error")
    axes[0, 1].set_xlabel("confidence")
    axes[0, 1].set_ylabel("absolute error")
    axes[0, 1].grid(alpha=0.25)

    order = ["low", "mid", "high"]
    by_bin_df = pd.DataFrame(by_bin).set_index("bin").loc[order].reset_index()

    axes[1, 0].bar(by_bin_df["bin"], by_bin_df["mae"], color="#55a868")
    axes[1, 0].set_title("MAE by confidence tertile")
    axes[1, 0].set_ylabel("MAE")
    axes[1, 0].grid(alpha=0.25, axis="y")

    axes[1, 1].bar(by_bin_df["bin"], by_bin_df["pearson"], color="#c44e52")
    axes[1, 1].set_title("Pearson r by confidence tertile")
    axes[1, 1].set_ylabel("Pearson r")
    axes[1, 1].grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(args.output_dir / "docking_robustness.png", bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {args.output_dir / 'docking_robustness.json'}")
    print(f"[OK] Saved: {args.output_dir / 'docking_robustness_by_bin.csv'}")
    print(f"[OK] Saved: {args.output_dir / 'DOCKING_ROBUSTNESS_REPORT.md'}")
    print(f"[OK] Saved: {args.output_dir / 'docking_robustness.png'}")


if __name__ == "__main__":
    main()
