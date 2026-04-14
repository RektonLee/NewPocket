#!/usr/bin/env python3
"""Analyze prediction bias/calibration on the homology-split test predictions.

This script is intentionally post-hoc: it does not retrain or alter the model.
It summarizes whether errors are concentrated in specific true-kcat ranges.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rmse(x: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="results/test_hom40_evaluation/test_predictions.csv",
        help="CSV with true_kcat_log10 and predicted_kcat_log10 columns.",
    )
    parser.add_argument(
        "--outdir",
        default="results/bias_calibration",
        help="Output directory for tables and figures.",
    )
    parser.add_argument("--bins", type=int, default=10, help="Number of quantile bins by true kcat.")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_path)
    required = {"true_kcat_log10", "predicted_kcat_log10"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["residual"] = df["predicted_kcat_log10"] - df["true_kcat_log10"]
    df["absolute_error"] = df["residual"].abs()

    # qcut can drop duplicate edges if many identical values exist.
    df["true_bin"] = pd.qcut(df["true_kcat_log10"], q=args.bins, duplicates="drop")

    rows = []
    for i, (interval, g) in enumerate(df.groupby("true_bin", observed=True), start=1):
        rows.append(
            {
                "bin": i,
                "true_range": str(interval),
                "n": int(len(g)),
                "true_mean": float(g["true_kcat_log10"].mean()),
                "pred_mean": float(g["predicted_kcat_log10"].mean()),
                "bias_pred_minus_true": float(g["residual"].mean()),
                "median_residual": float(g["residual"].median()),
                "mae": float(g["absolute_error"].mean()),
                "p90_abs_error": float(g["absolute_error"].quantile(0.9)),
                "rmse": rmse(g["residual"]),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "true_kcat_bin_bias.csv", index=False)

    overall = {
        "source_predictions": str(pred_path),
        "n": int(len(df)),
        "pearson_r": float(df["true_kcat_log10"].corr(df["predicted_kcat_log10"], method="pearson")),
        "mean_true": float(df["true_kcat_log10"].mean()),
        "mean_pred": float(df["predicted_kcat_log10"].mean()),
        "global_bias_pred_minus_true": float(df["residual"].mean()),
        "mae": float(df["absolute_error"].mean()),
        "p90_abs_error": float(df["absolute_error"].quantile(0.9)),
        "rmse": rmse(df["residual"]),
        "prediction_range": [float(df["predicted_kcat_log10"].min()), float(df["predicted_kcat_log10"].max())],
        "true_range": [float(df["true_kcat_log10"].min()), float(df["true_kcat_log10"].max())],
    }
    with (outdir / "bias_calibration_summary.json").open("w") as f:
        json.dump({"overall": overall, "bins": rows}, f, indent=2)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    ax = axes[0]
    ax.scatter(df["true_kcat_log10"], df["predicted_kcat_log10"], s=12, alpha=0.28, color="#355C7D", edgecolors="none")
    ax.plot(summary["true_mean"], summary["pred_mean"], "o-", color="#C06C84", lw=2, ms=5, label="Quantile-bin mean")
    lo = min(df["true_kcat_log10"].min(), df["predicted_kcat_log10"].min())
    hi = max(df["true_kcat_log10"].max(), df["predicted_kcat_log10"].max())
    ax.plot([lo, hi], [lo, hi], "--", color="#444444", lw=1, label="Ideal calibration")
    ax.set_xlabel("True log10(kcat)")
    ax.set_ylabel("Predicted log10(kcat)")
    ax.set_title("Prediction Calibration by True-kcat Range")
    ax.legend(frameon=True, fontsize=8)

    ax = axes[1]
    ax.axhline(0, color="#444444", lw=1, ls="--")
    ax.bar(summary["bin"], summary["bias_pred_minus_true"], color="#6C5B7B", alpha=0.86, label="Mean residual")
    ax.plot(summary["bin"], summary["mae"], "o-", color="#F67280", lw=2, ms=4, label="MAE")
    ax.set_xlabel("True-kcat quantile bin")
    ax.set_ylabel("log10(kcat) error")
    ax.set_title("Bias and MAE Across True-kcat Bins")
    ax.legend(frameon=True, fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / "fig_prediction_bias_calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(overall, indent=2))
    print(f"Wrote {outdir / 'true_kcat_bin_bias.csv'}")
    print(f"Wrote {outdir / 'fig_prediction_bias_calibration.png'}")


if __name__ == "__main__":
    main()
