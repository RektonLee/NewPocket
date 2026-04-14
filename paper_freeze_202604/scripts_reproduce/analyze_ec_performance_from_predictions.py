#!/usr/bin/env python3
"""
EC-wise performance analysis from an evaluated prediction CSV.

This script intentionally avoids re-loading the model. It aligns an existing
test dataset with the prediction file produced by src/test.py, verifies that
the ground-truth values match, then computes EC-wise metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_ec_class(ec_value) -> int:
    if ec_value is None:
        return 0
    ec_text = str(ec_value).strip()
    if not ec_text or ec_text.lower() in {"unknown", "nan", "none"}:
        return 0
    try:
        return int(ec_text.split(".")[0])
    except ValueError:
        return 0


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(pearsonr(y_true, y_pred)[0])


def compute_metrics(group: pd.DataFrame) -> dict:
    y_true = group["y_true"].to_numpy(dtype=np.float64)
    y_pred = group["y_pred"].to_numpy(dtype=np.float64)
    return {
        "N": int(len(group)),
        "Pearson_r": safe_pearson(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)) if len(group) >= 2 else float("nan"),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def build_aligned_dataframe(dataset_path: Path, prediction_csv: Path) -> pd.DataFrame:
    dataset = torch.load(dataset_path, weights_only=False)
    pred_df = pd.read_csv(prediction_csv)

    required = {"true_kcat_log10", "predicted_kcat_log10"}
    missing = required - set(pred_df.columns)
    if missing:
        raise ValueError(f"Prediction CSV missing required columns: {sorted(missing)}")
    if len(dataset) != len(pred_df):
        raise ValueError(
            f"Length mismatch: dataset has {len(dataset)} samples, "
            f"prediction CSV has {len(pred_df)} rows"
        )

    rows = []
    max_truth_delta = 0.0
    for idx, (data, pred_row) in enumerate(zip(dataset, pred_df.itertuples(index=False))):
        y_dataset = float(data.y.item())
        y_csv = float(getattr(pred_row, "true_kcat_log10"))
        max_truth_delta = max(max_truth_delta, abs(y_dataset - y_csv))
        rows.append(
            {
                "sample_idx": idx,
                "sample_id": getattr(data, "sample_id", f"sample_{idx}"),
                "ec": str(getattr(data, "ec", "Unknown")),
                "ec_class": parse_ec_class(getattr(data, "ec", None)),
                "y_true": y_dataset,
                "y_pred": float(getattr(pred_row, "predicted_kcat_log10")),
            }
        )

    if max_truth_delta > 1e-5:
        raise ValueError(
            "Dataset order and prediction CSV do not appear aligned: "
            f"max |dataset_y - csv_y| = {max_truth_delta:.6g}"
        )

    df = pd.DataFrame(rows)
    df["abs_error"] = (df["y_pred"] - df["y_true"]).abs()
    return df


def plot_ec_metrics(metrics_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = metrics_df[metrics_df["EC"] != "Overall"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=200)
    colors = plt.cm.Set2(np.linspace(0, 1, len(plot_df)))

    panels = [
        ("Pearson_r", "Pearson r", "A. Correlation by EC Class"),
        ("N", "Number of Samples", "B. Sample Distribution"),
        ("MAE", "MAE (log10 units)", "C. Mean Absolute Error"),
        ("R2", "R2", "D. R2 Score"),
    ]
    for ax, (col, xlabel, title) in zip(axes.ravel(), panels):
        ax.barh(plot_df["EC"], plot_df[col], color=colors, alpha=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("EC Class")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
        if col in {"Pearson_r", "R2"}:
            ax.axvline(0, color="black", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Correct EC-wise analysis from predictions.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/kcat_merged_hom40_test.pt"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/test_hom40_evaluation/test_predictions.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ec_analysis_corrected"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned = build_aligned_dataframe(args.dataset, args.predictions)
    aligned.to_csv(args.output_dir / "ec_aligned_predictions.csv", index=False)

    metric_rows = []
    for ec_class, group in aligned[aligned["ec_class"] > 0].groupby("ec_class"):
        row = {"EC": f"EC {int(ec_class)}"}
        row.update(compute_metrics(group))
        metric_rows.append(row)

    overall = {"EC": "Overall"}
    overall.update(compute_metrics(aligned[aligned["ec_class"] > 0]))
    metric_rows.append(overall)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(args.output_dir / "ec_wise_performance.csv", index=False)
    (args.output_dir / "ec_wise_performance.json").write_text(
        json.dumps(metric_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_ec_metrics(metrics_df, args.output_dir / "fig_ec_wise_performance_corrected.png")

    print(metrics_df.to_string(index=False))
    print(f"[OK] Saved corrected EC analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
