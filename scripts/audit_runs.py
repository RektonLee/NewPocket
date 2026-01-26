#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit training runs across outputs/ and experiments/.
Generates a unified registry with basic metrics and run_dir health.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
EXPERIMENTS = ROOT / "experiments"


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_yaml(path: Path) -> Optional[Dict[str, Any]]:
    if yaml is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def read_training_metrics(path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    try:
        df = pd.read_csv(path)
        if "R2" in df.columns:
            result["best_r2"] = float(df["R2"].max())
            result["last_r2"] = float(df["R2"].iloc[-1])
        if "Pearson" in df.columns:
            result["best_pearson"] = float(df["Pearson"].max())
            result["last_pearson"] = float(df["Pearson"].iloc[-1])
        if "Val_Loss" in df.columns:
            result["best_val_loss"] = float(df["Val_Loss"].min())
            result["last_val_loss"] = float(df["Val_Loss"].iloc[-1])
    except Exception:
        pass
    return result


def collect_outputs() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not OUTPUTS.exists():
        return rows
    for d in sorted(OUTPUTS.iterdir()):
        if not d.is_dir():
            continue
        row: Dict[str, Any] = {
            "source": "outputs",
            "output_dir": str(d),
        }
        run_dir_file = d / "run_dir.txt"
        if run_dir_file.exists():
            run_dir = run_dir_file.read_text().strip()
            row["run_dir"] = run_dir
            row["run_dir_exists"] = Path(run_dir).exists()
        metrics_path = d / "training_metrics.csv"
        if metrics_path.exists():
            row.update(read_training_metrics(metrics_path))
        config_path = d / "training_config.txt"
        if config_path.exists():
            row["has_config"] = True
        else:
            row["has_config"] = False
        rows.append(row)
    return rows


def collect_experiments() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not EXPERIMENTS.exists():
        return rows
    for exp_dir in sorted(EXPERIMENTS.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("experiments_"):
            continue
        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue
            row: Dict[str, Any] = {
                "source": "experiments",
                "exp_name": exp_dir.name,
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
            }
            metadata = read_yaml(run_dir / "metadata.yaml")
            if metadata:
                row.update({
                    "dataset_path": metadata.get("dataset_path"),
                    "save_dir": metadata.get("save_dir"),
                    "status": metadata.get("status"),
                    "final_r2": metadata.get("final_r2"),
                    "final_pearson": metadata.get("final_pearson"),
                    "best_val_loss": metadata.get("best_val_loss"),
                })
            metrics = read_json(run_dir / "metrics.json")
            if metrics:
                row["metrics_json"] = True
            rows.append(row)
    return rows


def main() -> None:
    rows = collect_outputs() + collect_experiments()
    out_csv = ROOT / "runs_registry.csv"
    if not rows:
        print("No runs found.")
        return
    # Normalize keys
    keys = sorted({k for row in rows for k in row.keys()})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved registry: {out_csv}")


if __name__ == "__main__":
    main()
