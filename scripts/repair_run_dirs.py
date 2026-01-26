#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repair missing experiments/ run directories referenced by outputs/*/run_dir.txt.
Creates metadata.yaml and metrics.json with best-known values from training_metrics.csv.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


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


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML not available; cannot write metadata.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def main() -> None:
    if yaml is None:
        print("PyYAML not installed; aborting.")
        return
    repaired = 0
    for d in sorted(OUTPUTS.iterdir()):
        if not d.is_dir():
            continue
        run_dir_file = d / "run_dir.txt"
        if not run_dir_file.exists():
            continue
        run_dir = Path(run_dir_file.read_text().strip())
        if run_dir.exists():
            continue
        run_dir.mkdir(parents=True, exist_ok=True)

        exp_name = run_dir.parents[0].name
        run_id = run_dir.name
        metrics = read_training_metrics(d / "training_metrics.csv")

        metadata = {
            "run_id": run_id,
            "exp_name": exp_name,
            "start_time": datetime.fromtimestamp(d.stat().st_mtime).isoformat(),
            "git_commit": "unknown",
            "save_dir": str(d),
            "dataset_path": "unknown",
            "model_version": "unknown",
            "graph_builder_version": "unknown",
            "status": "recovered",
            "note": "Recovered from outputs with missing run_dir",
            "best_val_loss": metrics.get("best_val_loss"),
            "final_r2": metrics.get("best_r2"),
            "final_pearson": metrics.get("best_pearson"),
        }

        write_yaml(run_dir / "metadata.yaml", metadata)
        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({
                "best_val_loss": metrics.get("best_val_loss"),
                "final_r2": metrics.get("best_r2"),
                "final_pearson": metrics.get("best_pearson"),
            }, f, indent=2, ensure_ascii=False)
        repaired += 1
        print(f"Repaired: {run_dir}")

    if repaired == 0:
        print("No missing run_dir to repair.")


if __name__ == "__main__":
    main()
