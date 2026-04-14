# Paper Freeze 202604

This directory is a manuscript-focused frozen package for the PocketGNN paper revision.
It copies only the files needed to inspect the current paper, core figures, core results, and post-hoc analysis scripts.
It does not contain raw datasets, model checkpoints, docking caches, or training outputs.

## Manuscript

- `manuscript/gemini.tex`: current main manuscript source
- `manuscript/supplementary_info.md`: current supplementary notes
- `manuscript/gemini.pdf`: compiled PDF from `gemini.tex`

## Core result sources

- `results_core/hom40_metrics.json`: main 40% homology-split metrics
- `results_core/hom40_predictions.csv`: main 40% homology-split predictions
- `results_core/baseline_significance.json`: shared-sample baseline comparison and significance tests
- `results_core/docking_robustness.json`: DiffDock confidence robustness diagnostics
- `results_core/ec_wise_corrected.csv`: corrected EC-wise metrics aligned to the main prediction file
- `results_core/true_kcat_bin_bias.csv`: true-kcat quantile bias/calibration table
- `results_core/bias_calibration_summary.json`: summary for prediction dynamic-range compression

## Reproduction scripts

- `scripts_reproduce/analyze_ec_performance_from_predictions.py`
- `scripts_reproduce/analyze_prediction_bias.py`
- Optional copied scripts if present: `analyze_significance.py`, `analyze_docking_robustness.py`

## Revision stance

Current paper revision is intentionally conservative:

- no model retraining
- no feature pipeline change during manuscript attack phase
- Integrated Gradients treated as exploratory only
- DiffDock confidence treated as an unreliable downstream error proxy, not direct pose-quality validation
- EC-wise performance described as heterogeneous
- calibration errors described as dynamic-range compression under homology split
