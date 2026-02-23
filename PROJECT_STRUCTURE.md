# Project Organization Guide

This document explains the project structure and file organization principles.

## Directory Structure

```
PGNN_clean/
├── README.md                    # Project overview and quick start
├── CLAUDE.md                    # AI assistant instructions
├── requirements.txt             # Python dependencies
├── .gitignore                   # Enhanced gitignore configuration
│
├── src/                         # Source code
│   ├── train.py                 # Main training script
│   ├── test.py                  # Testing/evaluation
│   ├── GNN_model.py            # Model definitions
│   ├── build_graph_dataset.py  # Dataset construction
│   ├── docking.py              # DiffDock integration
│   └── ...
│
├── scripts/                     # Utility scripts and benchmarks
│   ├── quick_benchmark.py      # Run all baselines
│   ├── setup_benchmark_env.sh  # Benchmark setup
│   ├── archived/               # Old/unused scripts
│   └── ...
│
├── docs/                        # All documentation
│   ├── README.md               # Documentation index
│   ├── dev/                    # Development guides
│   │   ├── DEV_GUIDE.md        # Main developer guide
│   │   ├── QUANTILE_REGRESSION_GUIDE.md
│   │   └── DEBUG_SUMMARY.md
│   ├── analysis/               # Diagnostic reports
│   │   ├── DIAGNOSIS_ANALYSIS.md
│   │   ├── DIAGNOSTIC_RESULTS.md
│   │   └── ENHANCEMENT_REPORT.md
│   ├── planning/               # Project planning & logs
│   │   ├── todo.md
│   │   ├── PROGRESS_REPORT.md
│   │   └── WhatIDid.md
│   └── paper/                  # Research notes
│       ├── 深挖到实现与数据流.md
│       ├── mmseqs.md
│       └── ...
│
├── paperwriting/                # Paper drafts and submission
│   ├── gemini.tex              # Main paper
│   ├── gemini.pdf              # Compiled PDF
│   └── figures/                # Paper figures (version controlled)
│
├── figures/                     # Generated figures (gitignored)
│   ├── README.md
│   └── archived/               # Historical figures
│
├── configs/                     # Configuration files
├── notebooks/                   # Jupyter notebooks (gitignored)
├── dockingcheck/               # Docking validation tools
│
└── [Gitignored Directories]    # Not committed to git
    ├── data/                   # Datasets and raw data
    ├── outputs/                # Training outputs
    ├── results/                # Experiment results
    ├── experiments/            # Experiment artifacts
    ├── DiffDock/              # External tool
    ├── benchmark_tools/       # Baseline implementations
    ├── wandb/                 # WandB logs
    └── logs/                  # Log files
```

## File Organization Principles

### 1. Documentation

**Where to put new docs:**
- **Development/technical docs** → `docs/dev/`
- **Analysis/diagnostic reports** → `docs/analysis/`
- **Planning/progress tracking** → `docs/planning/`
- **Research notes for paper** → `docs/paper/`
- **Paper drafts** → `paperwriting/`

**Keep at root:**
- `README.md` - Project overview (first thing people see)
- `CLAUDE.md` - AI assistant instructions

### 2. Code

**Structure:**
- **Core functionality** → `src/`
- **Utility scripts** → `scripts/`
- **One-time/deprecated scripts** → `scripts/archived/`
- **Configuration files** → `configs/`

**Naming:**
- Use descriptive names: `build_graph_dataset.py` not `build.py`
- Group related files with prefixes: `docking.py`, `docking_chai1.py`

### 3. Data & Outputs

**All large files must be gitignored:**
- `data/` - Input datasets
- `outputs/` - Training checkpoints and logs
- `results/` - Experiment results
- `experiments/` - Experiment artifacts
- `figures/` - Generated plots (except paper figures)

**Best practices:**
- Use `sample_manager.py` for path management
- Don't hardcode paths in scripts
- Keep small sample data in `sample_data/` if needed

### 4. External Tools

**Location:** Project root, gitignored
- `DiffDock/` - Molecular docking tool
- `benchmark_tools/` - Baseline implementations (CataPro, DLKcat, etc.)

**Setup:**
- Use environment variables or config files for paths
- Document setup in `scripts/setup_benchmark_env.sh`

### 5. Figures & Visualizations

**Generated figures** → `figures/` (gitignored)
**Paper figures** → `paperwriting/figures/` (version controlled)
**Old figures** → `figures/archived/`

### 6. Jupyter Notebooks

**Location:** `notebooks/` (gitignored by default)

To version control specific notebooks:
```bash
git add -f notebooks/important_analysis.ipynb
```

## Maintenance Tasks

### Weekly Cleanup

1. **Archive old scripts:**
   ```bash
   mv scripts/old_script.py scripts/archived/
   ```

2. **Move old figures:**
   ```bash
   mv figures/*.png figures/archived/
   ```

3. **Update documentation index:**
   - Update `docs/README.md` when adding new docs

### Before Commits

1. **Check untracked files:**
   ```bash
   git status
   ```

2. **Verify no large files:**
   ```bash
   find . -size +10M -type f | grep -v "^\./\."
   ```

3. **Lint important files:**
   ```bash
   # Check for TODOs
   grep -r "TODO\|FIXME" src/ scripts/ --exclude-dir=archived
   ```

## .gitignore Strategy

The enhanced `.gitignore` is organized into sections:

1. **Python** - Standard Python ignores
2. **Data & Large Files** - Critical section to prevent large commits
3. **External Tools** - Third-party dependencies
4. **Generated Outputs** - Figures, logs, etc.
5. **Environment & Tools** - venv, wandb, etc.
6. **IDE & Editor** - Editor-specific files
7. **Jupyter Notebooks** - Notebook checkpoints
8. **OS-specific** - OS artifacts

**Key features:**
- Organized with clear section headers
- Whitelisting for specific files (`!paperwriting/gemini.pdf`)
- Comments explaining critical ignores

## Common Tasks

### Adding a New Script

```bash
# Development script
vim scripts/my_new_tool.py

# One-time migration script (will archive after running)
vim scripts/migrate_old_data.py
# After running:
mv scripts/migrate_old_data.py scripts/archived/
```

### Adding Documentation

```bash
# Technical guide
vim docs/dev/NEW_FEATURE_GUIDE.md

# Research note
vim docs/paper/experiment_results.md

# Update index
vim docs/README.md  # Add link to new doc
```

### Managing Figures

```bash
# Generate new figure
python scripts/plot_results.py  # Outputs to figures/

# For paper
cp figures/important_plot.png paperwriting/figures/
git add paperwriting/figures/important_plot.png

# Archive old figures
mv figures/old_*.png figures/archived/
```

## Migration Completed

This organization was implemented on 2026-02-10, consolidating:
- 24 scattered markdown files → `docs/` subdirectories
- Root-level images → `figures/archived/`
- Enhanced `.gitignore` with clear sections
- Added README files for navigation

## Questions?

Refer to:
- `docs/dev/DEV_GUIDE.md` - Comprehensive development guide
- `README.md` - Project overview
- `CLAUDE.md` - AI assistant instructions
