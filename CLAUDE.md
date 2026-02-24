# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **See also**: `PROJECT_DOCUMENTATION.md` for comprehensive project documentation including background, literature review, experimental results, and progress tracking. That document is the single source of truth for project status.

---

## Project Overview

**PocketGNN** is an enzyme kinetics prediction project that uses Graph Neural Networks to predict enzyme catalytic constants ($k_{cat}$) from protein pocket 3D structures and substrate information.

**Core Task**: Predict $\log_{10}(k_{cat})$ values from enzyme active site (pocket) structures combined with substrate SMILES.

**Tech Stack**: PyTorch + PyTorch Geometric + RDKit + WandB + ESM-2 (optional sequence embeddings)

---

## Building and Running

### Environment Setup
```bash
pip install -r requirements.txt
```

Key dependencies: PyTorch, PyTorch Geometric, RDKit, BioPython, WandB, Transformers

### Training
```bash
# Basic training
python src/train.py --dataset data/processed/kcat_train.pt --save_dir outputs/exp001

# With sequence embeddings (ESM-2)
python src/train.py \
  --dataset data/processed/kcat_full_1213.pt \
  --use_seq_embedding \
  --seq_embedding_path data/processed/esm_embeddings.pt \
  --save_dir outputs/kcat_with_esm

# With advanced options
python src/train.py \
  --dataset data/processed/kcat_full_1213.pt \
  --loss huber \
  --scheduler plateau \
  --pooling_type set2set \
  --dropout 0.3 \
  --weight_decay 1e-4
```

### Testing
```bash
python src/test.py \
  --test_dataset data/processed/kcat_test_fixed.pt \
  --model outputs/kcat_20251213_151558/best_model.pt
```

### Dataset Building
```bash
# Build graph dataset from CSV (with sample_id, substrate_smiles, kcat_value columns)
python src/build_graph_dataset.py \
  --input data/raw/kcat_data.csv \
  --output data/processed/kcat_train.pt
```

### Data Diagnosis (Run BEFORE training)
```bash
# Analyze if graph features contain sufficient information for prediction
python src/analyze_feature_label_relation.py \
  --dataset data/processed/kcat_train.pt \
  --save_dir outputs/diagnosis
```

### Benchmark Comparison
```bash
# Compare against SOTA baselines (CataPro, CatPred, DLKcat, UniKP)
python scripts/quick_benchmark.py
```

---

## Architecture & Data Flow

### 1. Data Pipeline

**Input CSV** → **Build Graph Dataset** → **.pt File** → **Train** → **Model Checkpoint**

- **Input CSV Format**: Must contain `sample_id`, `substrate_smiles`, `kcat_value` columns
- **Graph Construction**: Uses `src/build_graph_dataset.py`
  - Builds radius graph (4.0 Å threshold) from pocket PDB files
  - Computes 52-dim node features (element, residue, ligand flag, distances, electronic features)
  - Computes 24-dim edge features (16-dim RBF distance + 4-dim bond angles + 4-dim dihedral angles)
- **Output Format**: List of PyTorch Geometric `Data` objects saved as `.pt` file

### 2. Model Architecture

**Active Model**: `PocketGNNKcatOnly` (in `src/GNN_model.py`)

```
Input Graph (52-dim nodes, 24-dim edges)
  ↓
Node Encoder: Linear(52, 128)
  ↓
3x GAT Layers (4 heads, first layer uses edge_attr)
  ↓
Graph Pooling (mean/global_attention/set2set)
  ↓
[Optional] Sequence Embedding Fusion (ESM-2)
  ↓
MLP Regressor (3 layers)
  ↓
Output: log10(kcat) [single value]
```

**Key Features**:
- 24-dim edge features (RBF + angles + dihedrals) - critical for performance
- Multi-head attention (4 heads) in GAT layers
- Three pooling options: mean, global_attention, set2set
- Optional ESM-2 sequence embedding fusion (Late Fusion with Projection)

**Legacy Models** (for reference only):
- `PocketGNN`: Basic GCN model
- `PocketGNNWithAttention`: Dual-task (kcat + Km) with temperature
- `PocketGNNWithAttentionNoTemp`: Dual-task without temperature
- `PocketGNN_Gated`: GatedGraphConv variant

### 3. Data Format

**PyTorch Geometric Data Object**:
- `x`: `[N, 52]` - Node features (element, residue, ligand flag, distances, electronic, properties)
- `edge_index`: `[2, E]` - Edge connectivity (bidirectional)
- `edge_attr`: `[E, 24]` - Edge features (16-dim RBF + 4-dim angles + 4-dim dihedrals)
- `pos`: `[N, 3]` - 3D atomic coordinates
- `y`: `[1]` - Label ($\log_{10}(k_{cat})$)
- `seq_embedding`: `[1, D]` - Optional ESM-2 sequence embedding (D=640 or 1280)
- `sample_id`, `pdb_id`, `ec` - Metadata (removed before training to avoid collate errors)

### 4. Molecular Docking (DiffDock)

The project uses **DiffDock** (diffusion-based docking) for generating protein-ligand complexes:
- Input: Protein PDB + Substrate SMILES
- Output: Docked complex with confidence scores
- Location: `src/docking.py` (also `src/docking_chai1.py` for Chai-1 variant)
- Note: Requires DiffDock installation and `DIFFDOCK_PATH` environment variable

---

## Key Files & Structure

**Core Training/Evaluation**:
- `src/train.py` - Main training script (WandB integration, metrics, checkpointing)
- `src/test.py` - Testing/evaluation script
- `src/GNN_model.py` - Model definitions (`PocketGNNKcatOnly` is the active model)

**Data Processing**:
- `src/build_graph_dataset.py` - Builds .pt datasets from CSV
- `src/data_loader.py` - Data loading utilities
- `src/graph_builder_rbf.py` - Graph construction with RBF edge features

**Molecular Docking & Structure**:
- `src/docking.py` - DiffDock integration for protein-ligand docking
- `src/docking_chai1.py` - Chai-1 docking variant
- `src/pose_validation.py` - Validate docking poses
- `src/generate_pdb_fixed.py` - Generate PDB structures from sequences

**Sequence Embeddings**:
- `src/generate_esm_embeddings.py` - Generate ESM-2 embeddings from protein sequences

**Diagnosis & Analysis**:
- `src/analyze_feature_label_relation.py` - Data-level diagnostic (feature correlations, baseline models)
- `src/evaluate.py` - Model evaluation utilities

**Benchmarking** (in `scripts/`):
- `quick_benchmark.py` - Run all SOTA baselines
- `run_catapro_baseline.py` - CataPro model
- `run_dlkcat.py` - DLKcat model
- `run_unikp.py` - UniKP model
- `benchmark_comparison.py` - Compare results across methods

**Utilities**:
- `src/metadata_utils.py` - Experiment metadata management
- `src/sample_manager.py` - Sample path management
- `src/quantile_loss.py` - Quantile regression loss (uncertainty quantification)

**Output Directories**:
- `outputs/` - Training outputs (models, logs, visualizations)
- `results/` - Experiment results and predictions
- `data/processed/` - Processed .pt datasets (gitignored)
- `data/raw/` - Raw CSV data (gitignored)

---

## Training Configuration

### Command-line Arguments

**Required**:
- `--dataset`: Path to .pt dataset file

**Model Architecture**:
- `--hidden_dim`: Hidden dimension (default: 128)
- `--num_layers`: Number of GAT layers (default: 3)
- `--heads`: Number of attention heads (default: 4)
- `--pooling_type`: Pooling method - `mean`, `global_attention`, or `set2set` (default: `mean`)
- `--dropout`: Dropout rate (default: 0.1)

**Training Parameters**:
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-3)
- `--epochs`: Max epochs (default: 500)
- `--weight_decay`: L2 regularization (default: 1e-4)

**Loss & Scheduler**:
- `--loss`: Loss function - `mse`, `huber`, or `quantile` (default: `mse`)
- `--scheduler`: LR scheduler - `none`, `plateau`, or `cosine` (default: `none`)
- `--quantiles`: For quantile regression, e.g., `0.05,0.5,0.95`

**Sequence Embeddings**:
- `--use_seq_embedding`: Enable ESM-2 sequence embedding fusion
- `--seq_embedding_path`: Path to ESM-2 embeddings .pt file (dict of {sample_id: tensor})

**Diagnostic Tests**:
- `--label_permutation`: Run label permutation test (should give ~0 performance)
- `--frozen_encoder`: Freeze encoder and only train MLP head
- `--load_checkpoint`: Load pretrained model weights

**Output**:
- `--save_dir`: Output directory (auto-generated if not specified)
- `--exp_name`: Experiment name for organization

### Recommended Training Commands

**Basic training**:
```bash
python src/train.py \
  --dataset data/processed/kcat_full_1213.pt \
  --save_dir outputs/baseline
```

**With regularization and better scheduler**:
```bash
python src/train.py \
  --dataset data/processed/kcat_full_1213.pt \
  --loss huber \
  --scheduler plateau \
  --weight_decay 1e-4 \
  --dropout 0.3
```

**With sequence embeddings**:
```bash
python src/train.py \
  --dataset data/processed/kcat_full_1213.pt \
  --use_seq_embedding \
  --seq_embedding_path data/processed/esm_embeddings.pt \
  --pooling_type set2set
```

**With uncertainty quantification**:
```bash
python src/train.py \
  --dataset data/processed/kcat_full_1213.pt \
  --loss quantile \
  --quantiles 0.05,0.5,0.95
```

---

## Diagnostic Testing

The project includes scientific diagnostic tools to validate models and data quality:

### 1. Label Permutation Test
**Purpose**: Verify model truly uses graph representations (not pipeline bugs)
```bash
python src/train.py --dataset <path> --label_permutation
```
**Expected**: Pearson ≈ 0, R² ≈ 0 (if model relies on graph structure)

### 2. Frozen Encoder Test
**Purpose**: Check if encoder representations are saturated
```bash
python src/train.py \
  --dataset <path> \
  --frozen_encoder \
  --load_checkpoint <checkpoint_path>
```
**Expected**:
- Performance ≈ baseline → encoder is saturated (features are linearly usable)
- Performance drops → encoder needs end-to-end optimization

### 3. Feature-Label Correlation Analysis
**Purpose**: Answer "Does the data contain sufficient information?" (model-agnostic)
```bash
python src/analyze_feature_label_relation.py \
  --dataset <path> \
  --save_dir outputs/diagnosis
```

**Output**:
- `feature_correlations.csv`: Correlation of graph statistics with kcat
- `model_baselines.csv`: Simple model performance (Linear, Random Forest)
- Visualization plots

**Interpretation**:
- Max |Pearson| < 0.1 AND RF Pearson < 0.3 → **Data lacks information**
- Max |Pearson| ≥ 0.3 AND RF Pearson ≥ 0.5 → **Data contains signal, problem may be in model**

---

## Experiment Tracking

### WandB Integration
Training automatically logs to WandB with:
- Training/validation loss curves
- Metrics (MAE, RMSE, R², Pearson)
- Hyperparameters
- Model architecture info
- Prediction visualizations

### Local Outputs
Each training run creates:
- `best_model.pt` - Best model checkpoint (based on validation loss)
- `config.json` - Training configuration
- `metrics.json` - Final metrics
- `loss_curve.png`, `metrics_curve.png` - Training curves
- `kcat_prediction_scatter.png` - True vs Predicted scatter plot
- `kcat_density.png` - Prediction density plot
- TensorBoard logs in `outputs/`

### Runs Registry
`runs_registry.csv` tracks all experiments with metadata and performance metrics.

---

## Important Notes

### Data Assumptions
1. **Edge Features Required**: Input graphs MUST have 24-dim edge features. Missing edge_attr will cause dimension mismatch errors in the first GAT layer.
2. **Label Format**: Labels (`y`) should be $\log_{10}(k_{cat})$, not raw kcat values.
3. **Sequence Embedding Matching**: If using `--use_seq_embedding`, embeddings must be matched via `sample_id`.

### Model Compatibility
- **Current Model**: `PocketGNNKcatOnly` outputs a single value (kcat only)
- **Legacy Models**: Some older models output 2 values (kcat + Km) - be careful when loading checkpoints
- `src/pred_range_fixed.py` may need updates to work with single-task models

### Temperature Feature
The current canonical pipeline (`PocketGNNKcatOnly`) explicitly removes temperature features. This is a deliberate design choice but may be reconsidered in future work.

### Data Management
- **DO NOT commit** large files (`.csv`, `.pt`, `.pdb`) to git
- Use `sample_manager.py` to manage sample paths instead of hardcoding
- Data files should go in `data/` (gitignored)
- Results should go in `results/` or `outputs/` (gitignored)

### DiffDock Configuration
`DIFFDOCK_PATH` in `src/docking.py` needs manual setup or environment variable configuration.

---

## Scientific Background

### Node Features (52 dimensions)
1. **Element Type** (10-dim): One-hot encoding for C, N, O, S, P, F, Cl, Br, I, H
2. **Residue Type** (21-dim): One-hot for 20 standard amino acids + ligand (LIG)
3. **Ligand Flag** (1-dim): Binary indicator (0 or 1)
4. **Min Distance** (1-dim): Distance to nearest neighbor atom
5. **Electronic Features** (16-dim): Electron configuration features based on atomic number
6. **Atomic Properties** (3-dim): Mass, electronegativity, radius

### Edge Features (24 dimensions)
1. **RBF Distance Encoding** (16-dim): Gaussian RBF encoding of interatomic distances (0-8Å)
2. **Bond Angles** (4-dim): Angles formed by edge atoms and common neighbors
3. **Dihedral Angles** (4-dim): Torsion angles for edge-related atom quartets

### Graph Construction
- **Method**: Radius graph with 4.0 Å cutoff
- **Edges**: Bidirectional (undirected graph)
- **Source**: Pocket PDB files (extracted from docked protein-ligand complexes)

### Evaluation Metrics
- **R²** (Coefficient of Determination): Measures explained variance
- **Pearson Correlation**: Linear correlation between true and predicted values
- **MAE** (Mean Absolute Error): Average absolute difference in log10 space
- **RMSE** (Root Mean Squared Error): Root of average squared difference

### Sequence Embedding Fusion
- **Source**: ESM-2 protein language model (640 or 1280 dimensions)
- **Method**: Late Fusion with Projection
  - Project ESM-2 embeddings to hidden_dim (128) via Linear + ReLU + Dropout
  - Concatenate with graph-level features from pooling
  - Feed combined features into final MLP regressor
- **Motivation**: Captures sequence context beyond local pocket structure

---

## Reference Documentation

For more detailed information, refer to:
- `DEV_GUIDE.md` - Comprehensive developer guide (single source of truth)
- `README.md` - Project overview
- `QUANTILE_REGRESSION_GUIDE.md` - Uncertainty quantification guide
- `scripts/QUICK_START.md` - SOTA benchmark quick start
- `scripts/BENCHMARK_STATUS.md` - Benchmark tool status
- `scripts/EXPLAIN_BENCHMARK.md` - Benchmark methodology explanation
