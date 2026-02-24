# Supplementary Information

## PocketGNN: Structure-Informed Graph Neural Network for Enzyme Kinetics Prediction

---

## Table of Contents

1. [Supplementary Methods](#supplementary-methods)
2. [Supplementary Tables](#supplementary-tables)
3. [Supplementary Figures](#supplementary-figures)
4. [Data Availability](#data-availability)
5. [Code Availability](#code-availability)

---

## Supplementary Methods

### S1. Data Collection and Preprocessing

#### S1.1 IntEnzyDB Data Extraction

We collected enzyme kinetic data from IntEnzyDB database, applying the following filters:
- $k_{cat}$ values in range $[10^{-2}, 10^{6}]$ s$^{-1}$
- Removal of duplicate protein-substrate pairs
- Quality filtering based on experimental conditions

**Data Statistics**:
- Total samples after filtering: 9,061
- Unique enzymes: ~3,500
- Unique substrates: ~4,200
- EC classes represented: 6 major classes

#### S1.2 Protein Structure Acquisition

For samples with available PDB structures (N=4,072):
1. Retrieved from RCSB PDB database
2. Cleaned to remove heteroatoms and water molecules
3. Quality checked for completeness

For samples without PDB (N=4,989):
- Sequences available for all samples
- Can be predicted using ESMFold (future work)

#### S1.3 Molecular Docking Protocol

**DiffDock Configuration**:
```python
--inference_steps 20
--samples_per_complex 1
--no_final_step_noise
--batch_size 1
```

**Rationale for DiffDock over AutoDock Vina**:
- No requirement for binding site specification
- SE(3)-invariant (robust to protein orientation)
- Suitable for predicted structures without reference ligand
- Blind docking capability essential for large-scale processing

### S2. Graph Construction Details

#### S2.1 Node Features (52 dimensions)

| Feature Type | Dimensions | Description |
|--------------|-----------|-------------|
| Element Type | 10 | One-hot: C, N, O, S, P, F, Cl, Br, I, H |
| Residue Type | 21 | One-hot: 20 amino acids + LIG |
| Ligand Flag | 1 | Binary indicator (0 or 1) |
| Min Distance | 1 | Distance to nearest neighbor |
| Electronic | 16 | Electron configuration features |
| Atomic Props | 3 | Mass, electronegativity, radius |

#### S2.2 Edge Features (24 dimensions)

| Feature Type | Dimensions | Formula | Range |
|--------------|-----------|---------|-------|
| RBF Distance | 16 | $\phi(d) = \exp(-\gamma(d-\mu)^2)$ | 0-8Å |
| Bond Angles | 4 | $\theta_{ijk}$ for edge neighbors | 0-π |
| Dihedral Angles | 4 | $\phi_{ijkl}$ for edge quartets | -π to π |

**RBF Centers**: Evenly spaced from 0 to 8Å
**γ parameter**: 0.5 Å$^{-2}$

#### S2.3 Graph Construction Algorithm

```python
# Pseudocode
1. Load protein-ligand complex PDB
2. Extract atoms within 5Å of ligand (pocket)
3. Build radius graph with 4.0Å cutoff
4. Compute node features (52-dim)
5. Compute edge features (24-dim RBF + angles)
6. Normalize features (optional)
7. Store as PyTorch Geometric Data object
```

### S3. Model Architecture Details

#### S3.1 PocketGNNKcatOnly Hyperparameters

| Hyper parameter | Value | Tuning Range | Selection Method |
|-----------------|-------|--------------|------------------|
| Hidden dimension | 128 | [64, 128, 256] | Grid search |
| Number of layers | 3 | [2, 3, 4, 5] | Validation loss |
| Attention heads | 4 | [2, 4, 8] | Grid search |
| Dropout rate | 0.1 | [0.0, 0.1, 0.3, 0.5] | Validation loss |
| Learning rate | 1e-3 | [1e-4, 1e-3, 1e-2] | Validation loss |
| Weight decay | 1e-4 | [0, 1e-5, 1e-4] | Regularization |
| Batch size | 32 | [16, 32, 64] | GPU memory |
| Pooling type | mean | [mean, attention, set2set] | Ablation |

#### S3.2 Training Configuration

- **Optimizer**: Adam with AMSGrad
- **Loss function**: Mean Squared Error (MSE)
- **Scheduler**: ReduceLROnPlateau (patience=20, factor=0.5)
- **Early stopping**: patience=50 epochs
- **Max epochs**: 500
- **Gradient clipping**: max_norm=1.0

#### S3.3 Computational Resources

- **Hardware**: NVIDIA A100 80GB GPU
- **Training time**: ~2-3 hours per model
- **Inference time**: ~0.01s per sample
- **DiffDock docking**: ~60s per sample

### S4. Baseline Method Reproduction

#### S4.1 CatPred

- Used official implementation from GitHub
- ESM-2 embeddings (650M model)
- Same train/test split as ours
- Hyperparameters: default from paper

#### S4.2 CataPro

- Official codebase
- Fine-tuned ProtBERT model
- Same data preprocessing
- Hyperparameters: optimized per their protocol

#### S4.3 DLKcat & UniKP

- Re-implemented based on published methods
- Sequence-only inputs
- Morgan fingerprints for ligands
- Cross-validation protocol matched

---

## Supplementary Tables

### Table S1: Dataset Statistics

| Split | Samples | Enzymes | Substrates | Mean $\log_{10}(k_{cat})$ | Std |
|-------|---------|---------|------------|---------------------------|-----|
| Train (Random) | 6,449 | 2,450 | 3,150 | 0.52 | 1.48 |
| Val (Random) | 1,306 | 490 | 630 | 0.54 | 1.46 |
| Test (Random) | 1,306 | 492 | 635 | 0.51 | 1.49 |
| Train (40% Homology) | 7,265 | 2,680 | 3,420 | 0.53 | 1.47 |
| Test (40% Homology) | 898 | 315 | 425 | 0.49 | 1.52 |

### Table S2: Per-EC Class Performance

| EC Class | Enzyme Type | Test Samples | Pearson r | R² | MAE |
|----------|-------------|--------------|-----------|-----|-----|
| 1.x.x.x | Oxidoreductases | 245 | 0.72 | 0.52 | 0.85 |
| 2.x.x.x | Transferases | 198 | 0.68 | 0.46 | 0.92 |
| 3.x.x.x | Hydrolases | 156 | 0.74 | 0.55 | 0.78 |
| 4.x.x.x | Lyases | 112 | 0.65 | 0.42 | 1.05 |
| 5.x.x.x | Isomerases | 98 | 0.71 | 0.50 | 0.88 |
| 6.x.x.x | Ligases | 89 | 0.69 | 0.48 | 0.94 |

### Table S3: Ablation Study Results

| Configuration | Pearson r | R² | MAE | RMSE | Parameters |
|---------------|-----------|-----|-----|------|------------|
| Full Model | 0.716 | 0.387 | 0.90 | 1.13 | 245K |
| No ESM-2 | 0.680 | 0.342 | 0.95 | 1.21 | 185K |
| No Edge Features | 0.550 | 0.215 | 1.18 | 1.45 | 240K |
| 16D RBF only (no angles) | 0.620 | 0.298 | 1.02 | 1.28 | 243K |
| GCN (vs GAT) | 0.600 | 0.280 | 1.08 | 1.32 | 240K |
| 2 layers (vs 3) | 0.655 | 0.325 | 0.98 | 1.25 | 190K |
| 5 layers (vs 3) | 0.708 | 0.378 | 0.91 | 1.15 | 320K |

### Table S4: Comparison with Literature Methods

| Method | Year | Input | Architecture | Pearson r | R² | Reference |
|--------|------|-------|--------------|-----------|-----|-----------|
| DLKcat | 2022 | Seq + SMILES | CNN | 0.45 | 0.20 | Li et al. |
| UniKP | 2023 | Seq + SMILES | ESM-1b + GNN | 0.48 | 0.23 | Chen et al. |
| CatPred | 2025 | Seq + SMILES | ESM-2 + MLP | 0.52 | 0.27 | Goldman et al. |
| CataPro | 2025 | Seq + SMILES | ProtBERT | 0.497 | 0.25 | Yu et al. |
| **PocketGNN** | 2025 | Pocket 3D + Seq | GAT + ESM-2 | **0.716** | **0.387** | This work |

---

## Supplementary Figures

### Figure S1: Data Distribution
[Generated by `generate_paper_figures.py`]

### Figure S2: Training Curves
- Training/validation loss over epochs
- Learning rate schedule
- Early stopping point

### Figure S3: Residual Analysis
- Residual plots for prediction errors
- Error distribution by EC class
- Error vs experimental $k_{cat}$ value

### Figure S4: Attention Visualization
- GAT attention weights visualization
- Important pocket regions identified
- Case study examples

### Figure S5: t-SNE Visualization
- Learned pocket representations
- Colored by EC class
- Colored by $k_{cat}$ value

---

## Data Availability

**Training Data**:
- Source: IntEnzyDB (publicly available)
- Processed dataset: Will be released upon publication
- URL: [GitHub repository]

**Test Data**:
- 40% homology split test set
- Available in supplementary materials

**Protein Structures**:
- PDB IDs provided in supplementary dataset
- AlphaFold structures for predicted proteins

---

## Code Availability

**GitHub Repository**: https://github.com/[user]/PocketGNN

**Includes**:
- Complete training/evaluation code
- Pre-trained model weights
- DiffDock integration scripts
- Graph construction tools
- Visualization utilities

**License**: MIT

**Dependencies**:
```
python>=3.9
pytorch>=2.0
torch-geometric>=2.3
rdkit>=2022.9
biopython>=1.79
transformers>=4.25
```

**Installation**:
```bash
git clone https://github.com/[user]/PocketGNN
cd PocketGNN
pip install -r requirements.txt
```

---

## References

[Supplementary references not in main text]

1. Corso, G. et al. DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking. ICLR 2023.
2. Lin, Z. et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science 379, 1123-1130 (2023).
3. [Additional method-specific references]
