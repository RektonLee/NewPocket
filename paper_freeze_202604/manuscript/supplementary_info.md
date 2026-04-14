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
| Dropout rate | 0.3 | [0.0, 0.1, 0.3, 0.5] | Validation loss |
| Learning rate | 1e-3 | [1e-4, 1e-3, 1e-2] | Validation loss |
| Weight decay | 1e-4 | [0, 1e-5, 1e-4] | Regularization |
| Batch size | 32 | [16, 32, 64] | GPU memory |
| Pooling type | set2set | [mean, attention, set2set] | Ablation |

#### S3.2 Training Configuration

- **Optimizer**: Adam with AMSGrad
- **Loss function**: Mean Squared Error (MSE)
- **Scheduler**: ReduceLROnPlateau (patience=20, factor=0.5)
- **Early stopping**: patience=50 epochs
- **Max epochs**: 500
- **Gradient clipping**: max_norm=1.0

#### S3.3 Computational Resources

- **Hardware**: NVIDIA GPU nodes (training and docking performed on local GPU workstations and cluster GPUs)
- **Training time**: few hours per model, depending on split size and checkpoint selection
- **Inference time**: seconds-level per batch on GPU
- **DiffDock docking**: minute-level per sample in single-sample inference mode

### S4. Baseline Method Reproduction

#### S4.1 CatPred

- Used official implementation from GitHub
- ESM-2 embeddings (650M model)
- Evaluated on our benchmark CSV and aligned on shared samples for paired comparison
- Hyperparameters: default from paper

#### S4.2 CataPro

- Official codebase
- Fine-tuned ProtBERT model
- Predictions aligned to our sample IDs before paired comparison
- Hyperparameters: optimized per their protocol

#### S4.3 DLKcat & UniKP

- Re-implemented based on published methods
- Sequence-only inputs
- Morgan fingerprints for ligands
- Cross-validation protocol matched

---

## Supplementary Tables

### Table S1: Verified data files used in this study

| File / split | Samples | Role |
|--------------|---------|------|
| `data/processed/kcat_full_1213.pt` | 9,059 | Full processed graph dataset used for random-split style experiments |
| `data/processed/kcat_merged_hom40_train.pt` | 8,620 | Homology-aware training set |
| `data/processed/kcat_merged_hom40_val.pt` | 996 | Homology-aware validation set |
| `data/processed/kcat_merged_hom40_test.pt` | 898 | Homology-aware test set used for main OOD evaluation |
| `data/raw/kcat_test_results.csv` | 3,444 | Benchmark CSV for shared-sample baseline comparison |
| `data/processed/kcat_test_new_diffdock.csv` | 940 | DiffDock subset used for docking robustness analysis |

### Table S2: Per-EC Class Performance

| EC Class | Enzyme Type | Test Samples | Pearson r | R² | MAE |
|----------|-------------|--------------|-----------|-----|-----|
| EC 1 | Oxidoreductases | 703 | 0.667 | 0.350 | 0.900 |
| EC 2 | Transferases | 67 | 0.773 | 0.503 | 0.559 |
| EC 3 | Hydrolases | 36 | 0.885 | 0.376 | 0.932 |
| EC 4 | Lyases | 42 | 0.813 | 0.422 | 1.047 |
| EC 5 | Isomerases | 31 | 0.843 | 0.059 | 1.383 |
| EC 6 | Ligases | 18 | 0.973 | 0.615 | 0.932 |
| Overall | All annotated test samples | 898 | 0.716 | 0.387 | 0.901 |

EC 7 contains one sample and is omitted from the per-class metric rows. These values are computed from `results/test_hom40_evaluation/test_predictions.csv` aligned to `data/processed/kcat_merged_hom40_test.pt`.

### Table S3: Feature-scale diagnostic for the graph inputs

| Feature group | Approximate observed range | Notes for interpretation |
|---------------|----------------------------|--------------------------|
| Element and residue one-hot | 0-1 | Categorical indicators; zero-baseline interpolation has no direct physical meaning |
| Ligand flag | 0-1 | Binary indicator |
| Nearest-neighbor distance | 0.50-1.82 | Continuous local geometry descriptor |
| Electronic occupancy descriptors | 0-6 | Count-like descriptors, larger scale than one-hot features |
| Atomic mass | 12.01-32.06 | Continuous physicochemical property, not standardized in current model |
| Electronegativity | 2.55-3.44 | Continuous physicochemical property |
| Atomic radius | 0.66-1.05 | Continuous physicochemical property |
| RBF distance edge features | approximately 0-1 | Bounded radial basis encoding |
| Angle and dihedral edge summaries | -1 to 1 | Cosine-based geometry summaries |

Training and inference use the same feature construction, so the predictive evaluation is internally consistent. However, raw gradient magnitudes can be affected by these mixed scales; quantitative attribution should therefore be interpreted cautiously unless re-evaluated with train-set-fitted standardization and group perturbation.

### Table S4: Shared-sample benchmark comparison

| Shared subset | Method | N | Pearson r | R² | MAE | RMSE |
|---------------|--------|---|-----------|-----|-----|------|
| CatPred overlap | CatPred | 3,439 | 0.608 | 0.165 | 0.942 | 1.405 |
| CatPred overlap | **PocketGNN** | **3,439** | **0.864** | **0.745** | **0.479** | **0.775** |
| CataPro overlap | CataPro | 1,455 | 0.683 | 0.444 | 0.892 | 1.178 |
| CataPro overlap | **PocketGNN** | **1,455** | **0.868** | **0.752** | **0.487** | **0.787** |

### Table S5: Stratified paired significance on shared subsets

| Comparison | Stratification | Pearson advantage range | MAE advantage range | Significance |
|------------|----------------|-------------------------|---------------------|--------------|
| PocketGNN vs CatPred | Sequence length tertiles | 0.207-0.281 | 0.391-0.544 | all p = 0.0005 |
| PocketGNN vs CatPred | Ligand heavy atom tertiles | 0.244-0.286 | 0.413-0.547 | all p = 0.0005 |
| PocketGNN vs CataPro | Sequence length tertiles | 0.158-0.227 | 0.355-0.433 | all p = 0.0005 |
| PocketGNN vs CataPro | Ligand heavy atom tertiles | 0.167-0.208 | 0.394-0.418 | all p = 0.0005 |

### Table S6: Docking robustness on the DiffDock subset

| Metric | Result |
|--------|--------|
| Samples | 940 |
| Overall Pearson r | 0.853 |
| Overall R² | 0.720 |
| Overall MAE | 0.473 |
| confidence vs abs error | Pearson r = 0.0267, p = 0.413 |
| confidence vs abs error | Spearman rho = 0.0361, p = 0.268 |
| low-confidence tertile | Pearson r = 0.886, MAE = 0.449 |
| mid-confidence tertile | Pearson r = 0.835, MAE = 0.481 |
| high-confidence tertile | Pearson r = 0.834, MAE = 0.491 |

These results show that DiffDock confidence is not a reliable downstream error proxy among successfully constructed graphs. They should not be used as a substitute for direct pose-quality validation, because confidence is not the same measurement as experimental pose RMSD.

### Table S7: Error correlations and density-aware confidence bins

| Variable vs absolute error | Pearson r | p-value | Spearman rho | p-value |
|----------------------------|-----------|---------|--------------|---------|
| DiffDock confidence | 0.0267 | 0.413 | 0.0361 | 0.268 |
| Number of pocket nodes | -0.0450 | 0.168 | -0.0419 | 0.199 |
| Number of edges | -0.0405 | 0.214 | -0.0423 | 0.195 |

Confidence deciles have similar sample counts by construction; visible large-error points in dense scatter regions should therefore be interpreted as a sample-density effect rather than a monotonic confidence-error trend.

| Confidence decile | N | MAE | Median AE | 90th percentile AE |
|-------------------|---:|----:|----------:|-------------------:|
| -4.54 to -2.16 | 94 | 0.472 | 0.264 | 1.095 |
| -2.16 to -1.66 | 97 | 0.401 | 0.246 | 0.859 |
| -1.66 to -1.33 | 93 | 0.511 | 0.239 | 1.400 |
| -1.33 to -1.11 | 94 | 0.404 | 0.211 | 0.909 |
| -1.11 to -0.89 | 92 | 0.585 | 0.254 | 1.341 |
| -0.89 to -0.72 | 95 | 0.394 | 0.292 | 0.737 |
| -0.72 to -0.56 | 97 | 0.470 | 0.270 | 1.294 |
| -0.56 to -0.28 | 93 | 0.478 | 0.247 | 0.981 |
| -0.28 to -0.03 | 91 | 0.502 | 0.286 | 1.458 |
| -0.03 to 1.05 | 94 | 0.525 | 0.320 | 1.184 |

### Table S8: Prediction bias across true-kcat quantile bins

This post-hoc calibration diagnostic uses the same 898 predictions as the 40% homology-split main test. Residual is defined as predicted minus true $\log_{10}(k_{cat})$. The model compresses the dynamic range of predictions: low true-kcat enzymes are over-predicted, high true-kcat enzymes are under-predicted, and the central kinetic range is best calibrated.

| True-kcat bin | N | True mean | Pred mean | Bias | MAE | 90th percentile AE |
|---------------|---:|----------:|----------:|-----:|----:|-------------------:|
| -5.52 to -1.16 | 90 | -2.138 | -0.188 | 1.950 | 1.950 | 3.319 |
| -1.16 to -0.42 | 92 | -0.803 | 0.092 | 0.895 | 0.911 | 1.553 |
| -0.42 to 0.03 | 88 | -0.159 | 0.240 | 0.399 | 0.472 | 0.886 |
| 0.03 to 0.38 | 92 | 0.216 | 0.404 | 0.187 | 0.380 | 0.657 |
| 0.38 to 0.64 | 88 | 0.524 | 0.508 | -0.016 | 0.243 | 0.682 |
| 0.64 to 0.98 | 89 | 0.816 | 0.579 | -0.237 | 0.323 | 0.852 |
| 0.98 to 1.40 | 94 | 1.178 | 0.694 | -0.483 | 0.502 | 0.830 |
| 1.40 to 1.96 | 85 | 1.672 | 0.777 | -0.895 | 0.895 | 1.314 |
| 1.96 to 2.54 | 90 | 2.267 | 0.982 | -1.285 | 1.285 | 1.657 |
| 2.54 to 6.03 | 90 | 3.280 | 1.237 | -2.044 | 2.044 | 3.146 |

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

### Figure S6: Exploratory Feature Attribution

**Panel A** (`figS12_node_feature_importance.png`):
- Top-20 most important node features (52-dim atom descriptors)
- Exploratory gradient-based attribution on mixed-scale node inputs
- Not used as a quantitative mechanistic claim in the main text

**Panel B** (`figS12_edge_feature_importance.png`):
- Top-20 most important edge features (24-dim geometric encodings)
- Exploratory attribution over RBF, angle, and dihedral channels
- Requires normalization-aware group perturbation for rigorous quantitative comparison

**Panel C** (`figS12b_feature_type_importance.png`):
- Preliminary decomposition of edge-channel sensitivity
- Reported as exploratory only because raw gradient magnitudes are scale-dependent
- Future validation should use train-set-fitted feature standardization and group masking/permutation

### Figure S7: Prediction Bias and Calibration

`results/bias_calibration/fig_prediction_bias_calibration.png`

- Left: scatter plot with true-kcat quantile-bin means overlaid against the ideal calibration diagonal
- Right: mean residual and MAE across true-kcat quantile bins
- Supports the interpretation that homology-split errors are partly driven by dynamic-range compression rather than random noise alone

---

### S5. Interpretability Analysis Methods

#### S5.1 Exploratory Integrated Gradients

We used Integrated Gradients (IG)~\citep{sundararajan2017axiomatic} as an exploratory diagnostic rather than as definitive mechanistic evidence. This distinction is important because PocketGNN node features include one-hot categorical variables, count-like electronic descriptors, and continuous physicochemical quantities on different scales. A zero-vector baseline is convenient computationally, but it does not define a physically meaningful interpolation path for every categorical feature.

**Algorithm**:
For an input feature $x$ and baseline $x'$ (zero vector), the attribution is computed as:

$$\text{IG}(x) = (x - x') \times \int_{\alpha=0}^{1} \frac{\partial f(x' + \alpha(x - x'))}{\partial x} d\alpha$$

In practice, we approximate the integral using Riemann sum with $n=50$ steps:

$$\text{IG}(x) \approx (x - x') \times \frac{1}{n} \sum_{k=1}^{n} \frac{\partial f(x' + \frac{k}{n}(x - x'))}{\partial x}$$

**Implementation Details**:
- **Baseline**: Zero vector, used only for exploratory diagnostics
- **Interpolation steps**: 50
- **Feature types analyzed**:
  - Node features (52-dim): mixed categorical, count-like, and continuous atom descriptors
  - Edge features (24-dim): RBF, angle, and dihedral encodings
- **Aggregation**: Sum absolute attribution scores across all nodes/edges
- **Interpretation**: Relative percentages are not used as primary evidence because they can be affected by feature scale and baseline choice

**Sample Selection**:
We analyzed 100 test enzyme-substrate pairs for feature attribution aggregation.
The illustrative catalytic-residue visualization is reported as a separate case study.

**Computational Cost**: on the order of seconds per sample on GPU for 50-step attribution

#### S5.2 Feature Decomposition Caveat

For exploratory analysis, the 24-dimensional edge features were decomposed into three groups:

| Feature Type | Dimensions | Indices | Description |
|--------------|-----------|---------|-------------|
| RBF Distance Encoding | 16 | [0:16] | Gaussian radial basis functions (0-8Å) |
| Bond Angles | 4 | [16:20] | $\theta_{ijk}$ for edge neighbor pairs |
| Dihedral Angles | 4 | [20:24] | $\phi_{ijkl}$ for edge atom quartets |

For each group, raw absolute attribution scores can be summed across dimensions:

$$\text{Importance}(\text{group}) = \sum_{i \in \text{group}} |\text{IG}_i|$$

These relative contributions are useful for hypothesis generation, but not sufficient for a quantitative mechanistic claim. A stricter follow-up analysis should compare feature groups by masking or permuting whole groups under the actual main-model configuration, ideally after train-set-fitted feature standardization.

#### S5.3 Validation Against Known Catalytic Residues

For the ALDH2 case study (Figure 3), we validated model attention against experimentally characterized catalytic residues:

**Known catalytic residues** (from literature):
- Cys302: Catalytic nucleophile
- Glu268: Proton abstractor
- Asn169: Substrate orientation

**Model saliency analysis**:
We computed Input × Gradient saliency for each atom:

$$\text{Saliency}(i) = |x_i \times \nabla_{x_i} f(x)|$$

This case study is used qualitatively to verify that high-saliency regions coincide with known catalytic machinery, rather than as a standalone quantitative benchmark.

#### S5.4 Reproducibility

All interpretability analysis scripts are available in the repository:
- `scripts/analyze_feature_importance.py`: exploratory Integrated Gradients implementation
- `scripts/visualize_attention_weights.py`: Attention heatmap generation
- `scripts/select_interpretability_cases.py`: Representative case selection
- `scripts/investigate_outliers.py`: Error and outlier analysis

**Random seeds**: Fixed at 42 for all analyses to ensure reproducibility.

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
