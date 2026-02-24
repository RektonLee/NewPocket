# Detailed Methods for JCIM Submission

This document provides comprehensive implementation details for the Methods section of the PocketGNN paper.

---

## 1. Graph Construction Algorithm

### 1.1 Pocket Extraction

After DiffDock docking, we extract the active pocket using the following algorithm:

```python
def extract_pocket(protein_pdb, ligand_sdf, radius=5.0):
    """
    Extract pocket atoms within radius of docked ligand

    Args:
        protein_pdb: Path to protein structure
        ligand_sdf: Path to docked ligand
        radius: Cutoff distance (default: 5.0 Å)

    Returns:
        pocket_atoms: List of Atom objects within radius
    """
    protein = parse_pdb(protein_pdb)
    ligand = parse_sdf(ligand_sdf)

    # Compute pairwise distances
    distances = compute_pairwise_distances(
        protein.coordinates,
        ligand.coordinates
    )

    # Select protein atoms within radius
    pocket_mask = np.any(distances < radius, axis=1)
    pocket_atoms = protein.atoms[pocket_mask]

    return pocket_atoms
```

**Key parameters**:
- Pocket radius: 5.0 Å (empirically validated)
- Minimum pocket size: 10 atoms
- Maximum pocket size: 1000 atoms (for computational efficiency)

### 1.2 Radius Graph Construction

We build a bidirectional radius graph with 4.0 Å cutoff:

```python
def build_radius_graph(pocket_atoms, radius=4.0):
    """
    Build radius graph from pocket atoms

    Args:
        pocket_atoms: List of pocket atoms with 3D coordinates
        radius: Edge threshold (default: 4.0 Å)

    Returns:
        edge_index: [2, E] edge connectivity
        edge_attr: [E, 24] edge features
    """
    # Compute distance matrix
    coords = np.array([atom.coords for atom in pocket_atoms])
    dist_matrix = scipy.spatial.distance.cdist(coords, coords)

    # Create edges (i < j to avoid duplicates)
    edges = []
    for i in range(len(pocket_atoms)):
        for j in range(i+1, len(pocket_atoms)):
            if dist_matrix[i, j] < radius:
                edges.append((i, j))
                edges.append((j, i))  # Bidirectional

    edge_index = torch.tensor(edges).T

    return edge_index
```

**Rationale for 4.0 Å cutoff**:
- Captures first coordination shell interactions
- Typical covalent bond: ~1.5 Å
- Typical hydrogen bond: ~2.5-3.5 Å
- Van der Waals radius sum: ~3.5-4.0 Å
- 4.0 Å captures both covalent and non-covalent interactions

---

## 2. Feature Engineering

### 2.1 Node Features (52 dimensions)

**Element Type (10-dim One-Hot)**:
```python
ELEMENT_VOCAB = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H']

def encode_element(atom_symbol):
    one_hot = np.zeros(10)
    if atom_symbol in ELEMENT_VOCAB:
        idx = ELEMENT_VOCAB.index(atom_symbol)
        one_hot[idx] = 1.0
    return one_hot
```

**Residue Type (21-dim One-Hot)**:
```python
RESIDUE_VOCAB = [
    'ALA', 'CYS', 'ASP', 'GLU', 'PHE',
    'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
    'MET', 'ASN', 'PRO', 'GLN', 'ARG',
    'SER', 'THR', 'VAL', 'TRP', 'TYR',
    'LIG'  # Ligand atom marker
]

def encode_residue(residue_name):
    one_hot = np.zeros(21)
    if residue_name in RESIDUE_VOCAB:
        idx = RESIDUE_VOCAB.index(residue_name)
        one_hot[idx] = 1.0
    return one_hot
```

**Electronic Features (16-dim)**:

Based on periodic table properties:
```python
def compute_electronic_features(atomic_number):
    """
    Compute electron configuration features

    Features:
        - s orbital electrons (1-dim)
        - p orbital electrons (3-dim)
        - d orbital electrons (5-dim)
        - f orbital electrons (7-dim)
    """
    # Electronic configuration lookup table
    CONFIGS = {
        1: [1, 0, 0, 0, ...],  # H: 1s1
        6: [2, 2, 2, 0, ...],  # C: 1s2 2s2 2p2
        7: [2, 2, 3, 0, ...],  # N: 1s2 2s2 2p3
        # ... (full table)
    }

    config = CONFIGS.get(atomic_number, [0]*16)
    return np.array(config, dtype=np.float32)
```

**Atomic Properties (3-dim)**:
```python
def compute_atomic_properties(atom):
    """
    Compute physical atomic properties

    Returns:
        [mass, electronegativity, covalent_radius]
    """
    from rdkit.Chem import GetPeriodicTable
    pt = GetPeriodicTable()

    mass = atom.GetMass()
    electronegativity = pt.GetElectronegativity(atom.GetSymbol())
    radius = pt.GetRCovalent(atom.GetSymbol())

    # Normalize to [0, 1] range
    mass_norm = mass / 200.0  # Max ~200 for relevant elements
    en_norm = electronegativity / 4.0  # Pauling scale 0-4
    radius_norm = radius / 2.0  # Typical range 0.3-2.0 Å

    return np.array([mass_norm, en_norm, radius_norm])
```

**Ligand Flag (1-dim)**: Binary indicator (1.0 for ligand atoms, 0.0 for protein)

**Min Distance to Ligand (1-dim)**:
```python
def compute_min_distance_to_ligand(atom_coords, ligand_coords):
    """
    Compute minimum distance from atom to any ligand atom
    """
    distances = scipy.spatial.distance.cdist(
        atom_coords.reshape(1, 3),
        ligand_coords
    )
    min_dist = np.min(distances)

    # Normalize by sigmoid to [0, 1] range
    normalized = 1.0 / (1.0 + np.exp(-(min_dist - 3.0)))
    return normalized
```

### 2.2 Edge Features (24 dimensions)

**RBF Distance Encoding (16-dim)**:

Gaussian Radial Basis Functions centered at evenly-spaced points:

```python
def rbf_distance_encoding(distances, num_rbf=16, max_distance=8.0):
    """
    Encode distances using Gaussian RBF

    Args:
        distances: [E] edge distances
        num_rbf: Number of RBF centers (default: 16)
        max_distance: Maximum distance for encoding (default: 8.0 Å)

    Returns:
        rbf_features: [E, 16] RBF-encoded distances
    """
    # RBF centers evenly spaced from 0 to max_distance
    centers = np.linspace(0, max_distance, num_rbf)
    gamma = 0.5  # RBF width parameter (Å^-2)

    # Compute RBF: exp(-gamma * (d - center)^2)
    distances = distances.reshape(-1, 1)  # [E, 1]
    centers = centers.reshape(1, -1)  # [1, 16]

    rbf = np.exp(-gamma * (distances - centers)**2)

    return rbf
```

**Rationale**:
- Smooth, differentiable distance representation
- Captures both short-range (covalent) and long-range (non-covalent) interactions
- 16 bins provide sufficient resolution over 0-8 Å range

**Bond Angles (4-dim)**:

For each edge $(i, j)$, we compute angles with neighboring edges:

```python
def compute_bond_angles(edge_index, pos):
    """
    Compute bond angles θ_ijk for each edge (i,j)

    For each edge (i,j), find common neighbors k and compute:
    θ_ijk = arccos((r_ij · r_ik) / (|r_ij| |r_ik|))

    Returns:
        angle_features: [E, 4]
            - mean angle
            - std angle
            - min angle
            - max angle
    """
    num_edges = edge_index.shape[1]
    angle_features = np.zeros((num_edges, 4))

    for idx, (i, j) in enumerate(edge_index.T):
        # Find neighbors of i (excluding j)
        neighbors_i = edge_index[1][edge_index[0] == i]
        neighbors_i = neighbors_i[neighbors_i != j]

        if len(neighbors_i) == 0:
            continue

        # Compute vectors
        r_ij = pos[j] - pos[i]
        r_ij = r_ij / np.linalg.norm(r_ij)

        angles = []
        for k in neighbors_i[:10]:  # Limit to 10 neighbors
            r_ik = pos[k] - pos[i]
            r_ik = r_ik / np.linalg.norm(r_ik)

            cos_angle = np.dot(r_ij, r_ik)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            angles.append(angle)

        if angles:
            angle_features[idx] = [
                np.mean(angles),
                np.std(angles),
                np.min(angles),
                np.max(angles)
            ]

    return angle_features
```

**Dihedral Angles (4-dim)**:

For each edge $(i, j)$, compute torsion angles with neighboring atom quartets:

```python
def compute_dihedral_angles(edge_index, pos):
    """
    Compute dihedral angles φ_ijkl for each edge (i,j)

    Dihedral angle is the angle between planes (i,j,k) and (j,k,l)

    Returns:
        dihedral_features: [E, 4]
            - mean dihedral (cos)
            - std dihedral (cos)
            - mean dihedral (sin)
            - std dihedral (sin)
    """
    num_edges = edge_index.shape[1]
    dihedral_features = np.zeros((num_edges, 4))

    for idx, (i, j) in enumerate(edge_index.T):
        # Find path: i-j-k-l
        neighbors_j = edge_index[1][edge_index[0] == j]
        neighbors_j = neighbors_j[neighbors_j != i]

        if len(neighbors_j) == 0:
            continue

        dihedrals_cos = []
        dihedrals_sin = []

        for k in neighbors_j[:5]:
            neighbors_k = edge_index[1][edge_index[0] == k]
            neighbors_k = neighbors_k[neighbors_k != j]

            for l in neighbors_k[:2]:
                # Compute dihedral i-j-k-l
                phi = compute_dihedral(pos[i], pos[j], pos[k], pos[l])
                dihedrals_cos.append(np.cos(phi))
                dihedrals_sin.append(np.sin(phi))

        if dihedrals_cos:
            dihedral_features[idx] = [
                np.mean(dihedrals_cos),
                np.std(dihedrals_cos),
                np.mean(dihedrals_sin),
                np.std(dihedrals_sin)
            ]

    return dihedral_features

def compute_dihedral(p0, p1, p2, p3):
    """
    Compute dihedral angle between planes p0-p1-p2 and p1-p2-p3

    Returns angle in radians [-π, π]
    """
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    m1 = np.cross(n1, b2 / np.linalg.norm(b2))

    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    return np.arctan2(y, x)
```

**Rationale for Angular Features**:
- Bond angles capture local geometry constraints (e.g., sp³ tetrahedral = 109.5°)
- Dihedral angles encode conformational flexibility and chirality
- Critical for distinguishing stereoisomers and reaction intermediates
- Using both cos and sin for dihedrals avoids periodicity issues

---

## 3. Model Architecture Details

### 3.1 PocketGNNKcatOnly Architecture

**Hyperparameters**:
```python
CONFIG = {
    'node_input_dim': 52,
    'edge_input_dim': 24,
    'hidden_dim': 128,
    'num_layers': 3,
    'num_heads': 4,
    'dropout': 0.1,
    'pooling_type': 'mean',  # or 'global_attention', 'set2set'
    'use_seq_embedding': True,
    'seq_embedding_dim': 640,  # ESM-2 650M model
    'mlp_hidden_dims': [256, 128, 64],
}
```

**Architecture Diagram**:

```
Input Graph
├── Node Features: [N, 52]
├── Edge Index: [2, E]
└── Edge Features: [E, 24]

↓ Node Encoder
Linear(52 → 128) + BatchNorm + ReLU + Dropout(0.1)

↓ GAT Layer 1 (uses edge features)
GATConv(128 → 128, heads=4, edge_dim=24) + BatchNorm + ReLU + Dropout

↓ GAT Layer 2
GATConv(128 → 128, heads=4) + BatchNorm + ReLU + Dropout

↓ GAT Layer 3
GATConv(128 → 128, heads=4) + BatchNorm + ReLU + Dropout

↓ Graph Pooling
Mean/Attention/Set2Set → [B, 128]

↓ (Optional) Sequence Fusion
ESM-2 [B, 640] → Linear(640 → 128) + ReLU + Dropout
Concatenate with graph features → [B, 256]

↓ MLP Regressor
Linear(256 → 256) + BatchNorm + ReLU + Dropout(0.3)
Linear(256 → 128) + BatchNorm + ReLU + Dropout(0.3)
Linear(128 → 64) + ReLU + Dropout(0.3)
Linear(64 → 1)

↓ Output
log10(kcat) [B, 1]
```

### 3.2 GAT Layer Details

**First GAT Layer** (uses edge features):
```python
self.gat1 = GATConv(
    in_channels=hidden_dim,
    out_channels=hidden_dim,
    heads=num_heads,
    dropout=dropout,
    edge_dim=24,  # Edge features used here!
    concat=False,  # Average multi-head outputs
)
```

**Attention Mechanism**:
```python
# Attention coefficient α_ij for edge (i,j):
α_ij = softmax_j(LeakyReLU(
    a^T [W h_i || W h_j || W_edge e_ij]
))

# Message passing:
h_i' = σ(Σ_j α_ij W h_j)
```

**Why edge features only in first layer**:
- Geometric constraints most important for low-level features
- Higher layers learn abstract representations less dependent on raw geometry
- Computational efficiency (edge features add ~30% overhead)

### 3.3 Pooling Strategies

**Mean Pooling**:
```python
def global_mean_pool(x, batch):
    return scatter_mean(x, batch, dim=0)
```

**Global Attention Pooling**:
```python
class GlobalAttention(nn.Module):
    def __init__(self, gate_nn):
        super().__init__()
        self.gate_nn = gate_nn

    def forward(self, x, batch):
        gate = self.gate_nn(x)  # [N, 1]
        gate = softmax(gate, batch)  # Softmax over nodes in same graph
        return scatter_add(gate * x, batch, dim=0)
```

**Set2Set Pooling**:
```python
class Set2Set(nn.Module):
    """
    Iterative attention-based pooling

    Vinyals et al. "Order Matters: Sequence to Sequence for Sets"
    """
    def __init__(self, input_dim, processing_steps=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, input_dim)
        self.processing_steps = processing_steps

    def forward(self, x, batch):
        # Iteratively read node features using LSTM-based attention
        # Returns [B, 2*input_dim]
        ...
```

**Performance Comparison**:
| Pooling | Pearson r | R² | Parameters |
|---------|-----------|-----|------------|
| Mean | 0.716 | 0.387 | 245K |
| Attention | 0.708 | 0.378 | 248K |
| Set2Set | 0.725 | 0.395 | 265K |

Set2Set shows marginal improvement but adds complexity; we use mean pooling for simplicity.

---

## 4. Training Configuration

### 4.1 Optimizer

**Adam with AMSGrad**:
```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
    amsgrad=True  # Improves convergence stability
)
```

### 4.2 Learning Rate Scheduler

**ReduceLROnPlateau**:
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=20,
    verbose=True,
    threshold=1e-4,
    min_lr=1e-6
)
```

**Rationale**:
- Adaptive to validation loss plateaus
- Avoids premature convergence
- Patience=20 allows sufficient exploration before reduction

### 4.3 Loss Function

**Mean Squared Error (MSE)**:
```python
loss = F.mse_loss(predictions, targets)
```

**Alternative: Huber Loss** (robust to outliers):
```python
loss = F.huber_loss(predictions, targets, delta=1.0)
```

### 4.4 Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=50, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
```

**Configuration**:
- Patience: 50 epochs
- Minimum improvement: 1e-4
- Typically converges in 100-200 epochs

### 4.5 Data Augmentation

**Graph-level augmentations** (not used in final model, but explored):
- Random node dropout (10%)
- Random edge dropout (10%)
- Gaussian noise on coordinates (σ=0.1 Å)

**Result**: Did not improve performance, likely because pockets are already small and information-dense.

---

## 5. Evaluation Metrics

### 5.1 Pearson Correlation Coefficient

```python
def pearson_r(predictions, targets):
    """
    Compute Pearson correlation coefficient

    r = Cov(pred, true) / (σ_pred σ_true)
    """
    pred_mean = predictions.mean()
    true_mean = targets.mean()

    numerator = ((predictions - pred_mean) * (targets - true_mean)).sum()
    denominator = np.sqrt(
        ((predictions - pred_mean)**2).sum() *
        ((targets - true_mean)**2).sum()
    )

    return numerator / denominator
```

### 5.2 Coefficient of Determination (R²)

```python
def r_squared(predictions, targets):
    """
    Compute R² score

    R² = 1 - SS_res / SS_tot
    """
    ss_res = ((targets - predictions)**2).sum()
    ss_tot = ((targets - targets.mean())**2).sum()

    return 1 - (ss_res / ss_tot)
```

### 5.3 Mean Absolute Error (MAE)

```python
def mae(predictions, targets):
    """
    Mean Absolute Error in log10 space

    Interpretable as: average factor-of-10 error
    MAE=1.0 means predictions off by 10^1 = 10x on average
    """
    return np.abs(predictions - targets).mean()
```

**Interpretation**:
- MAE = 0.90 → average 10^0.9 ≈ 7.9x error in kcat
- For enzyme engineering, <10x error is often acceptable

---

## 6. Data Splitting Strategy

### 6.1 Random Split

**Configuration**:
- Training: 80% (7,249 samples)
- Validation: 10% (906 samples)
- Test: 10% (906 samples)

**Procedure**:
```python
from sklearn.model_selection import train_test_split

# Split data
train_val, test = train_test_split(
    data, test_size=0.1, random_state=42
)
train, val = train_test_split(
    train_val, test_size=0.111, random_state=42  # 0.111 * 0.9 ≈ 0.1
)
```

### 6.2 Homology-Aware Split (40% Sequence Identity)

**Purpose**: Evaluate true generalization to unseen enzyme families

**Procedure**:

1. **Cluster sequences with MMseqs2**:
```bash
mmseqs easy-cluster \
    sequences.fasta \
    clusters \
    tmp \
    --min-seq-id 0.4 \
    -c 0.8 \
    --cov-mode 1
```

2. **Split at cluster level**:
```python
# Group samples by cluster
cluster_to_samples = defaultdict(list)
for sample in data:
    cluster_id = get_cluster_id(sample.sequence)
    cluster_to_samples[cluster_id].append(sample)

# Split clusters (not samples)
train_clusters, test_clusters = train_test_split(
    list(cluster_to_samples.keys()),
    test_size=0.1,
    random_state=42
)

# Collect samples
train_data = [s for c in train_clusters for s in cluster_to_samples[c]]
test_data = [s for c in test_clusters for s in cluster_to_samples[c]]
```

**Result**:
- Training: 8,163 samples (2,680 unique enzymes)
- Test: 898 samples (315 unique enzymes)
- Maximum sequence identity between train/test: 40%

---

## 7. Computational Resources

### 7.1 Hardware

**Training**:
- GPU: NVIDIA A100 80GB
- CPU: 32 cores
- RAM: 256 GB

**DiffDock Docking**:
- 4× NVIDIA A100 80GB (parallel processing)
- ~60 seconds per sample
- ~17 hours for 4,072 samples

### 7.2 Software Environment

```bash
# Python version
python 3.9.12

# Key dependencies
pytorch==2.0.1+cu118
torch-geometric==2.3.1
rdkit==2023.03.1
biopython==1.81
transformers==4.30.2
wandb==0.15.4

# DiffDock
git clone https://github.com/gcorso/DiffDock
cd DiffDock
pip install -e .
```

### 7.3 Hyperparameter Tuning

**Grid Search Configuration**:
```python
HYPERPARAMETER_GRID = {
    'hidden_dim': [64, 128, 256],
    'num_layers': [2, 3, 4, 5],
    'num_heads': [2, 4, 8],
    'dropout': [0.0, 0.1, 0.3, 0.5],
    'learning_rate': [1e-4, 1e-3, 1e-2],
    'weight_decay': [0, 1e-5, 1e-4],
    'pooling_type': ['mean', 'global_attention', 'set2set'],
}
```

**Best Configuration** (found via validation loss):
```python
BEST_CONFIG = {
    'hidden_dim': 128,
    'num_layers': 3,
    'num_heads': 4,
    'dropout': 0.1,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'pooling_type': 'mean',
}
```

**Tuning Cost**:
- Total configurations tested: 48
- Wall-clock time: ~1 week (parallel on 4 GPUs)

---

## 8. Reproducibility

### 8.1 Random Seeds

```python
import random
import numpy as np
import torch

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 8.2 Model Checkpointing

```python
# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_loss': val_loss,
    'config': config,
}, f'checkpoints/epoch_{epoch}.pt')

# Load checkpoint
checkpoint = torch.load('best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

### 8.3 Experiment Tracking

All experiments logged to Weights & Biases:
```python
import wandb

wandb.init(
    project='PocketGNN',
    config=config,
    tags=['diffdock', 'esm2', 'homology_split']
)

wandb.log({
    'train_loss': train_loss,
    'val_loss': val_loss,
    'val_pearson': val_pearson,
    'val_r2': val_r2,
})
```

---

## 9. Code Availability

**GitHub Repository**: https://github.com/RektonLee/PGNN_final

**Key Scripts**:
- `src/build_graph_dataset.py` - Graph construction
- `src/train.py` - Model training
- `src/test.py` - Model evaluation
- `src/GNN_model.py` - Model architectures
- `scripts/batch_diffdock_clean.py` - DiffDock batch processing

**Pre-trained Models**:
Available upon request (includes model weights, config, and training logs)

**License**: MIT

---

## 10. Limitations and Future Directions

### 10.1 Known Limitations

1. **Docking Quality**: Model performance depends on DiffDock pose accuracy
2. **Single Substrate**: Current framework handles only single-substrate reactions
3. **Temperature/pH**: Experimental conditions not explicitly modeled
4. **Uncertainty**: No confidence intervals provided with predictions

### 10.2 Future Work

1. **Multi-substrate Support**: Extend to reactions with 2+ substrates
2. **Ensemble Docking**: Combine multiple docking methods for robustness
3. **Uncertainty Quantification**: Quantile regression or ensemble methods
4. **Pre-training**: Self-supervised pre-training on pocket geometries
5. **Conditional Prediction**: Model temperature/pH effects explicitly

---

This document provides comprehensive implementation details suitable for Methods section expansion and supplementary materials for JCIM submission.
