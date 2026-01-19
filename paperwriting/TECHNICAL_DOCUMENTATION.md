# PocketGNN: Technical Documentation for Academic Reference

> **Document Purpose**: 本文档是 PocketGNN 项目的完整技术参考手册，旨在为论文撰写、AI 辅助审稿和研究综述提供全面的 context。
> 
> **Last Updated**: 2026-01-19

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Scientific Background](#2-scientific-background)
3. [Methodology](#3-methodology)
4. [Model Architecture](#4-model-architecture)
5. [Data Pipeline](#5-data-pipeline)
6. [Training Framework](#6-training-framework)
7. [Experimental Results](#7-experimental-results)
8. [Codebase Structure](#8-codebase-structure)
9. [Related Work Positioning](#9-related-work-positioning)
10. [Limitations and Future Work](#10-limitations-and-future-work)
11. [Reproducibility](#11-reproducibility)

---

## 1. Project Overview

### 1.1 Core Task

**PocketGNN** 是一个跨模态深度学习框架，用于预测酶的催化常数（$k_{cat}$）。核心任务是：

$$\text{Input: } \mathcal{G}_{\text{pocket}}, \mathbf{s}_{\text{ESM}} \quad \rightarrow \quad \text{Output: } \log_{10}(k_{cat})$$

其中：
- $\mathcal{G}_{\text{pocket}}$: 酶活性口袋的 3D 几何图
- $\mathbf{s}_{\text{ESM}}$: ESM-2 提取的全局序列嵌入（可选）

### 1.2 Key Innovations

| Innovation | Description | Corresponding Paper Section |
|------------|-------------|----------------------------|
| **High-fidelity Pocket Graph** | 24维边特征（RBF + 角度 + 二面角），而非简单的距离图 | Methods 2.2.1 |
| **Cross-modal Late Fusion** | 局部 3D 几何 + 全局 1D 序列语义的融合 | Methods 2.2.3 |
| **Geometry-aware GAT** | 第一层 GAT 使用边特征调制注意力权重 | Architecture |
| **Multi-scale Representation** | 原子级 → 图级的层次表示 | Methods 2.2 |

### 1.3 Performance Summary

| Split Type | Pearson $r$ | $R^2$ | Note |
|------------|-------------|-------|------|
| Random (80/20) | **0.98** | 0.918 | In-distribution |
| Homology (<40% identity) | **0.63** | 0.379 | OOD generalization |

---

## 2. Scientific Background

### 2.1 Why $k_{cat}$ Prediction is Challenging

催化常数 $k_{cat}$ 是酶动力学的核心参数，定义为：

$$k_{cat} = \frac{V_{max}}{[E]_0}$$

其预测难度源于：

1. **Multi-scale Dependency**: $k_{cat}$ 同时依赖于：
   - 活性位点的精细 3D 构象（sub-Ångström 精度）
   - 全局蛋白质稳定性和动力学
   - 底物与过渡态的相互作用

2. **Data Sparsity**: 实验测量的 $k_{cat}$ 数据稀少且噪声大

3. **Heterogeneous Conditions**: 不同文献的实验条件（温度、pH、缓冲液）差异显著

### 2.2 Evolution of Computational Approaches (2022-2025)

酶动力学参数预测领域经历了快速发展，可分为以下几个阶段：

#### Phase 1: Sequence-based Methods (2022)

**代表作**: DLKcat (Nat. Catal. 2022)

```
Protein Sequence + Substrate SMILES → CNN/GNN → kcat
```

- **优势**: 简单、数据需求低
- **局限**: 缺乏 3D 结构信息，无法捕获空间构象对催化的影响

#### Phase 2: pLM-based Methods (2023)

**代表作**: UniKP (Nat. Commun. 2023)

```
Protein Sequence → ProtTrans/ESM → Embedding → kcat/Km/Ki
```

- **优势**: 利用预训练语言模型的进化信息
- **局限**: 黑箱表示，缺乏物理可解释性；对低同源序列泛化有限

#### Phase 3: Condition-aware Methods (2024)

**代表作**: MPEK (2024)

```
Sequence + Substrate + Temperature + pH + Organism → Multi-task NN → kcat, Km
```

- **关键洞察**: $k_{cat}$ 强依赖实验条件，不建模条件变量会导致同一酶-底物对在不同条件下标签冲突
- **局限**: 仍未利用 3D 结构信息

#### Phase 4: Structure-informed Methods (2025)

**代表作**: CatPred, CataPro, DEKP (Nat. Commun. & Brief. Bioinf. 2025)

```
Protein Structure (AlphaFold/ESMFold) + Substrate → GNN/Transformer → kcat
```

- **优势**: 引入 3D 结构信息，提升泛化能力
- **局限**: 
  - 全局结构可能引入与催化无关的噪声
  - 计算成本高（需要全蛋白结构）

#### Phase 5: Pocket-centric Methods (2025-2026)

**代表作**: GraphKcat, KcatNet (bioRxiv 2025), **PocketGNN (Ours)**

```
Active Pocket 3D Graph + Docked Substrate → Geometry-aware GNN → kcat
```

- **核心假设**: 催化效率主要由活性口袋的局部几何约束决定
- **优势**: 
  - 聚焦催化相关区域，减少噪声
  - 显式建模酶-底物结合构象

### 2.3 Key Challenges Identified from SOTA Analysis

基于对 2023-2025 SOTA 工作的分析，酶动力学预测面临以下核心挑战：

| Challenge | Evidence from Literature | PocketGNN's Approach |
|-----------|-------------------------|---------------------|
| **OOD Generalization** | CataPro 报告 40% 同源划分下性能显著下降 | 40% homology split 评测 + ESM 融合 |
| **Condition Heterogeneity** | MPEK 指出温度/pH 未建模导致标签冲突 | *当前未解决，是已知局限* |
| **Stability/Reproducibility** | NNKcat 质疑 DLKcat 等对随机种子敏感 | 固定种子 + 多次验证 |
| **Structure Noise** | DEKP/CatPred 用全局结构可能引入噪声 | Pocket-centric 设计 |
| **Binding Geometry** | GraphKcat 强调需要显式的结合构象 | DiffDock 对接 + 口袋提取 |

### 2.4 Our Hypothesis

> **Core Hypothesis**: 酶的催化效率主要由活性口袋的局部几何约束决定，但需要全局进化信息（来自 pLM）提供上下文。

数学表述：
$$k_{cat} \approx f(\mathcal{G}_{\text{pocket}}) \cdot g(\mathbf{s}_{\text{global}})$$

其中 $f$ 捕获局部物理化学约束，$g$ 捕获全局进化/稳定性信息。

---

## 3. Methodology

### 3.1 Data Curation Pipeline

```
IntEnzyDB → Filter (UniProt ID + SMILES + kcat) → Structure Acquisition → Docking → Pocket Extraction → Graph Construction
```

#### 3.1.1 Data Source

- **Primary Database**: IntEnzyDB（集成了 BRENDA, SABIO-RK 等）
- **Filtering Criteria**:
  - 有效的 UniProt ID
  - 可解析的底物 SMILES
  - $k_{cat} \in [10^{-2}, 10^{6}]$ s$^{-1}$（排除极端值）

#### 3.1.2 Structure Acquisition

```python
# Hierarchical structure acquisition (src/data_loader.py)
def get_structure(uniprot_id, sequence):
    # 1. Try local cache
    if exists(local_pdb_path):
        return load_local(local_pdb_path)
    
    # 2. Try PDB download
    if pdb_id := get_pdb_id_from_uniprot(uniprot_id):
        return download_pdb(pdb_id)
    
    # 3. ESMFold prediction
    return esmfold_predict(sequence)
```

#### 3.1.3 Molecular Docking — Tool Evolution & Current Status

> **⚠️ Note**: 本节记录了对接工具的演进历史和当前状态。详细分析见 `WhatAiDid.md`。

##### 工具演进时间线

| Phase | Tool | Status | Issue |
|-------|------|--------|-------|
| **Initial** | AutoDock Vina | 稳定但繁琐 | 需手动设定 grid box，blind docking 效果不稳定 |
| **Current** (`diffdock-integration` branch) | DiffDock | 使用中 | 存在潜在问题（见下） |
| **Recommended** | Chai-1 | 待评估 | 新一代 co-folding 模型 |

##### 当前实现：DiffDock

项目当前使用 **DiffDock**（基于扩散模型的分子对接工具）：

```python
# src/docking.py
def run_diffdock(protein_pdb, ligand_sdf, output_dir):
    """
    DiffDock 优势:
    - 不需要预先指定结合位点（减少手动配置）
    - 基于深度学习的 pose 生成
    - 输出包含置信度分数
    
    DiffDock 局限（已知问题）:
    - 物理合理性：可能生成原子重叠、键长异常的 pose
    - 手性问题：配体 stereochemistry 可能失真
    - 评分函数：binding affinity 预测不如传统方法可靠
    """
    cmd = [
        DIFFDOCK_PYTHON, f"{DIFFDOCK_PATH}/inference.py",
        "--protein_path", protein_pdb,
        "--ligand", ligand_sdf,
        "--out_dir", output_dir,
        "--inference_steps", str(inference_steps),
        "--samples_per_complex", str(samples_per_complex)
    ]
    subprocess.run(cmd)
```

##### 对接工具对比（供论文/Methods 参考）

| Tool | Method | Blind Docking (RMSD ≤2Å) | Known Pocket | Pros | Cons |
|------|--------|--------------------------|--------------|------|------|
| **AutoDock Vina** | 经验评分 + 搜索 | ~50-60% | ~65-70% | 开源稳定 | 需手动设口袋 |
| **DiffDock v2.2** | 扩散模型生成 | ~50-60% | Good | 减少手动设定 | 物理合理性问题 |
| **Chai-1** | AF-like co-folding | ~77-81% (+constraints: ~85%) | Excellent | 多模态、约束支持 | 资源需求高 |
| **Gnina** | DL scoring + Vina | ~55-65% | Good | Hybrid 方法 | 仍需设口袋 |

##### 推荐改进方向

**短期（最小改动）**: DiffDock + 传统评分重排序
```
DiffDock 生成 → Vina/Gnina 重排序 → Clash check → 选最佳 pose
```

**中长期（推荐）**: 引入 Chai-1 co-folding
```
蛋白序列 + 配体 SMILES → Chai-1 → 复合物结构 → 口袋提取
```

优势：
- 不需要单独的对接步骤
- 支持 apo 结构输入
- PoseBusters benchmark 表现优秀（~80% success rate）

```python
# Chai-1 使用示例（未来集成）
from chai_lab.chai1 import run_inference

run_inference(
    fasta_file="protein_ligand.fasta",  # 包含蛋白序列和配体 SMILES
    output_dir="output/",
    num_trunk_recycles=3,
    num_diffn_timesteps=200,
    use_esm_embeddings=True
)
```

**混合策略（最稳健）**:
1. 已知口袋 → AutoDock Vina（可靠基线）
2. 未知口袋/apo 结构 → Chai-1（新方案）
3. 验证/备用 → DiffDock + 重排序

#### 3.1.4 Pocket Extraction

活性口袋定义为 docked 配体周围 **5.0 Å** 内的所有蛋白质原子：

$$\mathcal{P} = \{a \in \text{Protein} \mid \min_{l \in \text{Ligand}} d(a, l) \leq 5.0 \text{ Å}\}$$

### 3.2 Graph Representation

#### 3.2.1 Node Features (52-dim)

节点特征是本项目的关键设计之一，包含原子级别的丰富化学描述：

| Feature Group | Dimension | Description |
|---------------|-----------|-------------|
| Element One-hot | 10 | `['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H']` |
| Residue One-hot | 21 | 20 standard AA + `LIG` (ligand marker) |
| Is Ligand | 1 | Binary flag |
| Min Distance | 1 | Distance to nearest neighbor |
| Electronic Structure | 16 | Electron configuration features based on atomic number |
| Atomic Properties | 3 | Mass, Electronegativity, Radius |

```python
# src/graph_builder_rbf.py
def build_node_features(atoms):
    x = np.hstack([
        element_onehot,      # [N, 10]
        residue_onehot,      # [N, 21]
        is_ligand,           # [N, 1]
        min_distance,        # [N, 1]
        electronic_features, # [N, 16]
        atomic_properties    # [N, 3]
    ])  # Total: [N, 52]
    return x
```

#### 3.2.2 Edge Features (24-dim) — **Key Innovation**

边特征是本项目区别于传统方法的核心创新：

| Feature Group | Dimension | Formula / Description |
|---------------|-----------|----------------------|
| **RBF Distance** | 16 | $\phi_i(d) = \exp(-\gamma (d - c_i)^2)$, centers $c_i \in [0, 8]$ Å |
| **Bond Angles** | 4 | $[\cos\theta_{\min}, \cos\theta_{\max}, \cos\theta_{\text{mean}}, n_{\text{angles}}/N]$ |
| **Dihedral Angles** | 4 | $[\cos\phi_{\min}, \cos\phi_{\max}, \cos\phi_{\text{mean}}, n_{\text{dihedrals}}/N]$ |

```python
# src/build_graph_dataset.py
def compute_angle_features(edge_index, pos):
    """
    对每条边 (i, j)，计算 i 与其他邻居形成的键角
    """
    angle_features = []
    for edge_idx in range(num_edges):
        center = row[edge_idx]
        neighbor = col[edge_idx]
        # Find other neighbors of center
        other_neighbors = get_neighbors(center, exclude=neighbor)
        # Compute cosine of angles
        cos_angles = [dot(edge_vec, other_vec) for other_vec in other_vecs]
        # Statistics: min, max, mean, count
        angle_features.append([min, max, mean, len/max_neighbors])
    return torch.stack(angle_features)  # [E, 4]

def compute_dihedral_features(edge_index, pos):
    """
    对每条边 (B, C)，计算 A-B-C-D 二面角
    """
    dihedral_features = []
    for edge_idx in range(num_edges):
        atom_b, atom_c = row[edge_idx], col[edge_idx]
        # Find A (neighbor of B, not C) and D (neighbor of C, not B)
        for atom_a in neighbors_of_b:
            for atom_d in neighbors_of_c:
                # Dihedral: A-B-C-D
                n1 = cross(vec_BA, vec_BC)
                n2 = cross(vec_CB, vec_CD)
                cos_dihedral = dot(n1, n2)
                dihedrals.append(cos_dihedral)
        dihedral_features.append([min, max, mean, count])
    return torch.stack(dihedral_features)  # [E, 4]
```

**为什么角度特征重要？**

传统的"bag-of-distances"表示无法区分具有相同距离分布但不同几何构型的口袋。角度和二面角特征捕获了：
- **Chirality**: 手性中心的立体化学
- **Conformational Flexibility**: 局部构象柔性
- **Transition State Geometry**: 过渡态稳定所需的精确几何

#### 3.2.3 Graph Construction

使用 **Radius Graph** 构建分子图：

$$E = \{(i, j) \mid d(i, j) \leq r_{\text{cut}}\}$$

其中 $r_{\text{cut}} = 4.0$ Å（无向图，双向边）。

### 3.3 ESM-2 Sequence Embedding (Optional Cross-modal Fusion)

#### 3.3.1 Embedding Generation

```python
# src/generate_esm_embeddings.py
def generate_embeddings(csv_path, output_path, model_name="esm2_t33_150M_UR50D"):
    """
    为 CSV 中的蛋白质序列生成 ESM-2 嵌入
    
    Supported models:
    - esm2_t33_150M_UR50D (faster, smaller)
    - esm2_t36_3B_UR50D (larger, more powerful)
    - facebook/esm2_t30_150M_UR50D (HuggingFace version)
    """
    model, alphabet, batch_converter = load_esm_model(model_name, device)
    
    embeddings = {}
    for sample_id, sequence in sequences:
        with torch.no_grad():
            results = model(tokens, repr_layers=[last_layer])
            # Mean pooling over sequence length (excluding special tokens)
            embedding = results["representations"][last_layer][:, 1:-1, :].mean(dim=1)
        embeddings[sample_id] = embedding.squeeze().cpu()
    
    torch.save(embeddings, output_path)  # {sample_id: tensor[dim]}
```

输出维度：
- ESM-2 150M: **640-dim**
- ESM-2 3B: **1280-dim** (if using 33 layers)

#### 3.3.2 Fusion Strategy: Late Fusion with Projection

为避免高维序列嵌入"淹没"低维图特征，使用投影层：

$$\mathbf{s}_{\text{proj}} = \text{ReLU}(\text{LayerNorm}(\mathbf{W}_{\text{proj}} \mathbf{s}_{\text{ESM}}))$$

其中 $\mathbf{W}_{\text{proj}} \in \mathbb{R}^{d_{\text{hidden}} \times d_{\text{ESM}}}$（例如 $128 \times 640$）。

---

## 4. Model Architecture

### 4.1 Overall Architecture

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                    PocketGNNKcatOnly                    │
                    └─────────────────────────────────────────────────────────┘
                                              │
        ┌─────────────────────────────────────┴─────────────────────────────────────┐
        │                                                                            │
        ▼                                                                            ▼
┌───────────────────┐                                                    ┌───────────────────┐
│   Local Branch    │                                                    │   Global Branch   │
│  (Pocket Graph)   │                                                    │  (ESM-2 Sequence) │
└───────────────────┘                                                    └───────────────────┘
        │                                                                            │
        ▼                                                                            ▼
┌───────────────────┐                                                    ┌───────────────────┐
│  Node Encoder     │                                                    │  Projection Layer │
│  Linear(52→128)   │                                                    │  Linear(640→128)  │
│  + ReLU           │                                                    │  + LayerNorm      │
└───────────────────┘                                                    │  + ReLU + Dropout │
        │                                                                └───────────────────┘
        ▼                                                                            │
┌───────────────────┐                                                                │
│  GAT Layers (×3)  │                                                                │
│  - Layer 1: edge  │                                                                │
│    attr (24-dim)  │                                                                │
│  - Layer 2-3: no  │                                                                │
│    edge attr      │                                                                │
│  + ELU activation │                                                                │
└───────────────────┘                                                                │
        │                                                                            │
        ▼                                                                            │
┌───────────────────┐                                                                │
│  Graph Pooling    │                                                                │
│  (mean/attention/ │                                                                │
│   set2set)        │                                                                │
└───────────────────┘                                                                │
        │                                                                            │
        │                        ┌───────────────────┐                               │
        └───────────────────────►│    Concatenate    │◄──────────────────────────────┘
                                 │  [graph ∥ seq]   │
                                 └───────────────────┘
                                          │
                                          ▼
                                 ┌───────────────────┐
                                 │   MLP Regressor   │
                                 │  LayerNorm        │
                                 │  Linear→ReLU→Drop │
                                 │  Linear→ReLU→Drop │
                                 │  Linear→1         │
                                 └───────────────────┘
                                          │
                                          ▼
                                 ┌───────────────────┐
                                 │   log₁₀(kcat)     │
                                 └───────────────────┘
```

### 4.2 Geometry-Aware Graph Attention (GAT)

第一层 GAT 使用 24 维边特征调制注意力权重：

$$\alpha_{ij} = \text{softmax}_j \left( \text{LeakyReLU}\left( \mathbf{a}^T [\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j \| \mathbf{W}_e \mathbf{e}_{ij}] \right) \right)$$

其中 $\mathbf{e}_{ij} \in \mathbb{R}^{24}$ 是边特征（RBF + 角度 + 二面角）。

```python
# src/GNN_model.py
class PocketGNNKcatOnly(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim=256, 
                 num_layers=6, heads=8, dropout=0.1, ...):
        # GAT layers
        self.att_layers = nn.ModuleList()
        for i in range(num_layers):
            # First layer uses edge features
            edge_dim = edge_input_dim if i == 0 else None
            self.att_layers.append(
                GATConv(
                    current_dim, 
                    hidden_dim // heads,  # per-head dimension
                    heads=heads,
                    dropout=dropout,
                    concat=(i < num_layers - 1),  # concat for all but last
                    edge_dim=edge_dim  # Only first layer
                )
            )
```

### 4.3 Pooling Strategies

支持三种图池化方法：

| Pooling Type | Formula | Output Dim | Use Case |
|--------------|---------|------------|----------|
| **Mean** (default) | $\mathbf{h}_G = \frac{1}{N}\sum_i \mathbf{h}_i$ | $d_{\text{hidden}}$ | Fast, baseline |
| **Global Attention** | $\mathbf{h}_G = \sum_i \alpha_i \mathbf{h}_i$ | $d_{\text{hidden}}$ | Learnable weights |
| **Set2Set** | LSTM-based attention | $2 \cdot d_{\text{hidden}}$ | Captures set structure |

```python
# Pooling implementation
if pooling_type == 'mean':
    self.readout = global_mean_pool
elif pooling_type == 'global_attention':
    self.gate_nn = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, 1)
    )
    self.readout = GlobalAttention(gate_nn=self.gate_nn)
elif pooling_type == 'set2set':
    self.readout = Set2Set(hidden_dim, processing_steps=3)
```

### 4.4 Model Hyperparameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `hidden_dim` | 256 | Hidden dimension |
| `num_layers` | 6 | Number of GAT layers |
| `heads` | 8 | Attention heads |
| `dropout` | 0.1 | Dropout rate |
| `pooling_type` | 'mean' | Pooling strategy |
| `use_seq_embedding` | False | Enable ESM-2 fusion |

---

## 5. Data Pipeline

### 5.1 Data Flow Overview

```
CSV (sample_id, substrate_smiles, kcat_value, [protein_sequence])
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Structure Acquisition (src/data_loader.py)                     │
│  - Local cache → PDB download → ESMFold prediction             │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Molecular Docking (src/docking.py)                             │
│  - DiffDock: protein + substrate → docked complex              │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Pocket Extraction (src/docking.py)                             │
│  - Extract atoms within 5Å of ligand                           │
│  - Combine protein + ligand atoms                              │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Graph Construction (src/build_graph_dataset.py)                │
│  - build_graph(): node features (52-dim)                       │
│  - radius_graph(): edge connections (cutoff=4.0Å)              │
│  - gaussian_rbf(): RBF distance encoding (16-dim)              │
│  - compute_angle_features(): angle features (4-dim)            │
│  - compute_dihedral_features(): dihedral features (4-dim)      │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Optional: ESM-2 Embedding (src/generate_esm_embeddings.py)     │
│  - protein_sequence → ESM-2 → mean pooled embedding           │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
PyTorch Geometric Data Object (.pt file)
    - x: [N, 52] node features
    - edge_index: [2, E] edge connections
    - edge_attr: [E, 24] edge features
    - pos: [N, 3] 3D coordinates
    - y: [1] log10(kcat) label
    - seq_embedding: [1, dim] (optional)
```

### 5.2 Key Scripts

| Script | Function | Input | Output |
|--------|----------|-------|--------|
| `build_graph_dataset.py` | Build training dataset | CSV + Pocket PDBs | `.pt` file |
| `generate_esm_embeddings.py` | Generate sequence embeddings | CSV with sequences | `.pt` dict |
| `docking.py` | Molecular docking | Protein PDB + SMILES | Docked complex |
| `analyze_feature_label_relation.py` | Data diagnosis | `.pt` dataset | Correlation analysis |

---

## 6. Training Framework

### 6.1 Training Script (`src/train.py`)

```bash
python src/train.py \
    --dataset data/processed/kcat_full.pt \
    --exp_name kcat_cross_modal_v1 \
    --loss huber \
    --scheduler plateau \
    --pooling_type set2set \
    --use_seq_embedding \
    --seq_embedding_path data/processed/esm_embeddings.pt \
    --weight_decay 1e-4 \
    --dropout 0.3 \
    --max_epochs 500
```

### 6.2 Loss Functions

| Loss | Formula | Use Case |
|------|---------|----------|
| **MSE** (default) | $\frac{1}{n}\sum(y - \hat{y})^2$ | Standard regression |
| **Huber** | $\begin{cases} \frac{1}{2}(y-\hat{y})^2 & \|y-\hat{y}\| \leq \delta \\ \delta(\|y-\hat{y}\| - \frac{\delta}{2}) & \text{otherwise} \end{cases}$ | Robust to outliers |
| **Quantile** | $\sum_q \rho_q(y - \hat{y}_q)$ | Uncertainty estimation |

### 6.3 Learning Rate Schedulers

```python
# Plateau scheduler (recommended)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)

# Cosine annealing
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
```

### 6.4 Evaluation Metrics

```python
def compute_metrics(y_true_log, y_pred_log):
    """
    Metrics computed on log10(kcat) scale
    """
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'Pearson': pearsonr(y_true, y_pred)[0]
    }
```

### 6.5 Experiment Tracking

- **WandB**: Online experiment tracking with tags, notes, and hyperparameter logging
- **TensorBoard**: Local visualization
- **Metadata Utils**: JSON-based experiment registry

---

## 7. Experimental Results

### 7.1 Main Results (From Paper)

#### In-Distribution Performance (Random 80/20 Split)

| Model | MSE | $R^2$ | Pearson $r$ |
|-------|-----|-------|-------------|
| Sequence Baseline | 2.227 | -30.15 | 0.18 |
| Sequence Enhanced | 0.28 | -0.32 | 0.82 |
| PGNN-Basic (Structure Only) | 0.79 | 0.665 | 0.95 |
| **PocketGNN (Cross-Modal)** | **0.19** | **0.918** | **0.98** |

#### Out-of-Distribution Performance (40% Homology Split)

| Model | Test Pearson $r$ | Test $R^2$ |
|-------|------------------|------------|
| PocketGNN (Cross-Modal) | **0.632** | 0.379 |

### 7.2 Ablation Studies

| Ablation | Effect on Performance |
|----------|----------------------|
| Remove angle/dihedral features | Pearson ↓ ~0.05 |
| Remove ESM-2 branch | Pearson ↓ ~0.03 (more for flexible proteins) |
| Use GCN instead of GAT | Pearson ↓ ~0.08 |
| Use simple distance edges | Pearson ↓ ~0.10 |

### 7.3 EC-wise Performance

| EC Class | Description | Performance |
|----------|-------------|-------------|
| EC 3 (Hydrolases) | Highest accuracy | ★★★★★ |
| EC 2 (Transferases) | High accuracy | ★★★★☆ |
| EC 1 (Oxidoreductases) | Good accuracy | ★★★☆☆ |
| EC 6 (Ligases) | Lower accuracy (complex mechanisms) | ★★☆☆☆ |

---

## 8. Codebase Structure

```
PGNN_clean/
├── src/                           # Core source code
│   ├── GNN_model.py              # Model definitions (PocketGNNKcatOnly)
│   ├── train.py                  # Training script
│   ├── test.py                   # Testing/inference script
│   ├── build_graph_dataset.py    # Dataset construction
│   ├── graph_builder_rbf.py      # Base graph building utilities
│   ├── docking.py                # DiffDock integration
│   ├── data_loader.py            # Data loading and structure processing
│   ├── generate_esm_embeddings.py# ESM-2 embedding generation
│   ├── analyze_feature_label_relation.py  # Data diagnosis
│   └── quantile_loss.py          # Quantile regression loss
│
├── data/                          # Data directory
│   └── processed/                 # Processed datasets (.pt files)
│
├── experiments/                   # Experiment outputs
│   └── {exp_name}/{run_id}/      # Individual run results
│
├── DiffDock/                      # DiffDock installation
│
├── paperwriting/                  # Paper-related files
│   ├── gemini.pdf                # Draft paper
│   └── TECHNICAL_DOCUMENTATION.md # This document
│
├── DEV_GUIDE.md                   # Developer guide
├── requirements.txt               # Dependencies
└── README.md                      # Project overview
```

### 8.1 Key Classes and Functions

```python
# src/GNN_model.py
class PocketGNNKcatOnly(nn.Module):
    """
    Main model class for kcat prediction.
    
    Args:
        node_input_dim: Node feature dimension (default: 52)
        edge_input_dim: Edge feature dimension (default: 24)
        hidden_dim: Hidden layer dimension (default: 256)
        num_layers: Number of GAT layers (default: 6)
        heads: Number of attention heads (default: 8)
        dropout: Dropout rate (default: 0.1)
        pooling_type: 'mean', 'global_attention', or 'set2set'
        use_seq_embedding: Enable ESM-2 fusion
        seq_embedding_dim: ESM-2 embedding dimension
    """

# src/build_graph_dataset.py
def enhanced_build_graph(atoms, temperature):
    """
    Build graph with enhanced 24-dim edge features.
    
    Returns:
        torch_geometric.data.Data with:
        - x: [N, 52] node features
        - edge_index: [2, E] edges
        - edge_attr: [E, 24] edge features (RBF + angle + dihedral)
        - pos: [N, 3] coordinates
    """

# src/docking.py
def run_diffdock(protein_pdb, ligand_sdf, output_dir):
    """Run DiffDock for molecular docking."""

def extract_pocket_pymol(protein_path, ligand_path, output_path, cutoff=5.0):
    """Extract pocket region with ligand atoms."""
```

---

## 9. Related Work Positioning

> **Note**: 本节整理了 2023-2025 年酶动力学参数预测领域的 SOTA 工作，按"**数据 → 表征 → 结构信息 → 泛化评测 → 不确定性/可用性**"这一主线梳理。

### 9.1 SOTA Methods Landscape (2023-2025)

#### 9.1.1 Tier 1: Community Benchmark & Framework

**CatPred** (Nature Communications, 2025-02-28) [[Paper]](https://www.nature.com/articles/s41467-025-57215-9)

> **定位**: 目前最像"社区基准 + 强 SOTA 框架"的工作

| Aspect | Description |
|--------|-------------|
| **Task Coverage** | 同时做 $k_{cat}$ / $K_m$ / $K_i$，而非单一任务 |
| **Data Contribution** | 提出较大覆盖的 benchmark 数据集（$k_{cat}$ ~23k, $K_m$ ~41k, $K_i$ ~12k） |
| **Key Innovation** | 强调在**训练集不相似序列**上的表现 + **不确定性估计** |
| **Input Modality** | 序列/结构 + 底物 SMILES（$k_{cat}$ 用所有反应物 SMILES 拼接） |

> 💡 **对 PocketGNN 的意义**: CatPred 是**核心对照基线**，它不仅是模型，还提供了更标准的数据与评测叙事。

---

**CataPro** (Nature Communications, 2025-03-20) [[Paper]](https://www.nature.com/articles/s41467-025-58038-4)

> **定位**: "严肃冷启动评测（低同源）+ 序列/底物 PLM 组合"

| Aspect | Description |
|--------|-------------|
| **Evaluation Design** | CD-HIT 按序列相似度（cutoff 0.4）分组做 10-fold 评测 |
| **Representation** | ProtT5 (protein) + MolT5 (substrate) + MACCS fingerprint → MLP |
| **Special Design** | $k_{cat}/K_m$ 先分别预测，再用"修正项"网络纠偏 |

> 💡 **对 PocketGNN 的意义**: CataPro 的分组/评测策略值得直接借鉴，特别是"**新酶/低同源**"场景的评估设计。

---

#### 9.1.2 Tier 2: Condition-Aware & Multi-Task

**MPEK** (PubMed, 2024) [[Paper]](https://pubmed.ncbi.nlm.nih.gov/39129365/)

> **定位**: "把实验条件因素纳入输入"——解决被很多工作忽略的关键问题

| Aspect | Description |
|--------|-------------|
| **Key Innovation** | 明确把 **pH、温度、organism** 编码成特征输入 |
| **Architecture** | 多任务结构同时预测 $k_{cat}$ 与 $K_m$ |
| **Reported Performance** | $k_{cat}$ Pearson 0.808, $K_m$ Pearson 0.777 |

> 💡 **对 PocketGNN 的意义**: MPEK 指出了**不建模条件变量 → 同一酶同一底物在不同实验条件下标签冲突**的问题。这是当前 PocketGNN 的已知局限之一。

---

#### 9.1.3 Tier 3: Structure-Informed GNN

**DEKP** (Briefings in Bioinformatics, 2025) [[Paper]](https://academic.oup.com/bib/article/26/2/bbaf187/8119324)

> **定位**: "把结构信息更系统地引入（GNN/结构图）"

| Aspect | Description |
|--------|-------------|
| **Motivation** | 纯序列缺失关键 3D/口袋信息 |
| **Approach** | 结构图 + GNN 提升泛化、缓解同源性差异 |

> 💡 **对 PocketGNN 的意义**: DEKP 是走"ESMFold/AlphaFold 结构 + pocket 图 + substrate 图"路线的已发表参照，**与 PocketGNN 方向最为接近**。

---

#### 9.1.4 Tier 4: Mutation/Variant Prediction

**EITLEM-Kinetics** (Cell Chemical Catalysis, 2024) [[Paper]](https://www.sciencedirect.com/science/article/pii/S2667109324002665)

> **定位**: 变体/突变体动力学预测（定向进化语境）

| Aspect | Description |
|--------|-------------|
| **Task Focus** | 突变体/多突变 → 动力学参数变化 |
| **Key Techniques** | Iterative transfer learning / ensemble / 低同源（<40%）鲁棒性 |

> 💡 **对 PocketGNN 的意义**: 如果扩展到"突变导致的 $k_{cat}$/$K_m$ 变化"或定向进化筛选，此工作是重要参考。

---

#### 9.1.5 Tier 5: Stability & Reproducibility Critique

**NNKcat** (Briefings in Bioinformatics, 2025) [[Paper]](https://academic.oup.com/bib/article/26/3/bbaf212/8131740)

> **定位**: 对 DLKcat 等早期工作的稳定性/过拟合质疑与改进

| Aspect | Description |
|--------|-------------|
| **Critique** | 早期工作对数据划分/随机种子敏感，存在稳定性问题 |
| **Proposal** | 更可控的 protein/substrate processor 设计与 "focused learning" |

> 💡 **对 PocketGNN 的意义**: 在写"现有 SOTA 的可靠性/可复现性问题"时，此工作提供重要论据。

---

#### 9.1.6 Tier 6: Emerging Preprints (2025) — Binding Geometry

**GraphKcat** (bioRxiv, 2025-05) [[Paper]](https://www.biorxiv.org/content/10.1101/2025.05.18.654694v1.full-text)

**KcatNet** (bioRxiv, 2025-03) [[Paper]](https://www.biorxiv.org/content/10.1101/2025.03.09.642294v3)

> **共同信号**: "序列 + SMILES 还不够 → 需要真实的 binding geometry"

| Work | Focus |
|------|-------|
| **GraphKcat** | 将酶-底物 3D 结合构象纳入图学习，pocket-informed augmentation |
| **KcatNet** | 几何深度学习做 genome-wide 预测 |

> 💡 **对 PocketGNN 的意义**: 这条趋势与 PocketGNN 的"对接 → 提 pocket 原子 → GNN"**几乎同构**。差别主要在于：
> - 如何规模化得到构象
> - 如何评测冷启动
> - 是否做不确定性/校准

---

### 9.2 Comprehensive Method Matrix

| Method | Year | Venue | Input Modality | Structure Info | Split Strategy | Temp/pH | UQ | Mutant Support |
|--------|------|-------|----------------|----------------|----------------|---------|-----|----------------|
| **DLKcat** | 2022 | Nat. Catal. | Seq + SMILES | ✗ | Random | ✗ | ✗ | ✗ |
| **UniKP** | 2023 | Nat. Commun. | Seq (ProtTrans) | ✗ | Random | ✗ | ✗ | ✗ |
| **MPEK** | 2024 | PubMed | Seq + SMILES + Conditions | ✗ | Random | ✓ | ✗ | ✗ |
| **EITLEM-Kinetics** | 2024 | Chem. Catal. | Seq + Mutation | ✗ | Low-homology | ✗ | ✗ | ✓ |
| **CatPred** | 2025 | Nat. Commun. | Seq/Struct + SMILES | Partial | OOD-aware | ✗ | ✓ | ✗ |
| **CataPro** | 2025 | Nat. Commun. | ProtT5 + MolT5 + MACCS | ✗ | CD-HIT 40% | ✗ | ✗ | ✗ |
| **DEKP** | 2025 | Brief. Bioinf. | Struct Graph + GNN | ✓ (Global) | Mixed | ✗ | ✗ | ✗ |
| **NNKcat** | 2025 | Brief. Bioinf. | Seq + SMILES | ✗ | Focused | ✗ | ✗ | ✗ |
| **GraphKcat** | 2025 | bioRxiv | Pocket + Substrate 3D | ✓ (Pocket) | TBD | ✗ | ✗ | ✗ |
| **KcatNet** | 2025 | bioRxiv | Geometric DL | ✓ | TBD | ✗ | ✗ | ✗ |
| **PocketGNN (Ours)** | 2026 | - | Pocket Graph + ESM | ✓ (Pocket) | Random + 40% Homology | ✗* | ✗* | ✗ |

> *Note: Temperature/pH 和 UQ 是 PocketGNN 的已知局限，计划在后续版本加入。

---

### 9.3 PocketGNN's Position in the Landscape

#### 9.3.1 Three Categories of "SOTA" (论文/实验策略)

| Category | Representative Works | PocketGNN's Relation |
|----------|---------------------|---------------------|
| **1. Benchmark & Reproducibility SOTA** | CatPred (数据+UQ+OOD), CataPro (冷启动评测) | 可借鉴其评测设计和数据划分策略 |
| **2. Condition-Aware Modeling** | MPEK (条件变量 + 多任务) | 当前未覆盖，是**已知局限** |
| **3. Structure/Mechanism-Aware** | DEKP (结构图/GNN), GraphKcat/KcatNet (pocket-binding) | **PocketGNN 的核心定位**，但边特征设计更精细 |

#### 9.3.2 Key Differentiators of PocketGNN

| Differentiator | vs. Sequence-based (DLKcat, UniKP) | vs. Full-structure (DEKP) | vs. Pocket-based (GraphKcat) |
|----------------|-----------------------------------|---------------------------|------------------------------|
| **Pocket Focus** | ✓ 增加局部 3D 信息 | ✓ 避免全局噪声 | ≈ 相似 |
| **24-dim Edge Features** | ✓ 更丰富的几何描述 | ✓ 更精细（角度+二面角） | ✓ 可能更精细* |
| **ESM-2 Fusion** | ≈ 类似 pLM 使用 | ✓ 额外的全局语义 | ? 取决于实现 |
| **DiffDock Docking** | ✓ 显式建模结合构象 | ✓ 更合理的口袋定义 | ≈ 相似思路 |

> *Note: 与 GraphKcat/KcatNet 的详细对比需待其正式发表后进行。

#### 9.3.3 Innovation Positioning for Paper Writing

基于 SOTA 分析，PocketGNN 最容易写出创新点的切入位置：

1. **几何边特征的精细化**
   - 现有 pocket-based 方法多用简单距离图
   - PocketGNN 的 24-dim 边特征（RBF + 角度 + 二面角）是**显式的立体化学编码**
   
2. **Cross-modal Fusion 的设计选择**
   - 不同于 CataPro 的简单拼接
   - 使用投影层避免维度不平衡

3. **可解释性分析**
   - Input × Gradient 可视化催化残基
   - 验证模型关注化学上有意义的位置

4. **潜在改进方向**（差异化叙事）
   - 与 MPEK 的差异：结构 vs. 条件建模
   - 与 CatPred 的差异：局部 vs. 全局结构
   - 与 GraphKcat 的差异：边特征设计 + ESM 融合策略

---

### 9.4 Recommended Baseline Comparison Set

如果要搭建一个完整的对照实验，建议选择以下对照组：

| Role | Method | Reason |
|------|--------|--------|
| **Classic Baseline** | DLKcat (2022) | 仍是很多后续论文的比较对象 |
| **PLM-era Strong Baseline** | UniKP (2023) | 纯序列 + pLM 的代表 |
| **Recent Framework (2025)** | CatPred + CataPro | 系统框架 + 冷启动评测 |
| **Structure-based Contrast** | DEKP (2025) | 结构/GNN 路线的对照 |
| **Condition-aware Contrast** | MPEK (2024) | 说明条件变量的重要性 |

这样 Related Work + Benchmark 叙事会非常完整：
```
CNN/GNN (DLKcat) → PLM (UniKP) → 2025 系统框架 (CatPred, CataPro) → 结构/口袋路线 (DEKP, PocketGNN)
```

---

### 9.5 References for Related Work Section

```
[1] Li et al. "DLKcat: deep learning for kcat prediction using protein sequence and substrate structure." Nat. Catal. 2022.
[2] Yu et al. "UniKP: a unified framework for the prediction of enzyme kinetic parameters." Nat. Commun. 2023.
[3] Gao et al. "CatPred: a comprehensive framework for deep learning in vitro enzyme kinetic parameters." Nat. Commun. 2025.
[4] Yang et al. "Robust enzyme discovery and engineering with deep learning using CataPro." Nat. Commun. 2025.
[5] MPEK: "MPEK: a multitask deep learning framework based on pretrained language models for enzymatic reaction kinetic parameters prediction." PubMed 2024.
[6] EITLEM-Kinetics: "A deep-learning framework for kinetic parameter prediction of mutant enzymes." Chem. Catal. 2024.
[7] DEKP: "DEKP: ... based on pretrained models and graph neural networks." Brief. Bioinform. 2025.
[8] NNKcat: "NNKcat: deep neural network to predict catalytic constants..." Brief. Bioinform. 2025.
[9] GraphKcat: bioRxiv 2025.
[10] KcatNet: bioRxiv 2025.
```

---

## 10. Limitations and Future Work

### 10.1 Current Limitations

#### 10.1.1 Limitation Matrix (vs. SOTA)

| Limitation | Description | SOTA Comparison | Priority | Potential Solution |
|------------|-------------|-----------------|----------|-------------------|
| **Condition Variables** | 未建模温度/pH/organism | MPEK 已解决 | **High** | 条件嵌入 + 多任务学习 |
| **Uncertainty Quantification** | 无预测置信度估计 | CatPred 已实现 | **High** | Quantile regression / MC Dropout |
| **OOD Generalization** | 40% 同源划分下 $R^2$ 降至 0.379 | CataPro 评测更严格 | **Medium** | Pocket geometry pre-training |
| **Data Quality** | 低质量/不完整数据的表现未知 | NNKcat 分析了稳定性 | **Medium** | Robust loss + Data cleaning |
| **Multi-substrate** | 只处理单一底物 | CatPred 支持多反应物 | **Low** | Graph-level substrate encoding |
| **Mutant Support** | 不支持突变体预测 | EITLEM-Kinetics 专门设计 | **Low** | 突变编码 + Transfer learning |

#### 10.1.2 Gap Analysis with Leading SOTA

**vs. CatPred (Nat. Commun. 2025)**:
- ❌ 缺少 uncertainty estimation
- ❌ 任务覆盖（只做 $k_{cat}$，未做 $K_m$/$K_i$）
- ✅ 更精细的口袋几何表示

**vs. MPEK (2024)**:
- ❌ 未建模实验条件（温度/pH/organism）
- ✅ 引入了 3D 结构信息

**vs. GraphKcat/KcatNet (bioRxiv 2025)**:
- ≈ 相似的 pocket-binding 构象思路
- ✅ 更精细的边特征（24-dim vs. 简单距离）
- ? 待其正式发表后详细对比

### 10.2 Architecture Upgrade Roadmap (from `revise_advice0118.md`)

当前架构被视为"工程上很强，但方法论上偏 pipeline"。建议的升级方向：

#### 1. Physics-aware Dual-stream Architecture
```
Geometry Stream          Electronic Stream
(distances, angles)      (charges, elements)
      │                        │
      ▼                        ▼
   GAT_geo                  GAT_ele
      │                        │
      └──── Cross-Attention ───┘
                  │
                  ▼
        Unified Representation
```

#### 2. Residual Graph Transformer Block
```python
# Pre-norm residual structure
h_new = h + GAT(LayerNorm(h))  # Instead of h = GAT(h)
```

#### 3. Hierarchical Pooling
```
Atom-level GAT → Residue-aware Pooling → Pocket-level Transformer
```

#### 4. Structure-Sequence Co-attention (vs. simple concatenation)
```
H_graph' = Attn(H_graph, H_seq)
H_seq'   = Attn(H_seq, H_graph)
```

---

## 11. Reproducibility

### 11.1 Environment Setup

```bash
# Create conda environment
conda create -n pocketgnn python=3.9
conda activate pocketgnn

# Install PyTorch and PyG
pip install torch==2.0.0 torchvision torchaudio
pip install torch-geometric torch-scatter torch-sparse torch-cluster

# Install other dependencies
pip install -r requirements.txt

# For DiffDock (optional, requires separate environment)
cd DiffDock
conda env create --file environment.yml
conda activate diffdock
```

### 11.2 Key Dependencies

```
torch>=2.0.0
torch-geometric>=2.3.0
transformers>=4.30.0  # For ESM-2
esm>=2.0.0            # Facebook ESM
rdkit>=2023.03.1
biopython>=1.81
wandb>=0.15.0
```

### 11.3 Training Reproducibility

```bash
# Standard training command
python src/train.py \
    --dataset data/processed/kcat_full.pt \
    --exp_name reproduce_main_result \
    --loss huber \
    --scheduler plateau \
    --pooling_type set2set \
    --use_seq_embedding \
    --seq_embedding_path data/processed/esm_embeddings.pt \
    --hidden_dim 256 \
    --num_layers 6 \
    --heads 8 \
    --dropout 0.1 \
    --weight_decay 1e-4 \
    --lr 5e-4 \
    --max_epochs 500

# Data split: 80/20 with seed=42 (fixed in code)
```

### 11.4 Expected Output

训练完成后，在 `experiments/{exp_name}/{run_id}/` 目录下会生成：

```
├── best_model.pt              # Best model checkpoint
├── training_config.txt        # Hyperparameters
├── loss_curve.png             # Training/validation loss
├── metrics_curve.png          # R², Pearson curves
├── kcat_prediction_scatter.png # True vs. Predicted
└── run_metadata.json          # Experiment metadata
```

---

## Appendix A: Mathematical Formulations

### A.1 Gaussian RBF Encoding

$$\phi_i(d) = \exp\left(-\gamma (d - c_i)^2\right), \quad i = 1, \ldots, 16$$

其中：
- $d$: 原子间距离
- $c_i \in [0, 8]$ Å: 等间距中心点
- $\gamma = 20$: 带宽参数

### A.2 Bond Angle Calculation

对于边 $(i, j)$ 和 $i$ 的另一个邻居 $k$：

$$\cos\theta_{ijk} = \frac{\vec{r}_{ij} \cdot \vec{r}_{ik}}{|\vec{r}_{ij}| |\vec{r}_{ik}|}$$

### A.3 Dihedral Angle Calculation

对于原子序列 A-B-C-D：

$$\cos\phi = \frac{\vec{n}_1 \cdot \vec{n}_2}{|\vec{n}_1| |\vec{n}_2|}$$

其中：
- $\vec{n}_1 = \vec{r}_{BA} \times \vec{r}_{BC}$
- $\vec{n}_2 = \vec{r}_{CB} \times \vec{r}_{CD}$

### A.4 Graph Attention with Edge Features

$$\mathbf{h}_i' = \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W} \mathbf{h}_j$$

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

$$e_{ij} = \text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j \| \mathbf{W}_e \mathbf{e}_{ij}]\right)$$

---

## Appendix B: Data Statistics

### B.1 Dataset Overview (IntEnzyDB-derived)

| Statistic | Value |
|-----------|-------|
| Total samples | ~10,000 |
| Unique enzymes | ~3,000 |
| EC classes covered | 6 |
| $\log_{10}(k_{cat})$ range | [-2, 6] |
| Average pocket size | ~300 atoms |
| Average edges per graph | ~2,800 |

### B.2 Feature Dimensionality Summary

| Feature | Dimension | Source |
|---------|-----------|--------|
| Node features | 52 | `graph_builder_rbf.py` |
| Edge features | 24 | `build_graph_dataset.py` |
| ESM-2 embedding | 640/1280 | `generate_esm_embeddings.py` |
| Hidden dimension | 256 | Model config |
| Final representation | 256-512 | Depends on pooling + fusion |

---

## Appendix C: Citation

If you use this work, please cite:

```bibtex
@article{li2026pocketgnn,
  title={PocketGNN: A Cross-Modal Framework Unifying Local 3D Pocket Geometry and Global Sequence Semantics for Enzyme Kinetic Prediction},
  author={Li, Zihao and Lu, Diannan},
  journal={TBD},
  year={2026}
}
```

---

## Appendix D: Paper Writing Strategy

基于对 2023-2025 SOTA 的深入分析，以下是论文写作的战略建议：

### D.1 Narrative Positioning Options

#### Option A: "Geometry-aware Representation" Focus

**Title Pattern**: *"PocketGNN: Geometry-aware Graph Neural Networks for Enzyme Kinetic Prediction"*

**Story Arc**:
1. 现有方法（DLKcat, UniKP）忽略 3D 结构
2. 全局结构方法（DEKP）引入噪声
3. PocketGNN 聚焦活性口袋 + 精细几何边特征
4. 实验证明角度/二面角特征的重要性（消融实验）

**优势**: 技术创新点清晰，易于量化贡献
**风险**: 可能被认为是"工程改进"而非"方法论创新"

---

#### Option B: "Cross-modal Fusion" Focus

**Title Pattern**: *"Bridging Local Geometry and Global Semantics: A Cross-modal Framework for Enzyme Kinetics"*

**Story Arc**:
1. 局部 3D 口袋捕获物理约束
2. 全局 ESM-2 嵌入提供进化上下文
3. Late Fusion with Projection 避免维度不平衡
4. 在 OOD 场景（40% 同源划分）展示融合的价值

**优势**: 符合 AI4Science 的"多模态融合"叙事
**风险**: 需要更强的消融实验支撑

---

#### Option C: "Practical Tool" Focus

**Title Pattern**: *"PocketGNN: A Practical Deep Learning Tool for Enzyme Engineering"*

**Story Arc**:
1. 酶工程需要可靠的 $k_{cat}$ 预测
2. 现有方法在 OOD 场景不可靠（引用 NNKcat 的批评）
3. PocketGNN 提供更稳定、可解释的预测
4. Case study 验证在实际酶上的应用

**优势**: 应用导向，易于被酶工程领域接受
**风险**: 理论深度可能不足

---

### D.2 Addressing Common Reviewer Concerns

基于 SOTA 分析，预测审稿人可能的问题及应对策略：

| Potential Concern | Evidence/Response |
|-------------------|-------------------|
| "Why not compare with CatPred?" | 添加 CatPred 作为 baseline（如数据/代码可用） |
| "What about temperature/pH?" | 明确列为 limitation，引用 MPEK 说明问题的复杂性 |
| "Is 40% homology split strict enough?" | 引用 CataPro 的 CD-HIT 策略，说明我们的设置是领域标准 |
| "Architecturally straightforward" | 强调 24-dim 边特征的物理意义，不只是 "another GNN" |
| "Where's the uncertainty?" | 列为 future work，或快速添加 MC Dropout/Quantile 实验 |

### D.3 Recommended Experiment Additions

根据 SOTA 分析，以下实验可以显著加强论文：

| Experiment | Effort | Impact | SOTA Reference |
|------------|--------|--------|----------------|
| **CD-HIT 40% split** 多次重复 | Low | High | CataPro |
| **Quantile regression / UQ** | Medium | High | CatPred |
| **vs. DLKcat/UniKP 直接对比** | Medium | High | Community standard |
| **条件变量消融** (如有数据) | Medium | Medium | MPEK |
| **EC-class 细分分析** | Low | Medium | CatPred |
| **Case study (ALDH2 已做)** | Done | High | Interpretability |

### D.4 Claim Calibration

基于 SOTA 分析，对论文中的 claim 进行校准：

| Original Claim | Calibrated Version | Reason |
|----------------|---------------------|--------|
| "State-of-the-art performance" | "Competitive performance with geometry-aware design" | CatPred/CataPro 可能有更强数据 |
| "Novel cross-modal fusion" | "Effective late fusion with projection layer" | 融合本身不新，投影设计是改进 |
| "First to use angle features" | "Explicit stereochemical encoding via angle/dihedral features" | GraphKcat 可能有类似想法 |
| "Generalizes to unseen enzymes" | "Maintains moderate correlation ($r$=0.63) under strict homology split" | $R^2$=0.379 不算很强 |

---

## Appendix E: Quick Reference for AI Context

当使用 AI 辅助论文写作时，可以提供以下摘要作为 context：

```
PROJECT: PocketGNN - Cross-modal enzyme kcat prediction

INPUTS:
- Active pocket 3D graph (52-dim nodes, 24-dim edges: RBF + angles + dihedrals)
- Optional: ESM-2 sequence embedding (640/1280-dim)

ARCHITECTURE:
- Geometry-aware GAT (edge features in first layer)
- Graph pooling (mean/attention/set2set)
- Late fusion with projection
- MLP regressor

PERFORMANCE:
- Random split: Pearson r=0.98, R²=0.918
- 40% homology split: Pearson r=0.63, R²=0.379

KEY INNOVATIONS:
1. 24-dim edge features (vs. simple distance)
2. Pocket-centric (vs. full protein)
3. DiffDock for binding pose
4. ESM-2 fusion for global context

MAIN BASELINES:
- DLKcat (2022): Seq + SMILES, no structure
- UniKP (2023): pLM-based, no 3D
- CatPred (2025): Full structure + UQ
- DEKP (2025): Structure graph + GNN

KNOWN LIMITATIONS:
- No temperature/pH modeling (cf. MPEK)
- No uncertainty quantification (cf. CatPred)
- OOD generalization needs improvement
```

---

*Document generated for academic reference and AI-assisted paper review.*
*Last updated: 2026-01-19*
*Includes SOTA analysis from sotawork.md (2023-2025 literature)*

