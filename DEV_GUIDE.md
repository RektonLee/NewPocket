# Developer Guide (DEV_GUIDE.md)

这份文档是本项目的唯一真理来源（Single Source of Truth）。它定义了项目的高层架构、核心流程和开发规范。

## 📌 TL;DR (项目概要)

- **核心任务**：基于酶的口袋结构（Pocket）预测酶的催化常数 ($k_{cat}$)。目前专注于 $k_{cat}$ 单任务预测。
- **核心模型**：`PocketGNNKcatOnly` - 一个基于图注意力网络（GAT）的 GNN，使用增强的边特征（距离+角度+二面角），支持 ESM-2 序列嵌入融合（Late Fusion with Projection）。
- **输入数据**：酶的 PDB 结构文件（经 Pocket 提取）+ 底物 SMILES + 可选：ESM-2 序列嵌入。
- **数据格式**：PyTorch Geometric `.pt` 文件，包含节点特征（52维）和边特征（24维）。
- **训练框架**：PyTorch + PyG，使用 WandB 进行实验记录。
- **当前状态**：训练流程稳定（`train.py`），预测流程（`pred_range_fixed.py`）可能需要适配最新的单任务模型。
- **分支策略**：直接在主分支开发，实验性代码应放在 `experiments/` 下或单独分支。

---

## 🔬 科学与机器学习目标

### 1. 任务定义
输入酶的活性中心（Pocket）的 3D 结构图和底物信息，回归预测 $\log_{10}(k_{cat})$ 值。

### 2. 输入 (Inputs)
- **图结构**：基于口袋原子构建的半径图（Radius Graph，距离阈值 4.0 Å）。
- **节点特征 (Node Features)**：52维（详见"模型架构详解"部分）
  - 元素类型 one-hot (10维)
  - 残基类型 one-hot (21维)
  - 配体标识 (1维)
  - 最近邻距离 (1维)
  - 电子结构特征 (16维)
  - 原子属性：质量、电负性、半径 (3维)
- **边特征 (Edge Features)**：24维，这是本项目的关键创新点之一。
  - 16维：RBF (Radial Basis Function) 距离编码。
  - 4维：键角特征 (Bond Angles)。
  - 4维：二面角特征 (Dihedral Angles)。
- **序列嵌入 (可选)**：ESM-2 生成的蛋白质序列嵌入（640 或 1280 维），通过 Late Fusion 与图特征融合。

### 3. 输出 (Outputs)
- **标量**：$\log_{10}(k_{cat})$。

---

## ⚙️ 标准流水线 (Canonical Pipeline)

这是项目目前**唯一权威**的数据流转过程：

1.  **数据准备 (Data Prep)**
    -   **脚本**：`src/build_graph_dataset.py`
    -   **输入**：包含 `sample_id`, `substrate_smiles`, `kcat_value` 的 CSV 文件。
    -   **逻辑**：
        1.  读取 CSV。
        2.  根据 SMILES 和 ID 寻找/生成 Pocket PDB 文件。
        3.  解析 PDB，调用 `enhanced_build_graph` 构建图数据。
        4.  计算 24维边特征（RBF + Angle + Dihedral）。
    -   **输出**：`.pt` 文件（List of Data objects）。

2.  **模型训练 (Training)**
    -   **脚本**：`src/train.py`
    -   **模型**：`GNN_model.PocketGNNKcatOnly` (硬编码指定)。
    -   **流程**：加载 `.pt` -> 加载序列嵌入（可选）-> 8/2 划分训练验证集（固定随机种子）-> Adam 优化器 -> Loss (MSE/Huber) -> 保存最佳模型。
    -   **命令行参数**：
      - **必需参数**:
        - `--dataset`: 数据集路径（`.pt` 文件）
      - **输出参数**:
        - `--save_dir`: 输出目录（可选，默认自动生成）
      - **训练参数**:
        - `--batch_size`: 批次大小（默认 32）
        - `--lr`: 学习率（默认 1e-3）
        - `--epochs`: 最大训练轮数（默认 500）
        - `--weight_decay`: L2 正则化系数（默认 1e-4）
        - `--dropout`: Dropout 率（默认 0.1）
      - **损失与调度器**:
        - `--loss`: 损失函数类型，`mse` 或 `huber`（默认 `mse`）
        - `--scheduler`: 学习率调度器，`none`、`plateau` 或 `cosine`（默认 `none`）
      - **模型架构参数**:
        - `--pooling_type`: 池化类型，`mean`、`global_attention` 或 `set2set`（默认 `mean`）
      - **序列嵌入参数**:
        - `--use_seq_embedding`: 启用 ESM-2 序列嵌入融合（flag，默认 False）
        - `--seq_embedding_path`: 序列嵌入文件路径（`.pt` 字典，格式: `{sample_id: tensor}`）
      - **其他**:
        - `--load_checkpoint`: 加载预训练模型权重（可选）
    -   **监控**：WandB (在线，支持 group/tags/notes) + TensorBoard (`outputs/` 目录)。
    -   **示例命令**:
      ```bash
      python src/train.py \
        --dataset data/processed/kcat_full_1213.pt \
        --save_dir outputs/kcat_enhanced_run \
        --weight_decay 1e-4 \
        --dropout 0.3 \
        --loss huber \
        --scheduler plateau \
        --pooling_type set2set \
        --use_seq_embedding \
        --seq_embedding_path data/processed/esm_embeddings.pt
      ```

3.  **预测/推理 (Inference)**
    -   **脚本**：`src/pred_range_fixed.py` (⚠️ 注意：此脚本逻辑可能滞后于训练脚本)
    -   **现状**：目前该脚本主要用于旧版双任务模型（$k_{cat}$ + $K_m$）的预测，包含了 Docking 和结构预测的全流程。若用于最新的 `PocketGNNKcatOnly` 模型，需要修改代码以适配单输出维度。

4.  **数据诊断 (Data Diagnosis)** (可选，但强烈推荐在训练前执行)
    -   **脚本**：`src/analyze_feature_label_relation.py`
    -   **目的**：在训练 GNN 之前，先回答"数据本身是否包含足够信息"这一核心科学问题
    -   **用法**：`python src/analyze_feature_label_relation.py --dataset <path> --save_dir <output_dir>`
    -   **输出**：特征相关性分析、基准模型性能报告（详见"诊断测试功能"部分）

---

## 📂 核心代码库 (Active Files)

仅列出当前活跃且核心的文件，忽略废弃或实验性代码。

| 文件路径 | 类型 | 描述 | 状态 |
| :--- | :--- | :--- | :--- |
| `src/train.py` | **Core** | 权威训练入口。定义了 Loss、Metrics 和 WandB 记录。 | ✅ Active |
| `src/GNN_model.py` | **Model** | 包含多个模型类。**`PocketGNNKcatOnly` 是当前主模型**。 | ✅ Active |
| `src/build_graph_dataset.py` | **Data** | 构建训练用的 `.pt` 数据集。实现了增强图构建逻辑。 | ✅ Active |
| `src/graph_builder_rbf.py` | **Utils** | 基础图构建工具，定义了 RBF 参数和基础构图逻辑。 | ✅ Active |
| `src/pred_range_fixed.py` | **Script** | 批量预测脚本。包含从序列到结构再到预测的全流程。 | ⚠️ Needs Update |
| `src/docking.py` | **Utils** | 处理分子对接（Docking）和 Pocket 提取的逻辑。 | ✅ Active |
| `src/analyze_feature_label_relation.py` | **Diagnostic** | 数据层诊断脚本。分析图统计特征与 kcat 的相关性，不依赖 GNN 模型。 | ✅ Active |
| `metadata_utils.py` | **Utils** | 管理实验元数据和结果记录。 | ✅ Active |

---

## 🧠 模型定义

目前代码库中存在多个模型类，请严格区分：

- **✅ ACTIVE (主模型)**:
  - **`PocketGNNKcatOnly`**: 专为 $k_{cat}$ 预测设计。不使用温度特征。输出维度为 1。支持 24维边特征、ESM-2 序列嵌入融合（Late Fusion with Projection）。

- **⚠️ LEGACY / EXPERIMENTAL (旧版/实验性)**:
  - `PocketGNN`: 基础 GCN 模型。
  - `PocketGNNWithAttention`: 双任务（$k_{cat}, K_m$）模型，包含温度特征融合。
  - `PocketGNNWithAttentionNoTemp`: 双任务模型，无温度特征。
  - `PocketGNN_Gated`: 使用 GatedGraphConv 的变体。

**注意**：在 `src/train.py` 中，`PocketGNNKcatOnly` 是被显式调用的模型。

---

## 🏗️ 模型架构详解 (`PocketGNNKcatOnly`)

### 1. 输入数据格式

#### 1.1 图结构输入
- **节点特征 (Node Features)**: `x` - Shape: `[N, 52]`
  - **52维节点特征的构成**（来自 `graph_builder_rbf.py`）:
    - **元素类型 (Element)**: 10维 one-hot 编码
      - 元素列表: `['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H']`
    - **残基类型 (Residue)**: 21维 one-hot 编码
      - 残基列表: `['ALA', 'ARG', 'ASN', ..., 'VAL', 'LIG']` (20种标准氨基酸 + 配体)
    - **配体标识 (Is Ligand)**: 1维 (0或1)
    - **最近邻距离 (Min Distance)**: 1维 (到最近原子的距离)
    - **电子结构特征 (Electronic)**: 16维 (基于原子序数的电子排布特征)
    - **原子属性 (Properties)**: 3维
      - 质量 (Mass)
      - 电负性 (Electronegativity)
      - 原子半径 (Radius)
    - **总计**: 10 + 21 + 1 + 1 + 16 + 3 = **52维**

- **边特征 (Edge Features)**: `edge_attr` - Shape: `[E, 24]`
  - **24维边特征的构成**（来自 `build_graph_dataset.py` 的 `enhanced_build_graph`）:
    - **RBF 距离编码 (RBF Distance)**: 16维
      - 使用 Gaussian RBF 将原子间距离（0-8Å）编码为 16 维特征
      - 公式: $\exp(-\gamma (d - c_i)^2)$，其中 $c_i$ 是 16 个等间距中心点
    - **键角特征 (Bond Angles)**: 4维
      - 计算边两端原子与共同邻居形成的键角
    - **二面角特征 (Dihedral Angles)**: 4维
      - 计算边相关的二面角（扭转角）
    - **总计**: 16 + 4 + 4 = **24维**

- **边连接 (Edge Index)**: `edge_index` - Shape: `[2, E]`
  - 使用半径图（Radius Graph）构建，距离阈值: 4.0 Å
  - 无向图（双向边）

- **坐标 (Positions)**: `pos` - Shape: `[N, 3]`
  - 原子的 3D 坐标（用于计算角度和二面角）

#### 1.2 序列嵌入输入（可选）
- **序列嵌入 (Sequence Embedding)**: `seq_embedding` - Shape: `[batch_size, seq_embedding_dim]`
  - 来源: ESM-2 模型生成的蛋白质序列嵌入
  - 维度: 通常为 640 或 1280 维（取决于使用的 ESM-2 模型）
  - 生成方式: 使用 `generate_esm_embeddings.py` 从 CSV 中的蛋白质序列生成
  - 匹配方式: 通过 `sample_id` 与图数据对齐

### 2. 模型架构与数据流

#### 2.1 节点编码层 (Node Encoder)
```
输入: x [N, 52]
  ↓
Linear(52, hidden_dim)  # hidden_dim 默认 128
  ↓
ReLU
  ↓
输出: x [N, 128]
```

#### 2.2 图注意力层 (GAT Layers)
- **层数**: 默认 3 层（可配置 `num_layers`）
- **注意力头数**: 默认 4 个头（可配置 `heads`）
- **第一层**: 使用边特征 `edge_attr` (24维)
  ```
  输入: x [N, 128], edge_index [2, E], edge_attr [E, 24]
    ↓
  GATConv(128, 128//4, heads=4, edge_dim=24, concat=True)
    ↓
  输出: x [N, 128]  # 4个头拼接后为 128 维
  ```
- **后续层**: 不使用边特征
  ```
  输入: x [N, 128], edge_index [2, E]
    ↓
  GATConv(128, 128, heads=4, concat=False)
    ↓
  输出: x [N, 128]
  ```
- **激活函数**: ELU (Exponential Linear Unit)

#### 2.3 图池化层 (Graph Pooling)
支持三种池化策略（通过 `--pooling_type` 参数选择）：

- **Mean Pooling** (默认):
  ```
  输入: x [N, 128], batch [N]
    ↓
  global_mean_pool
    ↓
  输出: graph_x [batch_size, 128]
  ```

- **Global Attention Pooling**:
  ```
  输入: x [N, 128], batch [N]
    ↓
  GlobalAttention(gate_nn)
    ↓
  输出: graph_x [batch_size, 128]
  ```
  - `gate_nn`: 两层 MLP，计算每个节点的注意力权重

- **Set2Set Pooling**:
  ```
  输入: x [N, 128], batch [N]
    ↓
  Set2Set(128, processing_steps=3)
    ↓
  输出: graph_x [batch_size, 256]  # Set2Set 输出维度是输入的 2 倍
  ```

#### 2.4 序列嵌入融合 (Late Fusion with Projection)

**关键优化**: 使用投影层避免特征维度不平衡

如果启用 `--use_seq_embedding`:

```
图特征: graph_x [batch_size, readout_dim]
  ↓
序列嵌入: seq_embedding [batch_size, seq_embedding_dim]  # 例如 [32, 640]
  ↓
投影层: Linear(seq_embedding_dim, hidden_dim) + ReLU + Dropout
  ↓
投影后: seq_x [batch_size, hidden_dim]  # 例如 [32, 128]
  ↓
拼接: torch.cat([graph_x, seq_x], dim=1)
  ↓
输出: combined_x [batch_size, readout_dim + hidden_dim]
```

**维度说明**:
- `readout_dim` 取决于池化类型:
  - Mean/GlobalAttention: `readout_dim = hidden_dim = 128`
  - Set2Set: `readout_dim = hidden_dim * 2 = 256`
- `hidden_dim` 默认 128
- **最终拼接维度**:
  - Mean/GlobalAttention + 序列: `128 + 128 = 256`
  - Set2Set + 序列: `256 + 128 = 384`

#### 2.5 回归 MLP
```
输入: combined_x [batch_size, mlp_input_dim]
  ↓
Linear(mlp_input_dim, hidden_dim)  # 例如 256 -> 128
  ↓
ReLU + Dropout
  ↓
Linear(hidden_dim, hidden_dim // 2)  # 128 -> 64
  ↓
ReLU + Dropout
  ↓
Linear(hidden_dim // 2, 1)  # 64 -> 1
  ↓
输出: log10(kcat) [batch_size, 1]
```

### 3. 完整数据流示例

假设 batch_size=32，使用 Set2Set 池化 + 序列嵌入（640维）:

```
1. 输入
   - x: [N_total, 52]  # 所有图的节点拼接
   - edge_index: [2, E_total]
   - edge_attr: [E_total, 24]
   - batch: [N_total]  # 标识每个节点属于哪个图
   - seq_embedding: [32, 640]

2. 节点编码
   x: [N_total, 52] → [N_total, 128]

3. GAT 层（3层）
   x: [N_total, 128] → [N_total, 128] → [N_total, 128] → [N_total, 128]

4. Set2Set 池化
   graph_x: [N_total, 128] → [32, 256]

5. 序列嵌入投影
   seq_embedding: [32, 640] → seq_x: [32, 128]

6. 特征融合
   combined_x: [32, 256] + [32, 128] → [32, 384]

7. MLP 回归
   [32, 384] → [32, 128] → [32, 64] → [32, 1]

8. 输出
   log10(kcat): [32, 1]
```

### 4. 模型超参数（默认值）

- `hidden_dim`: 128
- `num_layers`: 3
- `heads`: 4
- `dropout`: 0.1 (可配置 `--dropout`)
- `pooling_type`: 'mean' (可配置 `--pooling_type`)
- `use_seq_embedding`: False (需显式开启 `--use_seq_embedding`)
- `seq_embedding_dim`: 640 或 1280 (自动检测)

---

## 💾 数据集格式与假设

- **文件格式**：`torch.load` 可读取的 `.pt` 文件（List of Data objects）。

- **Data 对象属性**：
  - `x`: `[Num_Atoms, 52]` - 节点特征（详见"模型架构详解"部分）
  - `edge_index`: `[2, Num_Edges]` - 边连接索引（无向图，双向边）
  - `edge_attr`: `[Num_Edges, 24]` - 边特征（16维RBF + 4维角度 + 4维二面角）
  - `pos`: `[Num_Atoms, 3]` - 原子 3D 坐标
  - `y`: `[1]` - 标签值（$\log_{10}(k_{cat})$）
  - `seq_embedding`: `[1, seq_embedding_dim]` - 序列嵌入（可选，仅在启用 `--use_seq_embedding` 时存在）
  - `batch`: PyG 自动处理（在 DataLoader 中自动添加）

- **元数据属性**（训练前会被删除，避免 collate 错误）:
  - `sample_id`: 样本标识符（用于匹配序列嵌入）
  - `pdb_id`: PDB 文件标识
  - `ec`: EC 编号
  - `uniprot_id`: UniProt 标识符

**假设**：
- 输入图数据必须包含完整的边特征（24维），否则模型第一层会报错维度不匹配。
- 标签 `y` 已经是 $\log_{10}$ 处理后的值。
- 如果启用序列嵌入，`seq_embedding` 必须通过 `sample_id` 正确匹配。

---

## 📉 训练与评估逻辑

- **损失函数**：
  - 默认: `nn.MSELoss()`
  - 可选: `nn.HuberLoss()` (通过 `--loss huber` 启用，更鲁棒，适合噪声数据)
- **优化器**：
  - `optim.Adam` with weight decay (L2 正则化)
  - 默认学习率: `1e-3` (可配置 `--lr`)
  - 默认 weight decay: `1e-4` (可配置 `--weight_decay`)
- **学习率调度器** (可选):
  - `ReduceLROnPlateau`: 验证损失不下降时降低学习率 (`--scheduler plateau`)
  - `CosineAnnealingWarmRestarts`: 余弦退火 (`--scheduler cosine`)
- **评估指标**：
  - $R^2$ (决定系数)
  - Pearson Correlation (皮尔逊相关系数，已处理 NaN 和常数输入的情况)
  - MAE / RMSE
- **Checkpoints**：根据验证集 Loss 保存 `best_model.pt`。
- **数据集划分**：8:2 训练/验证集划分（固定随机种子，确保可重复性）。
- **输出产物**：
  - `loss_curve.png`, `metrics_curve.png` (训练曲线)
  - `kcat_prediction_scatter.png` (真实值 vs 预测值散点图)
  - `kcat_density.png` (密度图)

---

## 🌿 分支与实验策略

- **Master Branch**: 保持稳定，对应 `src/train.py` 当前的配置。
- **Experiments**:
  - 所有的实验配置和结果应保存在 `outputs/` 或 `experiments/` 目录下。
  - 使用 `exp_name` 参数区分不同实验（如 `kcat_attn_v1`）。
  - **禁止**直接修改 `src/` 下的核心代码来跑一次性实验，应通过参数控制或新建实验脚本。

---

## ⚠️ 遗留代码声明 (Legacy Code Disclaimer)

以下文件或代码块被识别为遗留或非核心路径，开发者在使用时需谨慎：
- `src/GNN_model.py` 中除 `PocketGNNKcatOnly` 以外的类（除非明确要进行对比实验）。
- `src/pred_range_fixed.py` 目前主要适配双任务模型，若用于单任务模型需自行修改输出解析逻辑。

---

## 🤖 AI 助手行为准则

未来的 AI 助手在维护此仓库时应遵循：

1.  **单一真理**：遇到架构疑问时，以本 `DEV_GUIDE.md` 为准，而不是随机查看代码注释。
2.  **不破坏现有格式**：生成的 Dataset 必须保持 24维边特征，否则会破坏现有模型兼容性。
3.  **中文优先**：代码注释、文档说明默认使用中文。
4.  **文件管理**：不要随意创建顶层文件，新脚本放入 `scripts/`，新逻辑放入 `src/`。
5.  **检查模型匹配**：在编写预测脚本时，务必检查模型输出维度（1 vs 2）与加载的模型类是否匹配。

---

## 🔬 诊断测试功能

`src/train.py` 提供了两个诊断测试功能，用于科学地评估模型和数据质量：

### Test 1: Label Permutation Test
- **目的**：验证模型是否真正使用图表示，而非 pipeline bug
- **用法**：`--label_permutation`
- **原理**：随机置换标签，如果模型真正依赖图表示，性能应降至接近 0
- **预期结果**：Pearson ≈ 0, R² ≈ 0

### Test 2: Frozen Encoder Test
- **目的**：诊断 encoder 表示是否已饱和（linearly-usable）
- **用法**：`--frozen_encoder --load_checkpoint <path>`
- **原理**：冻结 encoder，重置 MLP head，只训练 head。如果性能接近 baseline，说明 encoder 已饱和
- **预期结果**：
  - 性能 ≈ baseline → encoder 已饱和
  - 性能明显下降 → encoder 仍需端到端优化

### Test 3: Feature-Label Correlation Analysis (数据层诊断)
- **脚本**：`src/analyze_feature_label_relation.py`
- **目的**：回答"数据本身是否包含足够信息"这一核心科学问题（不依赖 GNN 模型）
- **用法**：`python src/analyze_feature_label_relation.py --dataset <path> --save_dir <output_dir>`
- **原理**：
  1. 将图结构压缩为统计特征（节点/边特征的均值/方差、几何半径等）
  2. 计算每个特征与 $k_{cat}$ 的 Pearson/Spearman 相关系数
  3. 使用简单模型（Linear/Ridge、Random Forest）测试数据本身的"信息上限"
- **输出**：
  - `feature_correlations.csv`: 所有特征的相关性排序
  - `model_baselines.csv`: 基准模型性能（Linear 和 RF）
  - 可视化图表（相关性散点图、预测散点图）
- **结果解读**：
  - 如果最大 |Pearson| < 0.1 且 RF Test Pearson < 0.3 → **数据表示本身信息不足**
  - 如果最大 |Pearson| >= 0.3 且 RF Test Pearson >= 0.5 → **数据包含足够信号，问题可能在模型**
- **重要性**：这是**数据层**的诊断，比模型层诊断更底层、更硬核

详细说明请参考 `Todiagnose.md` 和 `DIAGNOSTIC_RESULTS.md`。

---

## 🚧 已知瓶颈与开放问题

1.  **预测脚本不兼容**：`src/pred_range_fixed.py` 尚未完全适配 `PocketGNNKcatOnly`，直接运行可能会因为输出维度期望不一致而报错。需要重构以支持单任务/双任务模型的自动切换。
2.  **硬编码参数**：`src/train.py` 中部分模型超参数（如 hidden_dim=128）是硬编码的，建议改为命令行参数。
3.  **温度特征**：目前的 Canonical Pipeline (`PocketGNNKcatOnly`) 显式移除了温度特征的输入，这是一个明确的设计选择，但在未来可能需要重新评估。
4.  ~~**Pearson 相关系数 NaN 处理**~~：✅ 已修复，现在会正确处理常数输入的情况。
5.  ~~**数据集划分不一致**~~：✅ 已修复，使用固定随机种子（seed=42）确保可重复性。

##特征
Data(x=[321, 52], edge_index=[2, 2814], edge_attr=[2814, 24], pos=[321, 3], temperature=[1], y=[2], pdb_id='kcat_000002_61151_10A.pdb', sample_id='kcat_000002', ec='1.1.1.1')


##正在做的事！！！（是对话形式）
有个很科学很严谨的问题是，我的构建好的图(.pt)，有没有可能他的图表示本身就跟label没有强相关性（也就是我们模型无论怎么调，都由于数据本身、数据特征选取本身而导致无法改进）
科学问题：有没有可能「图本身就不包含 label 信息」？



##推荐训练配置
使用 Huber Loss + Early Stopping
python src/train.py --dataset data/processed/kcat_full_1213.pt \
    --loss_type huber \
    --patience 30 \
    --max_epochs 500

##测试配置
python src/test.py \
    --test_dataset data/processed/kcat_test_fixed.pt \
    --model outputs/kcat_20251213_151558/best_model.pt