# Developer Guide (DEV_GUIDE.md)

这份文档是本项目的唯一真理来源（Single Source of Truth）。它定义了项目的高层架构、核心流程和开发规范。

## 📌 TL;DR (项目概要)

- **核心任务**：基于酶的口袋结构（Pocket）预测酶的催化常数 ($k_{cat}$)。目前专注于 $k_{cat}$ 单任务预测。
- **核心模型**：`PocketGNNKcatOnly` - 一个基于图注意力网络（GAT）的 GNN，使用增强的边特征（距离+角度+二面角）。
- **输入数据**：酶的 PDB 结构文件（经 Pocket 提取）+ 底物 SMILES。
- **数据格式**：PyTorch Geometric `.pt` 文件，包含节点特征（52维）和边特征（24维）。
- **训练框架**：PyTorch + PyG，使用 WandB 进行实验记录。
- **当前状态**：训练流程稳定（`train.py`），预测流程（`pred_range_fixed.py`）可能需要适配最新的单任务模型。
- **分支策略**：直接在主分支开发，实验性代码应放在 `experiments/` 下或单独分支。

---

## 🔬 科学与机器学习目标

### 1. 任务定义
输入酶的活性中心（Pocket）的 3D 结构图和底物信息，回归预测 $\log_{10}(k_{cat})$ 值。

### 2. 输入 (Inputs)
- **图结构**：基于口袋原子构建的半径图（Radius Graph）。
- **节点特征 (Node Features)**：52维，包含原子类型（One-hot）、杂化方式等化学属性。
- **边特征 (Edge Features)**：24维，这是本项目的关键创新点之一。
  - 16维：RBF (Radial Basis Function) 距离编码。
  - 4维：键角特征 (Bond Angles)。
  - 4维：二面角特征 (Dihedral Angles)。

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
    -   **流程**：加载 `.pt` -> 8/2 划分训练验证集（固定随机种子）-> Adam 优化器 -> MSE Loss -> 保存最佳模型。
    -   **命令行参数**：
      - `--dataset`: 数据集路径
      - `--save_dir`: 输出目录（可选，默认自动生成）
      - `--exp_name`: 实验名称（用于 W&B group）
      - `--batch_size`: 批次大小（默认 32）
      - `--lr`: 学习率（默认 1e-3）
      - `--max_epochs`: 最大训练轮数（默认 500）
      - `--load_checkpoint`: 加载预训练模型权重（可选）
    -   **监控**：WandB (在线，支持 group/tags/notes) + TensorBoard (`outputs/` 目录)。

3.  **预测/推理 (Inference)**
    -   **脚本**：`src/pred_range_fixed.py` (⚠️ 注意：此脚本逻辑可能滞后于训练脚本)
    -   **现状**：目前该脚本主要用于旧版双任务模型（$k_{cat}$ + $K_m$）的预测，包含了 Docking 和结构预测的全流程。若用于最新的 `PocketGNNKcatOnly` 模型，需要修改代码以适配单输出维度。

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
| `metadata_utils.py` | **Utils** | 管理实验元数据和结果记录。 | ✅ Active |

---

## 🧠 模型定义

目前代码库中存在多个模型类，请严格区分：

- **✅ ACTIVE (主模型)**:
  - **`PocketGNNKcatOnly`**: 专为 $k_{cat}$ 预测设计。不使用温度特征。输出维度为 1。支持 24维边特征。

- **⚠️ LEGACY / EXPERIMENTAL (旧版/实验性)**:
  - `PocketGNN`: 基础 GCN 模型。
  - `PocketGNNWithAttention`: 双任务（$k_{cat}, K_m$）模型，包含温度特征融合。
  - `PocketGNNWithAttentionNoTemp`: 双任务模型，无温度特征。
  - `PocketGNN_Gated`: 使用 GatedGraphConv 的变体。

**注意**：在 `src/train.py` 中，`PocketGNNKcatOnly` 是被显式调用的模型。

---

## 💾 数据集格式与假设

- **文件格式**：`torch.load` 可读取的 `.pt` 文件。
- **Data 对象属性**：
  - `x`: [Num_Atoms, 52] (Node Features)
  - `edge_index`: [2, Num_Edges] (Connectivity)
  - `edge_attr`: [Num_Edges, 24] (Edge Features: 16 RBF + 4 Angle + 4 Dihedral)
  - `y`: [1] (Log10 kcat value)
  - `batch`: (PyG 自动处理)

**假设**：
- 输入图数据必须包含完整的边特征（24维），否则模型第一层会报错维度不匹配。
- 标签 `y` 已经是 $\log_{10}$ 处理后的值。

---

## 📉 训练与评估逻辑

- **损失函数**：`nn.MSELoss()` (当前使用 MSE，可考虑改为 Huber Loss)。
- **评估指标**：
  - $R^2$ (决定系数)
  - Pearson Correlation (皮尔逊相关系数，已处理 NaN 和常数输入的情况)
  - MAE / RMSE
- **Checkpoints**：根据验证集 Loss 保存 `best_model.pt`。
- **数据集划分**：使用固定随机种子（seed=42）确保可重复性。
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
