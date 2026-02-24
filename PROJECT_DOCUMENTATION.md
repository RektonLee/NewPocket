# PocketGNN 项目文档

> **最后更新**: 2026-02-24
> **项目状态**: 实验进行中 - DiffDock批量对接 + 模型训练
> **负责人**: 李子豪 (清华大学化工系)

---

## 1. 项目概述

### 1.1 研究目标

预测酶的催化周转数 $k_{cat}$（turnover number），这是衡量酶催化效率的关键动力学参数。

**核心任务**: 从酶活性位点（pocket）的3D结构 + 底物SMILES → 预测 $\log_{10}(k_{cat})$

### 1.2 技术路线

```
序列 (UniProt)
    ↓ ESMFold
蛋白质3D结构 (PDB)
    ↓ DiffDock (blind docking)
蛋白质-配体复合物
    ↓ 5Å cutoff
活性口袋图 (PyG Data)
    ↓ PocketGNN
k_cat 预测值
```

### 1.3 创新点

1. **Cross-Modal Fusion**: 融合局部3D几何（口袋图）与全局序列语义（ESM-2）
2. **24维几何边特征**: RBF距离(16) + 键角(4) + 二面角(4)
3. **DiffDock自动化**: 无需手动指定binding site，适合大规模预测结构处理

---

## 2. 文献综述

### 2.1 酶动力学预测方法演进

| 方法 | 年份 | 输入 | 核心技术 | 局限性 |
|------|------|------|----------|--------|
| DLKcat | 2022 | 序列+SMILES | CNN+Morgan FP | 无结构信息 |
| UniKP | 2023 | 序列+SMILES | ESM-1b embeddings | 无3D几何 |
| CatPred | 2025 | 序列+SMILES | ESM-2 + 不确定性 | 无口袋结构 |
| CataPro | 2025 | 序列+SMILES | PLM fine-tuning | 无3D几何 |
| **PocketGNN (Ours)** | 2025 | 口袋3D+序列 | GAT + ESM-2 fusion | - |

### 2.2 分子对接方法

| 方法 | 类型 | Binding Site | 优势 | 劣势 |
|------|------|--------------|------|------|
| AutoDock Vina | 物理打分 | **必须指定** | 快速、准确 | 需要先验知识 |
| GNINA | CNN打分 | 必须指定 | 学习打分函数 | 同上 |
| **DiffDock** | 扩散模型 | **不需要** | Blind docking | 略慢 |
| Chai-1 | 端到端 | 不需要 | 多模态 | 资源消耗大 |

### 2.3 关键参考文献

1. **DiffDock** (Corso et al., ICLR 2023): 扩散模型用于分子对接，SE(3)-invariant
2. **ESM-2** (Lin et al., Science 2023): 蛋白质语言模型，进化尺度预训练
3. **IntEnzyDB** (Yan et al., JCIM 2022): 酶动力学数据库，我们的数据来源
4. **GAT** (Veličković et al., ICLR 2018): 图注意力网络

---

## 3. 方法详解

### 3.1 数据处理流程

```
IntEnzyDB (CSV)
    ↓ 过滤: k_cat ∈ [10^-2, 10^6] s^-1
~9000 酶-底物对
    ↓ ESMFold/PDB获取结构
    ↓ DiffDock对接
    ↓ 5Å pocket提取
    ↓ build_graph_dataset.py
PyG Dataset (.pt文件)
```

### 3.2 图表示

**节点特征 (52维)**:
- 元素类型 (10维 one-hot): C, N, O, S, P, F, Cl, Br, I, H
- 残基类型 (21维 one-hot): 20种氨基酸 + LIG
- 配体标志 (1维): 0或1
- 最近邻距离 (1维)
- 电子特征 (16维): 基于原子序数
- 原子属性 (3维): 质量、电负性、半径

**边特征 (24维)**:
- RBF距离编码 (16维): 高斯基函数，0-8Å
- 键角特征 (4维): 邻居对形成的角度
- 二面角特征 (4维): 扭转角，捕捉手性

### 3.3 模型架构

```
PocketGNNKcatOnly:
    NodeEncoder: Linear(52 → 128)
    GAT Layers: 3层, 4头注意力
    Pooling: mean / global_attention / set2set
    [Optional] ESM-2 Fusion: Linear(640 → 128) + Concat
    MLP Head: 128 → 64 → 32 → 1
    Output: log10(k_cat)
```

### 3.4 为什么选择DiffDock

| 场景 | Vina | DiffDock |
|------|------|----------|
| 晶体结构（已知配体位置） | ✅ 更准确 | ✅ 可用 |
| 预测结构（未知配体位置） | ❌ 无法使用 | ✅ 自动搜索 |
| 大规模自动化处理 | ❌ 需手动设置 | ✅ 完全自动 |
| 对蛋白方向的鲁棒性 | ❌ 敏感 | ✅ SE(3)-invariant |

**我们的场景**: 序列→ESMFold→对接，**没有参考配体位置**，必须用DiffDock。

---

## 4. 实验结果

### 4.1 主要性能指标

| 数据划分 | Pearson r | R² | MAE | 样本数 |
|----------|-----------|-----|-----|--------|
| Random Split (IID) | 0.98 | 0.918 | 0.35 | ~1800 |
| 40% Homology Split (OOD) | **0.716** | 0.387 | 0.90 | 898 |

### 4.2 与SOTA方法对比 (40% Homology OOD)

| 方法 | Pearson r | 相对提升 |
|------|-----------|----------|
| CatPred (2025) | 0.52 | baseline |
| CataPro (2025) | 0.497 | -4.4% |
| **PocketGNN (Ours)** | **0.716** | **+37.7%** |

### 4.3 DiffDock vs Vina 验证实验

在5个Astex Diverse Set样本上的re-docking测试：

| 方法 | 成功率 | 平均RMSD | 最佳RMSD |
|------|--------|----------|----------|
| Vina (已知binding site) | 100% (5/5) | 1.31±0.61Å | 0.30Å |
| DiffDock (blind) | 80% (4/5) | 1.69±1.19Å | 0.38Å |

**注意**: 这个对比对Vina有利，因为Vina使用了参考配体位置设置search box。在真实的blind docking场景下，DiffDock是唯一可行的选择。

### 4.4 消融实验

| 配置 | Pearson r | 说明 |
|------|-----------|------|
| 完整模型 | 0.716 | GAT + 24D边特征 + ESM-2 |
| 去除角度/二面角特征 | 下降 | 简单距离不足以捕捉催化几何 |
| 去除ESM-2分支 | 下降 | 全局进化上下文重要 |
| Label Permutation Test | ~0 | 模型确实学习了图结构 |

---

## 5. 代码结构

```
PGNN_clean/
├── src/                          # 核心代码
│   ├── train.py                  # 训练脚本
│   ├── test.py                   # 测试脚本
│   ├── GNN_model.py              # 模型定义 (PocketGNNKcatOnly)
│   ├── build_graph_dataset.py    # 数据集构建
│   ├── graph_builder_rbf.py      # 图构建 (24D边特征)
│   ├── docking.py                # DiffDock封装
│   └── generate_esm_embeddings.py # ESM-2特征提取
├── scripts/                      # 实验脚本
│   ├── quick_vina_test.py        # Vina RMSD测试
│   ├── quick_diffdock_test.py    # DiffDock RMSD测试
│   └── compare_representations.py # 表示学习对比
├── paperwriting/                 # 论文
│   ├── gemini.tex                # 主论文
│   └── gemini.pdf                # 编译后PDF
├── data/                         # 数据 (gitignored)
│   ├── raw/                      # 原始CSV
│   └── processed/                # .pt数据集
├── outputs/                      # 训练输出 (gitignored)
├── results/                      # 实验结果
├── DiffDock/                     # DiffDock子模块
├── CLAUDE.md                     # AI助手指南
└── PROJECT_DOCUMENTATION.md      # 本文档
```

---

## 6. 待办事项与进展

### 6.1 已完成

- [x] 数据收集与预处理 (IntEnzyDB → ~9000样本)
- [x] DiffDock对接流程
- [x] PocketGNN模型实现与训练
- [x] Random/Homology split评估
- [x] SOTA方法对比 (CatPred, CataPro)
- [x] DiffDock vs Vina RMSD验证实验
- [x] 论文初稿完成

### 6.2 进行中 (2026-02-24)

- [x] **DiffDock批量对接训练集** - 4 GPU并行处理中
  - 总样本: 9061 (kcat_full_1213.csv)
  - 有PDB结构: 4072 (45%)
  - 无PDB结构: 4989 (55%) - 需要ESMFold预测
  - 预计完成: 2026-02-25 中午
  - 进度监控: `python scripts/check_diffdock_progress.py`

- [x] **自动化训练pipeline** - 运行中
  - 等待DiffDock完成 → 自动构建数据集 → 自动训练模型 → 自动评估
  - 脚本: `scripts/auto_pipeline.py`
  - 日志: `logs/auto_pipeline.log`

- [ ] 论文修订（根据实验结果调整claims）
- [ ] 补充更多ablation实验

### 6.3 未来工作

- [ ] 多底物反应支持
- [ ] 不确定性量化（Quantile Regression）
- [ ] 实验条件建模（温度、pH）
- [ ] 更大规模DiffDock vs Vina对比

---

## 7. 常见问题

### Q1: 为什么用log10(k_cat)而不是原始k_cat?

$k_{cat}$值跨越多个数量级（$10^{-2}$ 到 $10^{6}$ s⁻¹），对数变换后更接近正态分布，有利于回归模型训练。

### Q2: 40% homology split是什么意思?

使用MMseqs2按40%序列相似度聚类，确保测试集中的酶与训练集中任何酶的序列相似度都<40%。这模拟了预测全新酶家族的场景。

### Q3: 为什么DiffDock的RMSD比Vina高，还说DiffDock更好?

我们的测试是**re-docking**（已知配体位置），Vina利用了这个先验信息设置search box。在真实的**blind docking**场景（预测结构，未知配体位置），Vina无法使用，DiffDock是唯一选择。

### Q4: ESM-2 embedding怎么获取?

```bash
python src/generate_esm_embeddings.py \
  --fasta data/sequences.fasta \
  --output data/processed/esm_embeddings.pt
```

---

## 8. 联系方式

- **学生**: 李子豪 (l-zh21@mails.tsinghua.edu.cn)
- **导师**: 卢滇楠 教授 (ludiannan@tsinghua.edu.cn)
- **单位**: 清华大学化学工程系

---

## 附录

### A. 关键命令速查

```bash
# 训练
python src/train.py --dataset data/processed/kcat_full_1213.pt --save_dir outputs/exp001

# 测试
python src/test.py --test_dataset data/processed/kcat_test.pt --model outputs/exp001/best_model.pt

# 数据诊断
python src/analyze_feature_label_relation.py --dataset data/processed/kcat_train.pt

# DiffDock对接
python DiffDock/inference.py --protein_path protein.pdb --ligand "CCO" --out_dir output/
```

### B. 更新日志

| 日期 | 更新内容 |
|------|----------|
| 2024-02-24 | 创建文档；添加DiffDock vs Vina实验结果；更新论文 |
| 2024-02-24 | 完成blind docking vs re-docking分析 |
