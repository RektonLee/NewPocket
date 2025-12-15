# PocketGNN 论文提升计划与待补实验清单

这份文档记录了为了达到 JCIM 投稿标准，我们还需要补充的实验和分析工作。

## 1. 待补充实验 (Supplementary Experiments)

### 1.1 泛化能力验证 (Generalization)
- [ ] **同源划分测试 (Homology Split)**: 
    - 运行 `scripts/merge_and_split_by_homology.py` 按照 40% sequence identity 划分数据集。
    - 重新训练 PocketGNN 并记录性能。
    - 目标：即使性能下降，也要诚实报告，并在文中讨论这代表了 Zero-shot 能力的真实水平。
- [ ] **EC 类别细分分析 (EC-wise Performance)**:
    - 分析模型在不同 EC 大类 (1-6) 上的 MSE/R2。
    - 检查模型是否在某些特定酶类（如水解酶 EC 3）上表现更好，而在其他类上较差。

### 1.2 基准对比 (Benchmarks)
- [ ] **重现 SOTA 结果 (如果可能)**:
    - 尝试在同一数据集上运行 DLKcat 或 TurNuP 的开源代码（如果可用）。
    - 替代方案：详细列出我们数据集与他们数据集的异同，进行定性对比。

### 1.3 超参数敏感性 (Hyperparameter Sensitivity)
- [ ] **Grid Search / Sensitivity Analysis**:
    - 测试 GAT Layers 数量 (3, 4, 5, 6) 对性能的影响。
    - 测试 Cutoff Distance (4A, 5A, 6A) 的影响。
    - 将结果绘制成图表放入 SI。

## 2. 论文写作完善 (Writing Polish)

### 2.1 Supporting Information (SI) 撰写
- [ ] **S1: 数据集统计**: 酶类别分布图，底物分子量分布图。
- [ ] **S2: 详细模型架构**: 每一层的输入输出维度表。
- [ ] **S3: 详细特征列表**: 52维节点特征和 24维边特征的完整列表和物理意义解释。
- [ ] **S4: 额外消融实验**: 关于超参数的敏感性分析结果。

### 2.2 引用修正
- [ ] 确认 "Eitlem et al." 的准确引用（推测是 Kroll et al. 2021 或其他，需核实）。
- [ ] 确保所有参考文献格式符合 ACS (JCIM) 标准。

## 3. 下一步行动
建议按照 `1.1` -> `2.1` -> `1.2` 的顺序执行。优先完成同源划分实验，这是审稿人最可能挑战的点。

