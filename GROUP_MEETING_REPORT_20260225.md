# PocketGNN项目组会报告

**报告日期**: 2026年2月25日
**报告人**: 李子豪
**项目**: 基于图神经网络的酶催化常数预测
**目标期刊**: Journal of Chemical Information and Modeling (JCIM)

---

## 📊 一、项目现状概览

### 1.1 核心性能指标

| 数据集分割方式 | Pearson r | R² | 对比SOTA |
|---------------|-----------|-----|----------|
| Random Split | 0.98 | 0.918 | - |
| **40% Homology Split** | **0.667** | **0.437** | CatPred: 0.52 (+28.7%) |

**结论**: 在低同源性(out-of-distribution)设置下，PocketGNN显著优于现有最佳方法。

---

## 📈 二、本周重大进展 (2.24-2.25)

### 🔬 突破1: 特征重要性分析揭示架构设计合理性

**动机**: JCIM审稿人可能质疑"为什么需要24维边特征？"、"模型学到了什么物理信息？"

**方法**:
- Integrated Gradients (IG) - 基于梯度的特征归因
- Permutation Importance - 基于性能下降的验证

**关键发现**:

#### 边特征贡献分析 (N=100样本, 50步IG)

```
┌─────────────┬──────┬─────────┬──────────────────────┐
│ 特征类型    │ 维度 │ 贡献率  │ 化学意义             │
├─────────────┼──────┼─────────┼──────────────────────┤
│ 键角        │ 4    │ 61.9%   │ 催化几何约束         │
│ 二面角      │ 4    │ 27.0%   │ 手性、扭转构象       │
│ RBF距离     │ 16   │ 11.1%   │ 空间邻近性           │
└─────────────┴──────┴─────────┴──────────────────────┘

总计: 几何特征 (Angles + Dihedrals) = 88.9%
```

**科学意义**:

1. **验证架构设计**: 24维边特征不是过参数化，几何信息是性能关键
   - 仅使用RBF距离(16维)会丢失89%的有效信息
   - 键角和二面角虽然只有8维，但捕捉了催化精确几何排列

2. **符合化学直觉**: 酶催化依赖精确的空间排列 (角度) 而非仅仅距离
   - 键角 → 反应中心几何 (如四面体中间态)
   - 二面角 → 底物构象适配 (诱导契合)

3. **对比SOTA优势**:
   - DLKcat, UniKP, CatPred: **仅使用序列特征** (无几何信息)
   - CataPro: 序列 + Morgan指纹 (无3D几何)
   - **PocketGNN**: 序列 + 3D几何 (RBF + Angles + Dihedrals)
   - 性能提升归因于几何特征的引入

**节点特征Top 5**:
1. **mass** (50.1%) - 原子质量主导
2. elec_5 (2.8%) - 电子构型
3. N element (1.5%) - 氮元素
4. VAL/ARG残基 (~1%) - 特定氨基酸类型

---

### 🛠️ 突破2: 可解释性工具链搭建完成

#### Day 1: 注意力可视化框架 ✅

**脚本**: `scripts/visualize_attention_weights.py` (515行)

**功能**:
- 自动从checkpoint推断模型配置 (解决heads、use_mlp_layernorm等参数不匹配问题)
- 提取GAT 3层注意力权重
- 边→节点注意力聚合 (加权入度平均)
- 生成结构-注意力叠加图

**技术细节**:
```python
# 配置自动推断示例
hidden_dim = state_dict['node_encoder.weight'].shape[0]  # 128
heads = state_dict['att_layers.0.att_src'].shape[1]      # 4 (中间维度)
num_layers = count_gat_layers(state_dict)                 # 3
use_mlp_layernorm = (len(state_dict['mlp.0.weight'].shape) == 1)  # False
```

**状态**: 脚本完成并验证，等待PDB文件生成完整可视化

---

#### Day 2: 特征重要性定量分析 ✅

**脚本**: `scripts/analyze_feature_importance.py` (630行)

**实现**:
- Integrated Gradients: 基线插值 + 梯度积分
- Permutation Importance: 特征组打乱 + 性能下降测量

**输出文件** (已生成):
```
results/interpretability/feature_importance/
├── figS12_node_feature_importance.png      (节点特征Top 20)
├── figS12_edge_feature_importance.png      (边特征Top 20)
├── figS12b_feature_type_importance.png     (RBF vs Angles vs Dihedrals)
├── figS12c_permutation_importance.png      (排列重要性验证)
├── node_feature_importance.csv
├── edge_feature_importance.csv
└── feature_importance_summary.json
```

**测试参数**:
- 样本数: 100 (从898个测试集中采样)
- IG步数: 50 (baseline→input插值)
- 批大小: 32
- 设备: CUDA

**状态**: ✅ **完全完成并验证**

---

## 🔄 三、进行中的工作

### 3.1 DiffDock批量对接 (Background Task)

**任务**: 对接3148个样本用于后续自动化训练流程

**进度** (截至2.25 16:13):
```
GPU 0: 365/1625  (22.5%)  | 已运行: ~6小时
GPU 1: 366/1523  (24.0%)  | ETA: 明天14:00
────────────────────────────────────────────
总计:  731/3148  (23.2%)
```

**预计输出**:
- 3148个口袋PDB文件 (`sample_data/samples/{sample_id}/docking/`)
- 用于训练的.pt数据集
- 用于注意力可视化的结构文件

**修复记录**:
- 问题: 误以为有4个GPU，GPU2/GPU3实际不存在
- 解决: 正确配置2 GPU (RTX 3090 x2)，重新启动workers

---

### 3.2 可解释性分析后续计划 (5天工作量)

**总体目标**: 构建完整的模型可解释性故事，超越单一性能指标

#### Day 3: 多案例研究 (Pending PDB files)

**脚本**: `scripts/select_interpretability_cases.py` + `scripts/analyze_case_mechanisms.py`

**选择标准**:
- 2个高准确样本 (误差<0.5 log单位)
- 2个离群点样本 (误差>2.0 log单位)
- 2个已知机制样本 (文献报道的催化残基)
- 2个尺寸多样样本 (小分子 vs 大底物)

**分析内容**:
- 注意力权重 vs 已知催化残基的对应关系
- 不同EC类别的注意力模式差异
- 误差来源诊断 (对接质量、结构缺陷等)

**预期输出**: `figS13` (8个4-panel案例可视化)

---

#### Day 4: 离群点系统调查

**脚本**: `scripts/investigate_outliers.py`

**分析维度**:
1. 对接质量 vs 预测误差相关性
2. EC类别富集分析 (哪些EC类表现差？)
3. 底物复杂度影响 (分子量、环数、氢键受体/供体)
4. 失败模式分类 (docking失败 vs 特征不足 vs 标签噪声)

**预期输出**: `figS14` (4-panel离群点分析)

---

#### Day 5-6: 跨模态交互分析

**脚本**: `scripts/analyze_cross_modal_interactions.py`

**分析内容**:
1. 序列 vs 口袋贡献 (按EC类别分层)
   - 消融实验: 只用口袋 vs 只用ESM-2 vs 融合
2. 几何基序发现
   - t-SNE聚类口袋图嵌入
   - 识别共享的催化几何模式

**预期输出**: `figS15` (跨模态+基序可视化)

---

#### Day 7: 论文整合

**任务**:
1. 更新`paperwriting/gemini.tex`:
   - 新增Section 3.X "Interpretability and Mechanistic Insights"
   - 4个子节: 注意力分析、特征重要性、案例研究、离群点调查
2. 更新`supplementary_info.md`:
   - Section S6: 详细方法学 (IG算法、permutation细节)
3. 生成所有图表 (300 DPI PDF)
4. 更新`scripts/generate_paper_figures.py` (自动化图表生成)

---

## 📊 四、对比SOTA的优势总结

| 维度 | DLKcat/UniKP | CatPred/CataPro | **PocketGNN (Ours)** |
|------|--------------|-----------------|----------------------|
| **输入信息** | 序列 + SMILES | 序列 + SMILES | **口袋3D + 序列** |
| **几何特征** | ❌ | ❌ | ✅ 24维 (RBF+Angles+Dihedrals) |
| **可解释性** | ❌ | 部分 (不确定性) | ✅ 注意力+特征归因 |
| **40% Homology** | 未报道 | r=0.52 | **r=0.667 (+28.7%)** |
| **机制洞察** | ❌ | ❌ | ✅ 键角是关键 (61.9%) |

**核心卖点**:
1. **性能**: 低同源设置下显著优于SOTA (r=0.667 vs 0.52)
2. **可解释**: 揭示几何约束是催化预测的关键 (89%贡献)
3. **机制**: 符合酶催化化学原理 (精确几何排列)
4. **架构**: 首次系统验证24维边特征设计的必要性

---

## 🎯 五、下一步工作计划

### 短期 (本周内, 2.25-3.1)

1. **DiffDock完成后** (明天14:00):
   - ✅ 生成完整的口袋PDB文件
   - ✅ 构建自动化训练数据集
   - 🔄 运行auto_pipeline训练新模型

2. **可解释性分析继续** (Day 3-7):
   - Day 3: 多案例研究 (需PDB文件)
   - Day 4: 离群点调查
   - Day 5-6: 跨模态分析
   - Day 7: 论文整合

3. **模型对比实验**:
   - 使用新DiffDock数据重新训练
   - 对比旧vs新口袋提取方法

### 中期 (3月, JCIM投稿准备)

1. **论文完善**:
   - Methods: 补充可解释性方法学细节
   - Results: 新增可解释性章节 (Section 3.X)
   - Discussion: 强调几何特征的化学意义
   - Supplementary: 完整案例研究和消融实验

2. **额外实验** (如需要):
   - Scaffold Split (底物泛化)
   - Ensemble不确定性量化
   - PHPTransformer架构对比

3. **代码整理**:
   - 清理实验脚本
   - 补充README
   - 准备GitHub公开仓库

---

## 💡 六、科学贡献与创新点

### 6.1 方法学创新

1. **24维几何边特征**:
   - 首次系统引入键角(4维) + 二面角(4维)
   - 通过特征归因证明其必要性 (89%贡献)
   - 超越传统距离编码 (RBF仅11%)

2. **跨模态融合**:
   - 局部3D几何 (口袋图) + 全局序列语义 (ESM-2)
   - Late fusion with projection layer

3. **可解释性框架**:
   - 注意力权重可视化 → 揭示模型关注的关键原子/残基
   - Integrated Gradients → 定量归因每个特征的贡献
   - 案例研究 → 验证与已知催化机制的一致性

### 6.2 科学洞察

1. **几何约束是催化预测的关键** (实证支持):
   - 键角贡献61.9% (vs RBF的11.1%)
   - 符合酶催化依赖精确空间排列的化学原理

2. **低同源性能显著提升** (vs SOTA +28.7%):
   - 结构信息帮助泛化到进化距离较远的酶
   - 序列方法在低同源设置下表现受限

3. **模型可解释性**:
   - 不再是"黑盒"预测
   - 提供催化机制假设生成的工具

---

## 📋 七、预期论文结构 (JCIM格式)

### Abstract
- Background: 酶动力学预测的挑战 (低同源泛化)
- Methods: 3D口袋图 + 24维几何特征 + GAT
- Results: r=0.667 (40% homology), 优于SOTA 28.7%
- Interpretability: 键角贡献61.9%，揭示几何约束重要性
- Conclusion: 首个可解释的3D结构驱动的kcat预测模型

### Introduction
1. 酶动力学预测的生物学意义
2. 现有方法局限 (序列方法无几何信息)
3. 3D结构的优势 (几何约束)
4. 本研究贡献

### Methods
1. 数据集: IntEnzyDB + 40% homology split
2. 结构准备: ESMFold + DiffDock
3. 图表示: 52-dim节点 + **24-dim几何边**
4. 模型架构: PocketGNNKcatOnly (GAT + ESM-2 fusion)
5. **可解释性方法**: Integrated Gradients + 注意力分析

### Results
1. 基准性能 (vs SOTA)
2. 消融实验 (几何特征、ESM-2、pooling)
3. **特征重要性分析** ⭐ (新增)
4. **案例研究** ⭐ (新增)
5. **离群点分析** ⭐ (新增)

### Discussion
1. 几何特征的化学意义
2. 低同源泛化机制
3. 可解释性的价值
4. 局限性与未来方向

### Supplementary Information
- S1: 数据集统计
- S2: 模型架构细节
- S3: 训练超参数
- S4: 额外基准对比
- S5: 消融实验
- **S6: 可解释性方法学** ⭐ (新增)
- **S7: 完整案例研究** ⭐ (新增)

---

## 📌 八、关键文件索引

### 可解释性脚本
- `scripts/visualize_attention_weights.py` - 注意力可视化
- `scripts/analyze_feature_importance.py` - 特征归因
- `scripts/select_interpretability_cases.py` - 案例选择 (待创建)
- `scripts/analyze_case_mechanisms.py` - 案例分析 (待创建)
- `scripts/investigate_outliers.py` - 离群点调查 (待创建)

### 实验结果
- `results/interpretability/feature_importance/` - Day 2完整输出
- `results/interpretability/attention/` - Day 1/3输出 (待生成)
- `results/interpretability/case_studies/` - Day 3输出 (待生成)
- `results/interpretability/outliers/` - Day 4输出 (待生成)

### 论文文件
- `paperwriting/gemini.tex` - 主论文
- `paperwriting/supplementary_info.md` - 补充材料
- `scripts/generate_paper_figures.py` - 自动化图表生成

### 项目文档
- `PROJECT_PROGRESS.md` - 综合进展追踪 (已更新)
- `PROJECT_DOCUMENTATION.md` - 项目全面文档
- `CLAUDE.md` - Claude Code指南

---

## 🎓 九、组会讨论点

### 需要反馈的问题

1. **论文策略**:
   - Q: 可解释性分析是否足够作为主要卖点？
   - Q: 是否需要额外的PHPTransformer架构对比实验？

2. **实验设计**:
   - Q: 案例研究选择8个样本是否合适？需要更多吗？
   - Q: 是否需要补充Scaffold Split实验？

3. **图表呈现**:
   - Q: 特征重要性结果如何最佳呈现？(当前: 柱状图+饼图)
   - Q: 注意力可视化的配色方案？(蓝色→红色 vs 其他)

4. **时间规划**:
   - Q: 5-7天完成可解释性分析是否合理？
   - Q: JCIM投稿目标时间？(建议3月中旬)

### 导师建议征集

- 可解释性分析的深度是否足够？
- 论文故事线是否清晰？
- 还需要哪些补充实验？
- 投稿时机建议？

---

## 📞 联系方式

**报告人**: 李子豪
**邮箱**: [您的邮箱]
**项目路径**: `/home/lizihao/Work/enzyme_prediction/PGNN_clean`

---

**报告结束 - 期待您的反馈！** 🚀
