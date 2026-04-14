# PocketGNN 项目深度上下文文档

最后维护日期：2026-04-14

用途：这不是普通 README，而是给 **人类合作者、导师、审稿前自检、以及后续 AI agent** 使用的项目全景文档。目标是让任何接手者能快速理解：

1. 这个项目到底想证明什么。
2. 代码和数据是怎样连成证据链的。
3. 论文里哪些 claim 是稳的，哪些必须降调。
4. 现有结果为什么“不完美但可 defend”。
5. 如果人和 AI 围绕项目辩论，应围绕哪些可检验事实，而不是凭感觉争论。

相关维护文档：

- `EXPERIMENT_SETTINGS_AND_RESULTS.md`：实验设定、结果文件、图表位置索引。
- `PROJECT_CLEAR_MIND_20260413.md`：组会反馈后的论文收束策略。
- `paper_freeze_202604/`：论文冻结包，包含 manuscript、核心图、核心结果和复现脚本副本。

---

## 0. 一句话项目定义

PocketGNN 是一个用于酶 `kcat` 预测的 cross-modal 模型：

> 用 DiffDock/结构流程提取底物周围 active pocket，构建带距离、角度、二面角统计特征的 pocket graph；再融合 ESM-2 序列 embedding，用 GNN + sequence late fusion 预测 `log10(kcat)`。

这个项目最合理的论文定位不是“完美解决 kcat prediction”，而是：

> 在 leakage-aware / homology-aware 评估下，证明局部 pocket geometry 与全局 sequence semantics 的融合能保留可迁移趋势信号，同时诚实揭示 calibration、EC class heterogeneity、docking confidence 和 feature attribution 的限制。

---

## 1. 当前最稳的论文主线

### 1.1 主 claim

当前最稳的主 claim 是：

1. **Structure-aware representation 有价值**：模型不是只靠序列或 family prior；pocket graph 里有可学习的几何/电子信号。
2. **homology-aware held-out test 上仍有 trend signal**：主测试集 `N=898`，Pearson `r=0.716`。
3. **绝对误差和校准仍有限**：R2 `0.387`，MAE `0.901`，极端 kcat 区间存在 regression-to-the-mean。
4. **EC class 表现不均一**：EC2/EC6 相对更好，EC5 Pearson 高但 R2/MAE 差。
5. **DiffDock confidence 不是 downstream error proxy**：confidence 与 absolute error 几乎无相关。
6. **Integrated Gradients 不能作为强机制证明**：只能作为 exploratory attribution 或 qualitative case support。

### 1.2 推荐论文叙事

推荐主线：

> PocketGNN demonstrates that automated pocket-centric structural modeling can scale enzyme kinetic prediction beyond family-prior-dominated inference. Under a 40% homology-aware test set, the model preserves meaningful transferable trend signal, but calibration and failure-mode analyses show that robust mechanistic interpretation and extreme-rate prediction remain open challenges.

中文解释：

- 重点不是“我所有图都很漂亮”。
- 重点是“我知道哪里稳、哪里不稳，并且这些不稳点构成未来工作和方法意义”。
- 这样即使结果不完美，论文也不会显得硬拗。

### 1.3 不应再写的 claim

以下表达要避免：

- 模型解决了 kcat prediction。
- 所有 EC class 都强泛化。
- DiffDock confidence 与误差无关，所以 docking 质量不重要。
- Integrated Gradients 证明了催化机制。
- 某一类角度/二面角特征贡献占绝对主导。
- shared-sample baseline comparison 是严格同源外推 baseline comparison。

---

## 2. 项目代码结构：从原始数据到论文结果

### 2.1 高层 pipeline

```text
IntEnzyDB-derived kcat records
  -> filter valid UniProt / SMILES / positive kcat / kcat range
  -> protein structure: PDB or ESMFold
  -> ligand conformer: RDKit + UFF
  -> docking: DiffDock blind docking
  -> pocket extraction: atoms within 5.0 A of docked ligand
  -> graph construction: node features + RBF distance edges + angle/dihedral stats
  -> optional ESM-2 sequence embeddings
  -> PocketGNNKcatOnly training
  -> evaluation on hom40 test, shared baselines, EC-wise, docking robustness, calibration
  -> paper figures / tables
```

### 2.2 关键代码文件

| 文件 | 作用 | 当前重要性 |
|---|---|---|
| `src/graph_builder_rbf.py` | 构建基础 pocket graph；52 维节点特征、16 维 RBF 边特征 | 很重要，决定输入特征 |
| `src/build_graph_dataset.py` | 在基础 RBF 边特征上添加 angle/dihedral stats，得到 24 维边特征 | 很重要，支撑 geometry-enhanced claim |
| `src/GNN_model.py` | 模型定义，主类是 `PocketGNNKcatOnly` | 很重要 |
| `src/train.py` | 训练入口，保存 checkpoint、training_config、曲线 | 很重要 |
| `src/test.py` | 测试入口，生成 metrics、predictions、scatter/residual 图 | 很重要 |
| `scripts/analyze_ec_performance_from_predictions.py` | corrected EC-wise 分析 | 当前可信 |
| `scripts/analyze_prediction_bias.py` | calibration / dynamic-range compression 分析 | 当前可信 |
| `scripts/analyze_docking_robustness.py` | DiffDock confidence vs downstream error 分析 | 当前可信 |
| `scripts/analyze_significance.py` | shared-sample baseline paired significance | 当前可信，但解释边界要清楚 |
| `scripts/analyze_feature_importance.py` | IG/feature importance 分析 | 仅 exploratory，不作为强 claim |

### 2.3 代码层面的重要实现细节

#### 图构建

`src/graph_builder_rbf.py` 中核心设定：

- distance cutoff: `4.0 A`
- RBF centers: `16`
- RBF distance range: `0.0` to `8.0`
- RBF gamma: `20.0`
- fallback: 如果 `radius_graph` 不可用，手动双向建边。

`src/build_graph_dataset.py` 中增强边特征：

- 原始 `edge_attr`: 16 维 Gaussian RBF distance。
- 新增 angle stats: 4 维，包含 cosine min/max/mean/count-normalized。
- 新增 dihedral stats: 4 维，包含 cosine min/max/mean/count-normalized。
- 最终 `edge_attr`: 24 维。

#### 节点特征

主数据对象确认：

- `x`: `[num_nodes, 52]`
- `edge_index`: `[2, num_edges]`
- `edge_attr`: `[num_edges, 24]`
- `y`: `[1]`，表示 `log10(kcat)`

节点特征大致包括：

- element one-hot
- residue one-hot
- ligand/protein indicator
- nearest-neighbor distance
- electronic occupancy/count-like descriptors
- mass / electronegativity / radius
- secondary-structure one-hot
- normalized SASA-like scalar

这直接导致 feature attribution 的 caveat：这些特征不是同一种物理量，也不是都适合 zero-baseline interpolation。

#### 模型结构

主模型是 `PocketGNNKcatOnly`：

1. `Linear(node_input_dim, hidden_dim)` 编码节点。
2. 多层 `GATConv` 做 message passing。
3. 第一层 GAT 使用 edge attributes。
4. pooling 采用 `set2set`，输出维度是 `2 * hidden_dim`。
5. 如果启用 ESM embedding，将 `seq_embedding_dim -> hidden_dim` 投影。
6. graph vector 与 sequence vector 拼接。
7. MLP 输出单个 `log10(kcat)` prediction。

主模型参数：

| 参数 | 值 |
|---|---:|
| node_input_dim | 52 |
| edge_input_dim | 24 |
| hidden_dim | 128 |
| num_layers | 3 |
| heads | 4 |
| dropout | 0.3 |
| pooling | set2set |
| ESM seq embedding | yes |
| seq_embedding_dim | 640 |

---

## 3. 数据集与 split：必须讲清楚的地方

### 3.1 已确认数据规模

通过 `.pt` 文件实际加载确认：

| 文件 | 样本数 | 第一个样本结构 |
|---|---:|---|
| `data/processed/kcat_full_1213.pt` | 9059 | x `[321,52]`, edge_attr `[2814,24]`, y `[1]` |
| `data/processed/kcat_merged_hom40_train.pt` | 8620 | x `[321,52]`, edge_attr `[2814,24]`, y `[1]` |
| `data/processed/kcat_merged_hom40_val.pt` | 996 | x `[137,52]`, edge_attr `[1236,24]`, y `[1]` |
| `data/processed/kcat_merged_hom40_test.pt` | 898 | x `[151,52]`, edge_attr `[1224,24]`, y `[1]` |
| `data/processed/kcat_test_new_diffdock.pt` | 940 | x `[66,52]`, edge_attr `[482,24]`, y `[1]`, docking_confidence `[1]` |

### 3.2 当前主模型训练与测试关系

主 checkpoint：

- `outputs/kcat_enhanced_run/best_model.pt`

训练配置显示：

```text
Dataset Path: data/processed/kcat_full_1213.pt
Pooling Type: set2set
Use Seq Embedding: True
Seq Embedding Path: data/processed/esm_embeddings.pt
Batch Size: 32
Learning Rate: 0.001
Max Epochs: 500
Loss Type: mse
Dropout: 0.3
Weight Decay: 0.0001
Scheduler: plateau
```

最关键的解释：

- 当前主 checkpoint 是用 `kcat_full_1213.pt` 训练的。
- 主 OOD 指标是把这个 checkpoint 放到 `kcat_merged_hom40_test.pt` 上测试得到的。
- 因此可以说模型在 homology-aware held-out test 上评估，但不要把当前 checkpoint 说成“严格只用 hom40_train 训练出来”。
- 如果后续要从方法学上更干净，应该专门用 `kcat_merged_hom40_train.pt` / `val.pt` 重训并替换主结果；但当前论文攻坚阶段默认不重训。

### 3.3 为什么这点重要

这是项目最需要诚实处理的地方之一。

如果审稿人问：

> Did you train the final model only on the homology-aware training split?

当前最诚实回答应该是：

> The reported homology-split test metrics are obtained by evaluating the selected PocketGNN checkpoint on the 40% homology-aware held-out test set. For the current manuscript revision, we treat this as an external homology-aware stress test rather than claiming a fully retrained homology-split training protocol. A future stricter protocol would retrain exclusively on the homology-aware training split and report multi-seed results.

这句话很保守，但比被抓住“训练集混了 homology 测试相关样本吗？”要安全。若你希望论文 claim 更强，唯一根本办法是重训 hom40-only 模型。

---

## 4. 当前可信结果链

### 4.1 主 homology test

来源：`results/test_hom40_evaluation/test_metrics.json`

| N | Pearson r | R2 | MAE | RMSE |
|---:|---:|---:|---:|---:|
| 898 | 0.716 | 0.387 | 0.901 | 1.197 |

配套文件：

- `results/test_hom40_evaluation/test_predictions.csv`
- `results/test_hom40_evaluation/test_kcat_prediction_scatter.png`
- `results/test_hom40_evaluation/test_residuals.png`
- `results/test_hom40_evaluation/test_residual_histogram.png`

论文解释：

- Pearson 说明 trend/rank signal 仍存在。
- R2/MAE 说明绝对校准仍有限。
- 这不是“模型失败”，而是 OOD kinetic prediction 的真实难度。

### 4.2 Calibration / dynamic-range compression

来源：`results/bias_calibration/`

| 指标 | 值 |
|---|---:|
| global bias pred-true | -0.149 |
| true range | -5.52 to 6.03 |
| prediction range | -1.30 to 1.98 |
| lowest true-kcat decile bias | +1.95 |
| highest true-kcat decile bias | -2.04 |

解释：

- 模型在极端低速酶上高估，在极端高速酶上低估。
- 这叫 regression-to-the-mean / dynamic-range compression。
- 它解释了为什么 Pearson 可以还不错，但 R2/MAE 被极端区间拖累。

论文价值：

- 这是一个强 failure-mode analysis。
- 比单纯说“误差还大”更有说服力。
- 可以自然连接到 future work：calibration loss、heteroscedastic uncertainty、quantile regression、多任务条件建模。

### 4.3 Corrected EC-wise analysis

来源：`results/ec_analysis_corrected/`

| EC | N | Pearson r | R2 | MAE | 解释 |
|---|---:|---:|---:|---:|---|
| EC1 | 703 | 0.667 | 0.350 | 0.900 | 主体样本，代表整体趋势 |
| EC2 | 67 | 0.773 | 0.503 | 0.559 | 相对稳定 |
| EC3 | 36 | 0.885 | 0.376 | 0.932 | trend 强但误差不低 |
| EC4 | 42 | 0.813 | 0.422 | 1.047 | trend 尚可，MAE 偏高 |
| EC5 | 31 | 0.843 | 0.059 | 1.383 | 典型校准/小样本问题 |
| EC6 | 18 | 0.973 | 0.615 | 0.932 | 小样本下表现好，但需谨慎 |

论文解释：

- 不说 all EC classes strongly generalize。
- 说 performance is heterogeneous。
- EC5 是一个很好的讨论点：rank/trend preserved, calibration poor。

### 4.4 Shared-sample baseline significance

来源：`results/significance/significance_results.json`

| Comparison | N | PocketGNN r | Baseline r | PocketGNN MAE | Baseline MAE | p-value |
|---|---:|---:|---:|---:|---:|---:|
| PocketGNN vs CatPred | 3439 | 0.864 | 0.608 | 0.479 | 0.942 | 0.0002 |
| PocketGNN vs CataPro | 1455 | 0.868 | 0.683 | 0.487 | 0.892 | 0.0002 |

解释边界：

- 这是 paired shared-sample comparison。
- 优点：公平，因为同一批样本上比。
- 局限：不等同严格 homology OOD。
- 论文应该明确：用于 benchmark comparison，而 homology test 用于 OOD stress test。

### 4.5 Docking robustness

来源：`results/docking_robustness/docking_robustness.json`

| 指标 | 值 |
|---|---:|
| N | 940 |
| overall Pearson | 0.853 |
| overall R2 | 0.720 |
| overall MAE | 0.473 |
| confidence vs abs error Pearson | 0.0267, p=0.413 |
| confidence vs abs error Spearman | 0.0361, p=0.268 |

解释：

- 在 successfully constructed graphs 中，DiffDock confidence 不能预测下游 kcat 误差。
- 不能推出 docking quality 不重要。
- 更稳的说法是：confidence score is not a reliable downstream error proxy。

### 4.6 Interpretability / attribution

当前可保留：

- ALDH2 qualitative saliency case。
- IG 作为 exploratory diagnostic。
- feature-scale caveat。

当前不可保留：

- 百分比式强机制 claim。
- “IG 已经证明机制”。
- 把 one-hot/category feature 的 zero baseline integration 当成物理路径。

---

## 5. 图表与文件地图

### 5.1 主文图表

| 图/表 | 当前文件 | 可信度 | 注意事项 |
|---|---|---:|---|
| Architecture | `paperwriting/Architecture_EN.jpg` | 高 | 方法示意图 |
| DiffDock vs Vina table | `paperwriting/gemini.tex` 内表格 | 中 | 强调 automation，不夸大 DiffDock 精度 |
| DiffDock comparison fig | `paperwriting/diffdock_vs_vina_comparison.png` | 中 | redocking 小样本，不能泛化过强 |
| Main scatter | `paperwriting/test_kcat_prediction_scatter.png` | 高 | 对应 hom40 result |
| Corrected EC figure | `results/ec_analysis_corrected/fig_ec_wise_performance_corrected.png` | 高 | 替代旧 EC 图 |
| ALDH2 saliency | `paperwriting/visualize.png` | 中 | qualitative only |

### 5.2 Supplementary 图表

| 图/表 | 文件/位置 | 可信度 | 用法 |
|---|---|---:|---|
| Corrected EC table | `results/ec_analysis_corrected/ec_wise_performance.csv` | 高 | 描述 heterogeneity |
| Feature-scale table | `paperwriting/supplementary_info.md` | 高 | 支撑 attribution caveat |
| Docking robustness table | `results/docking_robustness/docking_robustness.json` | 高 | confidence 不是 error proxy |
| Density/confidence bins | `paperwriting/supplementary_info.md` | 中高 | 解释 scatter density effect |
| Bias/calibration table | `results/bias_calibration/true_kcat_bin_bias.csv` | 高 | 支撑 dynamic-range compression |
| Bias/calibration figure | `results/bias_calibration/fig_prediction_bias_calibration.png` | 高 | failure-mode figure |
| IG feature figures | `paperwriting/figS12_*.png` | 低到中 | exploratory only |

### 5.3 论文冻结包

`paper_freeze_202604/` 是当前最好交给人或 AI 的入口：

```text
paper_freeze_202604/
  README.md
  manuscript/
    gemini.tex
    supplementary_info.md
    gemini.pdf
  figures/main/
  figures/supplementary/
  results_core/
  scripts_reproduce/
  manifests/
```

如果一个新的 AI agent 要接手，优先读：

1. `PROJECT_DEEP_CONTEXT_FOR_HUMAN_AND_AI.md`
2. `EXPERIMENT_SETTINGS_AND_RESULTS.md`
3. `PROJECT_CLEAR_MIND_20260413.md`
4. `paperwriting/gemini.tex`
5. `paperwriting/supplementary_info.md`
6. `paper_freeze_202604/README.md`

---

## 6. 争议点与辩论框架

这一节的目标是让人和 AI “愈辩愈明”。每个争议点都给出：支持证据、反方质疑、当前最稳回答、如果要彻底解决需要做什么。

### 6.1 争议 A：这是否真的是严格 homology generalization？

支持证据：

- 有 `data/processed/kcat_merged_hom40_test.pt`，N=898。
- 主结果 JSON 明确记录 `test_dataset_path` 是 hom40 test。
- 结果是 r=0.716, R2=0.387, MAE=0.901。

反方质疑：

- 主 checkpoint 的训练配置显示训练数据是 `data/processed/kcat_full_1213.pt`。
- 如果 full dataset 与 hom40 test 有重叠或相近样本，strict OOD claim 可能过强。

当前最稳回答：

- 当前指标可以作为 homology-aware held-out test/stress test 报告。
- 不应声称主 checkpoint 完全由 homology train split 训练。
- 论文中需要谨慎措辞，避免“严格 OOD training protocol”的暗示。

彻底解决方案：

- 用 `kcat_merged_hom40_train.pt` 和 `kcat_merged_hom40_val.pt` 从零重训。
- 在 `kcat_merged_hom40_test.pt` 上测试。
- 最好做 3 seeds，报告 mean±std。

当前论文攻坚选择：

- 暂不重训；用诚实语言收束。

### 6.2 争议 B：R2 只有 0.387，够不够？

支持证据：

- Pearson 0.716 表明 trend signal 不弱。
- calibration 分析显示错误不是随机，而是 dynamic-range compression。
- kcat 本身实验噪声和条件异质性大。

反方质疑：

- R2/MAE 不够强，说明绝对预测不可靠。
- 如果用于工程设计，极端 kcat 恰恰重要。

当前最稳回答：

- 模型适合 scalable trend-aware screening，不适合单点高精度机制判断。
- 论文应强调 predictor/failure modes，而不是宣称 fully accurate kinetic oracle。

彻底解决方案：

- calibration-aware loss。
- quantile/uncertainty prediction。
- condition-aware modeling: pH、temperature、organism、assay conditions。
- more balanced sampling over kcat range。

### 6.3 争议 C：DiffDock confidence 与 error 无关是好事还是坏事？

支持证据：

- confidence vs abs error Pearson 0.0267, p=0.413。
- tertiles 中性能相对稳定。

反方质疑：

- confidence 不等于 pose quality。
- 不能说明 docking 质量不影响模型。
- 只分析 successfully constructed graphs，存在 selection bias。

当前最稳回答：

- 这只说明 confidence score 不是 downstream kcat error proxy。
- 不能替代真实 RMSD 或 holo pose validation。

彻底解决方案：

- 在有 holo ligand 的结构上验证 pose RMSD。
- 用真实 pose quality 分层，而非 DiffDock confidence。
- 做 docking perturbation / pose ensemble robustness。

### 6.4 争议 D：IG/feature importance 有没有意义？

支持证据：

- qualitative saliency case 能显示模型关注 pocket/catalytic region。
- IG 可作为探索性诊断。

反方质疑：

- node features 含 one-hot 和 categorical，zero-baseline path 没物理意义。
- feature scales 混合，raw gradient magnitude 不可直接比较。
- 旧 feature importance 脚本疑似模型配置不匹配。

当前最稳回答：

- 只保留 exploratory/qualitative，不作为机制证明。

彻底解决方案：

- group masking / permutation。
- train-set-fitted standardization。
- compare feature groups under matched perturbation protocol。
- 多样本稳定性分析，而不是单次 attribution。

### 6.5 争议 E：EC-wise 结果是否支持泛化？

支持证据：

- EC2/EC6 表现较好。
- 多个 EC 类别 Pearson 为正且不低。

反方质疑：

- EC5 R2=0.059, MAE=1.383。
- EC6 N=18，样本太小。
- Pearson 高可能掩盖 calibration bias。

当前最稳回答：

- EC-wise behavior is heterogeneous。
- 这不是失败，而是对任务难度的诚实展示。

彻底解决方案：

- 每个 EC 内做 calibration plots。
- 用更多样本或分层采样。
- 引入 EC-aware or family-aware calibration。

---

## 7. 论文每一部分应承担的功能

### 7.1 Abstract

功能：给出贡献，但不能过度承诺。

应该包含：

- pocket graph + ESM-2 fusion。
- random split upper bound。
- homology test r=0.716/R2=0.387/MAE=0.90。
- shared-sample baseline improvement。
- EC/docking/attribution limitations。

不应该包含：

- IG 百分比。
- all EC strong generalization。
- docking 质量无关。

### 7.2 Introduction

功能：解释为什么 kcat 预测需要 local geometry + global sequence。

要点：

- sequence-only methods capture evolutionary semantics but miss local stereochemistry。
- whole-structure methods may be noisy。
- pocket-level graph is a targeted structural view。
- leakage-aware evaluation is necessary。

### 7.3 Methods

功能：让读者知道数据和模型怎么来。

必须讲清楚：

- IntEnzyDB-derived data。
- PDB/ESMFold + RDKit + DiffDock + 5A pocket。
- node/edge feature dimensions。
- ESM-2 embedding late fusion。
- evaluation settings。

需要谨慎：

- 如果当前主 checkpoint 不是 hom40 train-only 训练，不要写成严格 homology training protocol。

### 7.4 Results

推荐顺序：

1. Main prediction performance。
2. Shared-sample baseline comparison。
3. Calibration / bias failure mode。
4. Docking robustness。
5. EC-wise heterogeneity。
6. Qualitative interpretability。

原因：

- 先建立性能锚点。
- 再说明对比优势。
- 再主动解释缺陷。
- 最后把解释性降级到 qualitative。

### 7.5 Discussion / Limitations

必须写的 limitations：

- homology split / training protocol 的边界。
- assay condition missing: pH, temperature, organism, experimental protocol。
- run-to-run variability: selected checkpoint not multi-seed mean±std。
- docking dependency: confidence not pose quality。
- feature scaling/attribution caveat。
- dynamic-range compression。

---

## 8. 如果现在继续做，优先级排序

### P0：不建议立刻做

- 不建议现在重训主模型，除非论文必须提升 strict homology claim。
- 不建议继续强化 IG。
- 不建议继续挖旧结果目录找漂亮图。

### P1：最值得做

1. **论文 narrative polish**：把 `gemini.tex` 改得更像一篇成熟论文，而不是结果堆叠。
2. **Figure/result manifest**：把所有主文和 supplementary 图表来源完全登记。
3. **Deprecated results 清单**：明确哪些旧结果不能再用。
4. **一页答辩口径**：导师问关键问题时怎么答。

### P2：如果还有时间

1. EC-specific calibration/bias plot。
2. group masking/permutation feature group sensitivity。
3. hom40-train-only retraining + 3 seeds。
4. uncertainty/quantile calibration。

### P3：未来工作

1. condition-aware kcat model。
2. pose ensemble / docking perturbation。
3. train-set-fitted feature standardization。
4. multi-task kcat/Km/Ki/Kcat/Km。

---

## 9. AI 接手时的操作指南

如果一个新的 AI agent 接手，应该按这个顺序工作：

1. 读 `PROJECT_DEEP_CONTEXT_FOR_HUMAN_AND_AI.md`。
2. 读 `EXPERIMENT_SETTINGS_AND_RESULTS.md`。
3. 读 `PROJECT_CLEAR_MIND_20260413.md`。
4. 读 `paperwriting/gemini.tex`。
5. 读 `paperwriting/supplementary_info.md`。
6. 不要先跑训练。
7. 不要默认相信 `results/` 里的所有图。
8. 如果要引用数字，优先从这些文件取：
   - `results/test_hom40_evaluation/test_metrics.json`
   - `results/test_hom40_evaluation/test_predictions.csv`
   - `results/ec_analysis_corrected/ec_wise_performance.csv`
   - `results/docking_robustness/docking_robustness.json`
   - `results/significance/significance_results.json`
   - `results/bias_calibration/bias_calibration_summary.json`
9. 如果发现旧图和新结果冲突，以 corrected/newer result 为准，并在文档里标注 deprecated。

---

## 10. 人类读者的快速判断

如果只花 5 分钟理解这个项目，应记住：

1. 这是一个 pocket-graph + ESM-2 的 kcat predictor。
2. 主结果：hom40 test N=898, Pearson 0.716, R2 0.387, MAE 0.901。
3. 结果不是完美，但有可迁移趋势。
4. 最大问题是 calibration / extreme kcat range compression。
5. EC-wise 不均一，EC5 是典型弱点。
6. DiffDock confidence 不能预测 downstream error。
7. IG 不能作为强机制证明。
8. 当前最稳投稿策略是“诚实、稳健、failure-aware”，不是“强行包装成全胜”。

---

## 11. 当前可 defend 的核心结论

最终可 defend 的版本：

> PocketGNN shows that scalable, docking-derived pocket graphs fused with protein language model embeddings can retain transferable signal for enzyme turnover prediction under homology-aware evaluation. The method is promising as a structure-aware screening model, while calibration analyses, EC-wise heterogeneity, and attribution caveats reveal clear limitations that should guide future model development.

中文对应：

> 这个项目的价值不是宣称 kcat 已经被解决，而是把“局部催化口袋几何 + 全局序列语义”这条路线做成了可扩展 pipeline，并在严格一些的评估下证明它有可迁移信号；同时诚实指出极端 kcat 校准、EC 类别差异、docking 置信度和解释性归因仍然是未解决问题。
