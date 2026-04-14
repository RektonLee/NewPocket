# PocketGNN 实验设定与结果索引

最后维护日期：2026-04-14

这个文档用于回答四类问题：

1. 主模型到底怎么训练的？
2. 训练/测试数据分别是什么？
3. 论文重要图表和结果文件在哪里？
4. 哪些结果是当前论文可信主线，哪些结果应该避免继续引用？

当前论文阶段的默认策略：**不重训主模型，不临时修改输入特征归一化管线，不把 Integrated Gradients 当作强机制证明。**

---

## 0. 当前论文主线

当前最稳的叙事是：

> PocketGNN 将 DiffDock/结构口袋图与 ESM-2 全局序列表示融合，在严格 40% homology split 下仍能捕捉可迁移的 kcat 趋势；但 calibration、EC-wise、docking confidence 和 attribution 诊断显示，该模型更适合作为 scalable structure-aware predictor，而不是完全可靠的机制解释模型。

论文中应该强调：

- 40% homology test 是主性能锚点。
- shared-sample baseline comparison 只能说明同样本覆盖下优于 CatPred/CataPro，不应说成严格 OOD baseline comparison。
- DiffDock confidence 不是下游误差 proxy，也不能替代真实 pose-quality validation。
- EC-wise 结果是 heterogeneous，不是所有 EC 类别都强泛化。
- IG/feature attribution 只作为 exploratory，不作为百分比式机制证明。

---

## 1. 主模型与训练设定

### 1.1 当前论文主 checkpoint

| 项目 | 值 |
|---|---|
| 主 checkpoint | `outputs/kcat_enhanced_run/best_model.pt` |
| 训练配置 | `outputs/kcat_enhanced_run/training_config.txt` |
| 训练曲线 | `outputs/kcat_enhanced_run/training_metrics.csv` |
| 训练输出图 | `outputs/kcat_enhanced_run/loss_curve.png`, `outputs/kcat_enhanced_run/metrics_curve.png` |
| 训练散点图 | `outputs/kcat_enhanced_run/kcat_prediction_scatter.png` |

注意：`outputs/kcat_enhanced_run/training_config.txt` 中 `Experiment Name` 写的是 `kcat_esm_full`，但保存目录是 `outputs/kcat_enhanced_run`。当前论文主结果引用 checkpoint 时以 `outputs/kcat_enhanced_run/best_model.pt` 为准。

### 1.2 训练命令

来源：`outputs/kcat_enhanced_run/training_config.txt`

```bash
python src/train.py \
  --dataset data/processed/kcat_full_1213.pt \
  --save_dir outputs/kcat_enhanced_run \
  --weight_decay 1e-4 \
  --dropout 0.3 \
  --scheduler plateau \
  --pooling_type set2set \
  --use_seq_embedding \
  --seq_embedding_path data/processed/esm_embeddings.pt
```

### 1.3 训练数据与 split 解释

| 用途 | 文件 | 说明 |
|---|---|---|
| 主模型训练输入 | `data/processed/kcat_full_1213.pt` | full/random-split 训练文件；用于训练 `outputs/kcat_enhanced_run/best_model.pt` |
| 序列 embedding | `data/processed/esm_embeddings.pt` | ESM-2 embedding 字典，训练时 late fusion 使用 |
| homology 外部测试 | `data/processed/kcat_merged_hom40_test.pt` | 当前主 OOD evaluation 的 898-sample test set |
| homology train split | `data/processed/kcat_merged_hom40_train.pt` | homology-aware train 文件；当前主论文 checkpoint 不是直接用它训练出来的 |
| homology val split | `data/processed/kcat_merged_hom40_val.pt` | homology-aware val 文件 |
| DiffDock robustness subset | `data/processed/kcat_test_new_diffdock.pt` / `.csv` | docking robustness 相关子集；论文中成功构图样本 N=940 |

关键解释：

- 当前主 checkpoint 是在 `kcat_full_1213.pt` 上训练的，并用 `kcat_merged_hom40_test.pt` 做严格 homology-split 外部测试。
- 因此论文里可以说“evaluated on 40% homology-split test set”，但不要暗示主 checkpoint 是专门用 homology train split 从零训练的，除非后续重训并替换主结果。
- 如果答辩/组会被问到：这是一个 selected trained model evaluated on a held-out homology-aware test set；shared baseline 另有 paired subset 分析。

### 1.4 模型架构参数

主测试结果 JSON 来源：`results/test_hom40_evaluation/test_metrics.json`

| 参数 | 值 |
|---|---:|
| model class | `PocketGNNKcatOnly` |
| node_input_dim | 52 |
| edge_input_dim | 24 |
| hidden_dim | 128 |
| num_layers | 3 |
| attention heads | 4 |
| dropout | 0.3 |
| pooling_type | `set2set` |
| use_seq_embedding | true |
| seq_embedding_dim | 640 |
| output | point prediction for `log10(kcat)` |

架构实现位置：

- 主模型类：`src/GNN_model.py`, `PocketGNNKcatOnly`
- 训练入口：`src/train.py`
- 测试入口：`src/test.py`

模型结构简述：

1. 52 维节点特征进入 linear node encoder。
2. GATConv 层进行图消息传递，第一层使用 edge attributes。
3. 图级 readout 使用 `Set2Set(hidden_dim, processing_steps=3)`，输出维度为 `2 * hidden_dim`。
4. ESM-2 sequence embedding 通过 `Linear(seq_embedding_dim, hidden_dim) + LayerNorm + ReLU + Dropout` 投影。
5. graph representation 与 projected sequence embedding 拼接后进入 MLP 输出 `log10(kcat)`。

### 1.5 训练超参数

来源：`outputs/kcat_enhanced_run/training_config.txt`

| 超参数 | 值 |
|---|---:|
| batch size | 32 |
| learning rate | 0.001 |
| max epochs | 500 |
| loss | MSE |
| dropout | 0.3 |
| weight decay | 0.0001 |
| scheduler | ReduceLROnPlateau |
| device | cuda:0 |

训练脚本行为：

- optimizer: Adam
- scheduler: `ReduceLROnPlateau(mode='min', factor=0.5, patience=args.patience)`
- gradient clipping: `clip_grad_norm_(..., max_norm=0.5)`
- prediction clamp: `[-10, 10]` in training/validation metric computation
- best model selection: validation loss lower and R2 not abnormal

---

## 2. 数据处理与输入特征

### 2.1 原始数据来源

| 数据 | 位置 | 说明 |
|---|---|---|
| raw kcat data | `data/raw/kcat_data.csv` | IntEnzyDB-derived raw/curated input |
| processed full graph data | `data/processed/kcat_full_1213.pt` | 主训练使用的图数据 |
| homology split fasta | `data/processed/kcat_merged_hom40_sequences.fasta` | MMseqs/homology split 相关序列 |
| cluster output | `data/processed/clusters_cluster.tsv` | homology clustering 结果 |

论文中的数据描述：

- source: IntEnzyDB
- target: experimental `kcat`
- transformed label: `log10(kcat)`
- filtered range: `kcat in [1e-2, 1e6] s^-1`
- processed paper dataset size in manuscript: 9,059 enzyme-substrate pairs
- main homology test size: 898

### 2.2 结构与 docking pipeline

| 步骤 | 工具/文件 | 说明 |
|---|---|---|
| enzyme structure | PDB or ESMFold | PDB 优先，缺失时可用 ESMFold 预测结构 |
| ligand conformer | RDKit + UFF | 从 SMILES 生成 3D conformer |
| docking | DiffDock | 盲对接，无需手动 binding-site box |
| pocket definition | within 5.0 A of top-ranked pose | 取 docked ligand 周围蛋白原子构建 pocket graph |
| graph construction | `src/build_graph_dataset.py`, `src/graph_builder_rbf.py` | 生成 PyTorch Geometric Data |

论文中 DiffDock 的稳妥表述：

- DiffDock 的优势是自动化 blind docking，适合大规模 predicted-structure pipeline。
- 小规模 redocking 对比不是为了证明 DiffDock 全场景优于 Vina，而是说明自动化流程可行。
- DiffDock confidence 不能当作真实 pose RMSD，也不是下游 kcat 误差的可靠 proxy。

### 2.3 图特征维度

| 特征组 | 维度/范围 | 说明 |
|---|---|---|
| node features | 52 | atom/residue/category/count/physicochemical mixed features |
| edge features | 24 | 16 RBF distance + 4 angle summary + 4 dihedral/torsion summary |
| sequence embedding | 640 | ESM-2 embedding, projected to hidden_dim before fusion |

重要 caution：

- node features 混合 one-hot、计数、mass/electronegativity/radius 等连续物理量。
- edge RBF 大体在 0-1，angle/dihedral 约在 -1 到 1。
- 训练和推理使用一致尺度，所以主预测结果内部一致。
- 但 raw gradient / zero-baseline IG 会受尺度和类别特征路径影响，不适合做强定量机制 claim。

---

## 3. 主结果文件索引

### 3.1 40% homology test 主结果

| 文件 | 用途 |
|---|---|
| `results/test_hom40_evaluation/test_metrics.json` | 主 test metrics，论文核心数字来源 |
| `results/test_hom40_evaluation/test_predictions.csv` | true/pred/residual/abs error，EC 和 calibration 分析输入 |
| `results/test_hom40_evaluation/test_kcat_prediction_scatter.png` | 主散点图候选 |
| `results/test_hom40_evaluation/test_residual_histogram.png` | 残差直方图 |
| `results/test_hom40_evaluation/test_residuals.png` | 残差分析图 |

核心数字：

| N | Pearson r | R2 | MAE | RMSE |
|---:|---:|---:|---:|---:|
| 898 | 0.716 | 0.387 | 0.901 | 1.197 |

当前论文主文使用的散点图副本：

- `paperwriting/test_kcat_prediction_scatter.png`

### 3.2 shared-sample baseline significance

| 文件 | 用途 |
|---|---|
| `results/significance/significance_results.json` | CatPred/CataPro paired shared-sample comparison |
| `results/significance/SIGNIFICANCE_REPORT.md` | 文本报告 |
| `results/stratified_significance/stratified_significance.json` | 按 sequence length / ligand heavy atoms 分层显著性 |

核心数字：

| Comparison | N | PocketGNN r | Baseline r | PocketGNN MAE | Baseline MAE | p-value |
|---|---:|---:|---:|---:|---:|---:|
| PocketGNN vs CatPred | 3439 | 0.864 | 0.608 | 0.479 | 0.942 | 0.0002 |
| PocketGNN vs CataPro | 1455 | 0.868 | 0.683 | 0.487 | 0.892 | 0.0002 |

使用边界：

- 这组结果是 shared-sample paired comparison。
- 不要称为严格同源外推 baseline comparison。
- 论文里可以说 fair paired comparison on shared benchmark subsets。

### 3.3 corrected EC-wise analysis

| 文件 | 用途 |
|---|---|
| `scripts/analyze_ec_performance_from_predictions.py` | corrected EC analysis 生成脚本 |
| `results/ec_analysis_corrected/ec_wise_performance.csv` | corrected EC 表 |
| `results/ec_analysis_corrected/ec_wise_performance.json` | corrected EC JSON |
| `results/ec_analysis_corrected/ec_aligned_predictions.csv` | 与 hom40 prediction 对齐后的逐样本文件 |
| `results/ec_analysis_corrected/fig_ec_wise_performance_corrected.png` | 当前应引用 EC-wise 图 |

corrected EC 核心结果：

| EC | N | Pearson r | R2 | MAE |
|---|---:|---:|---:|---:|
| EC1 | 703 | 0.667 | 0.350 | 0.900 |
| EC2 | 67 | 0.773 | 0.503 | 0.559 |
| EC3 | 36 | 0.885 | 0.376 | 0.932 |
| EC4 | 42 | 0.813 | 0.422 | 1.047 |
| EC5 | 31 | 0.843 | 0.059 | 1.383 |
| EC6 | 18 | 0.973 | 0.615 | 0.932 |
| Overall | 898 | 0.716 | 0.387 | 0.901 |

论文口径：

- performance is heterogeneous across EC classes。
- EC5 高 Pearson 但 R2/MAE 差，提示 calibration bias 或小样本不稳定。
- EC7 只有 1 个样本，不做 per-class metric。

### 3.4 docking robustness

| 文件 | 用途 |
|---|---|
| `scripts/analyze_docking_robustness.py` | docking confidence/error 分析脚本 |
| `results/docking_robustness/docking_robustness.json` | 核心 docking robustness 数字 |
| `results/docking_robustness/docking_robustness.png` | docking robustness 图 |
| `results/docking_robustness/docking_robustness_by_bin.csv` | confidence bins 表 |
| `results/docking_robustness/DOCKING_ROBUSTNESS_REPORT.md` | 报告 |

核心数字：

| Metric | Value |
|---|---:|
| N | 940 |
| overall Pearson r | 0.853 |
| overall R2 | 0.720 |
| overall MAE | 0.473 |
| confidence vs abs error Pearson r | 0.0267, p=0.413 |
| confidence vs abs error Spearman rho | 0.0361, p=0.268 |

论文口径：

- DiffDock confidence is not a reliable downstream error proxy among successfully constructed graphs。
- 不要把它解释为 docking quality 不重要。
- confidence 不等于真实 RMSD，也不等于 pocket 生物学正确性。

### 3.5 prediction bias / calibration

| 文件 | 用途 |
|---|---|
| `scripts/analyze_prediction_bias.py` | calibration/bias 分析脚本 |
| `results/bias_calibration/bias_calibration_summary.json` | 总结 JSON |
| `results/bias_calibration/true_kcat_bin_bias.csv` | true-kcat quantile bin 表 |
| `results/bias_calibration/fig_prediction_bias_calibration.png` | calibration 图 |

核心发现：

| 指标 | 值 |
|---|---:|
| N | 898 |
| global bias pred-true | -0.149 |
| true range | -5.52 to 6.03 |
| prediction range | -1.30 to 1.98 |
| lowest true-kcat decile bias | +1.95 |
| highest true-kcat decile bias | -2.04 |

论文口径：

- 模型存在 regression-to-the-mean / dynamic-range compression。
- 这解释了 Pearson 仍可用但 R2/MAE 在极端 kcat 区间受限。
- 这是 limitation + failure mode，不是要重训才能解释的问题。

---

## 4. 论文图表位置索引

### 4.1 主文图表

| 论文位置 | 当前文件 | 数据来源 | 状态 |
|---|---|---|---|
| Figure 1 architecture | `paperwriting/Architecture_EN.jpg` | 手工/流程图 | 可用 |
| Table DiffDock vs Vina | 主文 LaTeX table | paperwriting text / redocking result | 可用，但口径要强调 automation |
| Figure docking comparison | `paperwriting/diffdock_vs_vina_comparison.png` | redocking comparison | 可用，避免夸大 |
| Main hom40 scatter | `paperwriting/test_kcat_prediction_scatter.png` | `results/test_hom40_evaluation/` | 可用 |
| Corrected EC-wise figure | `results/ec_analysis_corrected/fig_ec_wise_performance_corrected.png` | corrected EC analysis | 可用，替代旧 EC 图 |
| ALDH2 saliency case | `paperwriting/visualize.png` | qualitative saliency | 可用，但只作 qualitative case |

### 4.2 Supplementary 图表

| Supplementary item | 当前文件/表 | 状态 |
|---|---|---|
| Table S2 corrected EC | `results/ec_analysis_corrected/ec_wise_performance.csv` | 可用 |
| Table S3 feature-scale diagnostic | `paperwriting/supplementary_info.md` 内表格 | 可用 |
| Table S6 docking robustness | `results/docking_robustness/docking_robustness.json` | 可用 |
| Table S7 density-aware confidence bins | `paperwriting/supplementary_info.md` 内表格 | 可用 |
| Table S8 prediction bias bins | `results/bias_calibration/true_kcat_bin_bias.csv` | 可用 |
| Figure S7 prediction bias | `results/bias_calibration/fig_prediction_bias_calibration.png` | 可用 |
| Figure S6 IG/feature attribution | `paperwriting/figS12_*.png` | exploratory only |

### 4.3 论文冻结包

| 目录 | 内容 |
|---|---|
| `paper_freeze_202604/manuscript/` | 当前 `gemini.tex`, `supplementary_info.md`, `gemini.pdf` |
| `paper_freeze_202604/figures/main/` | 主文核心图副本 |
| `paper_freeze_202604/figures/supplementary/` | supplementary 图副本 |
| `paper_freeze_202604/results_core/` | 论文核心结果 JSON/CSV |
| `paper_freeze_202604/scripts_reproduce/` | 关键分析脚本副本 |
| `paper_freeze_202604/manifests/` | manifest 和 checksums |

---

## 5. 明确不要再引用或需标注 deprecated 的结果

### 5.1 旧 EC-wise 图

| 项目 | 文件 |
|---|---|
| 旧图 | `results/ec_analysis/figS_ec_wise_performance.png` |
| 旧脚本 | `scripts/quick_ec_analysis.py` |
| 问题 | 使用旧模型/配置，结果与主 prediction 不对齐 |
| 当前处理 | 不再引用；使用 `results/ec_analysis_corrected/fig_ec_wise_performance_corrected.png` |

### 5.2 `results/ec_wise_analysis/ec_performance.json`

问题：旧分析存在 prediction length mismatch / placeholder 风险，不作为论文依据。

当前处理：使用 `results/ec_analysis_corrected/`。

### 5.3 强 IG 百分比 claim

旧 claim 类型：某类 angular/dihedral features 贡献占绝对主导。

问题：

- node features 混合 one-hot、计数、连续物理量。
- zero-baseline IG 对类别变量没有清楚物理路径。
- raw gradient magnitude 受 feature scale 影响。
- 旧 feature importance 脚本疑似模型/配置不匹配。

当前处理：

- 主文不使用百分比式机制结论。
- Supplementary 只保留 exploratory IG 方法说明。
- 如果以后要补，应使用 group perturbation/masking，并最好配合 train-set-fitted feature standardization。

### 5.4 shared baseline 的 OOD 误读

`results/significance/significance_results.json` 很有用，但它不是严格 homology split 的 baseline 对比。

当前处理：

- 可以说 paired shared-sample benchmark subset。
- 不说 严格同源外推 baseline comparison。

---

## 6. 当前 manuscript 文件

| 文件 | 用途 |
|---|---|
| `paperwriting/gemini.tex` | 当前主文 LaTeX |
| `paperwriting/gemini.pdf` | 当前编译 PDF |
| `paperwriting/supplementary_info.md` | 当前补充材料/方法与表格说明 |
| `PROJECT_CLEAR_MIND_20260413.md` | 论文策略和组会反馈处理口径 |
| `EXPERIMENT_SETTINGS_AND_RESULTS.md` | 本文档，实验设定和结果索引 |

LaTeX 编译命令：

```bash
cd paperwriting
pdflatex -interaction=nonstopmode gemini.tex
pdflatex -interaction=nonstopmode gemini.tex
```

最近一次已确认编译通过，输出 15 页 `paperwriting/gemini.pdf`。

---

## 7. 如果导师/审稿人问，推荐回答口径

### Q1: 你们主结果为什么 R2 只有 0.387？

推荐回答：

> This is a strict 40% homology-split evaluation, so the test set contains enzyme families substantially separated from training sequences. The model still preserves a transferable trend signal with Pearson r=0.716, but calibration analysis shows dynamic-range compression: extreme low-rate enzymes are over-predicted and extreme high-rate enzymes are under-predicted. Therefore R2 and MAE remain limited by the extreme kinetic ranges.

### Q2: DiffDock confidence 和误差没关系，是不是说明 docking 不重要？

推荐回答：

> No. DiffDock confidence is not experimental pose RMSD and should not be interpreted as direct docking quality. Our analysis only shows that, among successfully constructed pocket graphs, the DiffDock confidence score is not a reliable proxy for downstream kcat prediction error.

### Q3: Integrated Gradients 还能用吗？

推荐回答：

> We keep it only as exploratory attribution. Because node features mix one-hot categorical variables, counts, and continuous quantities, zero-baseline IG does not define a physically meaningful path for every feature. Quantitative mechanistic claims would require normalization-aware group perturbation or masking.

### Q4: EC5 为什么 Pearson 高但 R2/MAE 差？

推荐回答：

> This suggests trend preservation but poor calibration, likely amplified by small sample size and EC-specific distribution shift. We therefore describe EC-wise performance as heterogeneous rather than uniformly strong.

---

## 8. 下一步维护规则

每次新增实验或图表时，在本文档中更新三处：

1. 在第 3 节登记结果文件和核心数字。
2. 在第 4 节登记图表位置和论文使用状态。
3. 如果替代旧结果，在第 5 节把旧结果标记为 deprecated。

如果新实验改变主结果数字，必须同步更新：

- `paperwriting/gemini.tex`
- `paperwriting/supplementary_info.md`
- `PROJECT_CLEAR_MIND_20260413.md`
- `paper_freeze_202604/results_core/`
- `paper_freeze_202604/manifests/`
