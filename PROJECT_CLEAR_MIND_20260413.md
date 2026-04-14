# PocketGNN 项目与论文攻坚审计 2026-04-13

这份文件的目的不是继续堆结果，而是把当前项目收束成一套可投稿、可答辩、可补实验的判断。

## 0. 一句话判断

项目可以救，而且有清楚主线：**用自动 docking-to-pocket 流水线构建配体条件化的局部 3D 口袋图，再与全局序列语义融合，在 homology-aware split 下评估 kcat OOD 泛化。**

但论文现在不能继续依赖三类薄弱证据：

1. **EC-wise 旧图/旧表不能用**：旧图来自错误模型配置；另一套漂亮表格来自随机占位预测。
2. **Integrated Gradients 特征重要性不能作为强机制结论**：节点 one-hot/类别特征不适合用 zero-baseline IG 解释，现有 permutation 结果也显示加载/配置有问题。
3. **共同样本 baseline 显著性不能等价为 OOD 证明**：它是 shared-sample paired comparison，不是 40% homology split。

论文应该从“我模型全面机制性解释很强”收束为：

> PocketGNN 在严格同源性拆分下保留了可观趋势预测能力；在共享样本 benchmark 上优于现有 baseline；自动化结构预处理使 pocket-level 建模可规模化；进一步分析显示失败模式与 EC 类别、数据/结构质量有关。

## 0.1 组会三点反馈的最终处理口径

这轮论文攻坚采用 **稳健修稿优先**，不重训主模型，不临时改输入特征管线。

### 梯度积分

处理原则：**从定量机制证明降级为探索性分析**。

原因：

1. 节点输入混合了 one-hot 类别变量、电子层计数、原子质量、电负性、半径等不同类型特征。
2. zero-baseline Integrated Gradients 对 one-hot 类别变量没有明确物理插值路径。
3. 不同特征尺度会污染 raw gradient magnitude，因此不能把 IG 百分比当作严谨 feature importance。

论文执行：

1. 摘要和结论删除“某类几何特征占主导贡献”这类强定量表述。
2. 主文解释性只保留 qualitative case study。
3. Supplementary 中把 IG 标注为 exploratory diagnostic。

### 特征归一化

处理原则：**不作为当前训练 bug 修复，而作为 attribution limitation 处理**。

已核对事实：

1. one-hot 特征是 0/1。
2. 最近邻距离约 0.50-1.82。
3. 电子层计数约 0-6。
4. mass 约 12-32，electronegativity 约 2.55-3.44，radius 约 0.66-1.05。
5. edge RBF 大体在 0-1，angle/dihedral cosine 在 -1 到 1。

判断：

1. 训练和推理使用一致特征尺度，所以主性能结果内部一致。
2. 但 feature attribution 会受尺度影响，所以不能用 raw IG 做强机制结论。
3. future work 写 train-set-fitted feature standardization + group perturbation。

### 样本密度与误差

处理原则：**不解释为 confidence/error 的机制关系，而解释为 scatter density effect**。

已核对事实：

1. DiffDock confidence vs absolute error: Pearson r = 0.0267, p = 0.413。
2. n_nodes vs absolute error: Pearson r = -0.0450, p = 0.168。
3. n_edges vs absolute error: Pearson r = -0.0405, p = 0.214。
4. confidence decile 的 MAE/median/p90 没有单调趋势。

论文执行：

1. docking robustness 改写为 “DiffDock confidence is not a reliable downstream error proxy among successfully constructed graphs”。
2. 不写“docking 质量无关”这类过强结论。
3. Supplementary 加 density-aware decile 表，说明密集区域更容易看到大误差点是样本量效应。

## 1. 目前最可信的结果

### 1.1 主结果可信

来源：`results/test_hom40_evaluation/test_metrics.json`

| Setting | N | Pearson r | R2 | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| 40% homology test | 898 | 0.716 | 0.387 | 0.901 | 1.197 |

这个结果是论文最稳的锚点。R2 不高，但在 homology OOD 下可以讲成“趋势捕捉强，绝对误差仍有空间”。

### 1.2 共享样本 baseline 比较可用，但要降调

来源：`results/significance/significance_results.json`

| Comparison | N | PocketGNN r | Baseline r | PocketGNN MAE | Baseline MAE |
|---|---:|---:|---:|---:|---:|
| PocketGNN vs CatPred | 3439 | 0.864 | 0.608 | 0.479 | 0.942 |
| PocketGNN vs CataPro | 1455 | 0.868 | 0.683 | 0.487 | 0.892 |

这可以支持“同样本覆盖下优于 baseline”。但 `results/leakage_check/multi_threshold_leakage_report.json` 显示 `kcat_test_new` 有 73.9% 样本在 40% identity 阈值下有训练命中，所以不能把这组结果说成严格 OOD。

### 1.3 calibration/bias 诊断可用

来源：`results/bias_calibration/`

这一步不重训模型，只分析 `results/test_hom40_evaluation/test_predictions.csv`。

关键事实：

1. 全局 bias(pred-true) = -0.149，MAE = 0.901，RMSE = 1.197。
2. 真实 `log10(kcat)` 范围是 -5.52 到 6.03，但预测范围只有 -1.30 到 1.98。
3. 最低 true-kcat decile 被高估约 +1.95 log units。
4. 最高 true-kcat decile 被低估约 -2.04 log units。
5. 中间 true-kcat 区间校准最好，MAE 约 0.24-0.50。

论文口径：

> The model preserves transferable trend signal under homology split, but compresses the kinetic dynamic range. This regression-to-the-mean behavior explains why Pearson remains useful while R2/MAE are limited by extreme low- and high-rate enzymes.

这个分析可以支撑一个更 solid 的叙事：模型不是随机错，而是在 family-level OOD 下保守预测极端 kcat。

### 1.4 docking confidence 结果可用，但要换解释

来源：`results/docking_robustness/docking_robustness.json`

DiffDock confidence 与绝对误差几乎无相关：

| Variable | Pearson r | p |
|---|---:|---:|
| confidence vs abs error | 0.0267 | 0.413 |

这不是坏结果。正确表述是：

> 在成功构图的样本中，DiffDock confidence score 不是下游 kcat 误差的有效线性代理；模型性能在 confidence tertile 上相对稳定。

不要说成 docking 质量对任务无关。

因为 confidence 不等于真实 RMSD，也不等于 pocket 生物学正确性。

## 2. 必须修正或撤下的结果

### 2.1 EC-wise 图 A：旧图不能用

旧图：`results/ec_analysis/figS_ec_wise_performance.png`

生成脚本：`scripts/quick_ec_analysis.py`

问题：

1. 脚本强行用 `outputs/kcat_hom40_train_20251213_221248/best_model.pt`。
2. 脚本强行设定 `pooling_type='mean'` 和 `use_seq_embedding=False`。
3. 论文主结果模型是 `outputs/kcat_enhanced_run/best_model.pt`，配置是 `set2set + ESM`。

所以图 A 里 Pearson r 为负，不应解释为科学现象。它主要是错误模型/配置结果。

### 2.2 EC-wise 漂亮表格也不能用

旧结果：`results/ec_wise_analysis/ec_performance.json`

生成脚本：`scripts/analyze_ec_performance_simple.py`

问题：

1. 脚本读取 `results/test_final_evaluation/test_predictions.csv`。
2. 该文件 940 行，而 hom40 test dataset 是 898 行。
3. 脚本在长度不一致时直接使用 `y_true + random noise` 作为 placeholder prediction。

所以这套 EC 结果里 0.9+ 的 Pearson 基本是占位预测造成的，不能进论文。

### 2.3 已补正确 EC 分析

新增脚本：

`scripts/analyze_ec_performance_from_predictions.py`

新增输出：

`results/ec_analysis_corrected/ec_wise_performance.csv`

`results/ec_analysis_corrected/fig_ec_wise_performance_corrected.png`

正确结果如下：

| EC | N | Pearson r | R2 | MAE |
|---|---:|---:|---:|---:|
| EC 1 | 703 | 0.667 | 0.350 | 0.900 |
| EC 2 | 67 | 0.773 | 0.503 | 0.559 |
| EC 3 | 36 | 0.885 | 0.376 | 0.932 |
| EC 4 | 42 | 0.813 | 0.422 | 1.047 |
| EC 5 | 31 | 0.843 | 0.059 | 1.383 |
| EC 6 | 18 | 0.973 | 0.615 | 0.932 |
| Overall | 898 | 0.716 | 0.387 | 0.901 |

论文建议表述：

> Model behavior varies across EC classes. EC 2 and EC 6 show relatively favorable performance, whereas EC 5 has high rank/trend correlation but poor R2/MAE, suggesting systematic calibration bias or limited sample support.

不要再写：

> six EC classes all show very strong generalization.

### 2.4 Error analysis 也要重做

旧脚本：`scripts/analyze_error_correlations.py`

问题：

1. 默认找 `results/test_predictions/test_predictions.csv`，该文件不存在。
2. fallback 到 `results/tsne_analysis/tsne_embeddings.csv` 的 `prediction` 列。
3. 这个 prediction 接近常数，和主结果 `results/test_hom40_evaluation/test_predictions.csv` 不是一回事。

所以 `results/error_analysis/*` 当前不应进论文。应按 `results/test_hom40_evaluation/test_predictions.csv` 重跑。

## 3. 可解释性该怎么处理

### 3.1 当前 IG 结论不能作为强 claim

旧脚本：`scripts/analyze_feature_importance.py`

旧结论曾把角度/二面角通道写成主导性定量贡献。

问题分两层。

第一，方法论问题：

1. 节点特征里大量是 one-hot 类别变量，如元素、残基、电子构型分桶。
2. 从 zero baseline 到 one-hot 的连续积分路径没有清楚物理意义。
3. `mass`、`electronegativity` 这类连续特征的 IG 会受量纲/尺度影响。

第二，实现/结果问题：

1. `results/ablation_edge_features.txt` 显示快速消融中的 full model Pearson 只有 0.0072。
2. `feature_importance_summary.json` 里的 permutation baseline correlation 也是 0.0072。
3. 这说明解释性脚本加载的模型/数据配置不是主结果配置。

### 3.2 论文中建议替代说法

可以保留的弱表述：

> Preliminary attribution suggested sensitivity to angular edge channels, motivating a more rigorous group-ablation analysis.

不建议保留“角度特征占据绝对主导贡献”这类百分比式机制结论。

除非重做以下实验。

### 3.3 最小可行重做方案

优先级从高到低：

1. **Group permutation / group masking on the actual main model**：使用 `outputs/kcat_enhanced_run/best_model.pt`，按主结果测试流程加载 `set2set + ESM`，分别扰动 RBF、angle、dihedral、node continuous、node categorical groups。
2. **Retrain ablation**：RBF-only、RBF+angle、full edge、no ESM，每个至少 3 seeds。如果时间不够，至少 1 seed 并明确为 single-run ablation。
3. **Case study**：只做 1-2 个已知催化残基案例，用 Input x Gradient 或 attention/saliency 可视化，作为 qualitative supporting evidence，不作为定量机制结论。

## 4. 论文 novelty 与 significance 应该怎么讲

### 4.1 最稳 novelty

1. **Ligand-conditioned pocket graph for enzyme kcat**：不是全蛋白结构，也不是纯序列；是 substrate-docked local pocket。
2. **Automated blind docking-to-pocket preprocessing**：不需要手工 binding site annotation，适合大规模 enzyme-substrate pair。
3. **Local geometry + global pLM fusion**：局部 3D 约束和全局进化语义互补。
4. **Leakage-aware evaluation**：把 random split 当 upper bound，把 40% homology split 当主评估。
5. **Paired shared-sample baseline comparison**：不拿不同样本集合的指标硬比。

### 4.2 最稳 significance

1. 说明 sequence-family prior 很强，random split 和共享样本 benchmark 可能乐观。
2. 即使在 40% homology split 下，模型仍达到 Pearson 0.716，说明有可迁移信号。
3. 自动结构流水线让 structure-aware enzyme kinetics 从小规模人工分析变成可规模化预测。
4. 失败分析可以诚实展示任务难度：R2 中等、EC5 偏差大、绝对误差仍受数据异质性影响。

### 4.3 应主动降调的 claim

1. 不说“solves kcat prediction”。
2. 不说 docking 质量对任务无关。
3. 不说 IG 已经证明催化机制。
4. 不说所有 EC 类别都强泛化。
5. 不把 shared-sample comparison 说成严格同源外推 baseline comparison。

## 5. 建议论文结果结构

主文结果建议这样排：

1. **Main OOD performance**：40% homology split，主散点图。
2. **Baseline comparison**：shared-sample paired comparison，明确不是严格 homology split。
3. **Ablation / component validation**：只有在重做后保留；否则降为 supplementary 或删。
4. **EC-wise and failure analysis**：用 corrected EC，重点讲 heterogeneity。
5. **Docking robustness**：confidence score 不是 error proxy，pipeline 在 successful graph subset 上稳定。
6. **Qualitative interpretability**：案例级，不做百分比式强机制结论。

## 6. 需要补的最小实验/图表

### 必做

1. **替换 EC 图**
   - 已生成：`results/ec_analysis_corrected/fig_ec_wise_performance_corrected.png`
   - 改论文表格和 supplementary。

2. **重做 error analysis**
   - 输入必须是 `results/test_hom40_evaluation/test_predictions.csv`。
   - 输出 MAE 应该回到 0.900 左右，而不是旧的 1.264。

3. **重做 feature importance**
   - 用 group permutation/masking。
   - 不再用 node zero-baseline IG 作为主解释。

### 强烈建议

1. **补 calibration / bias plot**
   - 按 true kcat bins 看 prediction bias。
   - 解释为什么 Pearson 高但 R2/MAE 不够好。

2. **补 EC5 failure note**
   - EC5 Pearson 0.843 但 R2 0.059、MAE 1.383。
   - 这比“全都很好”更可信。

3. **补 leakage-aware baseline note**
   - 把 KNN/homology leakage 结果放 supplement。
   - 证明你不是不知道 family prior，而是主动控制和讨论它。

## 7. 项目整理建议

不要现在把 75G 项目整体迁移到新目录。论文攻坚期最稳的方法是 **freeze 一个 paper package**，只放论文需要的源文件、图、结果表、脚本清单。

建议新建：

```text
paper_freeze_202604/
  README.md
  manuscript/
    gemini.tex
    supplementary_info.md
  figures/
    main/
    supplementary/
  results_core/
    hom40_metrics.json
    hom40_predictions.csv
    baseline_significance.json
    ec_wise_corrected.csv
    docking_robustness.json
  scripts_reproduce/
    analyze_ec_performance_from_predictions.py
    analyze_significance.py
    analyze_docking_robustness.py
    test.py command notes
  manifests/
    data_manifest.md
    model_manifest.md
    figure_manifest.md
```

根目录整理建议：

1. `docs/archive_reports/`：放旧组会、morning、goodnight、status 类 md。
2. `archives/`：放 `*.tar.gz`。
3. `cache/`：放 `.p.npy`、`.score.npy`、`.so3*.npy`。
4. `results/`：只保留最终可引用结果；旧结果加 `deprecated_` 或放 `results/archive/`。
5. `scripts/`：最终只保留可复现实验脚本；debug/quick 脚本放 `scripts/archive/`。

## 8. 当前论文需要立即改的地方

文件：`paperwriting/gemini.tex`

建议改动：

1. Abstract 中删除或降调“某类几何特征贡献占主导”的强定量表述。
2. Contributions 第 3 点从 “attribution analyses show” 改成 “ablation/group-sensitivity analyses evaluate”。
3. EC-wise section 用 corrected EC 表。
4. Interpretability section 删除定量 IG 百分比，改为“preliminary group sensitivity / qualitative case study”。
5. Limitations 加一句：feature attribution for mixed categorical/continuous graph features is nontrivial; therefore quantitative mechanistic claims are based on ablation/group perturbation rather than raw node-level IG。

## 9. 最后给自己的判断

这个项目的价值不在“每张图都漂亮”。真正可站住的是：

> 在 kcat 这种噪声大、同源泄漏强、结构输入不完美的任务上，PocketGNN 给出了一条可规模化的 structure-aware 路径，并且在严格 homology split 下仍保留了可观预测趋势。

这已经足够作为论文主线。现在要做的是把不稳的解释性和错误脚本清掉，把故事从“过度证明机制”改成“诚实证明方法可行、优势边界清楚、下一步明确”。
