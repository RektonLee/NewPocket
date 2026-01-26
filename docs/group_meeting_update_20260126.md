# 组会进展汇报

- 时间：2026-01-26
- 项目：PocketGNN / kcat 预测 + DiffDock 对接

## 近期完成的关键工作
- DiffDock 对接链路稳定化：改为严格按 confidence 选择 best pose，并补充完整的 pose 元数据（rank/confidence 对齐）。
- redock 流程优化：支持 GPU 轮询、实时进度日志、每 100 个样本保留前 5 个完整中间输出以便复盘。
- EC/离群点分析：从 test_predictions 反查 EC（用 raw kcat_data.csv 匹配），生成 EC 分组统计与离群点清单。

## DiffDock/对接链路改动（要点）
- best pose 选择逻辑：优先最高 confidence（符合 DiffDock 官方排序）。
- docking_meta.json：新增 pose_entries（rank/confidence/filename），confidence_values 与 sdf_files 一一对齐。
- redock 脚本：输出 [i/total] 成功/失败/GPU，保留 tmp 输出到 sample_data/.../diffdock_output/tmp_keep。

## 模型测试结果（样例）
- 示例：outputs/test_best_model_20251229_090906/test_metrics.csv
  - MAE 0.8466, RMSE 1.1798, R2 0.4432, Pearson 0.6728
  - 测试样本数：1455（见 test_metrics.json）

## 展示图
- 预测散点图：outputs/test_best_model_20251229_090906/test_kcat_prediction_scatter.png
- 残差分布：outputs/test_best_model_20251229_090906/test_residual_histogram.png
- 残差图：outputs/test_best_model_20251229_090906/test_residuals.png

## EC 分组分析（基于 kcat_test_new + raw kcat_data 反查 EC）
- 分析输出目录：outputs/analysis/20260125_164316
- 匹配统计：唯一匹配 1443 / 1455；歧义 10；未匹配 2

### EC 主类表现
- EC1: R2≈0.441（n=442）
- EC2: R2≈0.400（n=415）
- EC3: R2≈0.282（n=311）
- EC4: R2≈0.569（n=142）
- EC5: R2≈0.518（n=106）
- EC6: R2≈-0.087（n=26，明显偏差）

### EC 子类示例（level2/level3）
- 见 ec_summary_level2.csv / ec_summary_level3.csv（按 count 排序）

## 离群点（左上/右下）
- 左上（真实低/预测高）例：kcat_test_3013, kcat_test_1034 等
- 右下（真实高/预测低）例：kcat_test_0030, kcat_test_0075 等
- 详见 outliers_left_up.csv / outliers_right_down.csv

### 左上 Top5（真实低/预测高）
- kcat_test_3298 | EC 3.6.1.7 | true=-5.81, pred=2.39, abs_err=8.20
- kcat_test_3217 | EC 2.7.1.105 | true=-2.74, pred=2.23, abs_err=4.97
- kcat_test_3013 | EC 1.1.1.37 | true=-2.32, pred=2.43, abs_err=4.74
- kcat_test_1150 | EC 1.1.1.282 | true=-1.17, pred=2.71, abs_err=3.88
- kcat_test_1763 | EC 2.5.1.17 | true=-1.82, pred=1.93, abs_err=3.75

### 右下 Top5（真实高/预测低）
- kcat_test_0525 | EC 2.7.1.1 | true=3.28, pred=-0.89, abs_err=4.17
- kcat_test_0597 | EC 3.5.2.6 | true=3.04, pred=-0.70, abs_err=3.74
- kcat_test_0145 | EC 3.5.1.9 | true=1.70, pred=-1.57, abs_err=3.27
- kcat_test_1742 | EC 5.4.3.2 | true=1.92, pred=-1.32, abs_err=3.24
- kcat_test_2260 | EC 5.4.2.11 | true=2.52, pred=-0.62, abs_err=3.14

## 可复现实验命令（需要时）
- EC/离群点分析（带 EC 反查）：
  - python3 scripts/analyze_test_predictions_ec.py \
    --predictions outputs/test_best_model_20251229_090906/test_predictions.csv \
    --test-csv data/processed/kcat_test_new.csv \
    --raw-csv data/raw/kcat_data.csv
- 通用 EC/离群点分析（已有 EC 字段的表）：
  - python3 scripts/analyze_ec_outliers.py --input data/processed/kcat_test_model.csv

## 需要注意的局限
- EC 反查是基于 sequence + log10_kcat 四舍五入匹配，存在少量歧义/未匹配样本（已标记）。
- 若测试集补充 smiles 字段，可提高 EC 反查唯一性与可信度。

## 产出脚本
- scripts/analyze_ec_outliers.py：EC 分组与离群点分析（通用）
- scripts/analyze_test_predictions_ec.py：将 test_predictions 反查 EC + 统计

## 还在进行 / 下一步
- 若需要：把测试集补充 smiles 字段，提升 EC 反查的唯一性（减少歧义匹配）。
- 针对 EC6 等低效类别做专项改进（数据清理/采样/多模态权重调整）。

## WhatIDid 摘要
- `src/docking.py` 用 ETKDGv3 做 3D 嵌入并保留 fallback，提高 RDKit 构象成功率。
- 2026-01-25：小规模验证（limit=5）完成，成功 4/5；全量重对接已后台启动，日志：`logs/full_redock_test_new.log`（nohup 方式）。
- 2026-01-25: 修复 DiffDock 输出元数据的 confidence 解析：支持负号、按文件名排序并记录 confidence_map/best_confidence，避免之前只取正值且无法对应文件的问题。
- 2026-01-25: redock 脚本增加实时进度输出（处理/成功/失败/保留数量），并在每 100 个样本保留前 5 个的完整中间输出（DiffDock tmp 目录迁移到样本的 diffdock_output/tmp_keep）。
- 2026-01-25: redock 增加 GPU 轮询分配（--gpu-ids），进度日志中显示 GPU。
- 2026-01-25: DiffDock best pose 选择逻辑改为优先取最高 confidence 的 pose，只有缺失 confidence 时才回退到 rank1。
- 2026-01-25: redock 修复已有 PDB 同路径拷贝导致的 SameFileError。
- 2026-01-25: docking_meta.json 对齐 DiffDock 输出结构：解析 rank/confidence（含负号），按 rank 排序写入 pose_entries，并在 confidence_values 中保留 None；best_confidence 取最高置信。
- 2026-01-25: 新增 EC 分析与离群点脚本 scripts/analyze_ec_outliers.py；已生成 outputs/analysis/20260125_150015（EC分组/离群点CSV）。
- 2026-01-25: 新增 scripts/analyze_test_predictions_ec.py：从 test_predictions.csv + kcat_test_new.csv + raw kcat_data.csv 反查 EC，并输出 EC 分组与离群点分析（含唯一/歧义匹配统计）。
