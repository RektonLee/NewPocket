# Quantile Regression 使用指南

## 概述

Quantile Regression（分位数回归）是一种不确定性量化（Uncertainty Quantification）方法，可以直接预测置信区间，而不需要训练多个模型或使用复杂的贝叶斯方法。

## 使用方法

### 基本命令

```bash
python src/train.py \
  --dataset data/processed/kcat_full_1213.pt \
  --save_dir outputs/kcat_quantile \
  --exp_name kcat_quantile \
  --loss quantile \
  --quantiles 0.05,0.5,0.95 \
  --use_seq_embedding \
  --seq_embedding_path data/processed/esm_embeddings.pt
```

### 参数说明

- `--loss quantile`: 使用分位数回归损失函数
- `--quantiles 0.05,0.5,0.95`: 指定要预测的分位数
  - `0.05`: 5% 分位数（置信区间的下界）
  - `0.5`: 50% 分位数（中位数，点预测）
  - `0.95`: 95% 分位数（置信区间的上界）

### 输出

模型会输出3个值：
- `q_low`: 5% 分位数（置信区间下界）
- `q_median`: 50% 分位数（中位数预测）
- `q_high`: 95% 分位数（置信区间上界）

### 评估指标

训练过程中会记录：
- **Coverage**: 真实值落在预测区间 [q_low, q_high] 内的比例（理想情况下应该接近 90%）
- **Interval Width**: 预测区间的平均宽度（越小越好，但需要保证 coverage）
- **Median MAE**: 中位数预测的 MAE（点预测的准确性）

### 可视化

训练完成后会生成：
- `kcat_prediction_with_ci.png`: 带置信区间的散点图（阴影区域表示 95% 置信区间）
- `kcat_prediction_scatter.png`: 传统散点图（使用中位数预测）

## 优势

1. **实现简单**：只需修改 loss 函数，不需要训练多个模型
2. **直接给出置信区间**：不需要后处理
3. **计算高效**：训练时间与普通回归相同
4. **可解释性强**：分位数有明确的统计意义

## 注意事项

- Coverage 应该接近 `quantiles[-1] - quantiles[0]`（例如 0.95 - 0.05 = 0.9）
- 如果 coverage 太低，说明模型过于自信（区间太窄）
- 如果 coverage 太高，说明模型过于保守（区间太宽）
- 可以通过调整 quantile levels 来平衡 coverage 和 interval width












