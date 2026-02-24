# PocketGNN 实验记录

> 最后更新: 2026-02-24

## 重要提示

⚠️ **本文档区分验证集和测试集结果**。验证集指训练过程中的 val split，测试集指独立的 `kcat_test_new.pt` (1455样本)。

---

## 关键发现 (基于测试集)

1. **测试集最佳模型**: `kcat_esm_norm_fix` 达到 R²=0.4432, Pearson=0.6728 (ESM+set2set)
2. **ESM 融合在测试集上有帮助**: 与验证集结论相反，测试集上 ESM 融合模型优于纯 GNN
3. **泛化差距明显**: 验证集 R²≈0.77 → 测试集 R²≈0.44，表明存在过拟合或分布偏移
4. **CataPro baseline**: R²=0.4444, Pearson=0.6827 (与 PocketGNN 相当)

---

## 测试集结果 (kcat_test_new.pt, 1455 样本)

| # | 模型 | R² | Pearson | MAE | RMSE | 配置 |
|---|------|-----|---------|-----|------|------|
| 1 | kcat_esm_norm_fix | **0.4432** | **0.6728** | 0.8466 | 1.1798 | ESM+set2set, dropout=0.1 |
| 2 | kcat_20251213_151558 | 0.3787 | 0.6317 | 0.8878 | 1.2462 | 纯GNN, dropout=0.1 |
| 3 | kcat_esm_full | 0.3373 | 0.5809 | 0.9814 | 1.2870 | ESM+set2set |
| 4 | kcat_20251213_204244 | 0.3253 | 0.5812 | 0.9228 | 1.2987 | 纯GNN |

### 同源性划分测试 (kcat_merged_hom40_test.pt, 898 样本)

| 模型 | R² | Pearson | MAE | 配置 |
|------|-----|---------|-----|------|
| kcat_enhanced_run | 0.3872 | 0.7160 | 0.9005 | ESM+set2set, dropout=0.3 |

---

## 验证集结果 (仅供参考，非最终评估)

| # | 实验组 | Run | 数据集 | 配置 | R² | Pearson | 备注 |
|---|--------|-----|--------|------|-----|---------|------|
| 1 | kcat_attn_v1 | run_06 | kcat_full.pt | h128/l3/lr0.003 | 0.7664 | 0.8808 | 验证集最佳 |
| 2 | kcat_esm_full | run_01 | kcat_full_1213.pt | h128/l3/lr0.001+ESM | 0.6966 | - | ESM+set2set |
| 3 | kcat_attn_v1 | run_07 | kcat_full.pt | h128/l3/lr0.003 | 0.6359 | 0.8007 | 同run_06配置 |

---

## 测试集最佳模型详情

### #1: kcat_esm_norm_fix (测试集最佳)

```yaml
模型路径: outputs/kcat_esm_norm_fix/best_model.pt
测试集: data/processed/kcat_test_new.pt (1455 样本)
模型: PocketGNNKcatOnly

# 模型配置
hidden_dim: 128
num_layers: 3
heads: 4
dropout: 0.1
pooling_type: set2set
use_seq_embedding: true
seq_embedding_dim: 640

# 测试结果
R²: 0.4432
Pearson: 0.6728
MAE: 0.8466
RMSE: 1.1798
```

### #2: kcat_20251213_151558 (纯GNN)

```yaml
模型路径: outputs/kcat_20251213_151558/best_model.pt
测试集: data/processed/kcat_test_new.pt (1455 样本)
模型: PocketGNNKcatOnly

# 模型配置
hidden_dim: 128
num_layers: 3
heads: 4
dropout: 0.1
use_seq_embedding: false

# 测试结果
R²: 0.3787
Pearson: 0.6317
MAE: 0.8878
RMSE: 1.2462
```

---

## 与 Baseline 对比 (在 kcat_test_new.csv)

| 方法 | R² | Pearson | MAE | RMSE |
|------|-----|---------|-----|------|
| **PocketGNN (ESM)** | 0.4432 | 0.6728 | 0.8466 | 1.1798 |
| **CataPro** | 0.4444 | 0.6827 | 0.8920 | 1.1784 |
| PocketGNN (纯GNN) | 0.3787 | 0.6317 | 0.8878 | 1.2462 |

结论: PocketGNN 与 CataPro 在测试集上表现相当。

---

## 实验组说明

### kcat_esm_norm_fix (测试集最佳)
- 模型: PocketGNNKcatOnly + ESM-2 嵌入
- 特点: ESM embedding 归一化修复
- 测试集 R²: 0.4432

### kcat_attn_v1 (验证集最佳)
- 模型: PocketGNNKcatOnly
- 特点: 只用图特征，不用 ESM
- 验证集 R²: 0.7664 (但测试集表现较差)

### kcat_enhanced_run (同源性划分)
- 模型: PocketGNNKcatOnly + ESM-2
- 数据集: 40% 同源性划分
- 测试集 R²: 0.3872 (在 hom40 测试集上)

---

## 数据集说明

### PT 数据文件 (PyTorch Geometric)

| 文件名 | 大小 | 样本数 | 说明 |
|--------|------|--------|------|
| kcat_full.pt | ~1.5 GB | 9124 | 早期完整数据集 (最佳模型用此) |
| kcat_full_1213.pt | 1.6 GB | 9061 | 12月13日版本 |
| kcat_train_after_new_clean.pt | 922 MB | ~5000 | 清洗后训练集 |
| kcat_merged_hom40_train.pt | 1.5 GB | - | 40% 同源性划分训练集 |
| kcat_test_new.pt | 102 MB | 1455 | 测试集图数据 |
| esm_embeddings.pt | 24.7 MB | - | ESM-2 序列嵌入 |

### CSV 数据文件

| 文件名 | 样本数 | 说明 |
|--------|--------|------|
| kcat_full_1213.csv | 9061 | 完整训练数据 (原始格式) |
| kcat_test_new.csv | 1455 | 测试集 (无 SMILES) |
| kcat_test_results.csv | 3444 | 仅用于提供 SMILES |

---

## 模型配置说明

### PocketGNNKcatOnly (推荐)
- 节点特征: 52 维 (元素、残基、配体标志、距离、电子特征)
- 边特征: 24 维 (16-dim RBF + 4-dim bond angles + 4-dim dihedrals)
- 隐藏维度: 128
- GAT 层数: 3
- 注意力头数: 4
- Pooling: set2set (测试集最佳) / mean / global_attention

### 测试集最佳训练配置
```bash
python src/train.py \
  --dataset data/processed/kcat_full_1213.pt \
  --use_seq_embedding \
  --seq_embedding_path data/processed/esm_embeddings.pt \
  --hidden_dim 128 \
  --num_layers 3 \
  --heads 4 \
  --lr 0.001 \
  --dropout 0.1 \
  --pooling_type set2set \
  --save_dir outputs/your_exp_name
```

---

## 待改进

1. **泛化能力**: 验证集 R²=0.77 → 测试集 R²=0.44，需要减少过拟合
   - 更强的正则化 (dropout, weight_decay)
   - 数据增强
   - 更大的训练集

2. **与 CataPro 持平**: PocketGNN (R²=0.44) ≈ CataPro (R²=0.44)
   - 需要改进才能超越 baseline

3. **同源性泛化**: 40% 同源性划分下 R²=0.39，仍有提升空间

