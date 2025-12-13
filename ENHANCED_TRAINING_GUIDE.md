# 增强版训练指南

本文档说明如何使用增强版的训练系统，包含正则化改进、模型结构优化和序列嵌入融合。

## 🎯 改进概览

### 阶段一：训练与正则化改进
- ✅ Weight Decay (L2正则化)
- ✅ 可调整的 Dropout
- ✅ Huber Loss (鲁棒损失函数)
- ✅ Learning Rate Scheduler

### 阶段二：模型结构改进
- ✅ 可切换的 Pooling 方式（mean / attention / set2set）
- ✅ GlobalAttention (带可学习的 gate 网络)

### 阶段三：ESM-2 序列嵌入融合
- ✅ Late Fusion 方式
- ✅ 离线嵌入提取
- ✅ 向后兼容

---

## 📖 使用方法

### 1. 基础训练（保持原有行为）

```bash
python train.py \
    --dataset kcat_train_after_new_clean.pt \
    --save_dir outputs/baseline
```

**说明**: 这将使用默认参数，行为与原始版本完全一致。

---

### 2. 启用正则化改进

#### 2.1 添加 Weight Decay

```bash
python train.py \
    --dataset kcat_train_after_new_clean.pt \
    --save_dir outputs/with_weight_decay \
    --weight_decay 1e-4
```

**推荐值**: `1e-4` 到 `1e-3`

---

#### 2.2 增加 Dropout

```bash
python train.py \
    --dataset kcat_train_after_new_clean.pt \
    --save_dir outputs/with_dropout \
    --dropout 0.3
```

**推荐值**: `0.3` 到 `0.4` (原始为 `0.1`)

---

#### 2.3 使用 Huber Loss

```bash
python train.py \
    --dataset kcat_train_after_new_clean.pt \
    --save_dir outputs/with_huber \
    --loss huber
```

**说明**: Huber Loss 对离群点更鲁棒，适合处理实验噪声。

---

#### 2.4 添加 Learning Rate Scheduler

```bash
# 使用 ReduceLROnPlateau（推荐）
python train.py \
    --dataset kcat_train_after_new_clean.pt \
    --save_dir outputs/with_plateau_scheduler \
    --scheduler plateau

# 使用 CosineAnnealing（可选）
python train.py \
    --dataset kcat_train_after_new_clean.pt \
    --save_dir outputs/with_cosine_scheduler \
    --scheduler cosine
```

---

### 3. 改进模型结构

#### 3.1 使用 GlobalAttention Pooling

```bash
python train.py \
    --dataset kcat_train_after_new_clean.pt \
    --save_dir outputs/with_attention_pooling \
    --pooling_type attention
```

**说明**: GlobalAttention 能够学习哪些节点对最终预测更重要，通常优于简单的平均池化。

---

### 4. 使用 ESM-2 序列嵌入（Late Fusion）

#### 步骤 1: 生成序列嵌入

```bash
python generate_esm_embeddings.py \
    --dataset kcat_train_after_new_clean.pt \
    --output data/esm_embeddings_650M.pt \
    --model esm2_t33_650M_UR50D
```

**注意**: 
- 需要先安装 ESM: `pip install fair-esm`
- 首次运行会自动下载模型（约 2.5GB）
- 生成嵌入可能需要较长时间（取决于数据集大小）

**可选模型**:
- `esm2_t33_650M_UR50D` (推荐，性能最好)
- `esm2_t33_150M` (较快，性能略低)
- `esm2_t30_150M_UR50D` (最快，性能最低)

#### 步骤 2: 使用嵌入训练

```bash
python train.py \
    --dataset kcat_train_after_new_clean.pt \
    --save_dir outputs/with_esm_embeddings \
    --use_seq_embedding \
    --seq_embedding_path data/esm_embeddings_650M.pt
```

---

### 5. 组合所有改进（推荐配置）

```bash
# 步骤 1: 生成 ESM 嵌入（如果还没有）
python generate_esm_embeddings.py \
    --dataset kcat_train_after_new_clean.pt \
    --output data/esm_embeddings.pt

# 步骤 2: 使用所有增强功能训练
python train.py \
    --dataset kcat_train_after_new_clean.pt \
    --save_dir outputs/full_enhanced \
    --weight_decay 1e-4 \
    --dropout 0.3 \
    --loss huber \
    --scheduler plateau \
    --pooling_type attention \
    --use_seq_embedding \
    --seq_embedding_path data/esm_embeddings.pt \
    --lr 1e-3 \
    --batch_size 32 \
    --max_epochs 500
```

---

## 🔧 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | str | 必需 | PyG 图数据集路径 |
| `--save_dir` | str | 必需 | 输出目录 |
| `--batch_size` | int | 32 | 批次大小 |
| `--lr` | float | 1e-3 | 学习率 |
| `--max_epochs` | int | 500 | 最大训练轮数 |
| `--weight_decay` | float | 0.0 | L2正则化系数 |
| `--dropout` | float | 0.1 | Dropout概率 |
| `--loss` | str | 'mse' | 损失函数 ('mse' 或 'huber') |
| `--scheduler` | str | None | 学习率调度器 (None, 'plateau', 'cosine') |
| `--pooling_type` | str | 'mean' | 池化方式 ('mean', 'attention', 'set2set') |
| `--use_seq_embedding` | flag | False | 是否使用序列嵌入 |
| `--seq_embedding_path` | str | None | 序列嵌入文件路径 |

---

## 📊 实验建议

### 消融实验（Ablation Study）

建议按以下顺序进行实验，以验证每个改进的贡献：

1. **Baseline**: 原始配置
   ```bash
   python train.py --dataset data.pt --save_dir exp1_baseline
   ```

2. **+ Weight Decay**:
   ```bash
   python train.py --dataset data.pt --save_dir exp2_wd --weight_decay 1e-4
   ```

3. **+ Dropout**:
   ```bash
   python train.py --dataset data.pt --save_dir exp3_wd_dropout --weight_decay 1e-4 --dropout 0.3
   ```

4. **+ Huber Loss**:
   ```bash
   python train.py --dataset data.pt --save_dir exp4_wd_dropout_huber \
       --weight_decay 1e-4 --dropout 0.3 --loss huber
   ```

5. **+ Scheduler**:
   ```bash
   python train.py --dataset data.pt --save_dir exp5_wd_dropout_huber_scheduler \
       --weight_decay 1e-4 --dropout 0.3 --loss huber --scheduler plateau
   ```

6. **+ Attention Pooling**:
   ```bash
   python train.py --dataset data.pt --save_dir exp6_full_no_esm \
       --weight_decay 1e-4 --dropout 0.3 --loss huber --scheduler plateau \
       --pooling_type attention
   ```

7. **+ ESM Embeddings**:
   ```bash
   python train.py --dataset data.pt --save_dir exp7_full_with_esm \
       --weight_decay 1e-4 --dropout 0.3 --loss huber --scheduler plateau \
       --pooling_type attention --use_seq_embedding --seq_embedding_path data/esm_emb.pt
   ```

---

## ✅ 向后兼容性

所有改进都是**向后兼容**的：

- ✅ 默认参数保持原有行为不变
- ✅ 可以逐个启用新功能
- ✅ 旧模型检查点仍可加载（如果不使用新功能）
- ✅ 数据集格式无需修改

**注意**: 使用序列嵌入时，模型结构会发生变化，旧的检查点将不兼容。

---

## 🚫 已避免的风险

按照设计原则，以下风险已被规避：

- ❌ 不会破坏现有训练流程
- ❌ 不会强制使用 homology split
- ❌ 不会引入 residue-level 图对齐
- ❌ 不会进行 end-to-end ESM 微调
- ❌ 不会在模型中隐式使用 test 信息

---

## 📝 文件清单

修改的文件：

1. `train.py` - 增强的训练脚本
2. `GNN_model.py` - 改进的模型定义

新增的文件：

3. `generate_esm_embeddings.py` - ESM-2 嵌入生成工具
4. `ENHANCED_TRAINING_GUIDE.md` - 本文档

---

## 🐛 故障排除

### 问题 1: CUDA Out of Memory (OOM)

**解决方案**:
- 减小 batch_size: `--batch_size 16`
- 使用较小的 ESM 模型: `--model esm2_t33_150M`
- 生成嵌入时使用 CPU: 在 `generate_esm_embeddings.py` 中修改 `device`

### 问题 2: 训练不稳定 / NaN Loss

**解决方案**:
- 降低学习率: `--lr 5e-4`
- 使用梯度裁剪（已默认启用）
- 检查数据中是否有异常值

### 问题 3: 序列嵌入文件不存在

**解决方案**:
- 确保先运行 `generate_esm_embeddings.py`
- 检查文件路径是否正确
- 如果不使用序列嵌入，移除 `--use_seq_embedding` 参数

---

## 💡 最佳实践

1. **从简单到复杂**: 先使用基础改进（weight decay, dropout），再尝试复杂功能（ESM 嵌入）
2. **记录实验**: 每次实验使用不同的 `save_dir`，便于对比
3. **监控指标**: 关注 TensorBoard 中的 Loss、R²、Pearson 等指标
4. **早停**: 如果验证集性能不再提升，可以提前终止训练
5. **超参数搜索**: 使用网格搜索或贝叶斯优化寻找最佳超参数

---

## 📧 反馈

如有问题或建议，请查看项目文档或联系开发者。
