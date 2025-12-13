# 测试指南

## 🧪 如何测试增强版代码

根据你的项目结构和数据格式，这里提供完整的测试方案。

---

## 📋 前提检查

### 1. 确认数据文件位置

根据你的 README，数据应该在：
```
data/processed/kcat_train.pt
data/processed/kcat_test_new.pt
```

数据格式（已确认）：
```python
Data(x=[206, 52], edge_index=[2, 1800], edge_attr=[1800, 24], 
     pos=[206, 3], temperature=[1], y=[1], 
     pdb_id='kcat_test_0004_61714_10A.pdb', 
     sample_id='kcat_test_0004', ec=1)
```

### 2. 检查文件是否存在

```bash
# 确认增强文件已创建
ls -lh train.py GNN_model.py generate_esm_embeddings.py

# 确认数据文件
ls -lh data/processed/kcat_train*.pt
```

---

## 🚀 测试方案

### 测试 1: 向后兼容性测试（最重要）

**目的**: 验证使用默认参数时，行为与原始版本一致。

```bash
# 1. 基础训练测试（10个 epoch 快速验证）
python train.py \
    --dataset data/processed/kcat_train.pt \
    --save_dir results/test_baseline \
    --batch_size 32 \
    --lr 1e-3 \
    --max_epochs 10
```

**预期结果**:
- ✅ 训练正常启动
- ✅ 无错误信息
- ✅ 输出类似: `Epoch 001 | Train Loss: X.XXXX | Val Loss: X.XXXX | R2: X.XXX | LR: 1.00e-03`
- ✅ 在 `results/test_baseline/` 生成文件

---

### 测试 2: 正则化改进测试

**目的**: 测试 Weight Decay + Dropout

```bash
python train.py \
    --dataset data/processed/kcat_train.pt \
    --save_dir results/test_regularization \
    --weight_decay 1e-4 \
    --dropout 0.3 \
    --max_epochs 10
```

**预期结果**:
- ✅ 看到日志: `Weight Decay: 0.0001`
- ✅ 看到日志: `Dropout: 0.3`
- ✅ 训练正常完成

---

### 测试 3: Huber Loss 测试

**目的**: 测试鲁棒损失函数

```bash
python train.py \
    --dataset data/processed/kcat_train.pt \
    --save_dir results/test_huber \
    --loss huber \
    --max_epochs 10
```

**预期结果**:
- ✅ 看到日志: `✅ 使用 Huber Loss（对离群点更鲁棒）`
- ✅ 训练正常完成

---

### 测试 4: Learning Rate Scheduler 测试

**目的**: 测试学习率调度器

```bash
python train.py \
    --dataset data/processed/kcat_train.pt \
    --save_dir results/test_scheduler \
    --scheduler plateau \
    --max_epochs 20
```

**预期结果**:
- ✅ 看到日志: `✅ 使用 ReduceLROnPlateau 调度器`
- ✅ 学习率在验证损失停止下降时降低
- ✅ 日志中 LR 会变化

---

### 测试 5: GlobalAttention Pooling 测试

**目的**: 测试改进的 pooling 方式

```bash
python train.py \
    --dataset data/processed/kcat_train.pt \
    --save_dir results/test_attention_pooling \
    --pooling_type attention \
    --max_epochs 10
```

**预期结果**:
- ✅ 看到日志: `Pooling Type: attention`
- ✅ 模型参数数量略有增加（gate 网络）
- ✅ 训练正常完成

---

### 测试 6: 组合测试（推荐）

**目的**: 测试多个改进同时启用

```bash
python train.py \
    --dataset data/processed/kcat_train.pt \
    --save_dir results/test_combined \
    --weight_decay 1e-4 \
    --dropout 0.3 \
    --loss huber \
    --scheduler plateau \
    --pooling_type attention \
    --max_epochs 20
```

**预期结果**:
- ✅ 所有配置正确显示在日志中
- ✅ 训练稳定
- ✅ 性能可能优于基线

---

### 测试 7: ESM-2 序列嵌入测试（需要准备数据）

#### 步骤 7.1: 检查数据是否包含序列信息

```bash
python -c "
import torch
data_list = torch.load('data/processed/kcat_train.pt', weights_only=False)
first_sample = data_list[0]
print('数据字段:', dir(first_sample))
print('是否有 sequence:', hasattr(first_sample, 'sequence'))
print('是否有 protein_sequence:', hasattr(first_sample, 'protein_sequence'))
"
```

#### 步骤 7.2: 如果有序列信息，生成 ESM 嵌入

**注意**: 这一步需要安装 `fair-esm` 和足够的 GPU 内存。

```bash
# 安装 ESM（如果还没有）
pip install fair-esm

# 生成嵌入（小数据集测试，使用较小的模型）
python generate_esm_embeddings.py \
    --dataset data/processed/kcat_train.pt \
    --output data/processed/esm_embeddings_test.pt \
    --model esm2_t33_150M
```

**注意事项**:
- 第一次运行会下载模型（~600MB for 150M, ~2.5GB for 650M）
- 如果数据没有序列信息，会使用零向量
- 生成时间取决于数据集大小

#### 步骤 7.3: 使用嵌入训练

```bash
python train.py \
    --dataset data/processed/kcat_train.pt \
    --save_dir results/test_with_esm \
    --use_seq_embedding \
    --seq_embedding_path data/processed/esm_embeddings_test.pt \
    --max_epochs 10
```

**预期结果**:
- ✅ 看到日志: `✅ 加载序列嵌入: data/processed/esm_embeddings_test.pt`
- ✅ 看到日志: `Use Seq Embedding: True`
- ✅ 模型参数数量增加（序列投影层）

---

## 🔍 验证测试结果

### 检查输出文件

每次训练后，应该在 `results/test_xxx/` 目录下看到：

```
results/test_xxx/
├── best_model.pt              # 最佳模型权重
├── loss_curve.png             # 损失曲线
├── metrics_curve.png          # R²和Pearson曲线
├── kcat_prediction_scatter.png # 预测vs真实散点图
├── kcat_density.png           # 密度图
├── training_metrics.csv       # 训练指标
└── events.out.tfevents.*      # TensorBoard日志
```

### 查看 TensorBoard

```bash
tensorboard --logdir results/
```

然后在浏览器打开 `http://localhost:6006`

### 比较性能

```bash
# 提取各个实验的最终 R²
python -c "
import pandas as pd
import os

experiments = [
    'test_baseline',
    'test_regularization', 
    'test_attention_pooling',
    'test_combined'
]

results = []
for exp in experiments:
    csv_path = f'results/{exp}/training_metrics.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        best_r2 = df['R2'].max()
        best_epoch = df['R2'].idxmax() + 1
        results.append({
            'Experiment': exp,
            'Best R²': f'{best_r2:.4f}',
            'Best Epoch': best_epoch
        })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
"
```

---

## 🐛 故障排除

### 问题 1: 找不到数据文件

**错误**: `FileNotFoundError: 数据文件不存在`

**解决**:
```bash
# 检查数据路径
ls data/processed/*.pt

# 如果路径不同，修改 --dataset 参数
python train.py --dataset 你的实际路径/kcat_train.pt ...
```

---

### 问题 2: CUDA Out of Memory

**错误**: `RuntimeError: CUDA out of memory`

**解决**:
```bash
# 减小 batch_size
python train.py --dataset ... --batch_size 16

# 或使用 CPU
python train.py --dataset ... --device cpu
```

注意：当前代码默认使用 `cuda:1`，如果你只有一个GPU，需要修改 `train.py` 中的：
```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```

---

### 问题 3: 模块导入错误

**错误**: `ModuleNotFoundError: No module named 'XXX'`

**解决**:
```bash
# 安装缺失的依赖
pip install torch torch-geometric torch-scatter torch-sparse
pip install tensorboard matplotlib seaborn pandas scipy scikit-learn

# ESM 相关（仅在使用序列嵌入时需要）
pip install fair-esm
```

---

### 问题 4: 数据格式不匹配

**错误**: `AttributeError: 'Data' object has no attribute 'sequence'`

**解决**: 
这是正常的，如果数据中没有序列信息，`generate_esm_embeddings.py` 会使用零向量。不影响其他功能的测试。

---

### 问题 5: 训练时出现 NaN

**错误**: `ValueError: 模型输出包含 NaN 值`

**可能原因**:
1. 学习率过大
2. 数据中有异常值
3. 梯度爆炸

**解决**:
```bash
# 降低学习率
python train.py --dataset ... --lr 5e-4

# 检查数据
python -c "
import torch
data_list = torch.load('你的数据.pt', weights_only=False)
for i, data in enumerate(data_list):
    if torch.isnan(data.x).any() or torch.isnan(data.y).any():
        print(f'样本 {i} 包含 NaN')
"
```

---

## ✅ 最小测试清单

如果时间有限，至少运行这些测试：

1. **基线测试** (必须通过)
   ```bash
   python train.py --dataset data/processed/kcat_train.pt \
       --save_dir results/quick_test --max_epochs 5
   ```

2. **正则化测试** (验证新功能)
   ```bash
   python train.py --dataset data/processed/kcat_train.pt \
       --save_dir results/quick_reg --weight_decay 1e-4 --dropout 0.3 --max_epochs 5
   ```

3. **Attention Pooling 测试** (验证模型改进)
   ```bash
   python train.py --dataset data/processed/kcat_train.pt \
       --save_dir results/quick_att --pooling_type attention --max_epochs 5
   ```

**预期**: 全部通过，无错误，性能相近或更好。

---

## 📊 完整实验方案（如果要发论文）

### 消融实验（Ablation Study）

```bash
# 实验1: 基线
python train.py --dataset data/processed/kcat_train.pt \
    --save_dir results/ablation/exp1_baseline --max_epochs 200

# 实验2: + Weight Decay
python train.py --dataset data/processed/kcat_train.pt \
    --save_dir results/ablation/exp2_wd --weight_decay 1e-4 --max_epochs 200

# 实验3: + Dropout
python train.py --dataset data/processed/kcat_train.pt \
    --save_dir results/ablation/exp3_wd_dropout \
    --weight_decay 1e-4 --dropout 0.3 --max_epochs 200

# 实验4: + Huber Loss
python train.py --dataset data/processed/kcat_train.pt \
    --save_dir results/ablation/exp4_wd_dropout_huber \
    --weight_decay 1e-4 --dropout 0.3 --loss huber --max_epochs 200

# 实验5: + Scheduler
python train.py --dataset data/processed/kcat_train.pt \
    --save_dir results/ablation/exp5_full_no_att \
    --weight_decay 1e-4 --dropout 0.3 --loss huber --scheduler plateau --max_epochs 200

# 实验6: + Attention Pooling
python train.py --dataset data/processed/kcat_train.pt \
    --save_dir results/ablation/exp6_full_no_esm \
    --weight_decay 1e-4 --dropout 0.3 --loss huber --scheduler plateau \
    --pooling_type attention --max_epochs 200

# 实验7: + ESM Embeddings（如果有序列数据）
# 先生成嵌入...
python train.py --dataset data/processed/kcat_train.pt \
    --save_dir results/ablation/exp7_full_with_esm \
    --weight_decay 1e-4 --dropout 0.3 --loss huber --scheduler plateau \
    --pooling_type attention --use_seq_embedding \
    --seq_embedding_path data/processed/esm_embeddings.pt --max_epochs 200
```

---

## 🎯 成功标准

### 向后兼容性
- ✅ 基线测试（默认参数）必须成功运行
- ✅ 性能与原始版本相近（R² 差异 < 5%）

### 新功能
- ✅ 所有新参数都能正常工作
- ✅ 不同配置下训练都能完成
- ✅ 输出文件正确生成

### 性能提升（可选）
- 📈 正则化改进: R² 提升 5-10%
- 📈 Attention Pooling: R² 提升 5-10%
- 📈 ESM Embeddings: R² 提升 10-20%

---

## 📝 测试记录模板

```markdown
## 测试日期: YYYY-MM-DD

### 环境信息
- Python版本: 
- PyTorch版本: 
- CUDA版本: 
- GPU型号: 

### 测试结果

| 实验 | 状态 | 最佳 R² | 最佳 Epoch | 备注 |
|------|------|---------|------------|------|
| 基线 | ✅ | 0.XXX | XX | |
| 正则化 | ✅ | 0.XXX | XX | |
| Attention | ✅ | 0.XXX | XX | |
| 组合 | ✅ | 0.XXX | XX | |

### 问题记录
- 无

### 结论
- 向后兼容性: ✅ 通过
- 新功能可用性: ✅ 通过
- 性能提升: 📈 提升 X%
```

---

## 📞 需要帮助？

如果遇到问题：
1. 查看上面的"故障排除"部分
2. 检查错误信息并搜索解决方案
3. 查看 `ENHANCED_TRAINING_GUIDE.md` 详细文档
4. 查看代码注释

---

**祝测试顺利！** 🚀
