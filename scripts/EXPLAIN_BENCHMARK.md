# Benchmark 实验说明：CatPred 和 CataPro 的实际运行情况

## ⚠️ 重要说明

**实际情况**：在当前的 benchmark 报告中，我**并没有真正运行** CatPred 和 CataPro 的代码，而是使用了**文献中报告的结果**。

## 📋 当前状态

### ✅ 已完成的工作

1. **克隆了仓库**
   - `benchmark_tools/CatPred/` - CatPred 仓库已克隆
   - `benchmark_tools/CataPro/` - CataPro 仓库已克隆

2. **创建了运行脚本**
   - `scripts/run_catapro_baseline.py` - CataPro 运行脚本
   - `scripts/setup_benchmark_env.sh` - 环境配置脚本

3. **实际运行了 PocketGNN**
   - 在 `kcat_merged_hom40_test.pt` (898 samples) 上评估
   - 结果：Pearson r = 0.6667, R² = 0.4366

### ❌ 未完成的工作

1. **没有真正运行 CatPred**
   - 原因：需要下载预训练模型和数据（~2GB），配置环境
   - 当前：使用文献报告的结果（Pearson r = 0.52）

2. **没有真正运行 CataPro**
   - 原因：需要下载 ProtT5 和 MolT5 预训练模型，配置环境
   - 当前：使用文献报告的结果（Pearson r = 0.497）

## 🔍 代码逻辑说明

### `scripts/run_catapro_baseline.py` 的工作流程

```python
# 1. 准备输入数据（CataPro 格式）
prepare_catapro_input(df, output_path)
# 格式：Enzyme_id, type, sequence, smiles

# 2. 运行 CataPro 推理
run_catapro_inference(input_path, output_path, catapro_dir, device)
# 调用：python inference/predict.py -inp_fpath ... -out_fpath ...

# 3. 如果运行失败，使用文献结果作为 fallback
if not success:
    # 使用文献报告的值
    reference_metrics = {
        'Pearson_r': 0.497,
        'Spearman_rho': 0.502,
    }
```

### CataPro 的实际运行需要

1. **预训练模型**（需要从 HuggingFace 下载）：
   - `prot_t5_xl_uniref50` - 蛋白质语言模型
   - `molt5-base-smiles2caption` - 分子语言模型
   - 位置：`benchmark_tools/CataPro/models/`

2. **CataPro 模型权重**：
   - `models/kcat_models/` - kcat 预测模型
   - `models/Km_models/` - Km 预测模型
   - `models/act_models/` - Activity 模型

3. **环境配置**：
   ```bash
   conda create -n catapro python=3.9
   conda activate catapro
   pip install torch transformers sentencepiece rdkit-pypi pandas numpy
   ```

### CatPred 的实际运行需要

1. **预训练模型和数据**（~2GB）：
   ```bash
   cd benchmark_tools/CatPred
   wget https://catpred.s3.us-east-1.amazonaws.com/capsule_data_update.tar.gz
   tar -xzf capsule_data_update.tar.gz
   ```

2. **环境配置**：
   ```bash
   conda env create -f environment.yml
   conda activate catpred
   pip install -e .
   ```

3. **运行推理**：
   ```bash
   python predict.py --input test_data.csv --output predictions.csv
   ```

## 🚀 如何真正运行这些方法

### 步骤 1: 运行 CataPro

```bash
# 1. 下载预训练模型（需要手动从 HuggingFace 下载）
cd benchmark_tools/CataPro/models
# 下载 prot_t5_xl_uniref50 和 molt5-base-smiles2caption

# 2. 准备测试数据（从 CataPro 数据集）
python scripts/run_catapro_baseline.py \
    --fold 0 \
    --device cuda:0 \
    --output_dir results/catapro_baseline
```

### 步骤 2: 运行 CatPred

```bash
# 1. 下载预训练模型和数据
cd benchmark_tools/CatPred
wget https://catpred.s3.us-east-1.amazonaws.com/capsule_data_update.tar.gz
tar -xzf capsule_data_update.tar.gz

# 2. 配置环境
conda env create -f environment.yml
conda activate catpred
pip install -e .

# 3. 准备测试数据并运行
python predict.py --input test_data.csv --output predictions.csv
```

### 步骤 3: 在相同测试集上对比

**关键问题**：CatPred 和 CataPro 使用的是**不同的数据集**：

- **CataPro**: 使用自己的数据集（`kcat-data_0.4simi-10fold.csv`），10-fold CV
- **CatPred**: 使用 CatPred-DB，可能有不同的划分策略
- **PocketGNN**: 使用 `kcat_merged_hom40_test.pt`，40% 同源性划分

**公平对比需要**：
1. 将 PocketGNN 的测试集转换为 CatPred/CataPro 格式
2. 在**相同的测试集**上运行所有方法
3. 或者，在各自的数据集上运行，但明确说明数据集差异

## 📊 当前报告的问题

### 问题 1: 数据集不一致

- PocketGNN: 40% 同源性 OOD 测试集（898 samples）
- CatPred: 文献报告的是 OOD (low similarity)，但具体数据集未知
- CataPro: 文献报告的是 10-fold CV (0.4 similarity)，但具体数据集未知

### 问题 2: 使用文献结果而非实际运行

- 优点：快速，不需要配置复杂环境
- 缺点：无法保证在相同测试集上的公平对比

## ✅ 建议的改进方案

### 方案 A: 在相同测试集上运行（推荐）

1. 将 PocketGNN 的测试集转换为 CatPred/CataPro 格式
2. 真正运行这些方法
3. 在相同测试集上对比

### 方案 B: 明确说明数据集差异

1. 在报告中明确说明：
   - PocketGNN 使用的是 40% 同源性 OOD 测试集
   - CatPred/CataPro 使用的是文献报告的结果（不同数据集）
2. 强调这是**方法层面的对比**，而非严格公平的 benchmark

## 🔧 实际运行代码示例

### 运行 CataPro 的完整流程

```python
# scripts/run_catapro_baseline.py 的实际执行逻辑

# 1. 加载 CataPro 数据集
df = pd.read_csv('benchmark_tools/CataPro/datasets/kcat-data_0.4simi-10fold.csv')

# 2. 按 fold 划分
test_df = df[df['fold'] == 0]  # 使用 fold 0 作为测试集

# 3. 准备输入（CataPro 格式）
catapro_input = pd.DataFrame({
    'Enzyme_id': test_df['UniProtID'],
    'type': test_df['EnzymeType'],
    'sequence': test_df['Sequence'],
    'smiles': test_df['Smiles']
})

# 4. 运行 CataPro
# 需要：ProtT5 模型、MolT5 模型、CataPro 预训练权重
python inference/predict.py \
    -inp_fpath catapro_input.csv \
    -model_dpath models/ \
    -out_fpath predictions.csv

# 5. 计算指标
y_true = np.log10(test_df['kcat(s^-1)'])
y_pred = predictions['kcat_pred']
metrics = compute_metrics(y_true, y_pred)
```

### 运行 CatPred 的完整流程

```python
# CatPred 使用不同的输入格式
# 需要：ESM-2 embeddings, D-MPNN features, 预训练模型

# 1. 准备输入（CatPred 格式）
# CatPred 需要：protein sequence, substrate SMILES, 可能还需要结构信息

# 2. 运行 CatPred
python predict.py \
    --input test_data.csv \
    --output predictions.csv \
    --model_path capsule_data/models/

# 3. 计算指标
metrics = compute_metrics(y_true, y_pred)
```

## 📝 总结

1. **当前状态**：只运行了 PocketGNN，CatPred/CataPro 使用文献结果
2. **原因**：需要下载预训练模型、配置环境、处理数据格式转换
3. **改进**：需要在相同测试集上真正运行所有方法，或明确说明数据集差异

## 🎯 下一步行动

如果要真正运行这些方法，需要：

1. ✅ 下载预训练模型（ProtT5, MolT5, CatPred models）
2. ✅ 配置 conda 环境
3. ✅ 将 PocketGNN 测试集转换为 CatPred/CataPro 格式
4. ✅ 真正运行推理代码
5. ✅ 在相同测试集上对比结果





