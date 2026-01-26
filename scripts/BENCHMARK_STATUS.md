# SOTA Benchmark 实际运行状态

**日期**: 2026-01-19
**状态**: 进行中

---

## ✅ 已完成的工作

### 1. 数据转换 ✅

- **测试集**: `data/processed/kcat_merged_hom40_test.pt` (898 samples)
- **成功匹配**: 733 samples 在原始 CSV 中找到
- **转换输出**:
  - `data/sota_benchmark/catpred_test_input.csv` (733 samples)
  - `data/sota_benchmark/catapro_test_input.csv` (733 samples)
  - `data/sota_benchmark/conversion_metadata.json`

**转换脚本**: `scripts/convert_testset_for_sota.py`

### 2. 仓库克隆 ✅

- `benchmark_tools/CatPred/` - 已克隆
- `benchmark_tools/CataPro/` - 已克隆

### 3. 运行脚本创建 ✅

- `scripts/run_catpred_real.py` - CatPred 实际运行脚本
- `scripts/run_catapro_baseline.py` - CataPro 运行脚本（已存在，需要更新）

---

## ⚠️ 待完成的工作

### 1. CatPred 环境配置

**问题**: `capsule_data_update.tar.gz` 下载不完整（只有 3.0M，应该 ~2GB）

**需要**:
```bash
cd benchmark_tools/CatPred
# 重新下载完整文件
wget https://catpred.s3.us-east-1.amazonaws.com/capsule_data_update.tar.gz
# 验证文件大小（应该 ~2GB）
# 解压
tar -xzf capsule_data_update.tar.gz
```

**然后**:
```bash
# 配置环境
conda env create -f environment.yml
conda activate catpred
pip install -e .
```

**运行**:
```bash
python scripts/run_catpred_real.py \
    --test_csv data/sota_benchmark/catpred_test_input.csv \
    --output_dir results/catpred_real
```

### 2. CataPro 环境配置

**需要下载预训练模型**（从 HuggingFace）:

```bash
cd benchmark_tools/CataPro/models

# 下载 ProtT5-XL
git lfs install
git clone https://huggingface.co/Rostlab/prot_t5_xl_uniref50
# 或使用 huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('Rostlab/prot_t5_xl_uniref50', local_dir='prot_t5_xl_uniref50')"

# 下载 MolT5-base
git clone https://huggingface.co/laituan245/molt5-base-smiles2caption
# 或
python -c "from huggingface_hub import snapshot_download; snapshot_download('laituan245/molt5-base-smiles2caption', local_dir='molt5-base-smiles2caption')"
```

**配置环境**:
```bash
conda create -n catapro python=3.9
conda activate catapro
pip install torch transformers sentencepiece rdkit-pypi pandas numpy
```

**运行**:
```bash
python scripts/run_catapro_baseline.py \
    --test_csv data/sota_benchmark/catapro_test_input.csv \
    --output_dir results/catapro_real
```

---

## 📊 当前对比状态

### PocketGNN ✅

- **已运行**: 在 `kcat_merged_hom40_test.pt` (898 samples) 上评估
- **结果**: 
  - Pearson r = 0.6667
  - R² = 0.4366
  - Spearman ρ = 0.6092
  - MAE = 0.8301
  - p_1mag = 0.6993

### CatPred ⏳

- **状态**: 数据下载不完整，需要重新下载
- **测试集**: 733 samples (从 898 中匹配到的)
- **预期**: 运行后得到实际结果

### CataPro ⏳

- **状态**: 需要下载预训练模型
- **测试集**: 733 samples (从 898 中匹配到的)
- **预期**: 运行后得到实际结果

---

## 🎯 下一步行动

### 优先级 1: 完成 CatPred 运行

1. 重新下载 `capsule_data_update.tar.gz`（完整文件 ~2GB）
2. 解压并配置环境
3. 运行 `scripts/run_catpred_real.py`
4. 收集结果并计算指标

### 优先级 2: 完成 CataPro 运行

1. 下载 ProtT5 和 MolT5 预训练模型
2. 配置环境
3. 运行 `scripts/run_catapro_baseline.py`
4. 收集结果并计算指标

### 优先级 3: 公平对比

1. 在相同的 733 个样本上对比所有方法
2. 生成对比报告和可视化
3. 更新 benchmark 文档

---

## 📝 注意事项

1. **数据集差异**: 
   - PocketGNN 测试集: 898 samples (40% homology split)
   - CatPred/CataPro 测试集: 733 samples (匹配到的子集)
   - 需要在报告中说明这个差异

2. **数据格式**:
   - CatPred 需要: `SMILES`, `sequence` 列
   - CataPro 需要: `Enzyme_id`, `type`, `sequence`, `smiles` 列
   - 已通过 `convert_testset_for_sota.py` 转换

3. **环境依赖**:
   - CatPred: 需要 conda 环境，依赖较多
   - CataPro: 需要 HuggingFace 模型下载（大文件）

---

## 🔧 快速开始

### 运行 CatPred

```bash
# 1. 确保数据完整
cd benchmark_tools/CatPred
wget https://catpred.s3.us-east-1.amazonaws.com/capsule_data_update.tar.gz
tar -xzf capsule_data_update.tar.gz

# 2. 配置环境
conda env create -f environment.yml
conda activate catpred
pip install -e .

# 3. 运行
cd ../../  # 回到项目根目录
python scripts/run_catpred_real.py \
    --test_csv data/sota_benchmark/catpred_test_input.csv \
    --output_dir results/catpred_real
```

### 运行 CataPro

```bash
# 1. 下载模型（在 benchmark_tools/CataPro/models/ 目录下）
# 使用 huggingface_hub 或 git lfs

# 2. 配置环境
conda create -n catapro python=3.9
conda activate catapro
pip install torch transformers sentencepiece rdkit-pypi

# 3. 运行
python scripts/run_catapro_baseline.py \
    --test_csv data/sota_benchmark/catapro_test_input.csv \
    --output_dir results/catapro_real
```

---

---

## 🎉 最新更新 (2026-01-19 14:00)

### ✅ CataPro 模型已就绪！

- **ProtT5-XL**: 11GB 已下载 ✅
- **MolT5-base**: 已存在 ✅
- **状态**: 可以立即运行 CataPro！

### ⏳ CatPred 数据下载中

- **状态**: wget 进程正在运行
- **文件**: `capsule_data_update.tar.gz`
- **预计大小**: ~2GB

### 📝 使用 HuggingFace 镜像

已设置 `HF_ENDPOINT=https://hf-mirror.com` 用于模型下载。

---

**最后更新**: 2026-01-19 14:00

