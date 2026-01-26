# 快速开始：运行 SOTA Benchmark

## 当前状态

✅ **CataPro 模型已就绪** - 可以立即运行
⏳ **CatPred 数据下载中** - 等待完成后可运行

---

## 运行 CataPro（立即可用）

```bash
# 1. 激活环境（如果还没创建）
conda create -n catapro python=3.9
conda activate catapro
pip install torch transformers sentencepiece rdkit-pypi pandas numpy

# 2. 运行 CataPro
cd /home/lizihao/Work/enzyme_prediction/PGNN_clean
python scripts/run_catapro_baseline.py \
    --test_csv data/sota_benchmark/catapro_test_input.csv \
    --output_dir results/catapro_real \
    --device cuda:0
```

---

## 运行 CatPred（等待数据下载完成）

```bash
# 1. 检查下载状态
cd benchmark_tools/CatPred
ls -lh capsule_data_update.tar.gz  # 应该 ~2GB

# 2. 解压数据
tar -xzf capsule_data_update.tar.gz

# 3. 配置环境
conda env create -f environment.yml
conda activate catpred
pip install -e .

# 4. 运行 CatPred
cd ../../  # 回到项目根目录
python scripts/run_catpred_real.py \
    --test_csv data/sota_benchmark/catpred_test_input.csv \
    --output_dir results/catpred_real
```

---

## 对比结果

运行完成后，使用以下脚本对比结果：

```bash
python scripts/benchmark_comparison.py \
    --results_dir results/ \
    --output_dir results/final_comparison
```

---

## 数据说明

- **测试集**: 733 个样本（从 898 个 PocketGNN 测试样本中匹配到的）
- **输入格式**: 已转换为 CatPred/CataPro 格式
- **位置**: `data/sota_benchmark/`

