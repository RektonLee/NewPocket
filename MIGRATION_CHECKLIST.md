# 项目迁移清单

> 生成时间: 2026-02-24
>
> 从当前服务器迁移到新服务器的完整清单

---

## 📊 迁移数据量估算

| 类别 | 大小 | 文件数 |
|------|------|--------|
| 处理后数据 (data/processed) | 6.8 GB | ~20 文件 |
| 样本数据 (PGNN_clean/sample_data) | 516 MB | ~3444 目录 |
| 样本数据 (PGNN/sample_data) | 待确认 | ~数千目录 |
| 模型输出 (outputs) | 71 MB | ~51 目录 |
| 文档 (docs) | < 10 MB | ~8 文件 |
| **总计** | **≈7.5 GB+** | - |

---

## ✅ 必须迁移的文件清单

### 1. 核心代码和配置

```bash
# 整个代码库（建议用 git clone）
PGNN_clean/
├── src/                    # 核心源代码
├── scripts/                # 脚本文件
├── requirements.txt        # 依赖
├── CLAUDE.md              # Claude AI 指令
├── PROJECT_DOCUMENTATION.md
├── README.md
└── .gitignore
```

**迁移方法**: Git clone 或 rsync

---

### 2. 关键数据文件

#### 2.1 处理后的数据集 (data/processed/)

**必须迁移**:
```
data/processed/
├── kcat_full_1213.csv           (3.7 MB)  ← 完整训练数据 CSV
├── kcat_full_1213.pt            (1.7 GB)  ← 完整训练数据 .pt
├── kcat_test_new.csv            (618 KB)  ← 测试集 CSV
├── kcat_test_new.pt             (103 MB)  ← 测试集 .pt
├── kcat_test_new_diffdock.csv   (396 KB)  ← DiffDock 测试集 CSV
├── kcat_test_new_diffdock.pt    (137 MB)  ← DiffDock 测试集 .pt
├── esm_embeddings.pt            (25 MB)   ← ESM-2 序列嵌入
└── esm_embeddings_test_new.pt   (2.2 MB)  ← 测试集 ESM 嵌入
```

**可选迁移** (如果需要同源性划分实验):
```
data/processed/
├── kcat_merged_hom40_train.pt   (1.6 GB)
├── kcat_merged_hom40_val.pt     (182 MB)
└── kcat_merged_hom40_test.pt    (161 MB)
```

---

#### 2.2 样本数据 (PDB 文件等)

**PGNN_clean 中的样本** (516 MB, ~3444 目录):
```bash
sample_data/samples/
├── kcat_test_0001/     # 测试集样本
├── kcat_test_0002/
├── ...
└── kcat_test_xxxx/
```

**PGNN 目录中的样本** (需要确认大小):
```bash
/home/lizihao/Work/enzyme_prediction/PGNN/sample_data/samples/
├── kcat_000002/        # 格式: kcat_xxxxxx
├── kcat_000003/
├── ...
└── kcat_xxxxxx/
```

**迁移命令示例**:
```bash
# 从 PGNN_clean 迁移
rsync -avz sample_data/samples/ user@new_server:/path/to/PGNN_clean/sample_data/samples/

# 从 PGNN 迁移 (如果需要训练集样本)
rsync -avz /home/lizihao/Work/enzyme_prediction/PGNN/sample_data/samples/ \
    user@new_server:/path/to/backup/PGNN_samples/
```

---

### 3. 训练好的模型 (outputs/)

**必须迁移的最佳模型**:

```
outputs/
├── kcat_esm_norm_fix/          # 测试集最佳 (R²=0.4432)
│   ├── best_model.pt
│   ├── config.json
│   └── metrics.json
│
├── kcat_20251213_151558/       # 纯 GNN 最佳 (R²=0.3787)
│   ├── best_model.pt
│   ├── config.json
│   └── metrics.json
│
└── kcat_enhanced_run/          # 同源性划分最佳 (R²=0.3872)
    ├── best_model.pt
    ├── config.json
    └── metrics.json
```

**可选**: 其他实验模型目录 (如果需要复现所有实验)

---

### 4. 文档和记录

```
docs/
├── EXPERIMENTS_LOG.md          # 实验记录
├── paper/                      # 论文相关
├── analysis/                   # 分析脚本
└── README.md

# 根目录文档
├── CLAUDE.md                   # Claude AI 指令
├── PROJECT_DOCUMENTATION.md    # 项目文档
├── DOCKING_METHODS_COMPARISON.md
├── README.md
└── runs_registry.csv           # 实验运行记录
```

---

### 5. 其他重要文件

```
├── requirements.txt            # Python 依赖
├── .gitignore
├── configs/                    # 配置文件
└── benchmark_tools/            # Baseline 工具
```

---

## 🚫 **不需要迁移** 的内容

```
├── wandb/                      # WandB 日志（可从云端获取）
├── logs/                       # 训练日志
├── temp/                       # 临时文件
├── tmp/                        # 临时文件
├── __pycache__/                # Python 缓存
├── *.pyc                       # 编译文件
├── experiments/                # 可选，旧实验数据
└── results/                    # 可选，部分结果文件
```

---

## 📝 迁移步骤建议

### 方案 A: 完整迁移 (推荐用于备份)

```bash
# 1. 在新服务器上克隆代码
cd /path/to/new/location
git clone <your-repo-url> PGNN_clean
cd PGNN_clean

# 2. 创建数据目录
mkdir -p data/processed sample_data/samples outputs

# 3. 从旧服务器传输数据
# 在旧服务器上执行:
rsync -avzP --include='*.csv' --include='*.pt' data/processed/ \
    user@new_server:/path/to/PGNN_clean/data/processed/

rsync -avzP sample_data/samples/ \
    user@new_server:/path/to/PGNN_clean/sample_data/samples/

rsync -avzP outputs/kcat_esm_norm_fix/ \
    outputs/kcat_20251213_151558/ \
    outputs/kcat_enhanced_run/ \
    user@new_server:/path/to/PGNN_clean/outputs/

# 4. 传输 PGNN 样本数据（如果需要）
rsync -avzP /home/lizihao/Work/enzyme_prediction/PGNN/sample_data/samples/ \
    user@new_server:/path/to/backup/PGNN_samples/

# 5. 传输文档
rsync -avzP docs/ \
    user@new_server:/path/to/PGNN_clean/docs/
```

### 方案 B: 最小化迁移 (仅核心文件)

```bash
# 只迁移关键文件
tar -czf pgnn_migration.tar.gz \
    data/processed/kcat_full_1213.{csv,pt} \
    data/processed/kcat_test_new.{csv,pt} \
    data/processed/kcat_test_new_diffdock.{csv,pt} \
    data/processed/esm_embeddings*.pt \
    outputs/kcat_esm_norm_fix/ \
    outputs/kcat_20251213_151558/ \
    docs/EXPERIMENTS_LOG.md \
    PROJECT_DOCUMENTATION.md \
    CLAUDE.md \
    runs_registry.csv

# 传输压缩包
scp pgnn_migration.tar.gz user@new_server:/path/to/

# 在新服务器解压
cd /path/to/PGNN_clean
tar -xzf ../pgnn_migration.tar.gz
```

### 方案 C: 分批迁移 (适合网络不稳定)

```bash
# 第1批: 小文件 + 代码
tar -czf batch1_code_docs.tar.gz src/ scripts/ docs/ *.md requirements.txt

# 第2批: CSV 数据
tar -czf batch2_csv.tar.gz data/processed/*.csv

# 第3批: 重要 .pt 文件
tar -czf batch3_datasets.tar.gz \
    data/processed/kcat_full_1213.pt \
    data/processed/kcat_test_new.pt \
    data/processed/esm_embeddings.pt

# 第4批: 模型
tar -czf batch4_models.tar.gz outputs/kcat_*

# 第5批: 样本数据（最大）
tar -czf batch5_samples.tar.gz sample_data/samples/
```

---

## 🔍 迁移后验证清单

### 1. 文件完整性检查

```bash
# 检查关键文件是否存在
ls -lh data/processed/kcat_full_1213.pt
ls -lh data/processed/kcat_test_new.pt
ls -lh outputs/kcat_esm_norm_fix/best_model.pt

# 检查文件大小是否正确
du -sh data/processed/
du -sh sample_data/samples/
du -sh outputs/
```

### 2. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 检查 PyTorch 和 CUDA
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# 检查 PyTorch Geometric
python -c "import torch_geometric; print(torch_geometric.__version__)"
```

### 3. 快速测试

```bash
# 测试数据加载
python -c "
import torch
data = torch.load('data/processed/kcat_test_new.pt')
print(f'Loaded {len(data)} samples')
print(f'First sample: {data[0]}')
"

# 测试模型加载
python src/test.py \
    --test_dataset data/processed/kcat_test_new.pt \
    --model outputs/kcat_esm_norm_fix/best_model.pt
```

### 4. 运行完整测试

```bash
# 使用最佳模型运行测试
python src/test.py \
    --test_dataset data/processed/kcat_test_new.pt \
    --model outputs/kcat_esm_norm_fix/best_model.pt

# 预期结果: R² ≈ 0.44, Pearson ≈ 0.67
```

---

## 📌 重要提示

1. **Git 管理**: 代码部分建议用 `git clone` 而不是直接复制，保持版本历史
2. **数据完整性**: 使用 `rsync -c` (checksum) 或 `md5sum` 验证大文件传输
3. **权限问题**: 确保新服务器上文件权限正确 (`chmod -R u+rwX`)
4. **路径依赖**: 检查代码中是否有硬编码的旧服务器路径
5. **环境变量**: 如果使用了 `DIFFDOCK_PATH` 等环境变量，需要在新服务器重新配置

---

## 🔗 相关文档

- [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) - 项目完整文档
- [CLAUDE.md](CLAUDE.md) - Claude AI 使用指南
- [docs/EXPERIMENTS_LOG.md](docs/EXPERIMENTS_LOG.md) - 实验记录
- [README.md](README.md) - 项目介绍

---

## ❓ 故障排查

### 问题1: 模型加载失败
```bash
# 检查 PyTorch 版本兼容性
python -c "import torch; print(torch.__version__)"
# 如果版本不同，可能需要重新训练或使用 torch.load(..., map_location='cpu')
```

### 问题2: 数据路径错误
```bash
# 检查 sample_manager.py 中的路径配置
grep -r "/home/lizihao" src/
# 手动修改硬编码路径
```

### 问题3: 缺少依赖
```bash
# 重新安装所有依赖
pip install -r requirements.txt --force-reinstall
```

---

**迁移完成后，请运行测试验证所有功能正常！**
