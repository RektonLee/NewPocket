# 项目迁移快速指南

> **目标**: 将 PocketGNN 项目从当前服务器迁移到新服务器
>
> **预计数据量**: ~2.5 GB (核心文件) + 2.2 GB (样本数据，可选)

---

## 📦 迁移脚本总览

我已经为你准备了 3 个自动化脚本：

| 脚本 | 用途 | 大小 | 位置 |
|------|------|------|------|
| [migrate_to_new_server.sh](migrate_to_new_server.sh) | 打包核心文件（代码+数据+模型） | ~2.5 GB | 在旧服务器运行 |
| [migrate_samples.sh](migrate_samples.sh) | 打包样本数据（可选） | ~2.2 GB | 在旧服务器运行 |
| [verify_migration.sh](verify_migration.sh) | 验证迁移完整性 | - | 在新服务器运行 |

---

## 🚀 迁移步骤（3步走）

### 第1步: 在旧服务器打包

```bash
# 进入项目目录
cd /home/lizihao/Work/enzyme_prediction/PGNN_clean

# 打包核心文件（必需）
bash migrate_to_new_server.sh
# 输出: pgnn_migration_YYYYMMDD_HHMMSS.tar.gz (~2.5 GB)

# 打包样本数据（可选，如果需要完整复现）
bash migrate_samples.sh
# 输出: pgnn_samples_clean_*.tar.gz (~516 MB)
#       pgnn_samples_train_*.tar.gz (~1.7 GB)
```

**打包内容清单**:

<details>
<summary>核心文件包 (pgnn_migration_*.tar.gz) - 展开查看</summary>

```
核心代码:
├── src/                                   # 源代码
├── scripts/                               # 脚本
├── docs/                                  # 文档
├── configs/                               # 配置
├── benchmark_tools/                       # Baseline 工具
├── requirements.txt                       # 依赖
├── CLAUDE.md                             # AI 指令
├── PROJECT_DOCUMENTATION.md              # 项目文档
└── README.md

关键数据 (6 个文件):
├── data/processed/kcat_test_new.csv       (618 KB)
├── data/processed/kcat_test_new.pt        (103 MB)
├── data/processed/kcat_full_1213.csv      (3.7 MB)
├── data/processed/kcat_full_1213.pt       (1.7 GB)
├── data/processed/kcat_test_new_diffdock.csv (396 KB)
└── data/processed/kcat_test_new_diffdock.pt  (137 MB)

最佳模型 (3 个):
├── outputs/kcat_esm_norm_fix/             # 测试集最佳 R²=0.44
├── outputs/kcat_20251213_151558/          # 纯GNN R²=0.38
└── outputs/kcat_enhanced_run/             # 同源性划分 R²=0.39
```
</details>

---

### 第2步: 传输到新服务器

```bash
# 方法A: 使用 scp (推荐)
scp pgnn_migration_*.tar.gz user@new_server:/path/to/destination/

# 方法B: 使用 rsync (支持断点续传)
rsync -avzP --progress pgnn_migration_*.tar.gz user@new_server:/path/to/destination/

# 如果打包了样本数据，也传输
scp pgnn_samples_*.tar.gz user@new_server:/path/to/destination/
```

**传输时间估算**:
- 2.5 GB @ 100 Mbps = 约 3-4 分钟
- 2.5 GB @ 10 Mbps = 约 30-40 分钟

---

### 第3步: 在新服务器解压和验证

```bash
# 登录新服务器
ssh user@new_server

# 创建项目目录
mkdir -p /path/to/PGNN_clean
cd /path/to/PGNN_clean

# 解压核心文件
tar -xzf /path/to/pgnn_migration_*.tar.gz

# 解压样本数据（如果需要）
tar -xzf /path/to/pgnn_samples_clean_*.tar.gz

# 运行验证脚本
bash verify_migration.sh
```

**验证脚本会检查**:
- ✓ 所有文件是否存在
- ✓ 文件大小是否正确
- ✓ Python 环境和库
- ✓ 数据文件能否加载
- ✓ 模型能否加载

---

## ✅ 验证通过后的测试

```bash
# 安装依赖（如果验证脚本提示缺少）
pip install -r requirements.txt

# 测试最佳模型
python src/test.py \
    --test_dataset data/processed/kcat_test_new.pt \
    --model outputs/kcat_esm_norm_fix/best_model.pt

# 预期输出:
# R² ≈ 0.4432
# Pearson ≈ 0.6728
# MAE ≈ 0.8466
```

---

## 📋 迁移文件清单（仅精简版）

### 必需文件 (6个数据文件)

- [x] `data/processed/kcat_test_new.csv` - 测试集 CSV
- [x] `data/processed/kcat_test_new.pt` - 测试集 PyG 格式
- [x] `data/processed/kcat_full_1213.csv` - 完整训练集 CSV
- [x] `data/processed/kcat_full_1213.pt` - 完整训练集 PyG 格式
- [x] `data/processed/kcat_test_new_diffdock.csv` - DiffDock 测试集 CSV
- [x] `data/processed/kcat_test_new_diffdock.pt` - DiffDock 测试集 PyG 格式

### 核心代码

- [x] `src/` - 所有源代码
- [x] `scripts/` - 脚本文件
- [x] `docs/` - 文档
- [x] `requirements.txt` - 依赖

### 最佳模型 (3个)

- [x] `outputs/kcat_esm_norm_fix/` - 测试集最佳 (R²=0.4432)
- [x] `outputs/kcat_20251213_151558/` - 纯GNN最佳 (R²=0.3787)
- [x] `outputs/kcat_enhanced_run/` - 同源性划分最佳 (R²=0.3872)

### 样本数据 (可选)

- [ ] `sample_data/samples/` - PGNN_clean 测试集样本 (516 MB, 3444个目录)
- [ ] `/PGNN/sample_data/samples/kcat_xxxxxx/` - 训练集样本 (1.7 GB, 10319个目录)

---

## 🔧 常见问题

### Q1: 打包失败怎么办？

```bash
# 检查磁盘空间
df -h .

# 如果空间不足，可以分批打包
tar -czf pgnn_code.tar.gz src/ scripts/ docs/ requirements.txt
tar -czf pgnn_data.tar.gz data/processed/*.csv data/processed/*.pt
tar -czf pgnn_models.tar.gz outputs/kcat_*
```

### Q2: 传输中断怎么办？

```bash
# 使用 rsync 支持断点续传
rsync -avzP --partial pgnn_migration_*.tar.gz user@new_server:/path/to/
```

### Q3: 新服务器上模型加载失败？

```bash
# 检查 PyTorch 版本
python -c "import torch; print(torch.__version__)"

# 如果版本不同，可以尝试 CPU 加载
python -c "
import torch
model = torch.load('outputs/kcat_esm_norm_fix/best_model.pt',
                   map_location='cpu')
print('加载成功')
"
```

### Q4: 需要 ESM embeddings 吗？

**不需要**。精简版迁移清单中没有包含 ESM embeddings 文件。如果需要从头训练带 ESM 的模型，需要额外下载：
- `data/processed/esm_embeddings.pt` (25 MB)
- `data/processed/esm_embeddings_test_new.pt` (2.2 MB)

### Q5: 需要样本数据吗？

**取决于使用场景**:
- ✅ **只需要测试模型**: 不需要样本数据，只需要 .pt 文件
- ✅ **需要重新训练**: 需要样本数据（或者可以从 CSV 重新生成）
- ✅ **需要可视化/分析**: 可能需要部分样本的 PDB 文件

---

## 📞 遇到问题？

查看详细文档:
- [MIGRATION_CHECKLIST.md](MIGRATION_CHECKLIST.md) - 完整迁移清单
- [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) - 项目文档
- [CLAUDE.md](CLAUDE.md) - Claude AI 使用指南

---

## 🎯 一键迁移命令（旧服务器）

```bash
# 全自动打包和传输（需要配置免密登录）
cd /home/lizihao/Work/enzyme_prediction/PGNN_clean

# 打包
bash migrate_to_new_server.sh

# 传输（替换为你的服务器地址）
NEW_SERVER="user@new_server_ip"
TARGET_PATH="/path/to/destination"

scp pgnn_migration_*.tar.gz ${NEW_SERVER}:${TARGET_PATH}/
scp verify_migration.sh ${NEW_SERVER}:${TARGET_PATH}/

echo "迁移完成！现在登录新服务器运行:"
echo "  ssh ${NEW_SERVER}"
echo "  cd ${TARGET_PATH}"
echo "  tar -xzf pgnn_migration_*.tar.gz"
echo "  bash verify_migration.sh"
```

---

**就是这么简单！3个脚本搞定迁移 🚀**
