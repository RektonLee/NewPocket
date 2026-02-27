# 新集群需要的新脚本清单

## ✅ 已提交到Git的新脚本 (直接 git pull 获取)

### 8-GPU并行对接:
1. **scripts/launch_8gpu_diffdock.sh** - 启动8个GPU并行worker
2. **scripts/check_8gpu_progress.py** - 监控8-GPU进度
3. **scripts/distribute_8gpu_tasks.py** - 智能任务分配
4. **scripts/reconcile_docking_status.py** - 对接状态核对

### 文档:
5. **CLUSTER_MIGRATION_GUIDE.md** - 迁移操作指南

**Commit**: 63b2d17 (2026-02-27)
**分支**: final-experiments-paper-writing

---

## 📦 需要同步的数据文件 (用rsync/scp，不走Git)

### 必需文件:
```bash
# 1. CSV数据 (必需)
data/processed/kcat_full_1213.csv

# 2. 待对接样本列表 (必需)
results/remaining_samples.txt

# 3. 口袋PDB文件 (必需，如果要复用)
sample_data/samples/*/docking/*.pdb
# 或者从PGNN旧目录
/home/lizihao/Work/enzyme_prediction/PGNN/sample_data/samples/*/kcat_*_10A.pdb
```

### 可选文件:
```bash
# 已对接的SDF文件 (可选，如果想继续增量对接)
sample_data/samples/*/kcat_*_diffdock.sdf

# 对接结果CSV (可选，用于断点续传)
results/diffdock_batch_results_from_0.csv
results/diffdock_batch_results_from_2036.csv
```

---

## 🚀 新集群快速设置

### 方法A: 全新开始 (推荐，如果新集群是空的)
```bash
# 1. 克隆项目
git clone https://github.com/RektonLee/NewPocket.git PGNN_clean
cd PGNN_clean
git checkout final-experiments-paper-writing
git pull  # 确保最新

# 2. 从旧集群同步数据
rsync -avz --progress \
  oldcluster:/home/lizihao/Work/enzyme_prediction/PGNN_clean/data/processed/kcat_full_1213.csv \
  data/processed/

rsync -avz --progress \
  oldcluster:/home/lizihao/Work/enzyme_prediction/PGNN_clean/results/remaining_samples.txt \
  results/

# 3. 同步口袋PDB (从PGNN旧目录)
rsync -avz --progress --include='*/' --include='*_10A.pdb' --exclude='*' \
  oldcluster:/home/lizihao/Work/enzyme_prediction/PGNN/sample_data/samples/ \
  sample_data/samples/

# 4. 启动对接
mkdir -p logs
./scripts/launch_8gpu_diffdock.sh
```

### 方法B: 已有项目更新
```bash
# 1. 拉取新脚本
cd PGNN_clean
git pull origin final-experiments-paper-writing

# 2. 给新脚本执行权限
chmod +x scripts/launch_8gpu_diffdock.sh
chmod +x scripts/check_8gpu_progress.py

# 3. 确认数据文件存在
ls -lh data/processed/kcat_full_1213.csv
ls -lh results/remaining_samples.txt

# 4. 启动
./scripts/launch_8gpu_diffdock.sh
```

---

## 🔍 验证清单

新集群上检查:
```bash
# ✅ 脚本存在
ls -lh scripts/launch_8gpu_diffdock.sh
ls -lh scripts/check_8gpu_progress.py

# ✅ 数据存在
ls -lh data/processed/kcat_full_1213.csv
wc -l results/remaining_samples.txt  # 应该是2421行

# ✅ 目录结构
mkdir -p logs results sample_data/samples

# ✅ Git状态
git log -1  # 应该看到 63b2d17 commit

# ✅ conda环境
conda activate pgnn  # 或你的环境名
which python
```

---

## ⚠️ 注意事项

1. **不要用Git同步大文件**:
   - ❌ `.pdb`, `.sdf`, `.csv` (除非很小)
   - ✅ 用 `rsync` 或 `scp`

2. **remaining_samples.txt很重要**:
   - 包含2421个待对接样本ID
   - 必须同步到新集群
   - 否则脚本不知道要对接哪些样本

3. **口袋PDB文件来源**:
   - 优先从 `/home/lizihao/Work/enzyme_prediction/PGNN/sample_data/samples/` (旧PGNN目录)
   - 文件命名: `kcat_XXXXXX_XXXXX_10A.pdb`
   - 这些是protein PDB，对接时会用到

4. **DiffDock环境**:
   - 新集群需要安装DiffDock
   - 设置 `DIFFDOCK_PATH` 环境变量
   - 或者修改 `src/docking.py` 中的路径配置
