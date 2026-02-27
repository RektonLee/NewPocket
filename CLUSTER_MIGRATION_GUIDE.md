# 8-GPU集群迁移操作指南

**目标**: 在新的8×3090集群上继续DiffDock对接任务

---

## 📋 迁移前准备 (当前集群)

### 1. 提交所有新脚本到Git
```bash
cd /home/lizihao/Work/enzyme_prediction/PGNN_clean

# 查看新创建的文件
git status

# 添加新脚本
git add scripts/launch_8gpu_diffdock.sh
git add scripts/check_8gpu_progress.py
git add scripts/distribute_8gpu_tasks.py
git add results/remaining_samples.txt
git add results/docking_status_report.json
git add results/DIFFDOCK_STATUS_COMPLETE_REPORT_zh.md

# 提交
git commit -m "feat: Add 8-GPU parallel DiffDock support

- launch_8gpu_diffdock.sh: Launch 8 parallel workers
- check_8gpu_progress.py: Monitor combined progress
- distribute_8gpu_tasks.py: Intelligent task distribution
- Include reconciliation reports and remaining samples list"

# 推送到远程
git push origin final-experiments-paper-writing
```

### 2. 确认待对接样本列表已生成
```bash
# 检查文件是否存在
ls -lh results/remaining_samples.txt
# 应该显示: 2421个样本ID
```

---

## 🚀 新集群操作 (8×3090)

### 1. 克隆/拉取项目
```bash
# 如果是新机器,先克隆
git clone <your-repo-url> PGNN_clean
cd PGNN_clean
git checkout final-experiments-paper-writing

# 如果已有项目,拉取最新脚本
cd PGNN_clean
git pull origin final-experiments-paper-writing
```

### 2. 检查环境
```bash
# 确认8个GPU可用
nvidia-smi

# 确认conda环境存在
conda env list | grep pgnn

# 激活环境
conda activate pgnn  # 或你的环境名
```

### 3. 创建必要目录
```bash
mkdir -p logs
mkdir -p results
mkdir -p sample_data/samples
```

### 4. 同步数据文件 (关键!)
```bash
# 从旧集群同步以下目录:
# 1. data/processed/kcat_full_1213.csv (必需)
# 2. sample_data/samples/*.pdb (已处理的pocket PDB文件)
# 3. results/remaining_samples.txt (待对接列表)

# 示例 (从旧集群执行):
rsync -avz --progress \
  data/processed/kcat_full_1213.csv \
  newcluster:/path/to/PGNN_clean/data/processed/

rsync -avz --progress \
  sample_data/samples/ \
  newcluster:/path/to/PGNN_clean/sample_data/samples/

rsync -avz --progress \
  results/remaining_samples.txt \
  newcluster:/path/to/PGNN_clean/results/
```

### 5. 启动8-GPU并行对接
```bash
# 给脚本执行权限 (如果还没有)
chmod +x scripts/launch_8gpu_diffdock.sh

# 启动!
./scripts/launch_8gpu_diffdock.sh
```

**预期输出**:
```
📊 Distribution Plan:
   Total target: 4072
   Remaining: 2421
   Per GPU: ~303 samples

🚀 Launching 8 GPU workers...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GPU 0: Starting from index 0 → logs/diffdock_gpu0_8gpu_run.log
   PID: 12345
GPU 1: Starting from index 509 → logs/diffdock_gpu1_8gpu_run.log
   PID: 12346
...
✅ All 8 workers launched!
```

### 6. 监控进度
```bash
# 方法1: 使用进度检查脚本 (推荐)
python scripts/check_8gpu_progress.py

# 方法2: 查看单个GPU日志
tail -f logs/diffdock_gpu0_8gpu_run.log

# 方法3: 检查进程是否还在运行
ps aux | grep batch_diffdock_clean.py

# 方法4: 直接统计SDF文件数量
find sample_data/samples -name "*_diffdock.sdf" | wc -l
```

### 7. 预期时间线
- 每个GPU: ~303 samples @ 1分钟/sample = **5小时**
- 总进度检查: 每30分钟运行一次 `check_8gpu_progress.py`
- 预计完成: 启动后 **5-6小时**

---

## ⚠️ 常见问题处理

### Q1: 如果某个GPU worker崩溃了怎么办?
```bash
# 1. 检查哪些GPU还在运行
ps aux | grep batch_diffdock_clean.py

# 2. 查看崩溃GPU的日志
tail -100 logs/diffdock_gpu{X}_8gpu_run.log

# 3. 手动重启该GPU worker
CUDA_VISIBLE_DEVICES=X nohup python scripts/batch_diffdock_clean.py \
  --start_from <start_idx> \
  --max_samples 303 \
  > logs/diffdock_gpu{X}_8gpu_run.log 2>&1 &
```

### Q2: 如何停止所有workers?
```bash
pkill -f batch_diffdock_clean.py
```

### Q3: 如何检查是否有重复对接?
```bash
# batch_diffdock_clean.py会自动跳过已存在的SDF文件
# 日志中会显示: "✓ Already docked: sample_xxx"
```

### Q4: 对接完成后需要做什么?
```bash
# 1. 确认进度达到95%以上
python scripts/check_8gpu_progress.py
# 应显示: Docked: 3868+/4072 (95%+)

# 2. 同步结果回旧集群 (如果需要)
rsync -avz --progress \
  sample_data/samples/*_diffdock.sdf \
  oldcluster:/path/to/PGNN_clean/sample_data/samples/

# 3. 旧集群的auto_pipeline会自动检测并开始训练
# (如果在新集群训练,直接运行 python scripts/auto_pipeline.py)
```

---

## 📊 进度里程碑

| 时间点 | 预期进度 | 操作 |
|--------|---------|------|
| 启动时 | 1651/4072 (40.5%) | 启动8-GPU workers |
| +1小时 | ~2100/4072 (51%) | 检查进度,确认无错误 |
| +2小时 | ~2550/4072 (63%) | 中期检查 |
| +3小时 | ~3000/4072 (74%) | 检查日志,确认稳定 |
| +4小时 | ~3450/4072 (85%) | 接近完成 |
| +5小时 | ~3900/4072 (96%) | **达到95%阈值!** |
| +6小时 | 4072/4072 (100%) | 全部完成 |

---

## 🔄 与旧集群同步 (Git工作流)

### 场景1: 新集群完成对接,需要同步结果
```bash
# 新集群上提交结果
cd PGNN_clean
git add sample_data/samples/*_diffdock.sdf
git commit -m "chore: Add DiffDock results from 8-GPU cluster (batch 2421 samples)"
git push origin final-experiments-paper-writing

# 旧集群上拉取
cd /home/lizihao/Work/enzyme_prediction/PGNN_clean
git pull origin final-experiments-paper-writing
```

**注意**: SDF文件通常很大,如果有.gitignore排除,用rsync同步:
```bash
# 从新集群同步到旧集群
rsync -avz --progress \
  newcluster:/path/to/PGNN_clean/sample_data/samples/*.sdf \
  /home/lizihao/Work/enzyme_prediction/PGNN_clean/sample_data/samples/
```

### 场景2: Claude在旧集群上更新了代码,新集群需要同步
```bash
# 旧集群: 提交并推送
git add <modified_files>
git commit -m "fix: Update batch processing script"
git push origin final-experiments-paper-writing

# 新集群: 拉取更新
git pull origin final-experiments-paper-writing
# 如果workers还在运行,可能需要重启受影响的workers
```

---

## 📝 快速命令备忘

```bash
# === 新集群启动 ===
git pull && ./scripts/launch_8gpu_diffdock.sh

# === 监控 (每30分钟) ===
python scripts/check_8gpu_progress.py

# === 检查workers ===
ps aux | grep batch_diffdock

# === 统计进度 ===
find sample_data/samples -name "*_diffdock.sdf" | wc -l

# === 停止所有 ===
pkill -f batch_diffdock_clean.py

# === 查看日志 ===
tail -f logs/diffdock_gpu0_8gpu_run.log
```

---

## ✅ 完成检查清单

迁移前 (旧集群):
- [ ] Git commit所有新脚本
- [ ] Git push到远程仓库
- [ ] 确认remaining_samples.txt存在

迁移后 (新集群):
- [ ] Git pull最新代码
- [ ] 同步data/和sample_data/目录
- [ ] 确认8个GPU可用 (nvidia-smi)
- [ ] 激活conda环境
- [ ] 启动8-GPU workers
- [ ] 第一次进度检查 (30分钟后)

对接完成后:
- [ ] 确认进度≥95% (3868+/4072)
- [ ] 同步SDF文件回旧集群 (如需要)
- [ ] 旧集群auto_pipeline自动触发训练

---

**预计完成时间**: 启动后5-6小时
**下一步**: 等待auto_pipeline自动开始训练 → 训练完成 → 分析结果 → 准备JCIM投稿
