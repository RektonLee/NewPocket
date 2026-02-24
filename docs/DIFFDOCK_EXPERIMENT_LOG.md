# DiffDock训练集对接实验日志

**日期**: 2026-02-24
**目标**: 使用DiffDock对kcat_full_1213.csv训练集进行批量对接，以替换旧的对接方法

---

## 实验背景

### 问题发现

使用旧模型（`kcat_after_new`）在新的测试集（`kcat_test_new_diffdock.pt`, 940个DiffDock对接样本）上测试，结果极差：
- Pearson r: **0.160** (几乎随机)
- R²: **0.021**

**原因分析**:
1. 旧模型在旧对接方法的数据上训练
2. 新测试集使用DiffDock对接
3. 对接方法的差异导致模型完全失效

### 解决方案

重新使用DiffDock对全部训练集进行对接，确保训练集和测试集使用相同的对接方法。

---

## 数据统计

### 训练集分析 (kcat_full_1213.csv)

| 类别 | 数量 | 百分比 | 状态 |
|------|------|--------|------|
| 总样本数 | 9061 | 100% | - |
| 有PDB结构 | 4072 | 45% | ✅ 可对接 |
| 无PDB结构 | 4989 | 55% | ❌ 无法对接 |

**无PDB结构的样本**:
- Index范围: 4072-9060
- 可能原因: 数据集后期添加，未完成结构预测
- 解决方案: 需要ESMFold预测或从其他来源获取

### 测试集统计

| 数据集 | 样本数 | 对接方法 | 用途 |
|--------|--------|----------|------|
| kcat_test_new_diffdock.pt | 940 | DiffDock | 最终测试 |
| kcat_train_after_new_clean.pt | 4987 | 旧方法 | 旧模型训练集 |

---

## 实验设置

### DiffDock配置

- **模型**: DiffDock (env2环境)
- **参数**:
  - `--inference_steps 20`
  - `--samples_per_complex 1`
  - `--no_final_step_noise`
- **输入**: 蛋白PDB + 底物SMILES
- **输出**: 对接复合物SDF文件

### 批量处理策略

**并行方案**: 4 GPU并行处理
- GPU 0: samples 0-1018
- GPU 1: samples 1018-2036
- GPU 2: samples 2036-3054
- GPU 3: samples 3054-4072

**脚本**: `scripts/batch_diffdock_clean.py`
- 从PGNN目录提取蛋白质结构
- 移除HETATM行（配体）
- 使用蛋白质+SMILES进行blind docking
- 保存结果到 `sample_data/samples/{sample_id}/docking/`

**启动方式**:
```bash
bash scripts/launch_parallel_diffdock.sh
```

---

## 时间估算

- **单样本处理时间**: ~60秒
- **总样本数**: 4072
- **并行GPU数**: 4
- **预计总时间**: 4072 × 60 / 4 / 3600 ≈ **17小时**
- **开始时间**: 2026-02-24 19:44
- **预计完成**: 2026-02-25 12:44

---

## 监控与管理

### 监控命令

```bash
# 查看整体进度
python scripts/check_diffdock_progress.py

# 查看实时日志
tail -f logs/diffdock_gpu0.log  # GPU 0
tail -f logs/diffdock_gpu*.log  # 所有GPU

# 检查运行进程
ps aux | grep batch_diffdock_clean
```

### 管理命令

```bash
# 停止所有worker
bash scripts/stop_diffdock.sh

# 重启worker
bash scripts/launch_parallel_diffdock.sh
```

---

## 自动化Pipeline

**脚本**: `scripts/auto_pipeline.py`

**功能**:
1. **等待DiffDock完成** (检查间隔5分钟)
2. **构建图数据集** (`build_graph_dataset.py`)
3. **训练模型** (300 epochs, plateau scheduler)
4. **评估测试集** (kcat_test_new_diffdock.pt)

**启动**:
```bash
nohup python scripts/auto_pipeline.py > logs/auto_pipeline.log 2>&1 &
```

**状态**:
- PID: 1691262
- 日志: `logs/auto_pipeline.log`
- 状态: 等待DiffDock完成

---

## 预期结果

### 成功指标

1. **对接完成率**: >95% (3868/4072)
2. **数据集构建**: `data/processed/kcat_train_diffdock.pt` 生成
3. **模型性能** (在测试集上):
   - Pearson r > 0.5 (显著优于0.160的baseline)
   - R² > 0.25

### 风险与备份方案

**风险1**: DiffDock失败率过高
- 备份方案: 使用部分成功样本训练，评估最小样本量

**风险2**: 训练数据太少（<2000样本）
- 备份方案: 结合旧数据集进行迁移学习

**风险3**: 新模型性能仍然很差
- 原因分析: DiffDock对接质量、特征提取、模型架构
- 需要diagnostic分析

---

## 后续工作

1. **处理无PDB样本** (4989个)
   - 使用ESMFold预测结构
   - 或寻找AlphaFold预测结构

2. **模型优化**
   - 尝试PHPTransformer架构
   - 添加ESM-2序列embedding
   - 调整超参数

3. **数据增强**
   - Multi-pose训练
   - 数据增强技术

---

## 实验记录

### 2026-02-24 19:30 - 启动批量对接

- 发现后半部分样本(index >4072)没有PDB
- 调整任务分配，只处理有PDB的4072个样本
- 4个GPU全部启动成功

### 2026-02-24 19:47 - 启动自动pipeline

- auto_pipeline.py 运行中
- 将自动处理后续的数据集构建和训练
- 预计明天中午完成全部流程

---

## 参考

- DiffDock论文: Corso et al., ICLR 2023
- 项目文档: `PROJECT_DOCUMENTATION.md`
- 脚本目录: `scripts/`
  - `batch_diffdock_clean.py`: 单个worker
  - `launch_parallel_diffdock.sh`: 启动4个worker
  - `stop_diffdock.sh`: 停止所有worker
  - `check_diffdock_progress.py`: 检查进度
  - `auto_pipeline.py`: 自动化pipeline
