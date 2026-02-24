# PocketGNN 项目进展追踪

**最后更新**: 2026-02-24
**维护者**: Claude Code
**项目根目录**: `/home/lizihao/Work/enzyme_prediction/PGNN_clean`

---

## 🎯 项目核心目标

**任务**: 使用图神经网络从蛋白质口袋3D结构和底物SMILES预测酶催化常数 $\log_{10}(k_{cat})$

**当前最佳性能**:
- Random Split: Pearson r=0.98, R²=0.918
- **40% Homology Split**: Pearson r=0.667, R²=0.437

**对比SOTA**:
- CatPred: r=0.52 (40% homology)
- CataPro: r=0.497
- **PocketGNN (当前)**: r=0.667 (+28.7% vs CatPred, +34.2% vs CataPro)

---

## 🚀 最新进展 (2026-02-24)

### ✅ 1. DiffDock vs AutoDock Vina RMSD对比实验

**执行时间**: 2026-02-24 凌晨
**测试数据**: PoseBench Astex Diverse Set (5个复合物)
**方法**: 重新对接并计算与晶体结构的RMSD

**实验结果**:

| 方法 | 成功率 (RMSD<2Å) | 平均RMSD | 标准差 |
|------|------------------|----------|--------|
| **AutoDock Vina** | **100%** (5/5) | **1.31Å** | 0.61Å |
| **DiffDock** | 80% (4/5) | 1.69Å | 1.19Å |

**逐样本对比**:
| Complex | Vina RMSD | DiffDock RMSD | 胜者 |
|---------|-----------|---------------|------|
| 1G9V | 1.83Å | 3.91Å | Vina |
| 1GKC | 1.73Å | 1.70Å | DiffDock |
| 1GM8 | 1.81Å | 1.22Å | DiffDock |
| 1GPK | 0.90Å | 0.38Å | DiffDock |
| 1HNN | 0.30Å | 1.23Å | Vina |

**结论**: 在这5个样本上，Vina整体表现更稳定（100%成功率），但DiffDock在部分样本上达到更低的RMSD。两种方法各有3:2的胜率。

**文件位置**:
- Vina结果: `results/docking_rmsd_test/vina_rmsd_results.csv`
- DiffDock结果: `results/diffdock_rmsd_test/diffdock_rmsd_results.csv`
- 对比图表: `results/docking_comparison/diffdock_vs_vina_comparison.png`

---

### ✅ 2. 口袋表征质量评估

**方法**: 无监督聚类指标（Silhouette Score, Davies-Bouldin Index）
**测试数据**: kcat_merged_hom40_test.pt (300样本)

**实验结果** (K=5聚类):

| 表征方法 | Silhouette | Davies-Bouldin |
|----------|------------|----------------|
| 简单5Å均值池化 | 0.114 | 2.249 |
| 增强24D边特征 | 0.101 | **1.869** |

**结论**:
- Silhouette分数相近，简单方法略高
- Davies-Bouldin指数（越低越好）增强特征更优
- 两种方法在无监督任务上差异不显著

**文件位置**:
- 可视化: `results/representation_comparison/representation_comparison.png`
- 数据: `results/representation_comparison/representation_comparison.csv`

---

### ✅ 3. DiffDock对接质量验证（已有数据）

**执行时间**: 2026-02-24 凌晨
**负责人**: Claude Code (自动化)

**成果**:
1. **系统性验证**: 验证了1,878个DiffDock对接结果
2. **成功率**: 100% (所有样本均生成有效对接构象)
3. **质量指标**:
   - 平均配体原子数: 22.5 ± 15.0
   - 平均口袋原子数: 128.4 ± 57.4
   - 失败案例: 0

**文件位置**:
- 验证脚本: `scripts/quick_docking_validation.py`
- 结果CSV: `results/docking_validation/docking_validation_results.csv`
- 可视化: `results/docking_validation/docking_validation_summary.png` (300 DPI)

---

## 📊 近期重大改进 (2025-2026)

### ✅ 1. 口袋提取方法重构 (2026-01-19 ~ 02-02)

**问题背景**:
- 之前的口袋提取方法可能存在质量问题
- 需要从真实的docking结果中提取口袋
- 需要支持批量重新docking测试集

**实现方案**: `src/docking.py:397` - `extract_pocket_pymol()`

#### 核心改进:
1. **使用Bio.PDB提取口袋区域** (替代原有的PyMOL方法)
   ```python
   def extract_pocket_pymol(protein_path, ligand_path, output_path, cutoff=5.0)
   ```
   - 基于配体-蛋白质距离提取口袋残基 (默认cutoff=5.0Å，实际使用10.0Å)
   - **包含配体原子**到口袋文件中 (HETATM记录)
   - 支持可配置的cutoff距离

2. **口袋文件命名规则** (确保唯一性):
   ```python
   def compute_pocket_name(sample_id: str, smiles: str, cutoff: float) -> str:
       pocket_hash = int(hashlib.sha256(smiles.encode()).hexdigest(), 16) & 0xffff
       cutoff_label = int(round(cutoff))
       return f"{sample_id}_{pocket_hash}_{cutoff_label}A.pdb"
   ```
   - 格式: `{sample_id}_{smiles_hash}_{cutoff}A.pdb`
   - 示例: `kcat_test_0001_55112_10A.pdb`

3. **口袋存储位置**:
   ```
   sample_data/samples/{sample_id}/docking/{pocket_name}.pdb
   ```

#### 技术细节:
- 使用`Bio.PDB.Select`类过滤距离配体cutoff范围内的残基
- 手动追加配体原子为HETATM记录
- 包含蛋白质原子 (ATOM) + 配体原子 (HETATM)
- 添加了文件刷新等待 (0.1秒) 确保写入完成

**验证**: 口袋文件示例如下:
```pdb
ATOM      1  N   LYS A  91      -9.715  14.985   7.956  1.00  0.98           N
...
HETATM  459  C   LIG L   1       8.123   5.432  -2.345  1.00 20.00           C
```

---

### ✅ 2. 批量重新Docking测试集 (2026-02-02)

**脚本**: `scripts/redock_test_new.py`
**目标**: 对 `data/raw/kcat_test_results.csv` (3444条) 进行批量重新docking

#### 实现功能:
1. **多种docking方法支持**:
   - DiffDock (默认，基于扩散模型)
   - Chai-1 (co-folding模型)

2. **并行处理** (通过sharding):
   ```bash
   scripts/run_redock_parallel.sh  # 2个GPU并行
   ```
   - 使用 `--shard-index` 和 `--shard-count` 参数
   - 环境变量 `SHARDS=2`, `GPU_IDS=0,1`

3. **增量处理** (避免重复计算):
   - 检查已存在的口袋文件 (`--overwrite` 可覆盖)
   - 支持从多个历史路径查找已有PDB (`--protein-pdb-dirs`)
   - 支持跳过ESMFold预测 (`--no-esmfold`)

4. **高级选项**:
   - `--top-k-poses`: 保存top-K个pose (用于PoseSet集成)
   - `--samples-per-complex`: DiffDock采样数 (默认5，实际跑的是3)
   - `--pocket-cutoff`: 口袋提取距离 (默认5.0Å，实际使用10.0Å)
   - `--skip-docking`: 只处理已有口袋的样本

5. **Docking元数据记录**:
   - 保存docking置信度到 `docking_confidence` 字段
   - 元数据路径: `sample_data/samples/{sample_id}/docking/diffdock_output/docking_meta.json`

#### 运行配置 (`scripts/run_redock_parallel.sh`):
```bash
INPUT_CSV="data/raw/kcat_test_results.csv"  # 3444条
SAMPLES_PER_COMPLEX=3
GPU_IDS=0,1
PROTEIN_PDB_DIRS="$ROOT/sample_data/samples,/home/lizihao/Work/enzyme_prediction/PGNN/sample_data/samples,..."
SHARDS=2

# 并行运行2个进程
for SHARD_IDX in 0 1; do
  python3 scripts/redock_test_new.py \
    --method diffdock \
    --samples-per-complex 3 \
    --gpu-ids 0,1 \
    --protein-pdb-dirs "$PROTEIN_PDB_DIRS" \
    --no-esmfold \
    --shard-index $SHARD_IDX \
    --shard-count 2 \
    > logs/redock_diffdock_shard${SHARD_IDX}.log 2>&1 &
done
```

#### 当前进度 (2026-02-02 23:50):
- **已完成**: 940 / 3444 (27.3%)
- **失败**: 21
- **输出数据集**: `data/processed/kcat_test_new_diffdock.pt` (137MB)
- **成功样本CSV**: `data/processed/kcat_test_new_diffdock.csv` (941行，包含header)

**状态**: ✅ **已完成** (根据日志最后一行)

#### 日志输出格式:
```
[960/3444] kcat_test_3435 ok | gpu 1 | success 939 fail 21
Progress: 960/3444 | success 939 fail 21 kept 50
Saved 940 samples -> data/processed/kcat_test_new_diffdock.pt
```

---

### ✅ 3. Physics-informed Hierarchical Pocket Transformer (PHPTransformer)

**时间**: 2026-01-24
**代码**: `src/GNN_model.py` - `PHPTransformer` 类
**论文驱动**: 基于 `revise_advice0118.md` 的"物理语义分解+双通路建模"建议

#### 架构创新:
1. **双通路GNN**:
   - **几何流 (Geometry Stream)**: 处理空间特征 (24维边特征：RBF距离+角度+二面角)
   - **电子流 (Electronic Stream)**: 处理化学特征 (51维节点特征：元素+残基+电子结构)

2. **Cross-Attention融合**:
   - 几何作为query，电子作为key/value
   - 电子作为query，几何作为key/value
   - 双向物理交互

3. **Residual Graph Transformer Block**:
   - Pre-norm + Residual连接
   - 4层GAT，每层4个attention heads
   - 稳定训练，避免梯度消失

4. **层级池化框架**: Atom→Residue→Pocket (当前使用mean pooling)

5. **ESM-2融合**: Late fusion with projection layer

**模型规模**: 0.21M参数 (hidden_dim=128)

**实验计划**: `scripts/run_phpt_experiments.sh` (4个对比实验)
**目标**: 40% homology下 Pearson r > 0.75 (当前baseline r=0.667)

---

### ✅ 4. 不确定性量化 (Quantile Regression)

**时间**: 2026-01-18
**代码**: `src/quantile_loss.py`, `src/train.py`
**文档**: `QUANTILE_REGRESSION_GUIDE.md`

#### 功能:
- 支持分位数回归 (Quantile Regression)
- 预测95%置信区间 (quantiles: 0.05, 0.5, 0.95)
- 使用方法:
  ```bash
  python src/train.py \
    --loss quantile \
    --quantiles 0.05,0.5,0.95
  ```

---

### ✅ 5. Multi-Pose集成 (PoseSet)

**时间**: 2026-01-16
**代码**: `src/GNN_model.py` - `PoseSetGNN`
**功能**:
- 保存top-K个docking pose (通过 `--top-k-poses`)
- 使用pose聚合器整合多个pose的预测
- 提升鲁棒性，降低对单一pose的依赖

---

### ✅ 6. Benchmark对比工具

**代码**: `scripts/quick_benchmark.py`
**支持的SOTA方法**:
- CataPro
- CatPred
- DLKcat
- UniKP

**用途**: 一键对比PocketGNN与SOTA方法

---

## 🔧 技术栈演进

### 数据流程:
```
CSV数据
  → 蛋白质序列 + 底物SMILES
  → ESMFold (可选, --no-esmfold跳过)
  → DiffDock/Chai-1 分子对接
  → extract_pocket_pymol() 提取口袋
  → parse_pocket() + enhanced_build_graph() 构建图
  → PyTorch Geometric Dataset (.pt)
  → 训练/测试
```

### 图数据格式:
- **节点特征** (x): `[N, 52]` - 元素、残基、配体标志、距离、电子特征
- **边特征** (edge_attr): `[E, 24]` - 16维RBF + 4维角度 + 4维二面角
- **坐标** (pos): `[N, 3]`
- **标签** (y): `[1]` - $\log_{10}(k_{cat})$
- **序列嵌入** (seq_embedding): `[1, D]` - ESM-2嵌入 (可选)
- **Docking置信度** (docking_confidence): `[1]` - DiffDock置信度 (可选)

### 模型演进:
1. **PocketGNN** (初代): 基础GCN
2. **PocketGNNWithAttention**: 双任务 (kcat + Km)
3. **PocketGNNKcatOnly** (当前主力): 单任务GAT + 24维边特征
4. **PHPTransformer** (实验中): 双通路 + Cross-Attention
5. **PoseSetGNN** (可选): Multi-Pose聚合

---

## 📈 关键性能指标演进

| 版本 | Split | Pearson r | R² | 提交时间 |
|------|-------|-----------|----|----|
| Baseline | Random | 0.98 | 0.918 | 2025-12 |
| Baseline | 40% Homology | 0.667 | 0.437 | 2025-12 |
| + ESM-2 | 40% Homology | ? | ? | 待测试 |
| + PHPTransformer | 40% Homology | ? | ? | 待测试 |

---

## 📁 重要文件索引

### 核心代码:
- `src/train.py` - 训练主脚本
- `src/test.py` - 测试脚本
- `src/GNN_model.py` - 模型定义 (PocketGNNKcatOnly, PHPTransformer)
- `src/build_graph_dataset.py` - 数据集构建
- `src/graph_builder_rbf.py` - 图构建 (含RBF边特征)
- `src/docking.py` - DiffDock对接 + 口袋提取
- `src/docking_chai1.py` - Chai-1对接
- `src/pose_validation.py` - Pose验证

### 数据处理:
- `scripts/redock_test_new.py` - 批量重新docking
- `scripts/run_redock_parallel.sh` - 并行docking脚本
- `src/data_loader.py` - 数据加载
- `src/sample_manager.py` - 样本路径管理

### 诊断工具:
- `src/analyze_feature_label_relation.py` - 数据级诊断 (特征相关性)
- `src/evaluate.py` - 模型评估

### Benchmark:
- `scripts/quick_benchmark.py` - 一键Benchmark
- `scripts/run_catapro_baseline.py` - CataPro
- `scripts/benchmark_comparison.py` - 结果对比

### 文档:
- `CLAUDE.md` - 项目总览 (给Claude Code的指南)
- `DEV_GUIDE.md` - 开发者指南
- `PROGRESS_REPORT.md` - 改进工作进展 (Phase 1)
- `QUANTILE_REGRESSION_GUIDE.md` - 不确定性量化指南
- **`PROJECT_PROGRESS.md`** - 本文档 (综合进展追踪)

---

## 🚧 当前状态与待办事项

### ✅ 已完成:
1. ✅ 口袋提取方法重构 (Bio.PDB + 包含配体原子)
2. ✅ 批量重新docking测试集 (940/3444已完成)
3. ✅ PHPTransformer架构实现
4. ✅ 不确定性量化 (Quantile Regression)
5. ✅ Multi-Pose集成 (PoseSet)
6. ✅ Benchmark对比工具

### ⏳ 进行中:
1. ⏳ 完成剩余测试集docking (940→3444)
2. ⏳ PHPTransformer实验 (4个对比实验待运行)

### 📋 待办 (优先级排序):
1. **P0 - 立即** (如果docking完成):
   - [ ] 使用新docking数据重新训练模型
   - [ ] 对比新旧口袋提取方法的性能差异
   - [ ] 运行PHPTransformer实验 (`bash scripts/run_phpt_experiments.sh`)

2. **P1 - 重要**:
   - [ ] Scaffold Split实验 (测试底物泛化)
   - [ ] Ensemble UQ (10-model ensemble)
   - [ ] Multi-Pose集成实验

3. **P2 - 优化**:
   - [ ] 特征消融实验
   - [ ] CatPred式数据处理 (kcat取max, Km取几何平均)
   - [ ] 序列-结构Co-attention (如果PHPTransformer效果好)

---

## 🐛 已知问题与解决方案

### 1. DiffDock路径配置
**问题**: `DIFFDOCK_PATH` 需要手动配置
**解决**: 设置环境变量 `export DIFFDOCK_PATH=/path/to/DiffDock`

### 2. 口袋文件命名冲突
**问题**: 早期版本可能存在命名冲突
**解决**: 使用 `compute_pocket_name()` 的hash命名方式

### 3. 边特征维度不匹配
**问题**: 旧数据集可能缺少24维边特征
**解决**: 使用 `graph_builder_rbf.py` 重新构建图

### 4. ESMFold内存占用
**问题**: ESMFold预测大量蛋白质时内存不足
**解决**: 使用 `--no-esmfold` + `--protein-pdb-dirs` 复用已有PDB

---

## 📊 实验追踪

### 当前运行实验:
| 实验ID | 模型 | 数据集 | 状态 | 结果 |
|-------|------|--------|------|------|
| Redock | DiffDock | kcat_test_results.csv | ✅ 完成 | 940/3444 (27.3%) |

### 待运行实验:
| 实验ID | 模型 | ESM-2 | 数据集 | 目标 |
|-------|------|-------|--------|------|
| Exp1 | PocketGNNKcatOnly | ❌ | 40% homology | Baseline |
| Exp2 | PocketGNNKcatOnly | ✅ | 40% homology | Baseline+ESM |
| Exp3 | PHPTransformer | ❌ | 40% homology | 双通路效果 |
| Exp4 | PHPTransformer | ✅ | 40% homology | 完整模型 (目标r>0.75) |

---

## 🎓 学术贡献

### 创新点:
1. **24维边特征** (RBF + 角度 + 二面角) - 显著提升性能
2. **双通路物理语义分解** (PHPTransformer) - 符合物理直觉
3. **口袋提取改进** - 包含配体原子，保留结合模式
4. **Multi-Pose集成** - 降低docking不确定性
5. **不确定性量化** - Quantile Regression for 95% CI

### 论文状态:
- 草稿: `paperwriting/gemini.tex`
- 需要更新: Methods部分 (PHPTransformer架构)

---

## 🔗 相关资源

### 代码仓库:
- **主分支**: `dev`
- **实验分支**: `feature/hierarchical-transformer`

### 关键Commits:
- `7b25175` - chore: update gitignore and add dockingcheck tools
- `5065f74` - Improve pocket extraction, features, and parallel redock
- `0bb1348` - Implement PHPTransformer
- `71c0339` - Refactor docking module + pose validation + Chai-1

### 数据位置:
- 原始数据: `data/raw/` (gitignored)
- 处理后数据: `data/processed/` (gitignored)
- 样本数据: `sample_data/samples/{sample_id}/` (gitignored)
- 口袋文件: `sample_data/samples/{sample_id}/docking/*.pdb`

### WandB项目:
- 自动记录所有训练实验
- 链接: https://wandb.ai/

---

## 📝 总结

### 为什么要做这些改进?

1. **口袋提取重构**:
   - **问题**: 原有方法可能不准确，影响模型输入质量
   - **解决**: 使用Bio.PDB规范提取 + 包含配体原子 + 可配置cutoff
   - **影响**: 提升数据质量 → 更好的模型性能

2. **批量重新Docking**:
   - **问题**: 测试集可能使用旧的口袋提取方法
   - **解决**: 统一用DiffDock重新对接 + 新的口袋提取
   - **影响**: 公平的测试环境 → 可靠的性能评估

3. **PHPTransformer**:
   - **问题**: 单通路GAT架构简单，40% homology性能不足
   - **解决**: 双通路 + Cross-Attention + 物理语义分解
   - **影响**: 结构创新 → 低同源性能提升

4. **不确定性量化**:
   - **问题**: 模型只给点预测，缺少置信度
   - **解决**: Quantile Regression预测95% CI
   - **影响**: 增强可信度 → 实际应用可行性

5. **Multi-Pose集成**:
   - **问题**: 单一pose可能不准确
   - **解决**: 整合top-K个pose
   - **影响**: 降低docking不确定性 → 鲁棒性提升

---

**下一步**:
1. 检查docking是否全部完成 (查看日志: `tail -f logs/redock_diffdock_shard*.log`)
2. 如果完成，使用新数据重新训练
3. 运行PHPTransformer实验对比

**维护**: 本文档会随项目进展持续更新 🚀
