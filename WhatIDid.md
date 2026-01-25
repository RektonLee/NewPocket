# PocketGNN 改进工作日志 (What I Did)

**时间**: 2026-01-24
**负责人**: Claude (AI Assistant)
**目标**: 改进PocketGNN的低同源序列表现，提升模型泛化能力

---

## 📋 任务分析与优先级规划

### 核心问题诊断
根据文档分析（Catpred.md, revise_advice0118.md, gemini.tex），当前项目存在以下主要问题：

1. **低同源表现不足**: 40%同源split下性能下降明显 (r=0.67, R²=0.44)
2. **模型架构偏"流水线"**: 缺乏结构创新，审稿人可能认为"方法论新意不足"
3. **缺少不确定性量化 (UQ)**: 不如CatPred有预测区间
4. **多构象/PoseSet未实现**: 依赖单一docking pose
5. **特征平衡问题**: ESM-2 (1280维) vs 图特征 (128-256维) 不平衡（已部分解决）

### 改进方案与优先级

#### 🔴 **P0 - 立即实施 (核心性能提升)**

1. **架构升级: Physics-informed Hierarchical Pocket Transformer (PHPT)**
   - [ ] 实现双通路GNN（几何流 + 电子流）
   - [ ] 添加Cross-Attention融合层
   - [ ] 实现Residual Graph Transformer Block（Pre-norm + Residual）
   - [ ] 层级池化（Atom → Residue → Pocket）
   - **预期提升**: 模型表达能力+30%，审稿评价从"工程"升级到"方法论"

2. **序列-结构双向交互（不是简单拼接）**
   - [ ] 实现Structure-Sequence Co-attention
   - [ ] 替代当前的late fusion
   - **预期提升**: 低同源split性能提升10-15%

#### 🟡 **P1 - 重要实验 (提升泛化)**

3. **严格的Scaffold Split实验**
   - [ ] 实现scaffold splitting（基于底物骨架）
   - [ ] 对比random/homology/scaffold三种split
   - **预期效果**: 证明模型真实泛化能力

4. **不确定性量化 (Ensemble + Quantile Regression)**
   - [ ] 实现10-model ensemble（aleatoric + epistemic）
   - [ ] 集成已有的quantile regression模块
   - **预期效果**: 提供预测置信区间，对齐CatPred

5. **多构象集成 (PoseSet)**
   - [ ] 为每个样本生成top-5 poses
   - [ ] 实现pose-level aggregation
   - **预期提升**: 鲁棒性+5-10%

#### 🟢 **P2 - 数据与特征工程**

6. **特征消融实验**
   - [ ] 系统性测试：距离/角度/扭转角/电子特征的贡献
   - [ ] 移除每类特征后的性能变化
   - **预期效果**: 明确特征重要性排序

7. **CatPred式数据处理**
   - [ ] kcat: 使用reactants concatenated SMILES
   - [ ] 实现重复测量聚合（kcat取max，Km取几何平均）
   - **预期提升**: 数据质量+噪声控制

---

## 📦 实施计划 (分支管理)

### Phase 1: 架构升级 (预计2-3天)
**分支**: `feature/hierarchical-transformer`

**任务清单**:
- [ ] 创建新模型类 `PHPTransformer` (Physics-informed Hierarchical Pocket Transformer)
- [ ] 实现双通路GNN（GeometryStream + ElectronicStream）
- [ ] 实现Cross-Attention融合模块
- [ ] 实现Residual + Pre-norm Transformer Block
- [ ] 实现层级池化（atom→residue→pocket）
- [ ] 更新训练脚本支持新模型
- [ ] 基准实验对比

**代码修改位置**:
- `src/GNN_model.py`: 新增 `PHPTransformer` 类
- `src/train.py`: 添加 `--model_type` 参数支持多模型选择

**实验配置**:
```bash
# 基准对比 (当前模型)
conda activate env2
Python3 src/train.py --dataset data/processed/kcat_full_1213.pt \
  --model_type PocketGNNKcatOnly \
  --cluster_tsv data/processed/mmseqs_40pct.tsv \
  --exp_name baseline_40pct

# 新模型实验
Python3 src/train.py --dataset data/processed/kcat_full_1213.pt \
  --model_type PHPTransformer \
  --cluster_tsv data/processed/mmseqs_40pct.tsv \
  --exp_name phpt_40pct
```

---

### Phase 2: 结构-序列双向交互 (预计1-2天)
**分支**: `feature/structure-seq-coattention`

**任务清单**:
- [ ] 实现Structure-Sequence Co-attention模块
- [ ] 替换当前的late fusion
- [ ] 对比实验（late fusion vs co-attention）

**代码修改位置**:
- `src/GNN_model.py`: 新增 `StructureSeqCoAttention` 模块

---

### Phase 3: 不确定性量化 (预计1-2天)
**分支**: `feature/ensemble-uq`

**任务清单**:
- [ ] 实现Gaussian NLL loss（输出mean + variance）
- [ ] 训练10个ensemble模型
- [ ] 计算aleatoric + epistemic uncertainty
- [ ] 可视化预测区间

**代码修改位置**:
- `src/quantile_loss.py`: 扩展为 `uncertainty_utils.py`，添加NLL loss
- `src/train.py`: 添加 `--output_variance` 和 `--ensemble_size` 参数
- 新增 `scripts/train_ensemble.sh`: 自动化ensemble训练脚本

---

### Phase 4: 多构象集成 (预计2-3天)
**分支**: `feature/pose-ensemble`

**任务清单**:
- [ ] 修改docking pipeline生成top-5 poses
- [ ] 实现pose-level feature aggregation
- [ ] 对比单pose vs multi-pose性能

**代码修改位置**:
- `src/docking.py`: 修改为返回多个poses
- `src/build_graph_dataset.py`: 支持多pose处理
- 新增 `src/pose_aggregation.py`: 实现pose ensemble策略

---

### Phase 5: Scaffold Split实验 (预计1天)
**分支**: `feature/scaffold-split`

**任务清单**:
- [ ] 实现scaffold splitting（使用RDKit的MurckoScaffold）
- [ ] 生成scaffold cluster文件
- [ ] 运行scaffold split实验

**代码修改位置**:
- 新增 `scripts/generate_scaffold_clusters.py`
- `src/train.py`: 复用已有的 `--cluster_tsv` 参数

---

## 📊 实验跟踪表

| 实验名称 | 分支 | Split类型 | Pearson (r) | R² | MAE | 状态 | 备注 |
|---------|------|----------|-------------|-----|-----|------|------|
| Baseline (当前) | diffdock-integration | 40% homology | 0.667 | 0.437 | - | ✅ 完成 | 论文中的基准 |
| PHPT模型 | feature/hierarchical-transformer | 40% homology | - | - | - | ⏳ 待运行 | 双通路+层级池化 |
| Co-Attention | feature/structure-seq-coattention | 40% homology | - | - | - | ⏳ 待运行 | 替换late fusion |
| Ensemble (N=10) | feature/ensemble-uq | 40% homology | - | - | - | ⏳ 待运行 | 不确定性量化 |
| Multi-Pose | feature/pose-ensemble | 40% homology | - | - | - | ⏳ 待运行 | Top-5 pose集成 |
| Scaffold Split | feature/scaffold-split | scaffold | - | - | - | ⏳ 待运行 | 底物泛化测试 |

---

## 🔬 当前基线性能 (Reference)

### Random Split (IID)
- Pearson r = 0.98
- R² = 0.918
- MAE = ~0.19

### 40% Homology Split (OOD)
- Pearson r = 0.667
- R² = 0.437
- **目标**: 提升到 r > 0.75, R² > 0.55

### SOTA对比 (40% homology)
- CatPred: r = 0.52
- CataPro: r = 0.497
- **PocketGNN (当前)**: r = 0.667
- **目标**: r > 0.75 (超越当前+12%以上)

---

## ✅ 已完成工作

### 2026-01-24 (之前的工作)
- [x] Extended `src/train.py` to support pre-split datasets and homology-aware split via MMseqs2
- [x] Added deterministic split controls (`--split_seed`, `--split_ratios`)
- [x] Updated `paperwriting/gemini.tex` (弱化随机划分、强调同源划分)
- [x] 新增 `scripts/audit_runs.py` 生成统一 `runs_registry.csv`
- [x] 新增 `scripts/run_homology_baselines.sh` (40/70/90 同源 split)

### 2026-01-24 (今日工作)
- [x] 阅读关键文档（Catpred.md, revise_advice0118.md, gemini.tex）
- [x] 分析当前代码结构（GNN_model.py, train.py）
- [x] 制定改进计划与优先级
- [x] 更新工作日志文档 (WhatIDid.md)
- [x] 创建 `feature/hierarchical-transformer` 分支
- [x] ✅ **完成PHPTransformer模型实现**:
  - [x] 实现双通路GNN（几何流 + 电子流）
  - [x] 实现Cross-Attention融合模块
  - [x] 实现Residual Graph Transformer Blocks（Pre-norm + Residual）
  - [x] 支持层级池化框架（当前用mean pooling作为placeholder）
- [x] 更新train.py支持--model_type参数
- [x] 更新metadata和wandb记录
- [x] 提交代码: commit `0bb1348`

---

## 📝 下一步行动 (Next Actions)

**今日计划 (Day 1 - 2026-01-24 下午)**:
1. ✅ 完成任务分析与规划
2. ✅ 创建 `feature/hierarchical-transformer` 分支
3. ✅ 实现 `PHPTransformer` 模型骨架
4. ✅ 实现双通路GNN（几何流+电子流）
5. ⏳ **运行基准对比实验** (下一步):
   - Baseline (PocketGNNKcatOnly) vs PHPTransformer
   - 使用40% homology pre-split数据集
   - 对比性能指标

**预期时间线**:
- **Week 1**: Phase 1-2 (架构升级 + Co-attention)
- **Week 2**: Phase 3-5 (UQ + PoseSet + Scaffold Split)
- **Week 3**: 论文更新 + 实验结果整理

---

## 🐛 已知问题与技术债

1. **温度特征**: 当前代码显式移除温度特征（见GNN_model.py L173），可能损失信息
2. **特征维度不平衡**: ESM-2 (1280维) vs 图特征 (128-256维) - 已部分解决（投影层）
3. **Docking依赖**: 依赖单一DiffDock pose，质量不稳定
4. **数据清洗**: 未实现CatPred式的重复测量聚合策略

---

## 📚 参考资料

- **CatPred (Nat. Commun. 2025)**: D-MPNN + ESM-2 + E-GNN, Ensemble UQ
- **revise_advice0118.md**: 物理语义分解 + 双通路建模 + 层级表征
- **论文 (gemini.tex)**: 当前版本，需要在Methods部分更新架构描述

---

*此文档持续更新，记录所有代码修改、实验结果和设计决策*

## 2026-01-24 (补充)
- 修复 DiffDock 调用参数（改用 protein_ligand_csv，输出目录查找支持 out_dir/complex_name），见 `src/docking.py`。
- 修复 `scripts/redock_test_new.py` 的 CSV 列名读取逻辑（支持带空格列名），并成功跑通 `--limit 1` 的 DiffDock 预演，生成 `data/processed/kcat_test_new_diffdock_preview.pt`（1 条样本）。
- 完成 DiffDock 输出持久化与置信度解析：在 `sample_data/.../docking/diffdock_output/` 保存多构象 SDF + `docking_meta.json`，并把最大 confidence 写入 `data.docking_confidence`。
- 预演成功：`scripts/redock_test_new.py --limit 1 --samples-per-complex 3 --gpu-ids 0,1` 生成 1 条样本与持久化输出。
- `scripts/redock_test_new.py` 支持使用已有 PDB 目录（`--protein-pdb-dirs`）并可禁用 ESMFold（`--no-esmfold`），避免重复预测。
- `scripts/redock_test_new.py` 增加预筛选（仅对确有 PDB 的样本做 docking），并将 RDKit 失败的 SMILES 追加到 `logs/rdkit_failed_smiles.txt`。
- `src/docking.py` 用 ETKDGv3 做 3D 嵌入并保留 fallback，提高 RDKit 构象成功率。
- 2026-01-25：小规模验证（limit=5）完成，成功 4/5；全量重对接已后台启动，日志：`logs/full_redock_test_new.log`（nohup 方式）。
- 2026-01-25: 修复 DiffDock 输出元数据的 confidence 解析：支持负号、按文件名排序并记录 confidence_map/best_confidence，避免之前只取正值且无法对应文件的问题。
- 2026-01-25: redock 脚本增加实时进度输出（处理/成功/失败/保留数量），并在每 100 个样本保留前 5 个的完整中间输出（DiffDock tmp 目录迁移到样本的 diffdock_output/tmp_keep）。
- 2026-01-25: redock 增加 GPU 轮询分配（--gpu-ids），进度日志中显示 GPU。
- 2026-01-25: DiffDock best pose 选择逻辑改为优先取最高 confidence 的 pose，只有缺失 confidence 时才回退到 rank1。
- 2026-01-25: redock 修复已有 PDB 同路径拷贝导致的 SameFileError。
