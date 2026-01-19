# WhatAiDid.md - AI 辅助开发变更记录

> 本文档记录了 AI（Cursor/Claude）辅助开发过程中的主要变更和时间线，便于追踪项目演进和复现实验。

---

## 📅 变更时间线

### 2026-01-19

#### 1. 创建项目技术文档 (`paperwriting/TECHNICAL_DOCUMENTATION.md`)

| 项目 | 内容 |
|------|------|
| **类型** | 文档创建 |
| **文件** | `paperwriting/TECHNICAL_DOCUMENTATION.md` |
| **目的** | 为论文撰写和 AI 辅助审稿提供完整的技术 context |
| **内容** | 项目概览、方法论、模型架构、数据流程、训练框架、实验结果、相关工作定位 |
| **行数** | ~1300 行 |

#### 2. 整合 SOTA 相关工作分析

| 项目 | 内容 |
|------|------|
| **类型** | 文档扩展 |
| **来源** | `paperwriting/sotawork.md` |
| **新增内容** | |
| - Section 2.2 | Evolution of Computational Approaches (2022-2025) |
| - Section 9 | 扩展至 6 个 Tier 的 SOTA 方法详细分析 + Comprehensive Method Matrix |
| - Appendix D | Paper Writing Strategy（叙事定位、审稿问题应对、实验建议） |
| - Appendix E | Quick Reference for AI Context |

#### 3. 对接工具评估与代码修复 ✅

| 项目 | 内容 |
|------|------|
| **类型** | 代码重构 + 新模块创建 |
| **问题** | AutoDock Vina → DiffDock 的迁移存在潜在问题 |
| **分析** | 见下方 [对接工具评估](#对接工具评估autodock-vina-vs-diffdock-vs-chai-1) |

**已完成的代码变更:**

| 文件 | 操作 | 变更内容 |
|------|------|----------|
| `src/docking.py` | 重构 | 修复硬编码路径、GPU ID 可配置化、添加 CLI 接口、改进错误处理 |
| `src/pose_validation.py` | **新建** | Pose 验证模块：clash check、stereochemistry、bond geometry |
| `src/docking_chai1.py` | **新建** | Chai-1 对接模块：完整的 API 和 CLI 支持 |

---

## 🔧 对接工具评估：AutoDock Vina vs DiffDock vs Chai-1

### 背景：项目对接工具演进

| 时间 | 工具 | 状态 | 问题 |
|------|------|------|------|
| 初始版本 | AutoDock Vina | 稳定但繁琐 | 需手动设定 grid box/pocket，对 blind docking 效果不稳定 |
| `diffdock-integration` 分支 | DiffDock | 当前使用 | 存在潜在 bug：pose 物理合理性、手性问题、盲对接泛化问题 |
| 建议引入 | Chai-1 | 待评估 | 新一代 co-folding 模型，支持约束条件 |

### 工具对比分析

| 工具 | 核心方法 | Blind Docking 成功率 (RMSD ≤2Å) | 已知 Pocket 成功率 | 优点 | 缺点 |
|------|----------|--------------------------------|-------------------|------|------|
| **AutoDock Vina** | 经验评分 + 搜索算法 | ~50-60% | ~65-70% | 开源、稳定、社区广 | 需手动设定口袋，柔性处理弱 |
| **DiffDock** | 扩散模型生成 | ~50-60% (v2.2) | 较好 | 生成多样性好，减少手动设定 | 物理合理性问题，评分函数弱 |
| **Chai-1** | AF-like co-folding | ~77-81% (with constraints ~85%) | 优秀 | 多模态支持，约束条件 | 资源需求高，后处理需要 |

### DiffDock 当前代码中的潜在问题

基于 `src/docking.py` 的代码审查，发现以下潜在问题：

#### 1. **路径依赖问题** (Line 23-27)
```python
# 当前实现
DIFFDOCK_PATH = os.path.abspath(os.environ.get("DIFFDOCK_PATH", default_diffdock_path))
```
- 问题: 依赖环境变量或默认路径，配置错误会导致运行失败
- 建议: 添加启动时路径验证

#### 2. **输出文件解析脆弱** (Line 494-518)
```python
# 当前实现
for f in os.listdir(output_dir):
    if f.endswith('.sdf'):
        if 'rank1' in f.lower() or 'rank_1' in f.lower():
            output_files.insert(0, file_path)
```
- 问题: 依赖文件名模式匹配，DiffDock 版本更新可能改变输出格式
- 建议: 解析 DiffDock 的 confidence score 文件，而非依赖文件名

#### 3. **缺少物理合理性检查** ⚠️ **关键问题**
当前代码流程：
```
DiffDock output → sdf_to_pdb → extract_pocket_pymol → Done
```
**缺失步骤**:
- ❌ 原子间 clash 检测（是否有原子重叠）
- ❌ 键长/键角合理性检查
- ❌ 配体手性验证（DiffDock 可能生成错误的 stereoisomer）
- ❌ 结合能量评估

**建议添加**:
```python
def validate_pose(pose_pdb):
    """验证 pose 的物理合理性"""
    # 1. Clash check
    # 2. Bond geometry check  
    # 3. Stereochemistry check
    return is_valid, issues
```

#### 4. **硬编码 GPU ID** (Line 586)
```python
gpu_id=1  # 使用GPU 1（因为GPU 0被其他任务占用）
```
- 问题: 硬编码 GPU ID，在不同机器上可能失败
- 建议: 通过参数或环境变量配置

#### 5. **临时目录管理问题** (Line 554, 639)
```python
tmp_dir = f"tmp/{base_name}"  # 相对路径
# ...
subprocess.run(["rm", "-rf", tmp_dir], timeout=10)  # 可能残留
```
- 问题: 使用相对路径的临时目录，可能导致文件残留或冲突
- 建议: 使用 `tempfile.mkdtemp()` 管理

#### 6. **失败日志硬编码路径** (Line 632)
```python
with open("/home/lizihao/Work/enzyme_prediction/src/simple2/data/failed_samples.txt", "a") as f:
```
- 问题: 硬编码绝对路径，不可移植
- 建议: 使用配置文件或相对路径

### 推荐改进方案

#### 方案 A: DiffDock + 传统评分重排序（短期）

```
DiffDock 生成多个 pose → Vina/Gnina 评分重排序 → 物理合理性检查 → 选择最佳 pose
```

**优点**: 最小改动，保留现有代码
**缺点**: 仍依赖 DiffDock 的生成质量

#### 方案 B: 引入 Chai-1（推荐中长期）

```
蛋白序列/结构 + 配体 SMILES → Chai-1 co-folding → 复合物结构 → 口袋提取
```

**优点**: 
- 不需要单独的对接步骤
- 支持 apo 结构输入
- PoseBusters benchmark 表现优秀

**缺点**:
- 需要较大 GPU 资源（推荐 A100/H100）
- 安装配置较复杂

#### 方案 C: 混合策略（最稳健）

```
1. 已知口袋 → AutoDock Vina（可靠）
2. 未知口袋 → Chai-1 co-folding（新方案）
3. 验证/备用 → DiffDock + 重排序
```

### Chai-1 集成指南

```bash
# 安装 Chai-1
pip install chai_lab

# 基本使用示例
from chai_lab.chai1 import run_inference

# 准备输入
fasta_content = """
>protein|name=enzyme
MKFLILLFNILCLFPVLAADNHGVGPQGASGVDPITFDINSNQTGVQSLTNDGDNTVQWNSDKGSSYKVE...
>ligand|name=substrate
CCO  # 底物 SMILES
"""

# 运行预测
run_inference(
    fasta_file="input.fasta",
    output_dir="output/",
    num_trunk_recycles=3,
    num_diffn_timesteps=200,
    seed=42,
    use_esm_embeddings=True
)
```

### 后续待办事项

- [ ] 在小规模数据集上对比 Vina vs DiffDock vs Chai-1
- [x] ~~实现物理合理性检查模块（clash check, bond length/angle）~~ → `src/pose_validation.py`
- [ ] 评估 Chai-1 的 GPU 资源需求
- [ ] 考虑实现 RAPID-Net 口袋预测 + Vina 组合方案
- [ ] 测试 `docking_chai1.py` 在 A100/H100 上的性能

---

## 📁 文件变更记录

| 日期 | 文件 | 操作 | 说明 |
|------|------|------|------|
| 2026-01-19 | `paperwriting/TECHNICAL_DOCUMENTATION.md` | 创建 | 完整技术文档 |
| 2026-01-19 | `paperwriting/TECHNICAL_DOCUMENTATION.md` | 更新 | 整合 SOTA 相关工作 |
| 2026-01-19 | `WhatAiDid.md` | 创建 | 本文档 |
| 2026-01-19 | `src/docking.py` | **重构** | 修复硬编码问题、添加配置系统、CLI 接口 |
| 2026-01-19 | `src/pose_validation.py` | **新建** | Pose 验证：clash check、stereochemistry、bond geometry |
| 2026-01-19 | `src/docking_chai1.py` | **新建** | Chai-1 对接模块 |

---

## 📚 参考资源

### 对接工具

- [DiffDock GitHub](https://github.com/gcorso/DiffDock)
- [Chai-1 GitHub](https://github.com/chaidiscovery/chai-lab)
- [AutoDock Vina](https://vina.scripps.edu/)
- [Gnina](https://github.com/gnina/gnina)

### 相关论文

- DiffDock: Corso et al., ICLR 2023
- Chai-1: Chai Discovery, 2024
- PoseBusters benchmark: Buttenschoen et al., 2023 ([arXiv](https://arxiv.org/abs/2308.05777))

---

---

## 🆕 2026-01-19 代码变更详情

### `src/docking.py` 重构

**修复的问题:**

1. **硬编码路径** → 使用 `DockingConfig` dataclass 配置
2. **硬编码 GPU ID** → 支持环境变量 `DOCKING_GPU_ID` 或配置
3. **脆弱的输出解析** → 新增 `_find_best_pose()` 函数
4. **缺少验证** → 集成 `pose_validation` 模块
5. **临时目录管理** → 使用 `tempfile.mkdtemp()`
6. **失败日志硬编码** → 自动生成到 `logs/` 目录

**新增功能:**

```python
# 配置化 API
from docking import DockingConfig, set_config, run_diffdock

config = DockingConfig(
    gpu_id=0,
    inference_steps=20,
    samples_per_complex=5,
    validate_pose=True
)
set_config(config)

# CLI 支持
python src/docking.py check          # 检查安装
python src/docking.py dock --protein p.pdb --ligand l.sdf --output out/
python src/docking.py batch --csv data.csv --pdb-dir pdbs/ --output-dir pockets/
```

### `src/pose_validation.py` 新模块

**功能:**

| 函数 | 功能 |
|------|------|
| `check_clash_pdb()` | 蛋白质-配体 clash 检测 |
| `check_internal_clash_sdf()` | 配体内部 clash 检测 |
| `check_stereochemistry()` | 手性中心验证 |
| `check_bond_geometry()` | 键长/键角合理性检查 |
| `validate_pose()` | 综合验证（推荐使用） |
| `validate_pose_quick()` | 快速验证（仅内部 clash） |

**使用示例:**

```python
from pose_validation import validate_pose, ValidationResult

result: ValidationResult = validate_pose(
    protein_pdb="protein.pdb",
    ligand_sdf="ligand.sdf",
    original_smiles="CCO",  # 用于手性验证
    check_clash=True,
    check_stereo=True,
    check_geometry=True
)

print(result)  # 输出验证结果
print(result.is_valid)  # True/False
print(result.issues)    # 问题列表
print(result.metrics)   # 数值指标
```

### `src/docking_chai1.py` 新模块

**功能:**

| 函数 | 功能 |
|------|------|
| `check_chai1_available()` | 检查 Chai-1 是否安装 |
| `check_gpu_memory()` | 检查 GPU 显存是否足够 |
| `prepare_fasta_input()` | 准备 Chai-1 FASTA 输入 |
| `run_chai1_inference()` | 运行 Chai-1 推理 |
| `dock_with_chai1()` | 高级 API（推荐使用） |
| `dock_with_chai1_mock()` | Mock 实现（测试用） |

**使用示例:**

```python
from docking_chai1 import dock_with_chai1, Chai1Config

result = dock_with_chai1(
    protein_sequence="MKFLIL...",
    ligand_smiles="CCO",
    output_dir="output/",
    config=Chai1Config(num_diffn_timesteps=200),
    extract_pocket=True
)

if result['success']:
    print(f"Complex: {result['complex_pdb']}")
    print(f"Pocket: {result['pocket_pdb']}")
```

**CLI:**

```bash
# 检查安装
python src/docking_chai1.py --check

# 运行对接
python src/docking_chai1.py --sequence "MKFLIL..." --smiles "CCO" --output out/

# Mock 模式（测试）
python src/docking_chai1.py --sequence "MKFLIL..." --smiles "CCO" --output out/ --mock
```

---

*Last updated: 2026-01-19*

