# PocketGNN 增强工具包使用指南

## 📋 概述

根据你和老师的讨论，我实现了以下核心工具来解决对接准确性验证和图构建优化的问题：

1. **PDB 复合物筛选工具** (`dockingcheck/find_pdb_complexes.py`) - 找出数据集中有实验结构的样本
2. **Re-docking 验证工具** (`dockingcheck/redock_validation.py`) - 使用 RMSD 验证对接准确性
3. **增强版图构建模块** (`dockingcheck/graph_builder_enhanced.py`) - 几何截断 + 同残基原子聚合
4. **外部基准集清单抽取** (`dockingcheck/collect_external_complexes.py`) - PDBbind / Binding MOAD

---

## 🔧 工具 1: PDB 复合物筛选

### 功能
从你的 kcat/Km 数据集中，筛选出在 PDB 数据库中有真实实验结构的酶-底物复合物。

### 使用方法

```bash
# 基础用法（输出 EC 级别匹配）
python dockingcheck/find_pdb_complexes.py \
    --input data/processed/kcat_train.pt \
    --output results/pdb_matched_ec.csv

# 你也可以对测试集进行筛选
python dockingcheck/find_pdb_complexes.py \
    --input data/processed/kcat_test_new.pt \
    --output results/pdb_matched_test_ec.csv

# 需要样本级别输出时，加 --sample_output
python dockingcheck/find_pdb_complexes.py \
    --input data/processed/kcat_train.pt \
    --output results/pdb_matched_ec.csv \
    --sample_output results/pdb_matched_samples.csv
```

**常用参数**
- `--max_pdb_per_ec`: 每个 EC 最多检查多少个 PDB（默认 5）
- `--limit_ec`: 仅处理前 N 个 EC（用于快速试跑）

### 输出结果

生成 EC 级别 CSV，包含以下列：
- `ec`: EC 号
- `matched_pdb`: 匹配到的 PDB ID
- `method`: 实验方法（X-RAY DIFFRACTION, NMR 等）
- `resolution`: 分辨率（Å）
- `year`: 沉积年份
- `sample_count`: 该 EC 对应的样本数
- `sample_ids`: 该 EC 对应的样本ID（分号分隔）

如需样本级别输出，加 `--sample_output`。

### 工作原理

1. 从你的 `.pt` 数据集中提取 EC 号
2. 使用 RCSB PDB API 查询该 EC 号对应的所有 PDB 结构
3. 筛选包含配体的复合物结构
4. 获取结构元数据（分辨率、方法等）

### 示例输出

```
=== Summary ===
Total samples: 1455
Matched EC-PDB pairs: 234
Unique EC numbers with PDB: 89
Unique PDB IDs: 178

=== Top 10 Matched PDBs ===
          ec matched_pdb  resolution             method  sample_count
0          1        1ABC        1.8  X-RAY DIFFRACTION          320
1        3.2        2XYZ        2.1  X-RAY DIFFRACTION           45
...
```

---

## 🔧 工具 2: Re-docking 验证

### 功能
对已有实验结构的复合物进行 Re-docking 验证，计算 RMSD 来评估对接准确性。

### 环境准备

```bash
# 安装 AutoDockTools 和 Vina
conda install -c bioconda autodock-vina
conda install -c bioconda mgltools

# 或者使用 pip
pip install vina
```

### 使用方法

```bash
# 单个 PDB 验证
python dockingcheck/redock_validation.py \
    --pdb 1ABC \
    --output results/validation/

# 指定配体名称（如果 PDB 中有多个配体）
python dockingcheck/redock_validation.py \
    --pdb 1ABC \
    --ligand ATP \
    --output results/validation/
```

### 批量验证

```bash
# 从上一步的筛选结果中批量验证
# 首先创建一个简单的脚本
cat > scripts/batch_redock.sh << 'EOF'
#!/bin/bash
# 读取 pdb_matched_ec.csv，批量验证

while IFS=, read -r ec matched_pdb method resolution year sample_count sample_ids; do
    if [ "$matched_pdb" != "matched_pdb" ]; then  # 跳过表头
        echo "Validating $matched_pdb..."
        python dockingcheck/redock_validation.py \
            --pdb $matched_pdb \
            --output results/validation/${matched_pdb}/
    fi
done < results/pdb_matched_ec.csv
EOF

chmod +x scripts/batch_redock.sh
./scripts/batch_redock.sh
```

### 输出结果

```
=== Re-docking Validation Result ===
PDB ID: 1ABC
RMSD: 1.24 Å
Success: ✓ (threshold: 2.0 Å)
```

生成的文件：
- `1ABC_protein.pdb` - 提取的蛋白结构
- `1ABC_ligand_ref.pdb` - 配体真实位置（参考）
- `docked_ligand.pdbqt` - 对接结果

### RMSD 判断标准

- **RMSD < 2.0 Å**: 对接成功 ✓
- **RMSD 2.0-4.0 Å**: 对接可接受
- **RMSD > 4.0 Å**: 对接失败 ✗

---

## 🔧 工具 3: 增强版图构建

### 功能
优化图构建方式：在几何截断的基础上，保留完整残基，避免切断氨基酸的化学完整性。

### 对比原始方法

| 方法 | 描述 | 问题 |
|------|------|------|
| **原始方法** | 10Å 球形截断，只保留球内原子 | 可能切断残基，丢失化学信息 |
| **增强方法** | 10Å 截断 + 扩展到完整残基 | 保留残基完整性，增加化学语义 |

### 使用方法

#### 方法 A：直接替换原有模块

```python
# 在你的训练脚本中，替换导入
# from src.graph_builder_rbf import parse_pocket, build_graph
from dockingcheck.graph_builder_enhanced import parse_pocket_enhanced, build_graph_enhanced

# 使用新方法
atoms = parse_pocket_enhanced(
    pdb_path,
    cutoff=10.0,           # 几何截断半径
    expand_residues=True    # 扩展到完整残基
)

data = build_graph_enhanced(atoms, temperature=298.15)
```

#### 方法 B：对比实验

```python
# 构建两个数据集进行对比
from src.graph_builder_rbf import parse_pocket, build_graph
from dockingcheck.graph_builder_enhanced import parse_pocket_enhanced, build_graph_enhanced

# 原始方法
atoms_old = parse_pocket(pdb_path)
data_old = build_graph(atoms_old, temperature=298.15)

# 增强方法
atoms_new = parse_pocket_enhanced(pdb_path, cutoff=10.0, expand_residues=True)
data_new = build_graph_enhanced(atoms_new, temperature=298.15)

print(f"原始方法：{data_old.num_nodes} 个节点")
print(f"增强方法：{data_new.num_nodes} 个节点 (+{data_new.num_nodes - data_old.num_nodes})")
```

### 参数调节

```python
# 如果扩展后节点太多，可以减小截断半径
atoms = parse_pocket_enhanced(pdb_path, cutoff=8.0, expand_residues=True)

# 如果不想扩展残基，回退到纯几何截断
atoms = parse_pocket_enhanced(pdb_path, cutoff=10.0, expand_residues=False)

# 手动指定配体坐标（更精确）
ligand_coords = np.array([[x1, y1, z1], [x2, y2, z2], ...])
atoms = parse_pocket_enhanced(pdb_path, ligand_coords=ligand_coords, cutoff=10.0)
```

---

## 🔧 工具 4: 外部基准集清单抽取（PDBbind / Binding MOAD）

### 功能
从外部基准集的索引文件中抽取复合物 PDB ID，方便后续做 re-docking 验证。

### 使用方法

```bash
# PDBbind：解析 INDEX_general_PL.* 文件
python dockingcheck/collect_external_complexes.py \
    --pdbbind-index /path/to/INDEX_general_PL.2020 \
    --output results/pdbbind_ids.csv

# Binding MOAD：从列表/文本中抽取 PDB ID
python dockingcheck/collect_external_complexes.py \
    --moad-list /path/to/BindingMOAD_list.txt \
    --output results/moad_ids.csv

# 同时合并输出
python dockingcheck/collect_external_complexes.py \
    --pdbbind-index /path/to/INDEX_general_PL.2020 \
    --moad-list /path/to/BindingMOAD_list.txt \
    --output results/external_ids.csv
```

---

## 📊 完整工作流程

### Step 1: 筛选有 PDB 的样本

```bash
python dockingcheck/find_pdb_complexes.py \
    --input data/processed/kcat_train.pt \
    --output results/pdb_matched_ec.csv \
    --sample_output results/pdb_matched_samples.csv
```

### Step 2: 验证对接准确性

```bash
# 选择几个高分辨率的 PDB 进行验证
python dockingcheck/redock_validation.py --pdb 1ABC --output results/validation/1ABC/
python dockingcheck/redock_validation.py --pdb 2XYZ --output results/validation/2XYZ/
```

### Step 3: 分析 RMSD 结果

```python
import pandas as pd

# 收集所有验证结果
results = []
for pdb_id in ['1ABC', '2XYZ', ...]:
    result = validate(pdb_id)
    results.append(result)

df = pd.DataFrame(results)
print(f"对接成功率: {(df['rmsd'] < 2.0).mean() * 100:.1f}%")
print(f"平均 RMSD: {df['rmsd'].mean():.2f} Å")
```

### Step 4: 使用增强版图构建重新训练模型

```python
# 修改你的数据预处理脚本
from dockingcheck.graph_builder_enhanced import parse_pocket_enhanced, build_graph_enhanced

for idx, row in tqdm(df.iterrows(), total=len(df)):
    # ... 获取 pdb_path ...

    # 使用增强版图构建
    atoms = parse_pocket_enhanced(pdb_path, cutoff=10.0, expand_residues=True)
    data = build_graph_enhanced(atoms, temperature)
    data.y = torch.log10(torch.tensor([kcat, km], dtype=torch.float))
    dataset.append(data)

torch.save(dataset, 'data/processed/dataset_enhanced.pt')
```

### Step 5: 对比实验

```bash
# 训练原始模型
python src/train.py \
    --dataset data/processed/kcat_train.pt \
    --save_dir results/baseline/

# 训练增强模型
python src/train.py \
    --dataset data/processed/dataset_enhanced.pt \
    --save_dir results/enhanced/

# 对比结果
python src/evaluate.py --model results/baseline/best_model.pt
python src/evaluate.py --model results/enhanced/best_model.pt
```

---

## 🎓 教程：从零开始

### 场景 1: 我想验证现有对接流程是否准确

```bash
# 1. 筛选有 PDB 的样本
python dockingcheck/find_pdb_complexes.py \
    --input data/processed/kcat_train.pt \
    --output results/pdb_matched_ec.csv

# 2. 选择 10 个高质量的 PDB（分辨率 < 2.0 Å）
head -n 11 results/pdb_matched_ec.csv > results/top10_pdbs.csv

# 3. 批量验证
for pdb in $(tail -n +2 results/top10_pdbs.csv | cut -d',' -f4); do
    python dockingcheck/redock_validation.py --pdb $pdb --output results/validation/$pdb/
done

# 4. 查看结果
grep "RMSD" results/validation/*/output.log
```

### 场景 2: 我想优化图构建方式

```python
# 先做对比实验，看看扩展残基的影响
import torch
from src.graph_builder_rbf import parse_pocket, build_graph
from dockingcheck.graph_builder_enhanced import parse_pocket_enhanced, build_graph_enhanced

# 加载数据集
dataset_old = torch.load('data/processed/kcat_train.pt')

# 随机选择 10 个样本对比
import random
sample_indices = random.sample(range(len(dataset_old)), 10)

for idx in sample_indices:
    sample = dataset_old[idx]
    pdb_path = f"data/processed/pockets/{sample.pdb_id}"

    # 原始方法
    atoms_old = parse_pocket(pdb_path)

    # 增强方法
    atoms_new = parse_pocket_enhanced(pdb_path, cutoff=10.0, expand_residues=True)

    print(f"Sample {idx}: {len(atoms_old)} -> {len(atoms_new)} atoms "
          f"(+{len(atoms_new) - len(atoms_old)})")
```

### 场景 3: 我想找 AlphaFold 3 的预测结构

```python
# 暂时 AlphaFold 3 的结构数据库还不完善
# 可以使用 AlphaFold Server 手动提交预测

# 1. 准备输入：从你的数据集中提取序列和 SMILES
import torch
dataset = torch.load('data/processed/kcat_train.pt')

# 提取前 10 个样本的信息
with open('results/af3_submission.txt', 'w') as f:
    for i, sample in enumerate(dataset[:10]):
        f.write(f"Sample {i}:\n")
        f.write(f"EC: {sample.ec}\n")
        f.write(f"Sequence: {sample.get('sequence', 'N/A')}\n")
        f.write(f"Ligand: {sample.get('smiles', 'N/A')}\n\n")

# 2. 手动访问 https://alphafoldserver.com
# 3. 提交蛋白序列 + 配体 SMILES
# 4. 下载预测的复合物结构
```

---

## 🐛 常见问题

### Q1: "ModuleNotFoundError: No module named 'torch_cluster'"

```bash
# 安装 torch_cluster
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# 或者代码会自动回退到手动构建边的方法，不影响使用
```

### Q2: Re-docking 脚本报错 "prepare_receptor4.py not found"

```bash
# 需要安装 MGLTools
conda install -c bioconda mgltools

# 或者使用 Docker
docker pull ccsb/autodock-vina
```

### Q3: PDB 查询太慢

```python
# 在 find_pdb_complexes.py 中，限制每个 EC 号查询的 PDB 数量
# 修改第 138 行
for pdb_id in pdb_ids[:3]:  # 从 5 改为 3
```

### Q4: 增强版图构建后节点数暴增

```python
# 减小截断半径
atoms = parse_pocket_enhanced(pdb_path, cutoff=8.0, expand_residues=True)

# 或者只扩展边界残基（而不是所有残基）
# 这个功能可以后续添加
```

---

## 📈 预期效果

### Re-docking 验证
- **预期成功率**: 70-90%（RMSD < 2Å）
- **如果成功率低于 50%**: 可能需要调整对接参数或使用更准确的对接软件（如 Glide）

### 图构建优化
- **节点数增加**: 预计增加 10-30%
- **性能提升**: 预计 R² 提升 0.02-0.05（需要重新训练验证）

---

## 📝 下一步建议

1. **优先级 1**: 先跑 `find_pdb_complexes.py`，了解你的数据集有多少样本有 PDB 结构
2. **优先级 2**: 选择 5-10 个高质量的 PDB，用 `redock_validation.py` 验证对接准确性
3. **优先级 3**: 如果对接验证通过，再考虑使用 `graph_builder_enhanced.py` 优化图构建
4. **实验对比**: 使用增强版图构建重新训练模型，对比性能提升

---

## 💡 额外提示

- 所有脚本都支持 `--help` 参数查看详细用法
- 建议在服务器上运行（数据文件在服务器上）
- 可以先用小数据集测试，确保流程正确
- 记得更新 `experiments.md`，记录每次实验的参数和结果

祝实验顺利！🚀
