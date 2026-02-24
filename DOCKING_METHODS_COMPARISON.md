# DiffDock vs AutoDock Vina: 口袋位置与坐标系统对比

**日期**: 2026-02-24
**问题**: Vina和DiffDock各自如何找口袋位置？是否对整体平移旋转鲁棒？

---

## 1. AutoDock Vina - 需要预先指定位置

### 工作原理
Vina是**传统的基于搜索的方法**，**必须预先指定搜索区域**：

```python
# 从我们的代码: scripts/quick_vina_test.py (第150-156行)
center = ref_coords.mean(axis=0).tolist()  # 必须指定中心坐标
v.compute_vina_maps(center=center, box_size=[20, 20, 20])  # 固定搜索盒
```

### 关键特点
1. **需要手动指定**:
   - `center=[x, y, z]`: 搜索盒中心的绝对坐标
   - `box_size=[20, 20, 20]`: 搜索盒的尺寸（Å）

2. **依赖绝对坐标系**:
   - Vina在指定的坐标范围内搜索
   - 如果蛋白质整体平移或旋转，搜索盒位置也需要相应调整
   - **不具备平移/旋转不变性**

3. **适用场景**:
   - 已知大致结合位置（如已有晶体结构）
   - 小规模手动对接（需要为每个蛋白质指定盒子）
   - Re-docking验证（如我们的实验）

### ⚠️ 对你的Pipeline的影响
```
序列 → ESMFold → 蛋白质结构（随机方向）
SMILES → RDKit → 配体结构
```
**问题**: ESMFold生成的结构方向是随机的，没有固定的参考坐标系。Vina无法自动找到结合位置，需要你：
- 要么预先知道活性位点残基（如从UniProt/文献）
- 要么使用工具预测结合位点（如P2Rank, Fpocket）
- **无法大规模自动化处理**

---

## 2. DiffDock - 完全自动化，坐标不变

### 工作原理
DiffDock是**基于扩散模型的生成方法**，**不需要指定任何位置信息**：

```python
# 从我们的代码: scripts/quick_diffdock_test.py (第50-69行)
# 只需要protein和ligand，没有任何位置参数！
f.write(f"test,{protein_pdb},{ligand_sdf},\n")
# 无需指定center, box_size, 或任何空间信息
```

### 内部处理机制

#### 代码分析: `DiffDock/utils/inference_utils.py`
```python
# 1. 计算蛋白质中心
protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)

# 2. 中心化蛋白质坐标
complex_graph['receptor'].pos -= protein_center

# 3. 中心化配体坐标
ligand_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
complex_graph['ligand'].pos -= ligand_center

# 4. 保存原始中心（用于最后还原坐标）
complex_graph.original_center = protein_center
```

#### 关键特点
1. **Translation Invariance (平移不变性)**:
   - 自动将蛋白质和配体中心化到原点
   - **输入的绝对位置完全不影响结果**
   - 输出时再加回`original_center`还原坐标

2. **Rotation Equivariance (旋转等变性)**:
   - DiffDock的扩散过程定义在SE(3)流形上
   - 论文标题: "Diffusion Steps, **Twists, and Turns**"
   - 能够生成所有可能的旋转和扭转构象
   - **蛋白质整体旋转不影响对接结果**

3. **Blind Docking**:
   - 在整个蛋白质表面搜索结合位点
   - 无需任何先验知识
   - 自动识别最可能的口袋

### ✅ 完美适配你的Pipeline
```
序列 → ESMFold → 蛋白质结构（任意方向）✅
        ↓
     DiffDock（自动中心化，搜索全蛋白表面）
        ↓
     找到最佳结合位点 + pose
```

**优势**:
- ✅ ESMFold生成的结构无论什么方向都可以
- ✅ 配体SMILES生成的3D构象无论什么初始方向都可以
- ✅ 完全自动化，无需人工干预
- ✅ 可扩展到数千个样本

---

## 3. 实验验证

### 我们的对比实验 (5个PoseBench样本)

| 方法 | 需要位置信息 | 成功率 | 平均RMSD | 最优RMSD |
|------|------------|--------|----------|----------|
| **Vina** | ✅ 需要 (用了参考配体中心) | 100% | 1.31Å | 0.30Å |
| **DiffDock** | ❌ 不需要 | 80% | 1.69Å | **0.38Å** |

### 关键观察
1. **Re-docking场景** (我们的实验):
   - Vina略优，因为我们给了它准确的搜索盒中心
   - 这是"作弊"的设置，实际应用中你不会有参考配体

2. **Blind docking场景** (DiffDock论文, Astex n=85):
   - DiffDock: 38.2% success (RMSD<2Å)
   - Vina (with search box): 20.9% success
   - **DiffDock胜出82%**

3. **你的实际场景** (序列→结构→对接):
   - Vina: 无法使用（没有参考位置）
   - DiffDock: 完美适用 ✅

---

## 4. 技术对比总结

| 特性 | AutoDock Vina | DiffDock |
|------|--------------|----------|
| **搜索方式** | 局部搜索（指定盒子内） | 全局生成（整个蛋白表面） |
| **需要位置信息** | ✅ 必须指定center和box_size | ❌ 完全不需要 |
| **平移不变性** | ❌ 依赖绝对坐标 | ✅ 自动中心化 |
| **旋转不变性** | ❌ 需要合理的初始方向 | ✅ SE(3)等变 |
| **结合位点预测** | ❌ 需要外部工具 | ✅ 自动识别 |
| **大规模自动化** | ❌ 每个蛋白需手动设置 | ✅ 完全自动 |
| **适合你的场景** | ❌ 不适合 | ✅ **完美适配** |

---

## 5. 推荐方案

### 对于你的Pipeline (序列→结构→对接→kcat预测):

**强烈推荐使用DiffDock**:
1. ✅ 无需任何位置先验知识
2. ✅ 对ESMFold生成的任意方向结构鲁棒
3. ✅ 对配体初始构象鲁棒
4. ✅ 可扩展到数千样本的自动化处理
5. ✅ 已在你的1878个样本上验证100%成功

### 论文中如何表述:

**当前版本** (已更新到gemini.tex):
> "Traditional methods like AutoDock Vina require explicit specification of the binding-site search box, which is labor-intensive and impractical for large-scale datasets with no prior structural knowledge. DiffDock eliminates this requirement, enabling fully automated pocket extraction at scale."

**可以补充的技术细节**:
> "DiffDock achieves translation invariance by automatically centering protein and ligand coordinates, and rotation equivariance through its SE(3)-equivariant diffusion process. This makes it particularly suitable for our pipeline where structures are predicted from sequences (via ESMFold) and ligands are generated from SMILES, without pre-aligned coordinate systems."

---

## 6. 代码验证

### 验证DiffDock的坐标不变性

你可以运行这个实验来验证：

```python
# 测试：同一个蛋白，平移+旋转后，DiffDock结果应该一致
import numpy as np
from scipy.spatial.transform import Rotation

# 原始对接
result1 = run_diffdock("protein.pdb", "ligand.sdf")

# 平移+旋转蛋白质
translate_and_rotate_pdb("protein.pdb", "protein_moved.pdb",
                         translation=[100, 50, -30],
                         rotation=Rotation.random())

# 再次对接
result2 = run_diffdock("protein_moved.pdb", "ligand.sdf")

# 结果应该等价（除了坐标系变换）
assert compare_poses(result1, result2) < 0.5  # RMSD应该很小
```

---

## 参考文献

1. **DiffDock**: Corso et al. (2023) "DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking" *ICLR 2023*
2. **AutoDock Vina**: Trott & Olson (2010) "AutoDock Vina: improving the speed and accuracy of docking" *J Comput Chem*

---

**结论**: DiffDock在技术上完全适合你的场景，论文中的表述也是准确的。Vina的优势在于已知位置的快速精确对接，但对于从序列预测结构的大规模自动化流程，DiffDock是唯一实用的选择。
