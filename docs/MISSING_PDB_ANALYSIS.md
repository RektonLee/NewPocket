# 缺失PDB结构分析报告

**日期**: 2026-02-24
**问题**: kcat_full_1213.csv中4989个样本(55%)缺少蛋白质结构

---

## 数据统计

| 类别 | 数量 | 百分比 |
|------|------|--------|
| 有PDB结构 | 4072 | 45% |
| 无PDB结构 | 4989 | 55% |
| **总计** | 9061 | 100% |

## 缺失样本分布

### 按EC分类 (Top 10)

| EC编号 | 样本数 | 酶类型 |
|--------|--------|--------|
| 1.4.3.3 | 155 | Amine oxidase |
| 5.3.1.1 | 113 | Triose-phosphate isomerase |
| 5.1.2.2 | 110 | Epimerase |
| 1.1.1.1 | 95 | Alcohol dehydrogenase |
| 4.4.1.11 | 87 | Methionine γ-lyase |
| 2.7.1.74 | 85 | Kinase |
| 1.5.1.3 | 72 | Dihydrofolate reductase |
| 2.5.1.18 | 70 | Transferase |
| 1.14.20.1 | 68 | Oxidoreductase |
| 5.3.1.5 | 66 | Isomerase |

### Index范围

- **有PDB**: Index 0-4071
- **无PDB**: Index 4072-9060

这说明数据集可能是分批处理的，后半部分尚未完成结构获取。

---

## 可能的获取途径

### 1. AlphaFold Protein Structure Database

**优点**:
- 覆盖UniProt参考蛋白质组
- 高质量预测结构
- 可直接下载

**步骤**:
```python
# 从CSV中提取序列或UniProt ID
# 查询AlphaFold数据库
# 下载PDB文件
```

**限制**: 需要UniProt ID或序列信息

### 2. ESMFold预测

**优点**:
- 只需要序列
- 快速（~秒级）
- 质量接近AlphaFold2

**步骤**:
```bash
python scripts/generate_structures_esmfold.py \
  --csv data/processed/kcat_full_1213.csv \
  --start_index 4072 \
  --end_index 9061 \
  --output_dir sample_data/predicted_structures
```

**限制**: 需要GPU，批量处理需要时间

### 3. RoseTTAFold/其他工具

备选方案，但ESMFold通常更快更好

---

## 推荐方案

### 短期 (当前)

**使用现有4072个样本**进行训练和评估：
- ✅ 数据量足够 (vs 测试集940个样本)
- ✅ 可以完成实验验证
- ✅ 论文可以基于此完成

### 中期 (论文完成后)

**补充ESMFold预测**:
1. 对4989个样本批量预测结构
2. 使用DiffDock对接
3. 重新训练更大模型
4. 作为补充实验或后续工作

### 数据检查

首先确认CSV中是否有序列信息：

```python
import pandas as pd
df = pd.read_csv('data/processed/kcat_full_1213.csv')

# Check if sequence column exists
if 'sequence' in df.columns:
    no_pdb_df = df.iloc[4072:]
    has_seq = no_pdb_df['sequence'].notna().sum()
    print(f"Samples with sequence: {has_seq}/{len(no_pdb_df)}")
else:
    print("No sequence column found")
```

---

## 实施计划

### Phase 1: 验证当前工作流 ✅ 进行中

- [x] 使用4072个有PDB样本
- [x] DiffDock对接
- [ ] 训练模型
- [ ] 评估性能

### Phase 2: 补充结构 (可选)

1. **数据准备** (1天)
   - 检查CSV中序列信息
   - 准备ESMFold输入

2. **结构预测** (2-3天)
   - 批量ESMFold预测
   - 质量检查

3. **对接** (2-3天)
   - DiffDock批量对接
   - 结果整合

4. **重新训练** (1天)
   - 使用完整数据集
   - 评估性能提升

**总时间**: ~1周

---

## 成本效益分析

| 选项 | 样本数 | 时间成本 | 性能提升预期 |
|------|--------|----------|--------------|
| 仅用现有数据 | 4072 | 0 | Baseline |
| 补充ESMFold | 9061 | 1周 | +5-10% ? |

**建议**: 先完成现有数据的实验，评估效果后决定是否值得投入时间补充。

---

## 相关脚本

需要创建的工具：
- `scripts/generate_structures_esmfold.py` - ESMFold批量预测
- `scripts/check_alphafold_availability.py` - 检查AlphaFold数据库
- `scripts/download_alphafold_structures.py` - 批量下载

---

## 结论

**当前策略**:
✅ 使用4072个有PDB样本完成实验
✅ 在论文中说明数据集规模和局限性
⏳ 将剩余样本作为future work

这是**务实的选择**，因为：
1. 4072个样本足够训练
2. 节省时间专注于核心实验
3. 可以在论文完成后补充
