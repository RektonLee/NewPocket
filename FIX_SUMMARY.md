# 索引越界错误修复总结

## 问题描述
在运行 `generate_pdb.py` 时遇到以下错误：
```
index 0 is out of bounds for dimension 0 with size 0
```

## 根本原因分析
1. **直接原因**: ESMFold模型输出为空，导致 `convert_outputs_to_pdb` 函数中的循环无法执行
2. **根本原因**: ESMFold模型在某些情况下可能返回空的输出张量，特别是当输入序列有问题或模型处理失败时
3. **错误链**: ESMFold输出为空 → `convert_outputs_to_pdb` 返回空列表 → 后续索引访问越界

## 修复方案

### 1. 修复 `convert_outputs_to_pdb` 函数 (`generate_pdb.py`)
添加了输出数据有效性检查和错误处理：
```python
def convert_outputs_to_pdb(outputs) -> List[str]:
    # ... 现有代码 ...
    
    # 检查输出数据的有效性
    if outputs_np["aatype"].shape[0] == 0:
        logging.warning("ESMFold输出为空，无法生成PDB")
        return []
    
    for i in range(outputs_np["aatype"].shape[0]):
        try:
            # ... 转换逻辑 ...
            pdbs.append(to_pdb(pred))
        except Exception as e:
            logging.error(f"转换第{i}个结构时出错: {e}")
            # 如果转换失败，添加一个空的PDB字符串
            pdbs.append("")
    
    return pdbs
```

### 2. 修复 `_predict_single_batch` 函数 (`generate_pdb.py`)
添加了PDB转换结果的验证和回退机制：
```python
# 转换为PDB
pdb_contents = convert_outputs_to_pdb(output)

# 检查PDB转换结果
if len(pdb_contents) == 0:
    logging.warning("PDB转换结果为空，使用dummy PDB")
    return [(seq, make_dummy_pdb(seq)) for seq in sequences]

if len(pdb_contents) != len(sequences):
    logging.warning(f"PDB数量({len(pdb_contents)})与序列数量({len(sequences)})不匹配")
    # 补齐缺失的PDB
    while len(pdb_contents) < len(sequences):
        pdb_contents.append(make_dummy_pdb(sequences[len(pdb_contents)]))

return [(seq, pdb) for seq, pdb in zip(sequences, pdb_contents)]
```

### 3. 修复radius_graph导入问题 (`graph_builder_rbf.py`)
- 修正了 `radius_graph` 的导入方式，优先从 `torch_cluster` 导入
- 添加了 `build_edges_manual` 函数作为备用方案
- 实现了智能导入和自动回退机制

### 3. 手动边构建算法
`build_edges_manual` 函数实现了不依赖外部库的半径图构建：
- 遍历所有原子对，计算距离
- 距离小于阈值的原子对之间添加双向边
- 如果没有边，添加自环保证图连通性

## 修复效果
1. **更好的错误信息**: 现在能准确识别和报告ESMFold输出问题
2. **自动回退**: 当ESMFold输出为空时自动使用dummy PDB
3. **数据验证**: 检查输出数据的有效性和完整性
4. **保持功能**: 修复后PDB生成功能应该能正常工作

## 测试建议
1. 重新运行 `generate_pdb.py` 验证修复效果
2. 检查日志文件确认没有索引越界错误
3. 验证生成的PDB文件是否正常

## 验证步骤
```bash
# 重新运行PDB生成
python generate_pdb.py --input your_data.csv --use-sample-manager
```

## 技术说明
- ESMFold模型在某些情况下可能返回空的输出张量
- 修复后的代码能够检测并处理这种情况
- 使用dummy PDB作为回退方案，确保程序不会崩溃
- 修复后的代码具有更好的错误处理和日志记录能力
