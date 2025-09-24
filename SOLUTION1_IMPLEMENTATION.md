# 方案1实施：跳过失败的样本，确保数据质量

## 问题背景
用户担心dummy PDB会影响最终的预测结果，因为：
1. dummy PDB是简化的线性结构，缺少真实的3D结构信息
2. 会影响后续的docking和口袋提取
3. 最终影响GNN预测的准确性

## 方案1：跳过失败的样本

### 核心原则
- **不生成dummy PDB用于生产预测**
- **跳过所有失败的样本**
- **记录详细的失败信息**
- **确保只有有效的PDB结构进入后续预测**

### 主要修改

#### 1. `_predict_single_batch` 函数
```python
# 方案1：检查PDB转换结果，如果失败则抛出异常
if len(pdb_contents) == 0:
    logging.error("PDB转换结果为空，跳过这些序列")
    raise ValueError("ESMFold输出为空，无法生成有效的PDB结构")

if len(pdb_contents) != len(sequences):
    logging.error(f"PDB数量({len(pdb_contents)})与序列数量({len(sequences)})不匹配")
    raise ValueError(f"PDB转换结果数量不匹配: 期望{len(sequences)}个，实际{len(pdb_contents)}个")
```

#### 2. `predict_batch` 函数
```python
except Exception as e2:
    logging.error(f"Individual prediction failed for seq len={len(seq)}: {e2}")
    # 方案1：跳过失败的样本，不生成dummy PDB
    logging.error(f"跳过序列 {seq[:20]}... (长度: {len(seq)})")
    continue
```

#### 3. OOM处理
```python
# 单序列仍OOM，跳过该序列
logging.error(f"Single sequence OOM (len={len(sequences[0])}), skipping sequence")
raise ValueError(f"序列过长导致OOM，跳过序列: {len(sequences[0])}个氨基酸")
```

#### 4. 主函数错误处理
```python
except Exception as e:
    logging.error(f"Batch prediction failed: {e}")
    # 方案1：跳过失败的batch，记录失败信息
    for idx, proc_seq, sample_id, pdb_path, meta in processed_batch:
        if args.use_sample_manager:
            sample_manager.log_failure(sample_id, "structure_prediction", str(e), len(proc_seq))
        
        meta['status'] = 'failed'
        meta['error'] = str(e)
        report.append(meta)
        logging.error(f"[{idx+1}/{len(sequences)}] Failed {sample_id}: {str(e)}")
```

### 优势

1. **数据质量保证**
   - 只有有效的PDB结构进入后续预测
   - 避免dummy PDB影响预测准确性

2. **清晰的错误追踪**
   - 详细的失败日志
   - 在SampleManager中记录失败原因
   - 生成失败摘要报告

3. **便于调试**
   - 明确知道哪些样本失败了
   - 可以单独处理失败的样本
   - 便于优化和修复

4. **资源效率**
   - 不浪费时间处理无效数据
   - 避免后续管道的错误传播

### 使用方式

#### 方法1：使用修改脚本
```bash
python apply_solution1.py
```

#### 方法2：使用新的文件
```bash
cp generate_pdb_fixed.py generate_pdb.py
```

### 预期结果

运行修改后的 `generate_pdb.py` 时：

1. **成功的样本**：正常生成PDB文件
2. **失败的样本**：
   - 不会生成PDB文件
   - 在日志中记录失败原因
   - 在SampleManager中标记为失败状态
   - 在报告中统计失败数量

3. **最终报告**：
   ```
   Total sequences: 1000
   Successfully processed: 850
   Cached (skipped): 50
   Skipped (too long): 30
   Failed: 70
   ```

### 后续处理

对于失败的样本，您可以：

1. **分析失败原因**：
   - 查看日志文件了解具体错误
   - 检查失败摘要报告

2. **单独处理**：
   - 调整参数重新运行失败的样本
   - 使用不同的策略处理特定类型的失败

3. **继续预测**：
   - 只对成功生成的PDB文件运行后续预测
   - 确保预测结果的可靠性

### 注意事项

1. **备份原文件**：修改前请备份原始的 `generate_pdb.py`
2. **检查依赖**：确保所有必要的库都已安装
3. **监控资源**：跳过失败样本可能会减少最终的数据量
4. **日志分析**：定期检查日志文件了解失败模式

## 总结

方案1通过跳过失败的样本，确保了：
- **数据质量**：只有有效的PDB结构进入预测
- **结果可靠性**：避免dummy PDB影响最终预测
- **可追踪性**：详细的失败记录和报告
- **可维护性**：清晰的错误处理和日志记录

这个方案特别适合对预测准确性要求较高的场景。
