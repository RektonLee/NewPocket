# 直接模型预测功能

这个项目现在支持直接从已构建的图数据或pocket PDB文件进行模型预测，无需重新运行完整的pipeline。

## 🚀 快速开始

### 方法1: 使用已有的图数据文件（推荐）

如果你有已保存的图数据文件（.pt文件），这是最快的方法：

```bash
# 单个模型预测
python pred_direct.py \
    --graph-data /path/to/dataset.pt \
    --model /path/to/model.pt \
    --output results/single_model_test

# 多个模型比较
python pred_direct.py \
    --graph-data /path/to/dataset.pt \
    --model /path/to/model1.pt /path/to/model2.pt /path/to/model3.pt \
    --output results/model_comparison \
    --compare
```

### 方法2: 从pocket PDB文件预测

如果你有pocket PDB文件但需要重新构图：

```bash
# 基本用法
python pred_from_pockets.py \
    --pocket-dir /path/to/pocket/files \
    --model /path/to/model.pt \
    --output results/pocket_prediction

# 带样本信息（用于误差计算）
python pred_from_pockets.py \
    --pocket-dir /path/to/pocket/files \
    --model /path/to/model.pt \
    --sample-info /path/to/sample_info.csv \
    --output results/pocket_prediction_with_info \
    --compare
```

## 📁 可用的数据文件

根据扫描结果，你的项目中有以下可用的文件：

### 图数据文件 (.pt)
- `/home/lizihao/Work/enzyme_prediction/src/simple2/data/processed/dataset_DLKcat_S.pt`
- `/home/lizihao/Work/enzyme_prediction/src/simple2/data/processed/dataset_DLKcat.pt`
- 以及其他多个数据集文件

### 模型文件 (.pt)
- `/home/lizihao/Work/enzyme_prediction/src/simple2/outputs/nopqr_attention_rbf/best_model.pt`
- `/home/lizihao/Work/enzyme_prediction/src/simple2/outputs/nopqr_attention/best_model.pt`
- `/home/lizihao/Work/enzyme_prediction/src/simple2/outputs/nopqr/best_model.pt`
- 以及其他多个训练好的模型

## 🎯 实际使用示例

### 快速测试单个模型
```bash
python pred_direct.py \
    --graph-data /home/lizihao/Work/enzyme_prediction/src/simple2/data/processed/dataset_DLKcat_S.pt \
    --model /home/lizihao/Work/enzyme_prediction/src/simple2/outputs/nopqr_attention_rbf/best_model.pt \
    --output results/quick_test
```

### 比较多个模型性能
```bash
python pred_direct.py \
    --graph-data /home/lizihao/Work/enzyme_prediction/src/simple2/data/processed/dataset_DLKcat_S.pt \
    --model /home/lizihao/Work/enzyme_prediction/src/simple2/outputs/nopqr_attention_rbf/best_model.pt \
    --model /home/lizihao/Work/enzyme_prediction/src/simple2/outputs/nopqr_attention/best_model.pt \
    --model /home/lizihao/Work/enzyme_prediction/src/simple2/outputs/nopqr/best_model.pt \
    --output results/model_comparison \
    --compare
```

## 📊 输出文件说明

每个预测任务会生成以下文件：

- `predictions.csv` - 详细预测结果
- `stats.json` - 统计信息（包含误差指标）
- `model_comparison.csv` - 模型比较结果（如果使用--compare）
- `*.log` - 详细日志文件

## 🔧 参数说明

### pred_direct.py 参数
- `--graph-data`: 图数据文件路径（必需）
- `--model`: 模型文件路径，可指定多个（必需）
- `--output`: 输出目录（默认：direct_predictions）
- `--batch-size`: 批处理大小（默认：32）
- `--compare`: 比较多个模型性能

### pred_from_pockets.py 参数
- `--pocket-dir`: pocket PDB文件目录（必需）
- `--model`: 模型文件路径，可指定多个（必需）
- `--output`: 输出目录（默认：pocket_predictions）
- `--sample-info`: 样本信息CSV文件（可选）
- `--temperature`: 温度（默认：303.15K）
- `--compare`: 比较多个模型性能

## 💡 使用建议

1. **优先使用 pred_direct.py**：如果你有图数据文件，这是最快的方法
2. **模型比较**：使用 `--compare` 参数可以同时测试多个模型并生成比较报告
3. **样本信息**：如果提供样本信息文件，可以计算预测误差和性能指标
4. **批处理大小**：根据GPU内存调整 `--batch-size` 参数

## 🚨 注意事项

1. 确保模型文件与图数据的维度匹配
2. 图数据文件必须包含有效的图结构信息
3. 如果使用pocket文件，确保文件名格式正确（以`_10A.pdb`结尾）
4. 输出目录会自动创建，但请确保有写入权限

## 🎉 开始使用

运行以下命令查看详细的使用示例：

```bash
python example_usage.py
```

这将显示所有可用的数据文件和具体的命令示例。

