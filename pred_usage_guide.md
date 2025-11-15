# 改进后的预测脚本使用指南

## 🎯 概述

根据 `train_improved.py` 的改进，我已经更新了 `pred.py` 脚本，现在支持多种预测模式，可以自动适配不同类型的模型配置。

## 🚀 新增功能

### 1. 多种预测模式
- **original**: 原始模型配置
- **improved**: 改进模型配置  
- **auto**: 自动检测模型类型（推荐）
- **custom**: 自定义模型参数

### 2. 自动配置检测
- 根据模型目录名自动推断配置类型
- 支持从配置文件加载模型参数
- 智能匹配模型架构

### 3. 配置文件支持
- 训练时自动保存模型配置
- 预测时自动加载配置
- 支持手动指定配置文件

## 📋 使用方法

### 基础用法

```bash
# 自动检测模型类型（推荐）
python pred.py --dataset kcat_test_new.pt --model outputs/improved_training/best_model.pt --mode auto

# 原始模型预测
python pred.py --dataset kcat_test_new.pt --model outputs/kcat_after_new/best_model.pt --mode original

# 改进模型预测
python pred.py --dataset kcat_test_new.pt --model outputs/improved_training/best_model.pt --mode improved
```

### 高级用法

```bash
# 自定义模型配置
python pred.py --dataset kcat_test_new.pt --model outputs/kcat_after_new/best_model.pt \
    --mode custom --hidden_dim 256 --num_layers 4 --heads 8 --dropout 0.2

# 从配置文件加载
python pred.py --dataset kcat_test_new.pt --model outputs/improved_training/best_model.pt \
    --mode improved --load_config

# 指定配置文件
python pred.py --dataset kcat_test_new.pt --model outputs/improved_training/best_model.pt \
    --mode improved --config_file outputs/improved_training/model_config.json
```

## 🔧 参数说明

### 基础参数
- `--dataset`: 测试数据集路径
- `--model`: 模型文件路径
- `--save_dir`: 结果保存目录
- `--batch_size`: 批次大小

### 新增参数
- `--mode`: 预测模式 (original/improved/auto/custom)
- `--hidden_dim`: 隐藏层维度（custom模式）
- `--num_layers`: 层数（custom模式）
- `--heads`: 注意力头数（custom模式）
- `--dropout`: Dropout率（custom模式）
- `--config_file`: 配置文件路径
- `--load_config`: 从模型目录加载配置

## 📊 模型配置对比

| 配置类型 | hidden_dim | num_layers | heads | dropout | 适用场景 |
|----------|------------|------------|-------|---------|----------|
| 原始模型 | 128 | 3 | 4 | 0.1 | 基础训练 |
| 改进模型 | 256 | 4 | 8 | 0.2 | 提升泛化能力 |
| 高级模型 | 256 | 4 | 8 | 0.2 | 最佳性能 |

## 🔍 自动检测逻辑

### 目录名检测
- 包含 "improved" → 改进模型配置
- 包含 "advanced" → 高级模型配置
- 其他 → 原始模型配置

### 配置文件检测
- 优先从 `model_config.json` 加载
- 配置文件不存在时使用默认配置

## 💡 使用建议

### 1. 推荐使用自动模式
```bash
python pred.py --dataset kcat_test_new.pt --model your_model.pt --mode auto
```

### 2. 训练时保存配置
改进的训练脚本会自动保存模型配置到 `model_config.json`

### 3. 批量测试不同模型
```bash
# 测试原始模型
python pred.py --dataset kcat_test_new.pt --model outputs/kcat_after_new/best_model.pt --mode original --save_dir outputs/predictions/original

# 测试改进模型
python pred.py --dataset kcat_test_new.pt --model outputs/improved_training/best_model.pt --mode auto --save_dir outputs/predictions/improved
```

## 🎯 实际应用示例

### 场景1: 测试改进训练效果
```bash
# 1. 使用改进训练脚本训练模型
python train_improved.py --dataset kcat_train_after_new_clean.pt --save_dir outputs/improved_training --use_mixup --use_label_smoothing

# 2. 测试改进模型性能
python pred.py --dataset kcat_test_new.pt --model outputs/improved_training/best_model.pt --mode auto --save_dir outputs/predictions/improved_test
```

### 场景2: 对比不同模型性能
```bash
# 原始模型
python pred.py --dataset kcat_test_new.pt --model outputs/kcat_after_new/best_model.pt --mode original --save_dir outputs/predictions/comparison/original

# 改进模型
python pred.py --dataset kcat_test_new.pt --model outputs/improved_training/best_model.pt --mode auto --save_dir outputs/predictions/comparison/improved
```

### 场景3: 超参数搜索后的测试
```bash
# 使用超参数搜索找到的最佳配置
python pred.py --dataset kcat_test_new.pt --model outputs/best_config_training/best_model.pt \
    --mode custom --hidden_dim 512 --num_layers 5 --heads 16 --dropout 0.3 \
    --save_dir outputs/predictions/best_config
```

## ⚠️ 注意事项

1. **模型兼容性**: 确保模型文件与配置参数匹配
2. **配置文件**: 训练脚本会自动生成配置文件
3. **内存使用**: 更大的模型需要更多GPU内存
4. **预测时间**: 改进模型可能需要更长的预测时间
5. **评估模式**: 预测时模型自动设置为 `eval()` 模式，dropout等训练特性会被自动禁用

## 🔧 故障排除

### 常见问题
1. **模型加载失败**: 检查模型配置是否匹配
2. **内存不足**: 减小batch_size或使用更小的模型
3. **配置文件缺失**: 使用auto模式或手动指定参数

### 调试技巧
```bash
# 查看模型配置
cat outputs/improved_training/model_config.json

# 测试不同配置
python pred.py --dataset kcat_test_new.pt --model your_model.pt --mode custom --hidden_dim 128 --num_layers 3 --heads 4 --dropout 0.1
```

这个改进后的预测脚本现在完全支持 `train_improved.py` 的训练结果，可以自动适配不同的模型配置，大大简化了模型测试和比较的过程。
