# 模型更新总结

## 📅 日期：2024年9月28日

## 🎯 更新目标：适配增强的图特征和kcat专用预测

### ✅ 已完成的修改

#### 1. **新增模型类：`PocketGNNKcatOnly`**
- **位置**：`GNN_model.py` 第426-504行
- **特点**：
  - 专门用于kcat预测（输出维度：1）
  - 支持增强的边特征（24维：16维RBF + 4维键角 + 4维二面角）
  - 移除温度特征融合
  - 更深的MLP结构（3层：hidden_dim → hidden_dim → hidden_dim//2 → 1）

#### 2. **训练脚本适配：`train.py`**
- **模型选择**：使用 `PocketGNNKcatOnly` 替代 `PocketGNNWithAttention`
- **标签处理**：只使用kcat标签（`batch.y[:, 0:1]`）
- **维度检查**：自动检测节点和边特征维度
- **可视化更新**：只绘制kcat预测结果
- **默认路径**：使用 `kcat_dataset_enhanced1.pt`

#### 3. **图构建增强：`build_graph_dataset.py`**
- **边特征扩展**：从16维增加到24维
- **几何特征**：添加键角和二面角特征
- **输出文件**：`kcat_dataset_enhanced1.pt`

### 🔧 技术细节

#### **边特征构成（24维）**：
1. **RBF距离特征**：16维（原始）
2. **键角特征**：4维（cos_min, cos_max, cos_mean, num_angles）
3. **二面角特征**：4维（cos_min, cos_max, cos_mean, num_dihedrals）

#### **模型架构变化**：
- **输入**：节点特征52维，边特征24维
- **输出**：1维（只预测kcat）
- **MLP结构**：256 → 256 → 128 → 1
- **激活函数**：ReLU + Dropout

#### **训练流程变化**：
- **标签格式**：从 `[kcat, km]` 改为 `[kcat]`
- **损失函数**：MSE（保持不变）
- **评估指标**：R², MAE, RMSE, Pearson（针对kcat）

### 📊 预期改进

#### **特征增强收益**：
- **几何信息**：键角和二面角提供更丰富的分子几何描述
- **边特征**：从16维增加到24维（+50%信息量）
- **专业化**：专门针对kcat预测优化

#### **模型优化**：
- **单目标**：避免kcat和km之间的相互干扰
- **深度MLP**：更好的非线性拟合能力
- **无温度依赖**：简化模型，提高泛化性

### 🚀 使用方式

#### **1. 构建增强数据集**：
```bash
# 激活环境
source /home/lizihao/miniforge3/etc/profile.d/conda.sh
conda activate env2

# 运行图构建
python build_graph_dataset.py
```

#### **2. 训练kcat模型**：
```bash
# 训练增强模型
python train.py --dataset kcat_dataset_enhanced1.pt --save_dir outputs/kcat_enhanced_model
```

#### **3. 预期输出**：
- **模型文件**：`outputs/kcat_enhanced_model/best_model.pt`
- **训练曲线**：`loss_curve.png`, `metrics_curve.png`
- **预测结果**：`kcat_prediction_scatter.png`, `kcat_density.png`

### ⚠️ 注意事项

1. **数据兼容性**：确保使用增强版图构建脚本
2. **内存需求**：24维边特征会增加内存使用
3. **训练时间**：角度计算会增加图构建时间
4. **模型保存**：新模型与旧模型不兼容

### 📈 性能预期

- **准确率提升**：预期5-10%的R²提升
- **几何建模**：更好的分子几何理解
- **泛化能力**：单目标训练提高模型专注度

### 🔄 后续计划

1. **模型评估**：对比增强前后的性能
2. **超参数调优**：针对kcat预测优化
3. **Km模型开发**：类似的Km专用模型
4. **集成学习**：结合kcat和Km模型
