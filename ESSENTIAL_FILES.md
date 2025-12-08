# 必需文件列表

本项目用于酶动力学参数预测（kcat, Km），以下是核心必需文件：

## 核心必需文件

### 1. 图构建与数据处理
- `graph_builder_rbf.py` - 核心图构建逻辑，使用Gaussian RBF编码边特征
- `build_graph_dataset.py` - 增强版数据集构建，添加角度和二面角特征
- `data_loader.py` - 数据加载和蛋白质结构处理

### 2. 结构预测与对接
- `generate_pdb_fixed.py` - 使用ESMFold预测蛋白质结构，支持多GPU并行
- `docking.py` - 口袋提取与分子对接

### 3. 模型定义
- `GNN_model.py` - GNN模型定义（包含温度模块和无温度版本）

### 4. 训练与预测
- `train.py` - 模型训练脚本
- `pred_range_fixed.py` - 范围预测脚本，支持并行处理

### 5. 工具与辅助
- `sample_manager.py` - 样本管理器，处理文件组织和去重
- `evaluate.py` - 模型评估脚本

## 可选但推荐的文件

### 文档
- `TECHNICAL_DOCUMENTATION.md` - 技术文档
- `README_direct_prediction.md` - 使用指南
- 其他 `.md` 文档文件

### 辅助脚本
- `train_*.py` - 其他训练变体
- `pred_*.py` - 其他预测脚本
- `*.sh` - 并行处理脚本

## 依赖要求

- Python 3.10+
- PyTorch / PyTorch Geometric
- RDKit, BioPython, scikit-learn
- transformers (ESMFold)
- 可选：AutoDock Vina/ADFRsuite (用于对接)
