# 酶动力学预测 (PGNN Project Clean)

**开发者备忘录**

这个仓库是原 `PGNN` 项目的重构精简版。主要用于预测酶的 kcat 和 Km 值。

## 📁 目录结构

- **`src/`**: 核心代码
  - `GNN_model.py`: 模型定义 (PocketGNN, PocketGNNKcatOnly)
  - `train.py`: 训练脚本 (支持 WandB)
  - `evaluate.py`: 评估脚本
  - `data_loader.py`: 数据加载和处理
  - `graph_builder_rbf.py`: 图构建核心 (RBF 边特征)
  - `docking.py`: 分子对接逻辑
  - `sample_manager.py`: 样本管理工具
- **`scripts/`**: 运行脚本 (Shell scripts)
- **`configs/`**: 配置文件 (YAML/JSON)
- **`data/`**: 数据存放 (已被 gitignore，大文件放服务器)
- **`results/`**: 实验结果 (已被 gitignore)

## 🚀 常用命令

### 训练
```bash
# 确保在项目根目录下
python src/train.py --dataset data/processed/kcat_train.pt --save_dir results/experiments/exp001
```

### 预测
```bash
python src/pred_range_fixed.py --input data/test.csv --output results/predictions/
```

### 生成 PDB
```bash
python src/generate_pdb_fixed.py --input data/sequences.csv
```

## 🛠️ 环境依赖
见 `requirements.txt`。主要依赖：PyTorch, PyTorch Geometric, RDKit, WandB.

## 📝 开发规范
- **不要**把数据文件 (`.csv`, `.pt`, `.pdb`) 提交到 git。
- 每次重大修改后，更新 `experiments.md`。
- 优先使用 `sample_manager.py` 来管理样本路径，避免硬编码。


pred_range_fixed是docking的

data/processed/kcat_test_new.pt,PT文件类型: <class 'list'>
List length: 1455
第一个元素: Data(x=[206, 52], edge_index=[2, 1800], edge_attr=[1800, 24], pos=[206, 3], temperature=[1], y=[1], pdb_id='kcat_test_0004_61714_10A.pdb', sample_id='kcat_test_0004', ec=1)