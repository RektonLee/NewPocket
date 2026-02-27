# PocketGNN实验配置参考 (给新集群AI)

**来源**: PROJECT_PROGRESS.md 分析
**用途**: 告诉新集群AI当前工作的实验配置

---

## 🎯 当前最佳模型配置

### 模型: PocketGNNKcatOnly

**代码位置**: `src/GNN_model.py` (第476行提到的"当前主力")

**架构**:
```python
模型类: PocketGNNKcatOnly
- Node Encoder: Linear(52, 128)
- 3层GAT (Graph Attention Networks)
  - 每层4个attention heads
  - 第一层使用24维边特征 (edge_attr)
- Pooling: mean / global_attention / set2set (可选)
- [可选] ESM-2序列嵌入融合
- MLP Regressor: 3层全连接
- 输出: 单值 log10(kcat)
```

**参数规模**: ~0.21M (hidden_dim=128)

---

## 📊 训练数据配置

### 主要数据集

**1. 训练集** (40% Homology Split):
```
文件位置: data/processed/kcat_merged_hom40_train.pt
样本数: ~数千样本
Split方式: 40% sequence homology threshold
用途: 主训练集 (低同源性泛化测试)
```

**2. 测试集** (40% Homology Split):
```
文件位置: data/processed/kcat_merged_hom40_test.pt
样本数: 300-782样本 (根据图构建成功率)
用途: 最终性能评估
```

**3. 完整数据** (用于Random Split):
```
文件位置: data/processed/kcat_full_1213.pt
样本数: 4072样本 (有PDB的前4072条)
来源CSV: data/processed/kcat_full_1213.csv
用途: 数据增强、Random split baseline
```

**4. 测试集(新Docking)** (进行中):
```
文件位置: data/processed/kcat_test_new_diffdock.pt
样本数: 940/3444 (27.3%完成)
来源CSV: data/raw/kcat_test_results.csv
用途: 用新口袋提取方法重新评估
```

---

## 🔧 图数据格式 (输入到模型)

```python
PyTorch Geometric Data对象:
- x: [N, 52]              # 节点特征
  - 元素类型 (10-dim): C,N,O,S,P,F,Cl,Br,I,H
  - 残基类型 (21-dim): 20种氨基酸 + LIG (配体)
  - 配体标志 (1-dim): 0或1
  - 最小距离 (1-dim): 到最近邻原子
  - 电子特征 (16-dim): 基于原子序数
  - 原子属性 (3-dim): 质量、电负性、半径

- edge_index: [2, E]       # 边连接 (双向)
- edge_attr: [E, 24]       # 边特征 (关键!)
  - RBF距离编码 (16-dim): 0-8Å Gaussian RBF
  - 键角 (4-dim): 边原子与公共邻居的角度
  - 二面角 (4-dim): 扭转角

- pos: [N, 3]             # 3D坐标
- y: [1]                  # 标签: log10(kcat)
- seq_embedding: [1, D]   # [可选] ESM-2嵌入 (D=640或1280)
- docking_confidence: [1] # [可选] DiffDock置信度
```

**⚠️ 关键**: 24维边特征是性能提升的核心，必须存在！

---

## 🏆 当前最佳性能 (Baseline)

### Random Split (乐观上界):
```
Pearson r: 0.98
R²: 0.918
数据集: kcat_full_1213.pt (random split)
```

### 40% Homology Split (真实泛化):
```
Pearson r: 0.667
R²: 0.437
数据集: kcat_merged_hom40_test.pt
```

### 对比SOTA (在相同40% homology测试集):
```
PocketGNN:  r=0.667  R²=0.437
CataPro:    r=0.497  R²=0.444
CatPred:    r=0.520  R²=0.165
```

**结论**: PocketGNN在Pearson相关性上领先28-34%，但R²与CataPro持平

---

## 🔬 实验配置参考

### Baseline训练命令:
```bash
python src/train.py \
  --dataset data/processed/kcat_merged_hom40_train.pt \
  --hidden_dim 128 \
  --num_layers 3 \
  --heads 4 \
  --pooling_type mean \
  --dropout 0.1 \
  --batch_size 32 \
  --lr 1e-3 \
  --epochs 500 \
  --weight_decay 1e-4 \
  --loss mse \
  --scheduler plateau \
  --save_dir outputs/baseline_hom40
```

### 加ESM-2序列嵌入:
```bash
python src/train.py \
  --dataset data/processed/kcat_merged_hom40_train.pt \
  --use_seq_embedding \
  --seq_embedding_path data/processed/esm_embeddings.pt \
  --pooling_type set2set \
  --hidden_dim 128 \
  --save_dir outputs/baseline_hom40_esm
```

### 测试命令:
```bash
python src/test.py \
  --test_dataset data/processed/kcat_merged_hom40_test.pt \
  --model outputs/baseline_hom40/best_model.pt
```

---

## 📁 输出目录结构

### 训练输出 (每次运行):
```
outputs/
├── baseline_hom40/              # 实验目录
│   ├── best_model.pt            # 最佳模型checkpoint
│   ├── config.json              # 训练配置
│   ├── metrics.json             # 最终指标
│   ├── loss_curve.png           # 训练曲线
│   ├── metrics_curve.png        # 指标曲线
│   ├── kcat_prediction_scatter.png  # 真实vs预测散点图
│   └── kcat_density.png         # 预测密度图
```

### 结果输出:
```
results/
├── baseline_comparison/          # Baseline对比
│   ├── catapro_predictions.csv
│   ├── catpred_predictions.csv
│   └── baseline_comparison.json
├── interpretability/             # 可解释性分析
│   ├── attention_weights/        # 注意力权重可视化
│   └── feature_importance/       # 特征重要性
│       ├── figS12_node_feature_importance.png
│       ├── figS12_edge_feature_importance.png
│       └── feature_importance_summary.json
├── docking_validation/           # 对接质量验证
└── representation_comparison/    # 表征对比
```

### WandB记录:
- 所有训练自动上传到WandB
- 链接: https://wandb.ai/
- 记录内容: loss曲线、指标、超参数、可视化

---

## 🔧 数据处理流程

### 完整Pipeline:
```
1. CSV数据 (kcat_full_1213.csv)
   ↓
2. 蛋白质序列 + 底物SMILES
   ↓
3. [可选] ESMFold预测结构 (--no-esmfold跳过)
   ↓
4. DiffDock分子对接
   输出: sample_data/samples/{sample_id}/docking/diffdock_output/
   ↓
5. extract_pocket_pymol() 提取口袋 (10Å cutoff)
   输出: sample_data/samples/{sample_id}/docking/{sample_id}_XXXXX_10A.pdb
   包含: 蛋白质原子(ATOM) + 配体原子(HETATM)
   ↓
6. parse_pocket() + enhanced_build_graph() 构建图
   输出: 52维节点特征 + 24维边特征
   ↓
7. PyTorch Geometric Dataset (.pt)
   ↓
8. 训练/测试
```

### 口袋文件命名规则:
```python
格式: {sample_id}_{smiles_hash}_{cutoff}A.pdb
示例: kcat_000008_59520_10A.pdb

其中:
- sample_id: 样本ID (如kcat_000008)
- smiles_hash: SMILES的SHA256哈希(取后16位)
- cutoff: 口袋提取半径 (如10A)
```

---

## 🎯 当前进行中的工作

### 1. DiffDock批量对接 (优先级最高)
```
目标: 对接4072个样本 (前4072有PDB)
当前进度: 1651/4072 (40.5%)
剩余: 2421样本
样本列表: results/remaining_samples.txt
预计时间: 8-GPU集群上5小时完成

脚本: scripts/launch_8gpu_diffdock.sh
监控: python scripts/check_8gpu_progress.py
```

### 2. 可解释性分析 (论文核心)
```
已完成:
- ✅ 特征重要性分析 (figS12系列图)
- ✅ 注意力权重可视化脚本

核心发现:
- 几何特征(角度+二面角)贡献88.9%
- 键角是最重要特征(61.9%)
- 验证24维边特征设计合理

待完成:
- [ ] 多案例研究 (需要完整PDB)
- [ ] 离群点调查
- [ ] 跨模态交互分析
```

### 3. Baseline对比 (已完成)
```
已评估: CataPro, CatPred
结果文件: results/baseline_comparison/baseline_comparison.json
结论: PocketGNN与CataPro性能相当 (R²差0.0012)
```

---

## 🚀 新集群AI应该如何操作

### 如果要训练新模型:
```bash
# 1. 使用40% homology split数据
python src/train.py \
  --dataset data/processed/kcat_merged_hom40_train.pt \
  --save_dir outputs/exp_new_cluster

# 2. 测试
python src/test.py \
  --test_dataset data/processed/kcat_merged_hom40_test.pt \
  --model outputs/exp_new_cluster/best_model.pt
```

### 如果要继续DiffDock对接:
```bash
# 1. 启动8-GPU并行对接
./scripts/launch_8gpu_diffdock.sh

# 2. 监控进度
python scripts/check_8gpu_progress.py

# 3. 等待完成后自动触发训练 (如果配置了auto_pipeline)
```

### 如果要复现Baseline对比:
```bash
# CataPro评估
python scripts/run_catapro_baseline.py \
  --test_csv data/raw/kcat_test_results.csv

# CatPred评估
python scripts/evaluate_catpred_corrected.py
```

---

## 📝 重要文件路径速查

### 模型定义:
- `src/GNN_model.py` - PocketGNNKcatOnly (主力模型)

### 训练脚本:
- `src/train.py` - 主训练脚本
- `src/test.py` - 测试脚本

### 数据处理:
- `src/build_graph_dataset.py` - 构建.pt数据集
- `src/graph_builder_rbf.py` - 图构建(含24维边特征)
- `src/docking.py` - DiffDock对接 + 口袋提取

### 对接脚本:
- `scripts/batch_diffdock_clean.py` - 批量对接单GPU
- `scripts/launch_8gpu_diffdock.sh` - 8-GPU并行启动
- `scripts/check_8gpu_progress.py` - 进度监控

### 文档:
- `CLAUDE.md` - 项目总览
- `PROJECT_PROGRESS.md` - 进展追踪(最详细)
- `CLUSTER_MIGRATION_GUIDE.md` - 迁移指南

---

## ⚠️ 给新集群AI的注意事项

1. **边特征必需**: 输入图必须有24维edge_attr，否则模型会报错
2. **数据集split**: 40% homology是真实泛化，random split是乐观上界
3. **口袋PDB来源**: 从`/home/lizihao/Work/enzyme_prediction/PGNN/sample_data/samples/`获取
4. **模型版本**: 使用PocketGNNKcatOnly，不要用旧的PocketGNN或PocketGNNWithAttention
5. **ESM-2可选**: 加序列嵌入可能提升性能，但不是必需
6. **WandB自动记录**: 训练会自动上传到WandB，确保登录
7. **DiffDock路径**: 新集群需要设置`DIFFDOCK_PATH`环境变量

---

**最后更新**: 2026-02-27 (基于PROJECT_PROGRESS.md)
**维护**: 如有疑问参考PROJECT_PROGRESS.md或询问原集群AI
