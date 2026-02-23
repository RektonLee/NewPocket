# 项目增强改造报告

根据 `Enhance.md` 的要求，已完成对项目的增量式改造。以下是详细说明。

## 1. 修改文件列表

*   `GNN_model.py`: 核心模型文件，更新了 `PocketGNNKcatOnly` 类。
*   `train.py`: 训练脚本，增加了命令行参数解析、优化器配置、Loss配置、Scheduler配置以及序列Embedding加载逻辑。
*   `generate_esm_embeddings.py` (新增): 用于从CSV生成ESM-2序列嵌入的辅助脚本。

## 2. 修改点详细说明

### 2.1 训练与正则化改进 (`train.py`)

| 功能点 | 说明 | 向后兼容 | 默认开启 |
| :--- | :--- | :--- | :--- |
| **Weight Decay** | 在 Adam 优化器中加入 L2 正则化，参数 `--weight_decay`。 | ✅ 是 | ❌ (默认 1e-4) |
| **Dropout** | 模型 Dropout 率可配置，参数 `--dropout`。 | ✅ 是 | ❌ (默认 0.1) |
| **鲁棒 Loss** | 支持 Huber Loss 以应对噪声，参数 `--loss huber`。 | ✅ 是 | ❌ (默认 MSE) |
| **LR Scheduler** | 支持 ReduceLROnPlateau 或 Cosine，参数 `--scheduler`。 | ✅ 是 | ❌ (默认 None) |

> **注意**: `weight_decay` 默认值设为了 `1e-4` (原代码无)，这可能会轻微改变原默认行为，但属于推荐的改进。如需完全一致可设为 0。

### 2.2 模型结构改进 (`GNN_model.py`)

| 功能点 | 说明 | 向后兼容 | 默认开启 |
| :--- | :--- | :--- | :--- |
| **Pooling 策略** | 支持 `global_attention` 池化，参数 `--pooling_type`。 | ✅ 是 | ❌ (默认 mean) |
| **Late Fusion** | 支持在 MLP 输入前拼接 ESM 序列嵌入。 | ✅ 是 | ❌ (需参数开启) |

### 2.3 ESM-2 序列嵌入集成 (`train.py` & `GNN_model.py`)

| 功能点 | 说明 | 向后兼容 | 默认开启 |
| :--- | :--- | :--- | :--- |
| **Embedding 加载** | `train.py` 可加载外部 `.pt` 字典文件并匹配 ID。 | ✅ 是 | ❌ |
| **特征融合** | 模型自动根据输入维度调整 MLP 结构。 | ✅ 是 | ❌ |

## 3. 最小示例命令

以下命令展示了如何启用所有新特性：

```bash
# 1. (可选) 生成序列嵌入
python src/generate_esm_embeddings.py --csv path/to/data.csv --output data/esm_embeddings.pt

# 2. 启动增强训练
python src/train.py \
  --dataset data/processed/kcat_full_1213.pt \
  --save_dir outputs/kcat_enhanced_run \
  --weight_decay 1e-4 \
  --dropout 0.3 \
  --loss huber \
  --scheduler plateau \
  --pooling_type global_attention \
  --use_seq_embedding \
  --seq_embedding_path data/processed/esm_embeddings.pt
```

## 4. 验证与排查

*   **数据对齐**: 开启 `--use_seq_embedding` 时，脚本会尝试根据 `pdb_id` 或 `uniprot_id` 匹配 `Data` 对象与 Embedding 字典。匹配结果会打印在日志中。
*   **兼容性**: 不加任何新参数运行 `python src/train.py` 将保持原有行为（除了默认 weight_decay 现在是 1e-4）。
