#!/bin/bash
#
# Phase 1 实验: Baseline vs PHPTransformer 对比
# 使用40% homology pre-split数据集
# 2026-01-24
#

# 确保使用正确的环境
source ~/.bashrc
conda activate env2

# 设置Python别名（确保使用Python3）
PYTHON=Python3

# 数据集路径（40% homology pre-split）
TRAIN_DATA="data/processed/kcat_merged_hom40_train.pt"
VAL_DATA="data/processed/kcat_merged_hom40_val.pt"
TEST_DATA="data/processed/kcat_merged_hom40_test.pt"
ESM_EMB="data/processed/esm_embeddings.pt"

# 通用训练参数
COMMON_ARGS=(
    --train_dataset $TRAIN_DATA
    --val_dataset $VAL_DATA
    --test_dataset $TEST_DATA
    --batch_size 32
    --lr 5e-4
    --max_epochs 200
    --loss huber
    --scheduler plateau
    --patience 15
    --weight_decay 1e-4
    --dropout 0.15
    --hidden_dim 256
    --num_layers 4
    --heads 4
)

echo "=========================================="
echo "Phase 1: Baseline vs PHPTransformer 对比"
echo "=========================================="
echo "数据集: 40% homology split"
echo "训练集: $TRAIN_DATA"
echo "验证集: $VAL_DATA"
echo "测试集: $TEST_DATA"
echo ""

# ========== 实验1: Baseline (PocketGNNKcatOnly) - 无ESM ==========
echo "[实验1] Baseline (PocketGNNKcatOnly) - 无ESM"
$PYTHON src/train.py \
    "${COMMON_ARGS[@]}" \
    --model_type PocketGNNKcatOnly \
    --pooling_type mean \
    --exp_name phpt_exp1_baseline_no_esm

sleep 5

# ========== 实验2: Baseline (PocketGNNKcatOnly) - 有ESM ==========
echo "[实验2] Baseline (PocketGNNKcatOnly) - 有ESM"
$PYTHON src/train.py \
    "${COMMON_ARGS[@]}" \
    --model_type PocketGNNKcatOnly \
    --pooling_type mean \
    --use_seq_embedding \
    --seq_embedding_path $ESM_EMB \
    --exp_name phpt_exp2_baseline_with_esm

sleep 5

# ========== 实验3: PHPTransformer - 无ESM ==========
echo "[实验3] PHPTransformer (双通路GNN) - 无ESM"
$PYTHON src/train.py \
    "${COMMON_ARGS[@]}" \
    --model_type PHPTransformer \
    --exp_name phpt_exp3_phpt_no_esm

sleep 5

# ========== 实验4: PHPTransformer - 有ESM ==========
echo "[实验4] PHPTransformer (双通路GNN) - 有ESM"
$PYTHON src/train.py \
    "${COMMON_ARGS[@]}" \
    --model_type PHPTransformer \
    --use_seq_embedding \
    --seq_embedding_path $ESM_EMB \
    --exp_name phpt_exp4_phpt_with_esm

echo ""
echo "=========================================="
echo "所有实验完成！"
echo "查看结果: experiments/ 目录"
echo "查看WandB: https://wandb.ai/"
echo "=========================================="
