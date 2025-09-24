#!/bin/bash

# 简单的并行预测脚本
# 使用方法: ./run_parallel_simple.sh [num_chunks] [max_parallel]

# 默认参数
NUM_CHUNKS=${1:-8}       # 默认8个块
MAX_PARALLEL=${2:-4}     # 默认最大4个并行
INPUT_FILE="kcat_test_results.csv"
MODEL_PATH="/home/lizihao/Work/enzyme_prediction/src/simple2/outputs/nopqr_attention_rbf/best_model.pt"
BASE_OUTPUT_DIR="results/parallel_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "并行预测任务启动"
echo "输入文件: $INPUT_FILE"
echo "模型路径: $MODEL_PATH"
echo "输出目录: $BASE_OUTPUT_DIR"
echo "并行块数: $NUM_CHUNKS"
echo "最大并行数: $MAX_PARALLEL"
echo "=========================================="

# 检查输入文件
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ 错误: 输入文件 $INPUT_FILE 不存在"
    exit 1
fi

# 检查模型文件
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型文件 $MODEL_PATH 不存在"
    exit 1
fi

# 创建输出目录
mkdir -p "$BASE_OUTPUT_DIR"

# 计算每个块的大小
TOTAL_LINES=$(wc -l < "$INPUT_FILE")
TOTAL_SAMPLES=$((TOTAL_LINES - 1))  # 减去标题行
CHUNK_SIZE=$((TOTAL_SAMPLES / NUM_CHUNKS))
REMAINDER=$((TOTAL_SAMPLES % NUM_CHUNKS))

echo "总样本数: $TOTAL_SAMPLES"
echo "每块大小: $CHUNK_SIZE"
echo "余数: $REMAINDER"

# 启动并行任务
PIDS=()
START_IDX=1  # 从1开始，因为第0行是标题

for i in $(seq 1 $NUM_CHUNKS); do
    # 计算当前块的结束索引
    if [ $i -le $REMAINDER ]; then
        CURRENT_CHUNK_SIZE=$((CHUNK_SIZE + 1))
    else
        CURRENT_CHUNK_SIZE=$CHUNK_SIZE
    fi
    
    END_IDX=$((START_IDX + CURRENT_CHUNK_SIZE))
    OUTPUT_DIR="$BASE_OUTPUT_DIR/chunk_${i}_results"
    
    echo "启动块 $i: 样本 $START_IDX-$((END_IDX-1)) (共 $CURRENT_CHUNK_SIZE 个样本)"
    
    # 启动后台任务
    python pred_range.py \
        --input "$INPUT_FILE" \
        --model "$MODEL_PATH" \
        --output "$OUTPUT_DIR" \
        --start $START_IDX \
        --end $END_IDX \
        --use-sample-manager \
        --sample-data-dir "sample_data" \
        --temperature 303.15 \
        > "$BASE_OUTPUT_DIR/chunk_${i}.log" 2>&1 &
    
    PIDS+=($!)
    START_IDX=$END_IDX
    
    # 控制并行数
    if [ ${#PIDS[@]} -ge $MAX_PARALLEL ]; then
        echo "等待任务完成..."
        wait ${PIDS[0]}
        PIDS=("${PIDS[@]:1}")  # 移除第一个PID
    fi
done

# 等待所有任务完成
echo "等待所有任务完成..."
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo "=========================================="
echo "所有并行任务完成，开始合并结果..."

# 合并结果
FINAL_OUTPUT_DIR="$BASE_OUTPUT_DIR/final_results"
mkdir -p "$FINAL_OUTPUT_DIR"

# 合并predictions.csv
echo "合并预测结果..."
cat "$BASE_OUTPUT_DIR"/chunk_*_results/predictions.csv | head -1 > "$FINAL_OUTPUT_DIR/predictions.csv"
for i in $(seq 1 $NUM_CHUNKS); do
    CHUNK_FILE="$BASE_OUTPUT_DIR/chunk_${i}_results/predictions.csv"
    if [ -f "$CHUNK_FILE" ]; then
        tail -n +2 "$CHUNK_FILE" >> "$FINAL_OUTPUT_DIR/predictions.csv"
    fi
done

# 合并successful_predictions.csv
echo "合并成功预测结果..."
cat "$BASE_OUTPUT_DIR"/chunk_*_results/successful_predictions.csv | head -1 > "$FINAL_OUTPUT_DIR/successful_predictions.csv"
for i in $(seq 1 $NUM_CHUNKS); do
    CHUNK_FILE="$BASE_OUTPUT_DIR/chunk_${i}_results/successful_predictions.csv"
    if [ -f "$CHUNK_FILE" ]; then
        tail -n +2 "$CHUNK_FILE" >> "$FINAL_OUTPUT_DIR/successful_predictions.csv"
    fi
done

# 合并failed_predictions.csv
echo "合并失败预测结果..."
cat "$BASE_OUTPUT_DIR"/chunk_*_results/failed_predictions.csv | head -1 > "$FINAL_OUTPUT_DIR/failed_predictions.csv"
for i in $(seq 1 $NUM_CHUNKS); do
    CHUNK_FILE="$BASE_OUTPUT_DIR/chunk_${i}_results/failed_predictions.csv"
    if [ -f "$CHUNK_FILE" ]; then
        tail -n +2 "$CHUNK_FILE" >> "$FINAL_OUTPUT_DIR/failed_predictions.csv"
    fi
done

# 生成最终统计
echo "生成最终统计..."
python3 -c "
import pandas as pd
import os

# 读取合并后的结果
predictions_file = '$FINAL_OUTPUT_DIR/predictions.csv'
if os.path.exists(predictions_file):
    df = pd.read_csv(predictions_file)
    valid_predictions = df.dropna(subset=['kcat_pred', 'km_pred'])
    failed_predictions = df[df['kcat_pred'].isna() | df['km_pred'].isna()]
    
    stats = {
        '总样本数': len(df),
        '成功预测数': len(valid_predictions),
        '失败/跳过数': len(failed_predictions),
        '成功率': f'{len(valid_predictions)/len(df)*100:.1f}%',
        '并行块数': $NUM_CHUNKS
    }
    
    if len(valid_predictions) > 0:
        stats.update({
            'kcat预测范围': f'{valid_predictions[\"kcat_pred\"].min():.2f} - {valid_predictions[\"kcat_pred\"].max():.2f}',
            'Km预测范围': f'{valid_predictions[\"km_pred\"].min():.2e} - {valid_predictions[\"km_pred\"].max():.2e}'
        })
    
    pd.DataFrame([stats]).to_csv('$FINAL_OUTPUT_DIR/final_stats.csv', index=False)
    print('✅ 最终统计报告已生成')
"

echo "=========================================="
echo "✅ 并行预测任务完成！"
echo "结果保存在: $FINAL_OUTPUT_DIR/"
echo "=========================================="

# 显示结果文件
echo "生成的文件:"
ls -la "$FINAL_OUTPUT_DIR/"

# 显示简要统计
if [ -f "$FINAL_OUTPUT_DIR/final_stats.csv" ]; then
    echo ""
    echo "预测统计:"
    cat "$FINAL_OUTPUT_DIR/final_stats.csv"
fi
