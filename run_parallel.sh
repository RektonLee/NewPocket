#!/bin/bash

# 并行预测运行脚本
# 使用方法: ./run_parallel.sh [chunks] [max_parallel]

# 默认参数
CHUNKS=${1:-8}           # 默认8个块
MAX_PARALLEL=${2:-4}     # 默认最大4个并行
INPUT_FILE="kcat_test_results.csv"
MODEL_PATH="/home/lizihao/Work/enzyme_prediction/src/simple2/outputs/nopqr_attention_rbf/best_model.pt"
OUTPUT_DIR="results/parallel_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "并行预测任务启动"
echo "输入文件: $INPUT_FILE"
echo "模型路径: $MODEL_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "并行块数: $CHUNKS"
echo "最大并行数: $MAX_PARALLEL"
echo "=========================================="

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ 错误: 输入文件 $INPUT_FILE 不存在"
    exit 1
fi

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型文件 $MODEL_PATH 不存在"
    exit 1
fi

# 运行并行预测
python parallel_predict.py \
    --input "$INPUT_FILE" \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_DIR" \
    --chunks "$CHUNKS" \
    --max-parallel "$MAX_PARALLEL" \
    --use-sample-manager \
    --sample-data-dir "sample_data" \
    --temperature 303.15

# 检查运行结果
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "✅ 并行预测任务完成！"
    echo "结果保存在: $OUTPUT_DIR/final_results/"
    echo "=========================================="
    
    # 显示结果文件
    echo "生成的文件:"
    ls -la "$OUTPUT_DIR/final_results/"
    
    # 显示简要统计
    if [ -f "$OUTPUT_DIR/final_results/final_stats.csv" ]; then
        echo ""
        echo "预测统计:"
        cat "$OUTPUT_DIR/final_results/final_stats.csv"
    fi
else
    echo "❌ 并行预测任务失败"
    exit 1
fi
