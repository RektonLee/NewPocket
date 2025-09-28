#!/bin/bash

# 简单的并行PDB生成脚本
# 将数据分割成k个chunk，每个chunk独立运行generate_pdb_fixed.py

# 默认参数
NUM_CHUNKS=${1:-4}                    # 默认4个chunk
INPUT_FILE=${2:-"kcat_data_with_ids.csv"}
BASE_OUTPUT_DIR="parallel_pdb_output_$(date +%Y%m%d_%H%M%S)"
SEQUENCE_COLUMN="sequence"
SAMPLE_ID_COLUMN="sample_id"
MAX_LENGTH=400
TRUNCATE_MODE="skip"
GPUS="1"  # 使用GPU 1，避免GPU 0内存不足

echo "=========================================="
echo "并行PDB生成开始"
echo "输入文件: $INPUT_FILE"
echo "输出目录: $BASE_OUTPUT_DIR"
echo "Chunk数量: $NUM_CHUNKS"
echo "=========================================="

# 检查输入文件
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ 错误: 输入文件 $INPUT_FILE 不存在"
    exit 1
fi

# 创建输出目录
mkdir -p "$BASE_OUTPUT_DIR"

# 计算chunk大小
TOTAL_LINES=$(wc -l < "$INPUT_FILE")
TOTAL_SAMPLES=$((TOTAL_LINES - 1))  # 减去header行
CHUNK_SIZE=$((TOTAL_SAMPLES / NUM_CHUNKS))
REMAINDER=$((TOTAL_SAMPLES % NUM_CHUNKS))

echo "📊 总样本数: $TOTAL_SAMPLES"
echo "📊 每个chunk大小: $CHUNK_SIZE"
echo "📊 余数: $REMAINDER"

# 分割数据
echo "📂 分割数据到 $NUM_CHUNKS 个chunk..."
python3 -c "
import pandas as pd
import sys
import os

# 读取数据
df = pd.read_csv('$INPUT_FILE')
total_samples = len(df)
chunk_size = $CHUNK_SIZE
remainder = $REMAINDER
base_output_dir = '$BASE_OUTPUT_DIR'

# 分割数据
start_idx = 0
for i in range($NUM_CHUNKS):
    # 计算当前chunk的大小
    current_chunk_size = chunk_size + (1 if i < remainder else 0)
    end_idx = start_idx + current_chunk_size
    
    # 创建chunk
    chunk_df = df.iloc[start_idx:end_idx].copy()
    chunk_file = os.path.join(base_output_dir, f'chunk_{i+1}.csv')
    chunk_df.to_csv(chunk_file, index=False)
    
    print(f'创建chunk {i+1}: {chunk_file} ({len(chunk_df)} 个样本)')
    start_idx = end_idx
"

# 启动并行任务
echo "🚀 启动 $NUM_CHUNKS 个并行任务..."
PIDS=()

for i in $(seq 1 $NUM_CHUNKS); do
    CHUNK_FILE="$BASE_OUTPUT_DIR/chunk_${i}.csv"
    CHUNK_OUTPUT_DIR="$BASE_OUTPUT_DIR/chunk_${i}_output"
    
    echo "启动chunk $i: $CHUNK_FILE"
    
    # 启动后台任务
    python3 generate_pdb_fixed.py \
        --input "$CHUNK_FILE" \
        --sequence-column "$SEQUENCE_COLUMN" \
        --sample-id-column "$SAMPLE_ID_COLUMN" \
        --use-sample-manager \
        --sample-data-dir "$CHUNK_OUTPUT_DIR" \
        --gpus "$GPUS" \
        --max-length "$MAX_LENGTH" \
        --truncate-mode "$TRUNCATE_MODE" \
        --overwrite \
        > "$BASE_OUTPUT_DIR/chunk_${i}.log" 2>&1 &
    
    PIDS+=($!)
    echo "  PID: $!"
done

echo "⏳ 等待所有任务完成..."
echo "监控进程: ps aux | grep generate_pdb_fixed"

# 等待所有任务完成
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    chunk_num=$((i + 1))
    echo "等待chunk $chunk_num (PID: $pid)..."
    wait $pid
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ Chunk $chunk_num 完成"
    else
        echo "❌ Chunk $chunk_num 失败 (退出码: $exit_code)"
    fi
done

echo "=========================================="
echo "所有任务完成！"
echo "结果目录: $BASE_OUTPUT_DIR"
echo "=========================================="

# 显示结果统计
echo "📊 结果统计:"
for i in $(seq 1 $NUM_CHUNKS); do
    CHUNK_OUTPUT_DIR="$BASE_OUTPUT_DIR/chunk_${i}_output"
    if [ -d "$CHUNK_OUTPUT_DIR" ]; then
        PDB_COUNT=$(find "$CHUNK_OUTPUT_DIR" -name "*.pdb" | wc -l)
        echo "  Chunk $i: $PDB_COUNT 个PDB文件"
    else
        echo "  Chunk $i: 输出目录不存在"
    fi
done

# 合并结果（可选）
echo "🔄 合并结果..."
FINAL_OUTPUT_DIR="$BASE_OUTPUT_DIR/final_results"
mkdir -p "$FINAL_OUTPUT_DIR"

TOTAL_PDBS=0
for i in $(seq 1 $NUM_CHUNKS); do
    CHUNK_OUTPUT_DIR="$BASE_OUTPUT_DIR/chunk_${i}_output"
    if [ -d "$CHUNK_OUTPUT_DIR" ]; then
        # 复制PDB文件到最终目录
        find "$CHUNK_OUTPUT_DIR" -name "*.pdb" -exec cp {} "$FINAL_OUTPUT_DIR/" \;
        PDB_COUNT=$(find "$CHUNK_OUTPUT_DIR" -name "*.pdb" | wc -l)
        TOTAL_PDBS=$((TOTAL_PDBS + PDB_COUNT))
    fi
done

echo "✅ 合并完成！总共生成 $TOTAL_PDBS 个PDB文件"
echo "最终结果目录: $FINAL_OUTPUT_DIR"
