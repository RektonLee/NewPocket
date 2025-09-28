#!/bin/bash
# 真正的并行PDB生成脚本 - 启动多个独立的Python进程

# 设置参数
INPUT_CSV="${1:-your_data.csv}"
GPUS="${2:-0,1,2,3}"
OUTPUT_DIR="${3:-parallel_pdb_output}"
FINAL_OUTPUT_DIR="${4:-final_pdb_results}"

# 解析GPU列表
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "🚀 开始并行PDB生成"
echo "📁 输入文件: $INPUT_CSV"
echo "🖥️  使用GPU: $GPUS (共 $NUM_GPUS 个)"
echo "📁 输出目录: $OUTPUT_DIR"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 计算数据分块
TOTAL_LINES=$(wc -l < "$INPUT_CSV")
TOTAL_SAMPLES=$((TOTAL_LINES - 1))  # 减去header行
CHUNK_SIZE=$((TOTAL_SAMPLES / NUM_GPUS))
REMAINDER=$((TOTAL_SAMPLES % NUM_GPUS))

echo "📊 数据统计:"
echo "   总样本数: $TOTAL_SAMPLES"
echo "   GPU数量: $NUM_GPUS"
echo "   每块大小: $CHUNK_SIZE"
echo "   余数: $REMAINDER"

# 启动并行进程
PIDS=()
START_LINE=2  # 从第2行开始（跳过header）

for i in "${!GPU_ARRAY[@]}"; do
    GPU_ID="${GPU_ARRAY[$i]}"
    
    # 计算当前块的起始和结束行
    CURRENT_CHUNK_SIZE=$CHUNK_SIZE
    if [ $i -lt $REMAINDER ]; then
        CURRENT_CHUNK_SIZE=$((CHUNK_SIZE + 1))
    fi
    
    END_LINE=$((START_LINE + CURRENT_CHUNK_SIZE - 1))
    
    # 创建临时CSV文件
    TEMP_CSV="$OUTPUT_DIR/chunk_$((i+1)).csv"
    
    # 提取header和对应行数
    head -n 1 "$INPUT_CSV" > "$TEMP_CSV"
    sed -n "${START_LINE},${END_LINE}p" "$INPUT_CSV" >> "$TEMP_CSV"
    
    echo "📦 数据块 $((i+1)): 行 $START_LINE-$END_LINE (共 $CURRENT_CHUNK_SIZE 个样本)"
    
    # 启动后台进程
    (
        echo "🚀 启动数据块 $((i+1)) 在 GPU $GPU_ID"
        echo "📁 处理文件: $TEMP_CSV"
        
        python3 generate_pdb_fixed.py \
            --input "$TEMP_CSV" \
            --sequence-column "sequence" \
            --sample-id-column "sample_id" \
            --use-sample-manager \
            --sample-data-dir "$OUTPUT_DIR" \
            --gpus "$GPU_ID" \
            --max-length 400 \
            --truncate-mode "skip" \
            --fp16 \
            --overwrite \
            --output-dir "$OUTPUT_DIR/chunk_$((i+1))_gpu$GPU_ID" \
            --report-json "$OUTPUT_DIR/chunk_$((i+1))_report.json"
        
        echo "✅ 数据块 $((i+1)) 完成 (GPU $GPU_ID)"
    ) &
    
    PIDS+=($!)
    
    # 更新下一个块的起始行
    START_LINE=$((END_LINE + 1))
done

echo "⏳ 等待所有进程完成..."
echo "🔄 运行中的进程: ${PIDS[*]}"

# 等待所有进程完成
SUCCESS_COUNT=0
for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    GPU_ID="${GPU_ARRAY[$i]}"
    
    if wait $PID; then
        echo "✅ 数据块 $((i+1)) 成功完成 (GPU $GPU_ID)"
        ((SUCCESS_COUNT++))
    else
        echo "❌ 数据块 $((i+1)) 失败 (GPU $GPU_ID)"
    fi
done

echo ""
echo "🎯 处理结果: $SUCCESS_COUNT/$NUM_GPUS 个数据块成功"

# 生成合并报告
echo "📊 生成处理报告..."
python3 -c "
import os
import json
import glob
from datetime import datetime

output_dir = '$OUTPUT_DIR'
final_output_dir = '$FINAL_OUTPUT_DIR'

# 创建最终输出目录
os.makedirs(final_output_dir, exist_ok=True)

# 收集所有PDB文件
all_pdbs = []
chunk_reports = []

for i in range(1, $NUM_GPUS + 1):
    chunk_dirs = glob.glob(f'{output_dir}/chunk_{i}_gpu*')
    if chunk_dirs:
        chunk_dir = chunk_dirs[0]
        print(f'📁 处理数据块 {i} 输出: {chunk_dir}')
        
        # 收集PDB文件
        pdb_files = []
        for root, dirs, files in os.walk(chunk_dir):
            for file in files:
                if file.endswith('.pdb'):
                    pdb_files.append(os.path.join(root, file))
        
        all_pdbs.extend(pdb_files)
        print(f'   找到 {len(pdb_files)} 个PDB文件')
        
        # 读取chunk报告
        report_file = f'{output_dir}/chunk_{i}_report.json'
        if os.path.exists(report_file):
            try:
                with open(report_file, 'r') as f:
                    chunk_reports.append(json.load(f))
            except:
                pass

print(f'📊 总共找到 {len(all_pdbs)} 个PDB文件')

# 生成合并报告
report = {
    '处理时间': datetime.now().isoformat(),
    '总数据块数': $NUM_GPUS,
    '成功数据块数': $SUCCESS_COUNT,
    '总PDB文件数': len(all_pdbs),
    '数据块报告': chunk_reports
}

report_file = os.path.join(final_output_dir, 'merge_report.json')
with open(report_file, 'w') as f:
    json.dump(report, f, indent=2)

print(f'✅ 合并完成，报告保存到: {report_file}')
"

echo "✅ 并行PDB生成完成！"
echo "📁 输出目录: $OUTPUT_DIR"
echo "📁 最终结果: $FINAL_OUTPUT_DIR"
