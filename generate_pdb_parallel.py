# -*- coding: utf-8 -*-
"""
并行PDB生成脚本 - 数据分块并行处理
将数据分成多个块，每个块在指定的GPU上运行
"""

import os
import sys
import argparse
import pandas as pd
import subprocess
import time
from pathlib import Path

def split_data_into_chunks(csv_file, num_chunks, output_dir):
    """将数据分成多个块"""
    df = pd.read_csv(csv_file)
    total_rows = len(df)
    chunk_size = total_rows // num_chunks
    remainder = total_rows % num_chunks
    
    chunks = []
    start_idx = 0
    
    for i in range(num_chunks):
        # 计算当前块的大小
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        
        # 创建数据块
        chunk_df = df.iloc[start_idx:end_idx].copy()
        chunk_file = os.path.join(output_dir, f"chunk_{i+1}.csv")
        chunk_df.to_csv(chunk_file, index=False)
        
        chunks.append({
            'file': chunk_file,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'size': current_chunk_size
        })
        
        start_idx = end_idx
        print(f"创建数据块 {i+1}: {chunk_file} ({current_chunk_size} 个序列)")
    
    return chunks

def start_chunk_processing(chunk_info, gpu_id, base_output_dir, args):
    """启动数据块处理进程（非阻塞）"""
    chunk_file = chunk_info['file']
    chunk_id = os.path.basename(chunk_file).replace('.csv', '')
    output_dir = os.path.join(base_output_dir, f"chunk_{chunk_id}_gpu{gpu_id}")
    
    # 构建命令
    cmd = [
        'python3', 'generate_pdb_fixed.py',
        '--input', chunk_file,
        '--sequence-column', args.sequence_column,
        '--sample-id-column', args.sample_id_column,
        '--use-sample-manager',
        '--sample-data-dir', output_dir,
        '--gpus', str(gpu_id),  # 只使用一个GPU
        '--max-length', str(args.max_length),
        '--truncate-mode', args.truncate_mode,
        '--overwrite'
    ]
    
    if args.fp16:
        cmd.append('--fp16')
    
    print(f"🚀 启动数据块 {chunk_id} 在 GPU {gpu_id}")
    print(f"命令: {' '.join(cmd)}")
    
    # 启动进程（非阻塞，实时显示输出）
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
    return process

def wait_for_chunk_completion(process, chunk_id, gpu_id):
    """等待数据块处理完成，实时显示输出"""
    import threading
    import queue
    
    # 创建队列来收集输出
    output_queue = queue.Queue()
    
    def read_output():
        for line in iter(process.stdout.readline, ''):
            output_queue.put(f"[GPU {gpu_id}] {line.rstrip()}")
        process.stdout.close()
    
    # 启动输出读取线程
    output_thread = threading.Thread(target=read_output)
    output_thread.daemon = True
    output_thread.start()
    
    try:
        # 实时显示输出
        while process.poll() is None:
            try:
                line = output_queue.get(timeout=1)
                print(line)
            except queue.Empty:
                continue
        
        # 处理剩余输出
        while not output_queue.empty():
            line = output_queue.get_nowait()
            print(line)
        
        if process.returncode == 0:
            print(f"✅ 数据块 {chunk_id} 完成 (GPU {gpu_id})")
            return True, f"成功处理数据块 {chunk_id}"
        else:
            print(f"❌ 数据块 {chunk_id} 失败 (GPU {gpu_id})")
            return False, f"进程退出码: {process.returncode}"
            
    except Exception as e:
        print(f"💥 数据块 {chunk_id} 异常 (GPU {gpu_id}): {e}")
        process.kill()
        return False, str(e)

def run_chunk_processing(chunk_info, gpu_id, base_output_dir, args):
    """在指定GPU上处理数据块"""
    chunk_file = chunk_info['file']
    chunk_id = os.path.basename(chunk_file).replace('.csv', '')
    output_dir = os.path.join(base_output_dir, f"chunk_{chunk_id}_gpu{gpu_id}")
    
    # 构建命令
    cmd = [
        'python', 'generate_pdb_fixed.py',
        '--input', chunk_file,
        '--sequence-column', args.sequence_column,
        '--sample-id-column', args.sample_id_column,
        '--use-sample-manager',
        '--sample-data-dir', output_dir,
        '--gpus', str(gpu_id),  # 只使用一个GPU
        '--max-length', str(args.max_length),
        '--truncate-mode', args.truncate_mode,
        '--overwrite'
    ]
    
    if args.fp16:
        cmd.append('--fp16')
    
    print(f"🚀 启动数据块 {chunk_id} 在 GPU {gpu_id}")
    print(f"命令: {' '.join(cmd)}")
    
    # 运行命令
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1小时超时
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✅ 数据块 {chunk_id} 完成 (GPU {gpu_id}, 耗时: {end_time-start_time:.1f}s)")
            return True, f"成功处理 {chunk_info['size']} 个序列"
        else:
            print(f"❌ 数据块 {chunk_id} 失败 (GPU {gpu_id})")
            print(f"错误: {result.stderr}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"⏰ 数据块 {chunk_id} 超时 (GPU {gpu_id})")
        return False, "处理超时"
    except Exception as e:
        print(f"💥 数据块 {chunk_id} 异常 (GPU {gpu_id}): {e}")
        return False, str(e)

def merge_results(base_output_dir, chunks, final_output_dir):
    """合并所有数据块的结果"""
    print(f"🔄 合并结果到 {final_output_dir}")
    
    # 创建最终输出目录
    os.makedirs(final_output_dir, exist_ok=True)
    
    # 合并所有成功的PDB文件
    all_pdbs = []
    all_metadata = []
    
    for chunk_info in chunks:
        chunk_id = os.path.basename(chunk_info['file']).replace('.csv', '')
        chunk_output_dir = os.path.join(base_output_dir, f"chunk_{chunk_id}_gpu*")
        
        # 查找实际的输出目录
        import glob
        actual_dirs = glob.glob(chunk_output_dir)
        if not actual_dirs:
            print(f"⚠️  未找到数据块 {chunk_id} 的输出目录")
            continue
            
        chunk_dir = actual_dirs[0]
        print(f"📁 处理数据块 {chunk_id} 输出: {chunk_dir}")
        
        # 收集PDB文件
        pdb_files = []
        for root, dirs, files in os.walk(chunk_dir):
            for file in files:
                if file.endswith('.pdb'):
                    pdb_files.append(os.path.join(root, file))
        
        all_pdbs.extend(pdb_files)
        print(f"   找到 {len(pdb_files)} 个PDB文件")
    
    print(f"📊 总共找到 {len(all_pdbs)} 个PDB文件")
    
    # 生成合并报告
    report = {
        "total_chunks": len(chunks),
        "total_pdbs": len(all_pdbs),
        "chunks_info": chunks,
        "merge_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    report_file = os.path.join(final_output_dir, "merge_report.json")
    import json
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ 合并完成，报告保存到: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="并行PDB生成 - 数据分块处理")
    parser.add_argument('--input', type=str, required=True, help='输入CSV文件')
    parser.add_argument('--sequence-column', type=str, default='sequence', help='序列列名')
    parser.add_argument('--sample-id-column', type=str, default='sample_id', help='样本ID列名')
    parser.add_argument('--gpus', type=str, default='0,1', help='可用GPU列表')
    parser.add_argument('--max-length', type=int, default=400, help='最大序列长度')
    parser.add_argument('--truncate-mode', type=str, default='skip', help='超长序列处理方式')
    parser.add_argument('--fp16', action='store_true', help='使用FP16精度')
    parser.add_argument('--output-dir', type=str, default='parallel_output', help='输出目录')
    parser.add_argument('--final-output-dir', type=str, default='final_output', help='最终合并输出目录')
    
    args = parser.parse_args()
    
    print("🚀 并行PDB生成开始")
    print(f"输入文件: {args.input}")
    print(f"可用GPU: {args.gpus}")
    
    # 解析GPU列表
    gpu_list = [int(gpu.strip()) for gpu in args.gpus.split(',')]
    num_chunks = len(gpu_list)  # 块数等于GPU数量
    print(f"数据块数量: {num_chunks} (等于GPU数量)")
    
    # 在分割数据前全局注册样本
    import sample_manager
    shared_manager = sample_manager.SampleManager(args.output_dir)
    shared_manager.register_samples_from_csv(args.input)
    print("全局注册了所有样本，使用共享目录:", args.output_dir)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 分割数据
    print(f"📊 分割数据到 {num_chunks} 个块...")
    chunks = split_data_into_chunks(args.input, num_chunks, args.output_dir)
    
    # 并行处理每个块
    print(f"🔄 开始并行处理...")
    results = []
    
    # 启动所有进程
    processes = []
    for i, chunk_info in enumerate(chunks):
        gpu_id = gpu_list[i]  # 直接使用对应的GPU
        process = start_chunk_processing(chunk_info, gpu_id, args.output_dir, args)
        processes.append((i+1, gpu_id, process))
    
    # 等待所有进程完成
    print(f"⏳ 等待所有进程完成...")
    for chunk_id, gpu_id, process in processes:
        success, message = wait_for_chunk_completion(process, chunk_id, gpu_id)
        results.append({
            'chunk_id': chunk_id,
            'gpu_id': gpu_id,
            'success': success,
            'message': message
        })
    
    # 显示结果
    print("\n📊 处理结果:")
    successful_chunks = 0
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"{status} 数据块 {result['chunk_id']} (GPU {result['gpu_id']}): {result['message']}")
        if result['success']:
            successful_chunks += 1
    
    print(f"\n🎯 总结: {successful_chunks}/{len(chunks)} 个数据块成功处理")
    
    # 合并结果
    if successful_chunks > 0:
        merge_results(args.output_dir, chunks, args.final_output_dir)
        print(f"✅ 所有结果已合并到: {args.final_output_dir}")
    else:
        print("❌ 没有成功的数据块，跳过合并")

if __name__ == '__main__':
    main()
