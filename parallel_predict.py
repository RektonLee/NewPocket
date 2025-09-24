#!/usr/bin/env python3
"""
并行预测脚本 - 将数据集分成多个块，每个块独立运行预测
"""

import os
import sys
import pandas as pd
import argparse
import subprocess
import time
from pathlib import Path
import logging

def setup_logging(log_file):
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def split_dataset(input_csv, num_chunks, output_dir):
    """将数据集分成多个块"""
    df = pd.read_csv(input_csv)
    total_samples = len(df)
    chunk_size = total_samples // num_chunks
    remainder = total_samples % num_chunks
    
    logging.info(f"数据集总样本数: {total_samples}")
    logging.info(f"分成 {num_chunks} 个块，每块约 {chunk_size} 个样本")
    
    chunk_files = []
    start_idx = 0
    
    for i in range(num_chunks):
        # 计算当前块的大小（前remainder个块多分配一个样本）
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        
        # 创建块数据
        chunk_df = df.iloc[start_idx:end_idx].copy()
        
        # 保存块文件
        chunk_file = os.path.join(output_dir, f'chunk_{i+1:03d}.csv')
        chunk_df.to_csv(chunk_file, index=False)
        chunk_files.append(chunk_file)
        
        logging.info(f"块 {i+1}: 样本 {start_idx+1}-{end_idx} ({len(chunk_df)} 个样本) -> {chunk_file}")
        start_idx = end_idx
    
    return chunk_files

def run_chunk_prediction(chunk_file, chunk_id, model_path, base_output_dir, 
                        use_sample_manager=True, sample_data_dir='sample_data', 
                        temperature=303.15):
    """运行单个块的预测"""
    output_dir = os.path.join(base_output_dir, f'chunk_{chunk_id:03d}_results')
    
    cmd = [
        'python', 'pred_fullpipeline.py',
        '--input', chunk_file,
        '--model', model_path,
        '--output', output_dir,
        '--temperature', str(temperature)
    ]
    
    if use_sample_manager:
        cmd.extend(['--use-sample-manager', '--sample-data-dir', sample_data_dir])
    
    logging.info(f"启动块 {chunk_id} 预测: {' '.join(cmd)}")
    
    try:
        # 运行预测
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1小时超时
        
        if result.returncode == 0:
            logging.info(f"✅ 块 {chunk_id} 预测完成")
            return True, result.stdout
        else:
            logging.error(f"❌ 块 {chunk_id} 预测失败: {result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        logging.error(f"❌ 块 {chunk_id} 预测超时")
        return False, "Timeout"
    except Exception as e:
        logging.error(f"❌ 块 {chunk_id} 预测异常: {str(e)}")
        return False, str(e)

def run_parallel_predictions(chunk_files, model_path, base_output_dir, 
                           max_parallel=4, use_sample_manager=True, 
                           sample_data_dir='sample_data', temperature=303.15):
    """并行运行多个块的预测"""
    import concurrent.futures
    import threading
    
    results = {}
    completed_chunks = 0
    total_chunks = len(chunk_files)
    
    def run_chunk_with_id(chunk_file):
        chunk_id = int(os.path.basename(chunk_file).split('_')[1].split('.')[0])
        return chunk_id, run_chunk_prediction(
            chunk_file, chunk_id, model_path, base_output_dir,
            use_sample_manager, sample_data_dir, temperature
        )
    
    logging.info(f"开始并行预测，最大并行数: {max_parallel}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        # 提交所有任务
        future_to_chunk = {
            executor.submit(run_chunk_with_id, chunk_file): chunk_file 
            for chunk_file in chunk_files
        }
        
        # 处理完成的任务
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_file = future_to_chunk[future]
            try:
                chunk_id, (success, output) = future.result()
                results[chunk_id] = (success, output)
                completed_chunks += 1
                
                if success:
                    logging.info(f"✅ 块 {chunk_id} 完成 ({completed_chunks}/{total_chunks})")
                else:
                    logging.error(f"❌ 块 {chunk_id} 失败 ({completed_chunks}/{total_chunks})")
                    
            except Exception as e:
                chunk_id = int(os.path.basename(chunk_file).split('_')[1].split('.')[0])
                results[chunk_id] = (False, str(e))
                completed_chunks += 1
                logging.error(f"❌ 块 {chunk_id} 异常: {str(e)} ({completed_chunks}/{total_chunks})")
    
    return results

def merge_results(base_output_dir, num_chunks, final_output_dir):
    """合并所有块的结果"""
    logging.info("开始合并结果...")
    
    all_predictions = []
    all_successful = []
    all_failed = []
    
    for i in range(1, num_chunks + 1):
        chunk_dir = os.path.join(base_output_dir, f'chunk_{i:03d}_results')
        
        # 合并predictions.csv
        predictions_file = os.path.join(chunk_dir, 'predictions.csv')
        if os.path.exists(predictions_file):
            df = pd.read_csv(predictions_file)
            all_predictions.append(df)
            logging.info(f"合并块 {i} 的预测结果: {len(df)} 个样本")
        
        # 合并successful_predictions.csv
        successful_file = os.path.join(chunk_dir, 'successful_predictions.csv')
        if os.path.exists(successful_file):
            df = pd.read_csv(successful_file)
            all_successful.append(df)
        
        # 合并failed_predictions.csv
        failed_file = os.path.join(chunk_dir, 'failed_predictions.csv')
        if os.path.exists(failed_file):
            df = pd.read_csv(failed_file)
            all_failed.append(df)
    
    # 创建最终输出目录
    os.makedirs(final_output_dir, exist_ok=True)
    
    # 合并并保存结果
    if all_predictions:
        final_predictions = pd.concat(all_predictions, ignore_index=True)
        final_predictions.to_csv(os.path.join(final_output_dir, 'predictions.csv'), index=False)
        logging.info(f"✅ 合并完成: {len(final_predictions)} 个总样本")
    
    if all_successful:
        final_successful = pd.concat(all_successful, ignore_index=True)
        final_successful.to_csv(os.path.join(final_output_dir, 'successful_predictions.csv'), index=False)
        logging.info(f"✅ 成功预测: {len(final_successful)} 个样本")
    
    if all_failed:
        final_failed = pd.concat(all_failed, ignore_index=True)
        final_failed.to_csv(os.path.join(final_output_dir, 'failed_predictions.csv'), index=False)
        logging.info(f"✅ 失败样本: {len(final_failed)} 个样本")
    
    # 生成最终统计报告
    if all_predictions:
        valid_predictions = final_predictions.dropna(subset=['kcat_pred', 'km_pred'])
        failed_predictions = final_predictions[final_predictions['kcat_pred'].isna() | final_predictions['km_pred'].isna()]
        
        stats = {
            '总样本数': len(final_predictions),
            '成功预测数': len(valid_predictions),
            '失败/跳过数': len(failed_predictions),
            '成功率': f"{len(valid_predictions)/len(final_predictions)*100:.1f}%",
            '并行块数': num_chunks
        }
        
        if len(valid_predictions) > 0:
            stats.update({
                'kcat预测范围': f"{valid_predictions['kcat_pred'].min():.2f} - {valid_predictions['kcat_pred'].max():.2f}",
                'Km预测范围': f"{valid_predictions['km_pred'].min():.2e} - {valid_predictions['km_pred'].max():.2e}"
            })
        
        pd.DataFrame([stats]).to_csv(os.path.join(final_output_dir, 'final_stats.csv'), index=False)
        logging.info("✅ 最终统计报告已生成")

def main():
    parser = argparse.ArgumentParser(description="并行预测脚本")
    parser.add_argument('--input', type=str, required=True, help='输入CSV文件路径')
    parser.add_argument('--model', type=str, 
                       default="/home/lizihao/Work/enzyme_prediction/src/simple2/outputs/nopqr_attention_rbf/best_model.pt",
                       help='模型文件路径')
    parser.add_argument('--output', type=str, default='parallel_results', help='输出目录')
    parser.add_argument('--chunks', type=int, default=8, help='并行块数')
    parser.add_argument('--max-parallel', type=int, default=4, help='最大并行数')
    parser.add_argument('--temperature', type=float, default=303.15, help='温度（K）')
    parser.add_argument('--use-sample-manager', action='store_true', help='使用SampleManager')
    parser.add_argument('--sample-data-dir', type=str, default='sample_data', help='SampleManager基础目录')
    parser.add_argument('--skip-split', action='store_true', help='跳过数据分割，直接运行预测')
    parser.add_argument('--merge-only', action='store_true', help='只合并已有结果')
    
    args = parser.parse_args()
    
    # 设置日志
    log_file = os.path.join(args.output, 'parallel_prediction.log')
    os.makedirs(args.output, exist_ok=True)
    setup_logging(log_file)
    
    logging.info("=" * 60)
    logging.info("开始并行预测任务")
    logging.info(f"输入文件: {args.input}")
    logging.info(f"模型路径: {args.model}")
    logging.info(f"输出目录: {args.output}")
    logging.info(f"并行块数: {args.chunks}")
    logging.info(f"最大并行数: {args.max_parallel}")
    logging.info("=" * 60)
    
    if args.merge_only:
        # 只合并结果
        merge_results(args.output, args.chunks, os.path.join(args.output, 'final_results'))
        return
    
    if not args.skip_split:
        # 分割数据集
        chunk_files = split_dataset(args.input, args.chunks, args.output)
    else:
        # 使用已有的块文件
        chunk_files = [os.path.join(args.output, f'chunk_{i+1:03d}.csv') for i in range(args.chunks)]
        logging.info(f"使用已有的块文件: {chunk_files}")
    
    # 并行运行预测
    start_time = time.time()
    results = run_parallel_predictions(
        chunk_files, args.model, args.output, 
        args.max_parallel, args.use_sample_manager, 
        args.sample_data_dir, args.temperature
    )
    end_time = time.time()
    
    # 统计结果
    successful_chunks = sum(1 for success, _ in results.values() if success)
    failed_chunks = len(results) - successful_chunks
    
    logging.info("=" * 60)
    logging.info("并行预测完成")
    logging.info(f"成功块数: {successful_chunks}/{len(results)}")
    logging.info(f"失败块数: {failed_chunks}")
    logging.info(f"总耗时: {end_time - start_time:.2f} 秒")
    logging.info("=" * 60)
    
    # 合并结果
    final_output_dir = os.path.join(args.output, 'final_results')
    merge_results(args.output, args.chunks, final_output_dir)
    
    logging.info(f"🎉 所有任务完成！最终结果保存在: {final_output_dir}")

if __name__ == '__main__':
    main()
