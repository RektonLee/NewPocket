#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从kcat并行任务的日志文件中提取预测结果并生成CSV文件
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path

def extract_predictions_from_log(log_file):
    """从单个日志文件中提取预测结果"""
    predictions = []
    
    if not os.path.exists(log_file):
        print(f"日志文件不存在: {log_file}")
        return predictions
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取样本ID和预测结果
    # 匹配模式: Processing sample X/Y: sample_id
    sample_pattern = r'Processing sample \d+/\d+: (kcat_test_\d+)'
    # 匹配模式: Prediction completed - kcat: X.XX, Km: X.XXe-XX
    prediction_pattern = r'Prediction completed - kcat: ([\d.]+), Km: ([\d.e-]+)'
    
    # 找到所有样本ID
    sample_matches = re.findall(sample_pattern, content)
    # 找到所有预测结果
    prediction_matches = re.findall(prediction_pattern, content)
    
    print(f"在 {log_file} 中找到 {len(sample_matches)} 个样本, {len(prediction_matches)} 个预测结果")
    
    # 将样本ID和预测结果配对
    for i, sample_id in enumerate(sample_matches):
        if i < len(prediction_matches):
            kcat_pred = float(prediction_matches[i][0])
            km_pred_str = prediction_matches[i][1]
            
            # 解析Km值
            if 'e-' in km_pred_str:
                km_pred = float(km_pred_str)
            else:
                km_pred = float(km_pred_str)
            
            # 计算log10值
            km_pred_log10 = np.log10(km_pred) if km_pred > 0 else np.nan
            
            predictions.append({
                'sample_id': sample_id,
                'kcat_pred': kcat_pred,
                'km_pred': km_pred,
                'km_pred_log10': km_pred_log10
            })
        else:
            # 没有预测结果的样本（可能是跳过的）
            predictions.append({
                'sample_id': sample_id,
                'kcat_pred': np.nan,
                'km_pred': np.nan,
                'km_pred_log10': np.nan
            })
    
    return predictions

def main():
    # 设置路径
    base_dir = "/home/lizihao/Work/enzyme_prediction/PGNN/results/parallel_kcat_20250919_184103"
    input_csv = "/home/lizihao/Work/enzyme_prediction/PGNN/kcat_test_results.csv"
    output_csv = "/home/lizihao/Work/enzyme_prediction/PGNN/results/kcat_predictions_extracted.csv"
    
    # 读取原始输入文件
    print("读取原始输入文件...")
    original_df = pd.read_csv(input_csv)
    print(f"原始文件包含 {len(original_df)} 个样本")
    
    # 提取所有chunk的预测结果
    all_predictions = []
    
    for chunk_num in range(1, 9):  # chunk_1 到 chunk_8
        log_file = os.path.join(base_dir, f"chunk_{chunk_num}_results", "prediction.log")
        print(f"\n处理 chunk_{chunk_num}...")
        
        chunk_predictions = extract_predictions_from_log(log_file)
        all_predictions.extend(chunk_predictions)
    
    print(f"\n总共提取到 {len(all_predictions)} 个预测结果")
    
    # 转换为DataFrame
    predictions_df = pd.DataFrame(all_predictions)
    
    # 与原始数据合并
    print("\n合并预测结果与原始数据...")
    merged_df = original_df.merge(predictions_df, on='sample_id', how='left')
    
    # 重命名列以匹配原始格式
    result_df = merged_df[['sample_id', 'sequence', 'smiles', 'experimental value[log10]', 'km_pred_log10']].copy()
    result_df.columns = ['sample_id', 'sequence', 'smiles', 'experimental value[log10]', 'predicted value[log10]']
    
    # 统计信息
    total_samples = len(result_df)
    successful_predictions = result_df['predicted value[log10]'].notna().sum()
    failed_predictions = total_samples - successful_predictions
    
    print(f"\n统计信息:")
    print(f"总样本数: {total_samples}")
    print(f"成功预测: {successful_predictions}")
    print(f"失败/跳过: {failed_predictions}")
    print(f"成功率: {successful_predictions/total_samples*100:.1f}%")
    
    # 保存结果
    result_df.to_csv(output_csv, index=False)
    print(f"\n结果已保存到: {output_csv}")
    
    # 保存成功预测的样本
    successful_df = result_df.dropna(subset=['predicted value[log10]'])
    successful_csv = "/home/lizihao/Work/enzyme_prediction/PGNN/results/kcat_successful_predictions.csv"
    successful_df.to_csv(successful_csv, index=False)
    print(f"成功预测的样本已保存到: {successful_csv}")
    
    # 保存失败/跳过的样本
    failed_df = result_df[result_df['predicted value[log10]'].isna()]
    failed_csv = "/home/lizihao/Work/enzyme_prediction/PGNN/results/kcat_failed_predictions.csv"
    failed_df.to_csv(failed_csv, index=False)
    print(f"失败/跳过的样本已保存到: {failed_csv}")
    
    # 如果有实验值，计算比较统计
    if 'experimental value[log10]' in result_df.columns:
        comparison_df = result_df.dropna(subset=['experimental value[log10]', 'predicted value[log10]'])
        if len(comparison_df) > 0:
            error = comparison_df['predicted value[log10]'] - comparison_df['experimental value[log10]']
            mae = np.mean(np.abs(error))
            rmse = np.sqrt(np.mean(error**2))
            r2 = np.corrcoef(comparison_df['experimental value[log10]'], comparison_df['predicted value[log10]'])[0,1]**2
            
            print(f"\n比较分析 (有实验值的样本: {len(comparison_df)}):")
            print(f"平均绝对误差: {mae:.4f}")
            print(f"均方根误差: {rmse:.4f}")
            print(f"R²相关系数: {r2:.4f}")

if __name__ == "__main__":
    main()
