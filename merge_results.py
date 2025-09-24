#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手动合并并行预测结果的脚本
用于在部分chunk还在运行时合并已完成的结果
"""

import pandas as pd
import os
import glob
from pathlib import Path

def merge_parallel_results(base_dir, output_file=None):
    """
    合并并行预测结果
    
    Parameters:
    base_dir: 并行结果的基础目录
    output_file: 输出文件路径，如果为None则自动生成
    """
    base_path = Path(base_dir)
    
    # 查找所有chunk结果目录
    chunk_dirs = sorted(glob.glob(str(base_path / "chunk_*_results")))
    
    if not chunk_dirs:
        print(f"在 {base_dir} 中没有找到chunk结果目录")
        return None
    
    print(f"找到 {len(chunk_dirs)} 个chunk结果目录:")
    for chunk_dir in chunk_dirs:
        print(f"  - {chunk_dir}")
    
    # 合并predictions.csv
    all_predictions = []
    successful_predictions = []
    failed_predictions = []
    
    for chunk_dir in chunk_dirs:
        chunk_path = Path(chunk_dir)
        
        # 检查predictions.csv
        pred_file = chunk_path / "predictions.csv"
        if pred_file.exists():
            df = pd.read_csv(pred_file)
            all_predictions.append(df)
            print(f"  ✅ 读取 {pred_file}: {len(df)} 行")
        else:
            print(f"  ❌ 未找到 {pred_file}")
        
        # 检查successful_predictions.csv
        success_file = chunk_path / "successful_predictions.csv"
        if success_file.exists():
            df = pd.read_csv(success_file)
            successful_predictions.append(df)
            print(f"  ✅ 读取 {success_file}: {len(df)} 行")
        
        # 检查failed_predictions.csv
        failed_file = chunk_path / "failed_predictions.csv"
        if failed_file.exists():
            df = pd.read_csv(failed_file)
            failed_predictions.append(df)
            print(f"  ✅ 读取 {failed_file}: {len(df)} 行")
    
    # 合并所有结果
    if all_predictions:
        merged_predictions = pd.concat(all_predictions, ignore_index=True)
        print(f"\n合并后的总预测结果: {len(merged_predictions)} 行")
        
        # 生成输出文件名
        if output_file is None:
            output_file = base_path / "merged_predictions.csv"
        
        # 保存合并结果
        merged_predictions.to_csv(output_file, index=False)
        print(f"合并结果已保存到: {output_file}")
        
        # 生成统计信息
        valid_predictions = merged_predictions.dropna(subset=['kcat_pred', 'km_pred'])
        failed_predictions_merged = merged_predictions[merged_predictions['kcat_pred'].isna() | merged_predictions['km_pred'].isna()]
        
        stats = {
            'Total Samples': len(merged_predictions),
            'Successful Predictions': len(valid_predictions),
            'Failed/Skipped': len(failed_predictions_merged),
            'Success Rate': f"{len(valid_predictions)/len(merged_predictions)*100:.1f}%"
        }
        
        if len(valid_predictions) > 0:
            stats.update({
                'kcat Prediction Range': f"{valid_predictions['kcat_pred'].min():.2f} - {valid_predictions['kcat_pred'].max():.2f}",
                'Km Prediction Range': f"{valid_predictions['km_pred'].min():.2e} - {valid_predictions['km_pred'].max():.2e}"
            })
            
            # 如果有实验值，计算误差统计
            if 'experimental_km_log10' in valid_predictions.columns:
                comparison_data = valid_predictions.dropna(subset=['experimental_km_log10'])
                if len(comparison_data) > 0:
                    stats.update({
                        'Samples with Experimental Values': len(comparison_data),
                        'Mean Absolute Error (log10)': f"{comparison_data['km_error_log10'].mean():.3f}",
                        'Median Absolute Error (log10)': f"{comparison_data['km_error_log10'].median():.3f}",
                        'Mean Relative Error': f"{comparison_data['km_error_relative'].mean():.3f}",
                        'R² Correlation': f"{comparison_data['km_pred_log10'].corr(comparison_data['experimental_km_log10'])**2:.3f}"
                    })
        
        # 保存统计信息
        stats_file = base_path / "merged_stats.csv"
        pd.DataFrame([stats]).to_csv(stats_file, index=False)
        print(f"统计信息已保存到: {stats_file}")
        
        # 打印统计信息
        print("\n=== 合并结果统计 ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        return merged_predictions
    else:
        print("没有找到任何预测结果文件")
        return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python merge_results.py <并行结果目录> [输出文件]")
        print("示例: python merge_results.py results/parallel_km_test_20250920_232209")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = merge_parallel_results(base_dir, output_file)
    
    if result is not None:
        print(f"\n✅ 合并完成！共处理 {len(result)} 个样本")
    else:
        print("\n❌ 合并失败")
