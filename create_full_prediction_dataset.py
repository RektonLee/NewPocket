#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建包含kcat和km预测值的完整数据集
"""

import pandas as pd
import numpy as np
import os

def create_full_prediction_dataset(original_csv, predictions_csv, output_csv):
    """
    创建包含kcat和km预测值的完整数据集
    
    Parameters:
    original_csv: 原始数据文件路径
    predictions_csv: 预测结果文件路径
    output_csv: 输出文件路径
    """
    
    # 读取原始数据
    print(f"读取原始数据: {original_csv}")
    original_df = pd.read_csv(original_csv)
    print(f"原始数据: {len(original_df)} 行, 列: {list(original_df.columns)}")
    
    # 读取预测结果
    print(f"读取预测结果: {predictions_csv}")
    predictions_df = pd.read_csv(predictions_csv)
    print(f"预测结果: {len(predictions_df)} 行, 列: {list(predictions_df.columns)}")
    
    # 只保留预测成功的样本（kcat_pred和km_pred都不为NaN）
    successful_predictions = predictions_df.dropna(subset=['kcat_pred', 'km_pred'])
    print(f"成功预测的样本: {len(successful_predictions)} 个")
    
    # 创建结果DataFrame，包含kcat和km预测值
    result_data = []
    
    for _, pred_row in successful_predictions.iterrows():
        sample_id = pred_row['sample_id']
        
        # 在原始数据中找到对应的行
        original_row = original_df[original_df['sample_id'] == sample_id]
        
        if len(original_row) > 0:
            original_row = original_row.iloc[0]  # 取第一行
            
            # 创建新行，包含kcat和km预测值
            new_row = {
                'sample_id': sample_id,
                'sequence': original_row['sequence'],
                'smiles': original_row['smiles'],
                'experimental_km_log10': original_row['experimental_km_log10'],
                'predicted_kcat': pred_row['kcat_pred'],
                'predicted_km': pred_row['km_pred'],
                'predicted_km_log10': pred_row['km_pred_log10']
            }
            
            result_data.append(new_row)
        else:
            print(f"警告: 在原始数据中未找到样本 {sample_id}")
    
    # 创建结果DataFrame
    result_df = pd.DataFrame(result_data)
    
    # 保存结果
    result_df.to_csv(output_csv, index=False)
    print(f"结果已保存到: {output_csv}")
    print(f"最终数据集: {len(result_df)} 行")
    
    # 显示统计信息
    print("\n=== 数据集统计 ===")
    print(f"总样本数: {len(result_df)}")
    print(f"有实验值的样本: {len(result_df.dropna(subset=['experimental_km_log10']))}")
    
    # 计算预测误差（如果有实验值）
    comparison_data = result_df.dropna(subset=['experimental_km_log10'])
    if len(comparison_data) > 0:
        errors = np.abs(comparison_data['predicted_km_log10'] - comparison_data['experimental_km_log10'])
        print(f"平均绝对误差 (log10): {errors.mean():.3f}")
        print(f"中位数绝对误差 (log10): {errors.median():.3f}")
        print(f"相关系数 R²: {comparison_data['predicted_km_log10'].corr(comparison_data['experimental_km_log10'])**2:.3f}")
    
    # 显示预测值范围
    print(f"kcat预测值范围: {result_df['predicted_kcat'].min():.2f} 到 {result_df['predicted_kcat'].max():.2f}")
    print(f"km预测值范围 (log10): {result_df['predicted_km_log10'].min():.3f} 到 {result_df['predicted_km_log10'].max():.3f}")
    
    return result_df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("用法: python create_full_prediction_dataset.py <原始数据CSV> <预测结果CSV> <输出CSV>")
        print("示例: python create_full_prediction_dataset.py km_test_data_with_values.csv merged_predictions.csv full_prediction_dataset.csv")
        sys.exit(1)
    
    original_csv = sys.argv[1]
    predictions_csv = sys.argv[2]
    output_csv = sys.argv[3]
    
    result = create_full_prediction_dataset(original_csv, predictions_csv, output_csv)
    
    print(f"\n✅ 完成！生成了包含 {len(result)} 个成功预测样本的完整数据集")
