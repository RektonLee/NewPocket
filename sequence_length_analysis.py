#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
序列长度分布统计脚本
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_sequence_lengths(csv_file):
    """分析序列长度分布"""
    
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 计算每个序列的长度
    df['sequence_length'] = df['sequence'].str.len()
    
    # 定义长度区间
    bins = [0, 400, 500, 600, 700, 800, 900, 1000, float('inf')]
    bin_labels = ['<400', '400-500', '500-600', '600-700', '700-800', '800-900', '900-1000', '1000+']
    
    # 统计每个区间的数量
    df['length_category'] = pd.cut(df['sequence_length'], bins=bins, labels=bin_labels, right=False)
    
    # 统计结果
    length_stats = df['length_category'].value_counts().sort_index()
    
    print("序列长度分布统计:")
    print("=" * 50)
    total_sequences = len(df)
    
    for category, count in length_stats.items():
        percentage = (count / total_sequences) * 100
        print(f"{category:>10}: {count:>4} 个 ({percentage:>5.1f}%)")
    
    print("-" * 50)
    print(f"{'总计':>10}: {total_sequences:>4} 个")
    
    # 基础统计信息
    print("\n基础统计信息:")
    print("=" * 50)
    print(f"最短序列长度: {df['sequence_length'].min()}")
    print(f"最长序列长度: {df['sequence_length'].max()}")
    print(f"平均序列长度: {df['sequence_length'].mean():.1f}")
    print(f"中位数序列长度: {df['sequence_length'].median():.1f}")
    print(f"标准差: {df['sequence_length'].std():.1f}")
    
    # 绘制分布图
    plt.figure(figsize=(12, 8))
    
    # 子图1: 直方图
    plt.subplot(2, 2, 1)
    plt.hist(df['sequence_length'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('序列长度')
    plt.ylabel('频次')
    plt.title('序列长度分布直方图')
    plt.grid(True, alpha=0.3)
    
    # 子图2: 箱线图
    plt.subplot(2, 2, 2)
    plt.boxplot(df['sequence_length'])
    plt.ylabel('序列长度')
    plt.title('序列长度箱线图')
    plt.grid(True, alpha=0.3)
    
    # 子图3: 长度区间柱状图
    plt.subplot(2, 2, 3)
    length_stats.plot(kind='bar', color='lightcoral')
    plt.xlabel('长度区间')
    plt.ylabel('序列数量')
    plt.title('各长度区间序列数量')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 子图4: 累积分布
    plt.subplot(2, 2, 4)
    sorted_lengths = np.sort(df['sequence_length'])
    cumulative_freq = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
    plt.plot(sorted_lengths, cumulative_freq, linewidth=2, color='green')
    plt.xlabel('序列长度')
    plt.ylabel('累积频率')
    plt.title('序列长度累积分布')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sequence_length_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存详细统计结果到CSV
    detailed_stats = df[['sample_id', 'sequence_length', 'length_category']].copy()
    detailed_stats = detailed_stats.sort_values('sequence_length')
    detailed_stats.to_csv('sequence_length_stats.csv', index=False)
    
    # 保存汇总统计
    summary_stats = pd.DataFrame({
        '长度区间': length_stats.index,
        '序列数量': length_stats.values,
        '百分比': [(count / total_sequences) * 100 for count in length_stats.values]
    })
    
    print(f"\n详细结果已保存到:")
    print(f"- sequence_length_stats.csv (每个序列的详细信息)")
    print(f"- sequence_length_analysis.png (可视化图表)")
    
    return summary_stats, df

if __name__ == "__main__":
    # 分析序列长度
    summary, data = analyze_sequence_lengths('km_test_data.csv')
    
    print("\n汇总统计表:")
    print(summary.to_string(index=False))