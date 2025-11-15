#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据分布诊断脚本
分析训练集和测试集的数据分布差异，找出模型性能差的原因
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
import os

def analyze_data_distribution():
    """分析训练集和测试集的数据分布"""
    print("🔍 开始数据分布分析...")
    
    # 加载数据
    train_data = torch.load('kcat_train_after_new_clean.pt', weights_only=False)
    test_data = torch.load('kcat_test_new.pt', weights_only=False)
    
    print(f"训练集大小: {len(train_data)}")
    print(f"测试集大小: {len(test_data)}")
    
    # 提取y值
    train_y = torch.cat([data.y for data in train_data]).numpy().flatten()
    test_y = torch.cat([data.y for data in test_data]).numpy().flatten()
    
    # 转换为原始尺度
    train_y_original = np.power(10, train_y)
    test_y_original = np.power(10, test_y)
    
    print("\n=== 数据分布统计 ===")
    print("训练集 (log10尺度):")
    print(f"  范围: {train_y.min():.3f} 到 {train_y.max():.3f}")
    print(f"  均值: {train_y.mean():.3f}, 标准差: {train_y.std():.3f}")
    print(f"  中位数: {np.median(train_y):.3f}")
    
    print("\n测试集 (log10尺度):")
    print(f"  范围: {test_y.min():.3f} 到 {test_y.max():.3f}")
    print(f"  均值: {test_y.mean():.3f}, 标准差: {test_y.std():.3f}")
    print(f"  中位数: {np.median(test_y):.3f}")
    
    print("\n训练集 (原始尺度):")
    print(f"  范围: {train_y_original.min():.2e} 到 {train_y_original.max():.2e}")
    print(f"  均值: {train_y_original.mean():.2e}, 标准差: {train_y_original.std():.2e}")
    
    print("\n测试集 (原始尺度):")
    print(f"  范围: {test_y_original.min():.2e} 到 {test_y_original.max():.2e}")
    print(f"  均值: {test_y_original.mean():.2e}, 标准差: {test_y_original.std():.2e}")
    
    # 计算重叠度
    train_min, train_max = train_y.min(), train_y.max()
    test_min, test_max = test_y.min(), test_y.max()
    
    overlap_min = max(train_min, test_min)
    overlap_max = min(train_max, test_max)
    overlap_range = max(0, overlap_max - overlap_min)
    total_range = max(train_max, test_max) - min(train_min, test_min)
    overlap_ratio = overlap_range / total_range if total_range > 0 else 0
    
    print(f"\n=== 分布重叠分析 ===")
    print(f"训练集范围: [{train_min:.3f}, {train_max:.3f}]")
    print(f"测试集范围: [{test_min:.3f}, {test_max:.3f}]")
    print(f"重叠范围: [{overlap_min:.3f}, {overlap_max:.3f}]")
    print(f"重叠比例: {overlap_ratio:.1%}")
    
    # 创建可视化
    create_distribution_plots(train_y, test_y, train_y_original, test_y_original)
    
    return {
        'train_y': train_y,
        'test_y': test_y,
        'overlap_ratio': overlap_ratio,
        'train_stats': {
            'mean': train_y.mean(),
            'std': train_y.std(),
            'min': train_y.min(),
            'max': train_y.max()
        },
        'test_stats': {
            'mean': test_y.mean(),
            'std': test_y.std(),
            'min': test_y.min(),
            'max': test_y.max()
        }
    }

def create_distribution_plots(train_y, test_y, train_y_original, test_y_original):
    """创建分布对比图"""
    os.makedirs('outputs/diagnosis', exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. log10尺度分布对比
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 直方图对比
    axes[0, 0].hist(train_y, bins=50, alpha=0.7, label='训练集', color='blue', density=True)
    axes[0, 0].hist(test_y, bins=50, alpha=0.7, label='测试集', color='red', density=True)
    axes[0, 0].set_xlabel('kcat (log10)')
    axes[0, 0].set_ylabel('密度')
    axes[0, 0].set_title('数据分布对比 (log10尺度)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 箱线图对比
    data_to_plot = [train_y, test_y]
    axes[0, 1].boxplot(data_to_plot, labels=['训练集', '测试集'])
    axes[0, 1].set_ylabel('kcat (log10)')
    axes[0, 1].set_title('数据分布箱线图 (log10尺度)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 原始尺度分布对比
    axes[1, 0].hist(np.log10(train_y_original), bins=50, alpha=0.7, label='训练集', color='blue', density=True)
    axes[1, 0].hist(np.log10(test_y_original), bins=50, alpha=0.7, label='测试集', color='red', density=True)
    axes[1, 0].set_xlabel('kcat (log10)')
    axes[1, 0].set_ylabel('密度')
    axes[1, 0].set_title('数据分布对比 (原始尺度转log10)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 累积分布函数
    train_sorted = np.sort(train_y)
    test_sorted = np.sort(test_y)
    train_cdf = np.arange(1, len(train_sorted) + 1) / len(train_sorted)
    test_cdf = np.arange(1, len(test_sorted) + 1) / len(test_sorted)
    
    axes[1, 1].plot(train_sorted, train_cdf, label='训练集', color='blue', linewidth=2)
    axes[1, 1].plot(test_sorted, test_cdf, label='测试集', color='red', linewidth=2)
    axes[1, 1].set_xlabel('kcat (log10)')
    axes[1, 1].set_ylabel('累积概率')
    axes[1, 1].set_title('累积分布函数对比')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/diagnosis/data_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 详细分析图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 散点图：训练集vs测试集
    train_indices = np.random.choice(len(train_y), min(1000, len(train_y)), replace=False)
    test_indices = np.random.choice(len(test_y), min(1000, len(test_y)), replace=False)
    
    axes[0].scatter(train_y[train_indices], np.ones_like(train_y[train_indices]), 
                   alpha=0.6, s=20, label='训练集', color='blue')
    axes[0].scatter(test_y[test_indices], np.ones_like(test_y[test_indices]) * 1.1, 
                   alpha=0.6, s=20, label='测试集', color='red')
    axes[0].set_xlabel('kcat (log10)')
    axes[0].set_ylabel('数据集')
    axes[0].set_title('数据点分布对比')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 分位数对比
    quantiles = np.linspace(0, 1, 21)
    train_quantiles = np.quantile(train_y, quantiles)
    test_quantiles = np.quantile(test_y, quantiles)
    
    axes[1].plot(quantiles, train_quantiles, 'o-', label='训练集', color='blue', markersize=4)
    axes[1].plot(quantiles, test_quantiles, 'o-', label='测试集', color='red', markersize=4)
    axes[1].set_xlabel('分位数')
    axes[1].set_ylabel('kcat (log10)')
    axes[1].set_title('分位数对比')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 统计信息对比
    stats_data = {
        '指标': ['样本数', '均值', '标准差', '最小值', '最大值', '中位数'],
        '训练集': [f'{len(train_y):,}', f'{train_y.mean():.3f}', f'{train_y.std():.3f}', 
                  f'{train_y.min():.3f}', f'{train_y.max():.3f}', f'{np.median(train_y):.3f}'],
        '测试集': [f'{len(test_y):,}', f'{test_y.mean():.3f}', f'{test_y.std():.3f}', 
                  f'{test_y.min():.3f}', f'{test_y.max():.3f}', f'{np.median(test_y):.3f}']
    }
    
    axes[2].axis('tight')
    axes[2].axis('off')
    table = axes[2].table(cellText=list(zip(*[stats_data['指标'], stats_data['训练集'], stats_data['测试集']])),
                         colLabels=['指标', '训练集', '测试集'],
                         cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    axes[2].set_title('统计信息对比')
    
    plt.tight_layout()
    plt.savefig('outputs/diagnosis/detailed_analysis.png', dpi=300, bbox_inights='tight')
    plt.close()
    
    print("✅ 可视化图表已保存到 outputs/diagnosis/")

def suggest_solutions(analysis_results):
    """基于分析结果提供解决方案建议"""
    print("\n=== 解决方案建议 ===")
    
    overlap_ratio = analysis_results['overlap_ratio']
    train_stats = analysis_results['train_stats']
    test_stats = analysis_results['test_stats']
    
    print(f"数据重叠度: {overlap_ratio:.1%}")
    
    if overlap_ratio < 0.3:
        print("❌ 严重的数据分布不匹配问题！")
        print("\n🔧 建议解决方案:")
        
        print("\n1. 数据重新划分:")
        print("   - 将训练集和测试集合并，重新按EC号或蛋白质相似性划分")
        print("   - 确保训练集和测试集有相似的kcat值分布")
        
        print("\n2. 数据增强:")
        print("   - 对测试集范围的数据进行过采样")
        print("   - 使用SMOTE等技术生成合成样本")
        
        print("\n3. 模型调整:")
        print("   - 使用域适应技术 (Domain Adaptation)")
        print("   - 在测试集上微调模型")
        print("   - 使用对抗训练减少域间差异")
        
        print("\n4. 评估策略:")
        print("   - 在重叠范围内评估模型性能")
        print("   - 分别报告不同kcat范围的性能")
        
    elif overlap_ratio < 0.6:
        print("⚠️ 存在中等程度的数据分布不匹配")
        print("\n🔧 建议解决方案:")
        print("   - 使用数据标准化/归一化")
        print("   - 在重叠范围内重点评估")
        print("   - 考虑使用更鲁棒的损失函数")
        
    else:
        print("✅ 数据分布匹配度较好")
        print("   - 问题可能在其他方面，如特征质量、模型复杂度等")

def main():
    """主函数"""
    print("🚀 开始数据分布诊断...")
    
    # 分析数据分布
    analysis_results = analyze_data_distribution()
    
    # 提供解决方案建议
    suggest_solutions(analysis_results)
    
    print("\n✅ 诊断完成！")
    print("📊 查看 outputs/diagnosis/ 目录中的可视化图表")

if __name__ == "__main__":
    main()

