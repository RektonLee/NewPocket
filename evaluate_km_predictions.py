#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算kcat预测效果的详细评估指标
包括R²、MAE、RMSE、MAPE等学术评价方法
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set journal-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.shadow': False,
    'figure.dpi': 300
})

def calculate_metrics(y_true, y_pred):
    """计算各种评估指标"""
    # 移除NaN值
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {}
    
    # 基本统计
    n = len(y_true_clean)
    
    # 1. R² (决定系数)
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    # 2. 皮尔逊相关系数
    pearson_r, pearson_p = stats.pearsonr(y_true_clean, y_pred_clean)
    
    # 3. 斯皮尔曼相关系数
    spearman_r, spearman_p = stats.spearmanr(y_true_clean, y_pred_clean)
    
    # 4. 平均绝对误差 (MAE)
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    
    # 5. 均方根误差 (RMSE)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    
    # 6. 平均绝对百分比误差 (MAPE)
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
    
    # 7. 对称平均绝对百分比误差 (sMAPE)
    smape = np.mean(2 * np.abs(y_true_clean - y_pred_clean) / (np.abs(y_true_clean) + np.abs(y_pred_clean))) * 100
    
    # 8. 最大误差
    max_error = np.max(np.abs(y_true_clean - y_pred_clean))
    
    # 9. 中位数绝对误差
    median_ae = np.median(np.abs(y_true_clean - y_pred_clean))
    
    # 10. 误差的标准差
    error_std = np.std(y_true_clean - y_pred_clean)
    
    # 11. 误差的偏度
    error_skew = stats.skew(y_true_clean - y_pred_clean)
    
    # 12. 误差的峰度
    error_kurtosis = stats.kurtosis(y_true_clean - y_pred_clean)
    
    # 13. 在特定误差范围内的样本比例
    error_1 = np.sum(np.abs(y_true_clean - y_pred_clean) <= 1.0) / n * 100
    error_2 = np.sum(np.abs(y_true_clean - y_pred_clean) <= 2.0) / n * 100
    error_3 = np.sum(np.abs(y_true_clean - y_pred_clean) <= 3.0) / n * 100
    
    return {
        '样本数': n,
        'R²': r2,
        '皮尔逊相关系数': pearson_r,
        '皮尔逊p值': pearson_p,
        '斯皮尔曼相关系数': spearman_r,
        '斯皮尔曼p值': spearman_p,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE (%)': mape,
        'sMAPE (%)': smape,
        '最大误差': max_error,
        '中位数绝对误差': median_ae,
        '误差标准差': error_std,
        '误差偏度': error_skew,
        '误差峰度': error_kurtosis,
        '误差≤1.0的样本比例 (%)': error_1,
        '误差≤2.0的样本比例 (%)': error_2,
        '误差≤3.0的样本比例 (%)': error_3
    }

def plot_predictions(y_true, y_pred, title="Prediction Performance Evaluation"):
    """Plot prediction performance charts with journal-quality styling"""
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    # Calculate key metrics
    r2 = r2_score(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    pearson_r, _ = stats.pearsonr(y_true_clean, y_pred_clean)
    
    # Define professional color palette
    colors = {
        'primary': '#2E86AB',      # Professional blue
        'secondary': '#A23B72',    # Deep magenta
        'accent': '#F18F01',       # Orange
        'success': '#C73E1D',      # Red
        'text': '#2C3E50',         # Dark gray
        'light': '#ECF0F1',        # Light gray
        'perfect': '#E74C3C'       # Red for perfect line
    }
    
    # Create figure with journal-quality layout
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.patch.set_facecolor('white')
    
    # 1. Scatter plot with professional styling
    scatter = axes[0, 0].scatter(y_true_clean, y_pred_clean, 
                                alpha=0.7, s=25, c=colors['primary'], 
                                edgecolors='white', linewidth=0.5)
    
    # Perfect prediction line
    min_val, max_val = min(y_true_clean.min(), y_pred_clean.min()), max(y_true_clean.max(), y_pred_clean.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 
                    color=colors['perfect'], linestyle='--', linewidth=2.5, 
                    label='Perfect Prediction', alpha=0.8)
    
    # Styling
    axes[0, 0].set_xlabel('Experimental Values (log₁₀)', fontsize=13, fontweight='bold', color=colors['text'])
    axes[0, 0].set_ylabel('Predicted Values (log₁₀)', fontsize=13, fontweight='bold', color=colors['text'])
    axes[0, 0].set_title('Predicted vs Experimental Values', fontsize=14, fontweight='bold', color=colors['text'], pad=20)
    axes[0, 0].legend(frameon=True, fancybox=False, shadow=False, fontsize=11)
    
    # Professional metrics box
    metrics_text = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}\nr = {pearson_r:.4f}'
    axes[0, 0].text(0.05, 0.95, metrics_text, transform=axes[0, 0].transAxes, 
                    fontsize=11, verticalalignment='top', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=colors['light'], 
                             edgecolor=colors['primary'], alpha=0.9, linewidth=1.5))
    
    # 2. Residual plot with enhanced styling
    residuals = y_pred_clean - y_true_clean
    axes[0, 1].scatter(y_true_clean, residuals, alpha=0.7, s=25, 
                       c=colors['secondary'], edgecolors='white', linewidth=0.5)
    axes[0, 1].axhline(y=0, color=colors['perfect'], linestyle='--', linewidth=2.5, alpha=0.8)
    
    # Styling
    axes[0, 1].set_xlabel('Experimental Values (log₁₀)', fontsize=13, fontweight='bold', color=colors['text'])
    axes[0, 1].set_ylabel('Residuals (Predicted - Experimental)', fontsize=13, fontweight='bold', color=colors['text'])
    axes[0, 1].set_title('Residual Plot', fontsize=14, fontweight='bold', color=colors['text'], pad=20)
    
    # Residual statistics box
    residual_std = np.std(residuals)
    residual_mean = np.mean(residuals)
    residual_text = f'Mean = {residual_mean:.4f}\nStd = {residual_std:.4f}'
    axes[0, 1].text(0.05, 0.95, residual_text, transform=axes[0, 1].transAxes, 
                    fontsize=11, verticalalignment='top', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=colors['light'], 
                             edgecolor=colors['secondary'], alpha=0.9, linewidth=1.5))
    
    # 3. Enhanced histogram with gradient colors
    n, bins, patches = axes[1, 0].hist(residuals, bins=50, alpha=0.8, edgecolor='white', linewidth=0.8)
    
    # Apply gradient colors to histogram bars
    for i, (patch, count) in enumerate(zip(patches, n)):
        intensity = count / max(n)
        patch.set_facecolor(plt.cm.Blues(0.3 + 0.7 * intensity))
    
    # Add reference lines
    axes[1, 0].axvline(x=0, color=colors['perfect'], linestyle='--', linewidth=2.5, alpha=0.8, label='Zero Line')
    axes[1, 0].axvline(x=residual_mean, color=colors['accent'], linestyle='-', linewidth=2.5, 
                       label=f'Mean = {residual_mean:.3f}')
    
    # Styling
    axes[1, 0].set_xlabel('Residuals', fontsize=13, fontweight='bold', color=colors['text'])
    axes[1, 0].set_ylabel('Frequency', fontsize=13, fontweight='bold', color=colors['text'])
    axes[1, 0].set_title('Residual Distribution', fontsize=14, fontweight='bold', color=colors['text'], pad=20)
    axes[1, 0].legend(frameon=True, fancybox=False, shadow=False, fontsize=11)
    
    # Distribution statistics box
    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)
    dist_text = f'Skewness = {skewness:.3f}\nKurtosis = {kurtosis:.3f}'
    axes[1, 0].text(0.05, 0.95, dist_text, transform=axes[1, 0].transAxes, 
                    fontsize=11, verticalalignment='top', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=colors['light'], 
                             edgecolor=colors['accent'], alpha=0.9, linewidth=1.5))
    
    # 4. Enhanced Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Residual Q-Q Plot', fontsize=14, fontweight='bold', color=colors['text'], pad=20)
    
    # Style the Q-Q plot
    for line in axes[1, 1].get_lines():
        if line.get_marker() == 'o':
            line.set_markersize(4)
            line.set_markerfacecolor(colors['primary'])
            line.set_markeredgecolor('white')
            line.set_markeredgewidth(0.5)
        else:
            line.set_color(colors['perfect'])
            line.set_linewidth(2.5)
    
    # Normality test results box
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(residual_mean, residual_std))
    norm_text = f'Shapiro-Wilk:\np = {shapiro_p:.2e}\nKS test:\np = {ks_p:.2e}'
    axes[1, 1].text(0.05, 0.95, norm_text, transform=axes[1, 1].transAxes, 
                    fontsize=10, verticalalignment='top', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=colors['light'], 
                             edgecolor=colors['success'], alpha=0.9, linewidth=1.5))
    
    # Professional overall title
    fig.suptitle(f'{title}\n(n = {len(y_true_clean):,} samples)', 
                 fontsize=18, fontweight='bold', color=colors['text'], y=0.98)
    
    # Adjust layout with professional spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
    
    # Save with high quality
    plt.savefig('/home/lizihao/Work/enzyme_prediction/PGNN/results/km_prediction_evaluation.png', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()

def main():
    # 读取数据
    print("读取Km预测结果...")
    df = pd.read_csv('/home/lizihao/Work/enzyme_prediction/PGNN/km_test_data_with_values.csv')
    
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # 提取实验值和预测值
    y_true = df['experimental_km_log10'].values
    y_pred = df['predicted_km_log10'].values
    
    print(f"\n实验值范围: {np.nanmin(y_true):.3f} 到 {np.nanmax(y_true):.3f}")
    print(f"预测值范围: {np.nanmin(y_pred):.3f} 到 {np.nanmax(y_pred):.3f}")
    
    # 计算评估指标
    print("\n计算评估指标...")
    metrics = calculate_metrics(y_true, y_pred)
    
    # 打印结果
    print("\n" + "="*60)
    print("Km预测效果评估结果")
    print("="*60)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'p值' in key:
                print(f"{key}: {value:.2e}")
            elif '比例' in key or 'MAPE' in key or 'sMAPE' in key:
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # 保存结果到CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('/home/lizihao/Work/enzyme_prediction/PGNN/results/km_evaluation_metrics.csv', 
                      index=False)
    print(f"\n评估指标已保存到: km_evaluation_metrics.csv")
    
    # Generate plots
    print("\nGenerating prediction performance charts...")
    plot_predictions(y_true, y_pred, "Km Prediction Performance Evaluation")
    print("Charts saved to: km_prediction_evaluation.png")
    
    # 按误差范围分析
    print("\n" + "="*60)
    print("按误差范围分析")
    print("="*60)
    
    error = np.abs(y_pred - y_true)
    error_ranges = [
        (0, 0.5, "误差 ≤ 0.5"),
        (0.5, 1.0, "0.5 < 误差 ≤ 1.0"),
        (1.0, 2.0, "1.0 < 误差 ≤ 2.0"),
        (2.0, 3.0, "2.0 < 误差 ≤ 3.0"),
        (3.0, float('inf'), "误差 > 3.0")
    ]
    
    for min_err, max_err, label in error_ranges:
        if max_err == float('inf'):
            count = np.sum(error > min_err)
        else:
            count = np.sum((error > min_err) & (error <= max_err))
        percentage = count / len(error) * 100
        print(f"{label}: {count} 个样本 ({percentage:.1f}%)")
    
    # 异常值分析
    print("\n" + "="*60)
    print("异常值分析")
    print("="*60)
    
    # 使用IQR方法识别异常值
    Q1 = np.percentile(error, 25)
    Q3 = np.percentile(error, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = (error < lower_bound) | (error > upper_bound)
    outlier_count = np.sum(outliers)
    outlier_percentage = outlier_count / len(error) * 100
    
    print(f"异常值数量: {outlier_count} 个 ({outlier_percentage:.1f}%)")
    print(f"异常值范围: 误差 < {lower_bound:.3f} 或 误差 > {upper_bound:.3f}")
    
    if outlier_count > 0:
        print(f"最大异常误差: {np.max(error[outliers]):.3f}")
        print(f"最小异常误差: {np.min(error[outliers]):.3f}")

if __name__ == "__main__":
    main()
