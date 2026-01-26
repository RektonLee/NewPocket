#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 UniKP 和 DLKcat 在 test_new 数据集上的结果创建散点图
包含 R² 和 Pearson r 指标
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import os
import sys


def plot_scatter(y_true, y_pred, model_name, save_path, title_suffix=""):
    """
    创建散点图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        model_name: 模型名称（用于标题）
        save_path: 保存路径
        title_suffix: 标题后缀
    """
    # 计算指标
    r2 = r2_score(y_true, y_pred)
    pearson_r, _ = pearsonr(y_true, y_pred)
    
    # 创建散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, s=20)
    
    # 添加完美预测线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # 设置标签和标题
    plt.xlabel('True kcat (log10)', fontsize=12)
    plt.ylabel('Predicted kcat (log10)', fontsize=12)
    plt.title(f'{model_name} Test Set Prediction{title_suffix}\nR² = {r2:.3f}, Pearson r = {pearson_r:.3f}', fontsize=13)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ {model_name} 散点图已保存: {save_path}")
    print(f"   R² = {r2:.4f}, Pearson r = {pearson_r:.4f}")
    
    return r2, pearson_r


def main():
    # 项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # UniKP 结果路径
    unikp_pred_path = os.path.join(project_root, 'results/unikp_test_new_20260118_185213/unikp_predictions.csv')
    unikp_save_dir = os.path.join(project_root, 'results/unikp_test_new_20260118_185213')
    
    # DLKcat 结果路径
    dlkcat_pred_path = os.path.join(project_root, 'results/dlkcat_test_new_20260118_173826/dlkcat_predictions.csv')
    dlkcat_save_dir = os.path.join(project_root, 'results/dlkcat_test_new_20260118_173826')
    
    # 检查文件是否存在
    if not os.path.exists(unikp_pred_path):
        print(f"❌ UniKP 预测结果文件不存在: {unikp_pred_path}")
        return
    
    if not os.path.exists(dlkcat_pred_path):
        print(f"❌ DLKcat 预测结果文件不存在: {dlkcat_pred_path}")
        return
    
    # 读取 UniKP 结果
    print("📊 读取 UniKP 预测结果...")
    df_unikp = pd.read_csv(unikp_pred_path)
    # 过滤 NaN 值
    df_unikp = df_unikp.dropna(subset=['log_kcat_true', 'log_kcat_pred'])
    y_true_unikp = df_unikp['log_kcat_true'].values
    y_pred_unikp = df_unikp['log_kcat_pred'].values
    print(f"   样本数: {len(y_true_unikp)}")
    
    # 读取 DLKcat 结果
    print("📊 读取 DLKcat 预测结果...")
    df_dlkcat = pd.read_csv(dlkcat_pred_path)
    # 过滤 NaN 值
    df_dlkcat = df_dlkcat.dropna(subset=['log_kcat_true', 'log_kcat_pred'])
    y_true_dlkcat = df_dlkcat['log_kcat_true'].values
    y_pred_dlkcat = df_dlkcat['log_kcat_pred'].values
    print(f"   样本数: {len(y_true_dlkcat)}")
    
    # 创建 UniKP 散点图
    print("\n📈 创建 UniKP 散点图...")
    unikp_plot_path = os.path.join(unikp_save_dir, 'test_kcat_prediction_scatter.png')
    r2_unikp, pearson_unikp = plot_scatter(
        y_true_unikp, y_pred_unikp, 
        'UniKP', 
        unikp_plot_path,
        title_suffix=" (kcat_test_new)"
    )
    
    # 创建 DLKcat 散点图
    print("\n📈 创建 DLKcat 散点图...")
    dlkcat_plot_path = os.path.join(dlkcat_save_dir, 'test_kcat_prediction_scatter.png')
    r2_dlkcat, pearson_dlkcat = plot_scatter(
        y_true_dlkcat, y_pred_dlkcat, 
        'DLKcat', 
        dlkcat_plot_path,
        title_suffix=" (kcat_test_new)"
    )
    
    # 打印总结
    print("\n" + "="*60)
    print("📊 模型性能总结 (kcat_test_new)")
    print("="*60)
    print(f"UniKP:")
    print(f"  R² = {r2_unikp:.4f}")
    print(f"  Pearson r = {pearson_unikp:.4f}")
    print(f"\nDLKcat:")
    print(f"  R² = {r2_dlkcat:.4f}")
    print(f"  Pearson r = {pearson_dlkcat:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()

