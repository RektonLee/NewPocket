#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 DLKcat 官方测试集上运行 DLKcat 和 UniKP 基准测试
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# 项目路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

# 导入 DLKcat 和 UniKP 的预测器
from run_dlkcat import DLKcatPredictor, evaluate_predictions as dlkcat_evaluate
from run_unikp import UniKPPredictor, evaluate_predictions as unikp_evaluate


def shuffle_dataset(dataset, seed=1234):
    np.random.seed(seed)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    return [dataset[i] for i in indices]


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]


def load_and_split_dlkcat_data():
    """加载 DLKcat 数据集并按官方方式划分"""
    data_path = os.path.join(
        PROJECT_ROOT, 
        'benchmark_tools/DLKcat/DeeplearningApproach/Data/database/Kcat_combination_0918_wildtype_mutant.json'
    )
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # 按 DLKcat 官方方式划分
    dataset = shuffle_dataset(data, seed=1234)
    train_set, rest = split_dataset(dataset, 0.8)
    dev_set, test_set = split_dataset(rest, 0.5)
    
    return train_set, dev_set, test_set


def prepare_dataframe(data_list):
    """将数据列表转换为 DataFrame"""
    records = []
    for item in data_list:
        records.append({
            'sequence': item['Sequence'],
            'substrate_smiles': item['Smiles'],
            'kcat_value': float(item['Value']),
            'type': item.get('Type', 'wildtype'),
        })
    return pd.DataFrame(records)


def plot_scatter(y_true, y_pred, model_name, save_path, metrics):
    """创建散点图"""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, s=15)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('True log10(kcat)', fontsize=12)
    plt.ylabel('Predicted log10(kcat)', fontsize=12)
    plt.title(f'{model_name} on DLKcat Official Test Set\nR² = {metrics["r2"]:.3f}, Pearson r = {metrics["pearson_r"]:.3f}', fontsize=13)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 散点图已保存: {save_path}")


def run_dlkcat_benchmark(test_df, output_dir, device='cuda:1'):
    """运行 DLKcat 基准测试"""
    print("\n" + "=" * 60)
    print("🔬 运行 DLKcat 基准测试")
    print("=" * 60)
    
    # 数据清理
    df_clean = test_df.dropna(subset=['sequence', 'substrate_smiles', 'kcat_value']).copy()
    df_clean['kcat_value'] = pd.to_numeric(df_clean['kcat_value'], errors='coerce')
    df_clean = df_clean.dropna(subset=['kcat_value'])
    df_clean = df_clean[df_clean['kcat_value'] > 0]
    # 过滤多分子 SMILES
    df_clean = df_clean[~df_clean['substrate_smiles'].str.contains(r'\.', regex=True, na=False)]
    
    print(f"   有效样本数: {len(df_clean)}")
    
    # 准备数据
    sequences = df_clean['sequence'].tolist()
    smiles_list = df_clean['substrate_smiles'].tolist()
    y_true = np.log10(df_clean['kcat_value'].values + 1e-10)
    
    # 初始化预测器
    predictor = DLKcatPredictor(device=device)
    
    # 预测
    y_pred = []
    failed_count = 0
    for i, (seq, smiles) in enumerate(zip(sequences, smiles_list)):
        if (i + 1) % 200 == 0:
            print(f"   处理进度: {i+1}/{len(sequences)}")
        try:
            pred = predictor.predict_single(seq, smiles)
            y_pred.append(pred)
            if np.isnan(pred):
                failed_count += 1
        except Exception as e:
            y_pred.append(np.nan)
            failed_count += 1
    
    y_pred = np.array(y_pred)
    print(f"   预测完成，失败: {failed_count}")
    
    # 评估
    mask = ~np.isnan(y_pred)
    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]
    
    metrics = {
        'n_valid': len(y_true_valid),
        'r2': r2_score(y_true_valid, y_pred_valid),
        'pearson_r': pearsonr(y_true_valid, y_pred_valid)[0],
        'spearman_r': spearmanr(y_true_valid, y_pred_valid)[0],
        'rmse': np.sqrt(mean_squared_error(y_true_valid, y_pred_valid)),
        'mae': mean_absolute_error(y_true_valid, y_pred_valid),
    }
    
    print(f"\n📈 DLKcat 评估结果:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")
    
    # 保存结果
    results_df = pd.DataFrame({
        'sequence': sequences,
        'smiles': smiles_list,
        'log_kcat_true': y_true,
        'log_kcat_pred': y_pred,
    })
    results_path = os.path.join(output_dir, 'dlkcat_predictions.csv')
    results_df.to_csv(results_path, index=False)
    
    # 绘图
    plot_path = os.path.join(output_dir, 'dlkcat_scatter.png')
    plot_scatter(y_true_valid, y_pred_valid, 'DLKcat', plot_path, metrics)
    
    # 保存指标
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, 'dlkcat_metrics.csv'), index=False)
    
    return metrics, y_true, y_pred


def run_unikp_benchmark(test_df, output_dir, device='cuda:1'):
    """运行 UniKP 基准测试"""
    print("\n" + "=" * 60)
    print("🔬 运行 UniKP 基准测试")
    print("=" * 60)
    
    # 数据清理
    df_clean = test_df.dropna(subset=['sequence', 'substrate_smiles', 'kcat_value']).copy()
    df_clean['kcat_value'] = pd.to_numeric(df_clean['kcat_value'], errors='coerce')
    df_clean = df_clean.dropna(subset=['kcat_value'])
    df_clean = df_clean[df_clean['kcat_value'] > 0]
    # 过滤多分子 SMILES
    df_clean = df_clean[~df_clean['substrate_smiles'].str.contains(r'\.', regex=True, na=False)]
    
    print(f"   有效样本数: {len(df_clean)}")
    
    # 准备数据
    sequences = df_clean['sequence'].tolist()
    smiles_list = df_clean['substrate_smiles'].tolist()
    y_true = np.log10(df_clean['kcat_value'].values + 1e-10)
    
    # 初始化预测器
    predictor = UniKPPredictor(device=device)
    
    # 提取特征
    features = predictor.extract_features(sequences, smiles_list)
    
    # 交叉验证预测
    y_pred = predictor.cross_validate(features, y_true, n_splits=5)
    
    # 评估
    mask = ~np.isnan(y_pred)
    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]
    
    metrics = {
        'n_valid': len(y_true_valid),
        'r2': r2_score(y_true_valid, y_pred_valid),
        'pearson_r': pearsonr(y_true_valid, y_pred_valid)[0],
        'spearman_r': spearmanr(y_true_valid, y_pred_valid)[0],
        'rmse': np.sqrt(mean_squared_error(y_true_valid, y_pred_valid)),
        'mae': mean_absolute_error(y_true_valid, y_pred_valid),
    }
    
    print(f"\n📈 UniKP 评估结果:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")
    
    # 保存结果
    results_df = pd.DataFrame({
        'sequence': sequences,
        'smiles': smiles_list,
        'log_kcat_true': y_true,
        'log_kcat_pred': y_pred,
    })
    results_path = os.path.join(output_dir, 'unikp_predictions.csv')
    results_df.to_csv(results_path, index=False)
    
    # 绘图
    plot_path = os.path.join(output_dir, 'unikp_scatter.png')
    plot_scatter(y_true_valid, y_pred_valid, 'UniKP', plot_path, metrics)
    
    # 保存指标
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, 'unikp_metrics.csv'), index=False)
    
    return metrics, y_true, y_pred


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--run_dlkcat', action='store_true', help='运行 DLKcat')
    parser.add_argument('--run_unikp', action='store_true', help='运行 UniKP')
    parser.add_argument('--max_samples', type=int, default=None, help='最大样本数')
    args = parser.parse_args()
    
    # 如果两个都没指定，则两个都运行
    if not args.run_dlkcat and not args.run_unikp:
        args.run_dlkcat = True
        args.run_unikp = True
    
    print("=" * 60)
    print("在 DLKcat 官方测试集上进行基准测试")
    print("=" * 60)
    
    # 加载并划分数据
    train_set, dev_set, test_set = load_and_split_dlkcat_data()
    print(f"\n📊 数据集划分:")
    print(f"   Train: {len(train_set)}")
    print(f"   Dev: {len(dev_set)}")
    print(f"   Test: {len(test_set)}")
    
    # 转换为 DataFrame
    test_df = prepare_dataframe(test_set)
    
    # 限制样本数（用于测试）
    if args.max_samples:
        test_df = test_df.head(args.max_samples)
        print(f"\n⚠️ 使用前 {args.max_samples} 个样本进行测试")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(PROJECT_ROOT, 'results', f'dlkcat_official_benchmark_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    all_metrics = {}
    
    # 运行 DLKcat
    if args.run_dlkcat:
        dlkcat_metrics, _, _ = run_dlkcat_benchmark(test_df, output_dir, args.device)
        all_metrics['DLKcat'] = dlkcat_metrics
    
    # 运行 UniKP
    if args.run_unikp:
        unikp_metrics, _, _ = run_unikp_benchmark(test_df, output_dir, args.device)
        all_metrics['UniKP'] = unikp_metrics
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 基准测试总结 (DLKcat 官方测试集)")
    print("=" * 60)
    
    summary_data = []
    for model_name, metrics in all_metrics.items():
        print(f"\n{model_name}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"   {k}: {v:.4f}")
            else:
                print(f"   {k}: {v}")
        summary_data.append({'Model': model_name, **metrics})
    
    # 保存总结
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'benchmark_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ 总结已保存: {summary_path}")
    print(f"✅ 所有结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()

