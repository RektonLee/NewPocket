#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据层诊断脚本：分析图统计特征与 kcat 的相关性

目的：不依赖 GNN 模型，直接回答"数据本身是否包含足够信息"这一核心科学问题。

方法：
1. Feature-Label 统计相关性分析（Pearson/Spearman）
2. 简单模型基准测试（Linear/Ridge）
3. 上界测试（Random Forest，看数据本身的"信息上限"）

输出：
- feature_correlations.csv: 所有特征与 kcat 的相关性排序
- model_baselines.csv: 基准模型性能
- 可视化图表
"""

import torch
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')  # 确保在没有GUI的环境中使用
import matplotlib.pyplot as plt
import seaborn as sns

def extract_graph_stats(data):
    """
    从图数据中提取统计特征
    
    Returns:
        dict: 包含所有统计特征的字典
    """
    features = {}
    
    # 1. 基础几何特征
    pos = data.pos
    if pos.shape[0] > 0:
        centroid = pos.mean(dim=0)
        dists = torch.norm(pos - centroid, dim=1)
        features['geo_radius_max'] = dists.max().item()
        features['geo_radius_mean'] = dists.mean().item()
        features['geo_radius_std'] = dists.std().item()
    else:
        features['geo_radius_max'] = 0
        features['geo_radius_mean'] = 0
        features['geo_radius_std'] = 0
        
    features['num_nodes'] = data.x.shape[0]
    features['num_edges'] = data.edge_index.shape[1]
    features['edge_density'] = features['num_edges'] / (features['num_nodes'] * (features['num_nodes'] - 1) / 2 + 1e-6)

    # 2. 节点特征统计 (Node Features: 52 dim)
    # mean, std, max for each dimension
    x = data.x.float()
    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0)
    x_max = x.max(dim=0)[0]
    
    for i in range(x.shape[1]):
        features[f'node_feat_{i}_mean'] = x_mean[i].item()
        features[f'node_feat_{i}_std'] = x_std[i].item()
        features[f'node_feat_{i}_max'] = x_max[i].item()

    # 3. 边特征统计 (Edge Features: 24 dim)
    # mean, std for each dimension
    if data.edge_attr is not None and data.edge_attr.shape[0] > 0:
        ea = data.edge_attr.float()
        ea_mean = ea.mean(dim=0)
        ea_std = ea.std(dim=0)
        
        for i in range(ea.shape[1]):
            # 给部分关键维度起有意义的名字
            # 0-15: RBF distance features
            # 16-19: Angle features (min, max, mean, count_norm)
            # 20-23: Dihedral features (min, max, mean, count_norm)
            feat_name = f'edge_feat_{i}'
            if 16 <= i <= 19:
                suffix = ['min', 'max', 'mean', 'count'][i-16]
                feat_name = f'angle_{suffix}'
            elif 20 <= i <= 23:
                suffix = ['min', 'max', 'mean', 'count'][i-20]
                feat_name = f'dihedral_{suffix}'
            elif i < 16:
                feat_name = f'rbf_{i}'
                
            features[f'{feat_name}_mean'] = ea_mean[i].item()
            features[f'{feat_name}_std'] = ea_std[i].item()
    else:
        # Handle cases with no edges
        for i in range(24):  # Assuming 24 edge dims
            feat_name = f'edge_feat_{i}'
            features[f'{feat_name}_mean'] = 0.0
            features[f'{feat_name}_std'] = 0.0

    return features

def analyze_dataset(dataset_path, save_dir):
    """
    执行完整的特征-标签相关性分析
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"📂 Loading dataset from {dataset_path}...")
    dataset = torch.load(dataset_path, weights_only=False)
    print(f"✅ Loaded {len(dataset)} samples")
    
    # 提取所有样本的特征
    print("\n📊 Extracting statistical features from graphs...")
    data_list = []
    y_list = []
    
    for data in tqdm(dataset, desc="Processing graphs"):
        feats = extract_graph_stats(data)
        
        # 处理 label: 确保是标量
        if data.y.numel() > 1:
            # 假设第一个是 kcat（兼容旧的双任务数据格式）
            y_val = data.y.view(-1)[0].item()
        else:
            y_val = data.y.item()
            
        feats['y_kcat_log'] = y_val
        data_list.append(feats)
        
    df = pd.DataFrame(data_list)
    
    # 保存原始特征表
    csv_path = os.path.join(save_dir, 'extracted_features.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ Extracted features shape: {df.shape}")
    print(f"   Saved to: {csv_path}")
    
    # === Analysis 1: Correlation Analysis ===
    print("\n" + "="*60)
    print("Analysis 1: Feature-Label Correlations")
    print("="*60)
    corrs = []
    feature_cols = [c for c in df.columns if c != 'y_kcat_log']
    y = df['y_kcat_log']
    
    print(f"Computing correlations for {len(feature_cols)} features...")
    for col in tqdm(feature_cols, desc="Computing correlations"):
        x_col = df[col]
        # Skip constant columns
        if x_col.std() < 1e-6:
            continue
            
        try:
            p_r, p_p = pearsonr(x_col, y)
            s_r, s_p = spearmanr(x_col, y)
            
            corrs.append({
                'feature': col,
                'pearson_r': p_r,
                'pearson_p': p_p,
                'spearman_r': s_r,
                'spearman_p': s_p,
                'abs_pearson': abs(p_r),
                'abs_spearman': abs(s_r)
            })
        except Exception as e:
            print(f"Warning: Failed to compute correlation for {col}: {e}")
            continue
        
    corr_df = pd.DataFrame(corrs).sort_values('abs_pearson', ascending=False)
    corr_path = os.path.join(save_dir, 'feature_correlations.csv')
    corr_df.to_csv(corr_path, index=False)
    print(f"✅ Correlation analysis saved to: {corr_path}")
    
    print("\n📈 Top 10 correlated features (by |Pearson|):")
    print(corr_df[['feature', 'pearson_r', 'spearman_r', 'pearson_p']].head(10).to_string(index=False))
    
    # 统计显著相关的特征数量
    significant_pearson = corr_df[corr_df['pearson_p'] < 0.01]
    strong_corr = corr_df[corr_df['abs_pearson'] >= 0.3]
    print(f"\n📊 Summary:")
    print(f"   - Features with |Pearson| >= 0.3: {len(strong_corr)}")
    print(f"   - Features with p < 0.01: {len(significant_pearson)}")
    
    # Plot top correlations
    top_n = min(5, len(corr_df))
    top_feats = corr_df.head(top_n)['feature'].tolist()
    
    if top_n > 0:
        fig, axes = plt.subplots(top_n, 1, figsize=(10, 3*top_n))
        if top_n == 1:
            axes = [axes]
        
        for i, feat in enumerate(top_feats):
            axes[i].scatter(df[feat], df['y_kcat_log'], alpha=0.3, s=10)
            r_val = corr_df[corr_df['feature'] == feat]['pearson_r'].values[0]
            axes[i].set_xlabel(feat)
            axes[i].set_ylabel('log(kcat)')
            axes[i].set_title(f'{feat} vs log(kcat) (r={r_val:.3f})')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'top_correlations.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"✅ Top correlations plot saved to: {plot_path}")

    # === Analysis 2 & 3: Model Upper Bound ===
    print("\n" + "="*60)
    print("Analysis 2 & 3: Model Baseline (Upper Bound Test)")
    print("="*60)
    
    X = df[feature_cols].values
    # 简单的填充 NaN (如果有)
    X = np.nan_to_num(X)
    y = y.values
    
    # 划分数据集 (Seed 42，与训练脚本保持一致)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
    
    models = {
        'Ridge (Linear)': Ridge(alpha=1.0),
        'RandomForest (Non-linear)': RandomForestRegressor(
            n_estimators=100, 
            max_depth=None, 
            n_jobs=-1, 
            random_state=42,
            verbose=0
        )
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_pearson = pearsonr(y_test, y_test_pred)[0]
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        results.append({
            'Model': name,
            'Train R2': train_r2,
            'Test R2': test_r2,
            'Test Pearson': test_pearson,
            'Test MAE': test_mae
        })
        
        print(f"   Train R²: {train_r2:.4f}")
        print(f"   Test R²: {test_r2:.4f}")
        print(f"   Test Pearson: {test_pearson:.4f}")
        print(f"   Test MAE: {test_mae:.4f}")
        
        # Plot prediction scatter
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_test_pred, alpha=0.5, s=20)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect')
        plt.xlabel('True log(kcat)', fontsize=12)
        plt.ylabel('Predicted log(kcat)', fontsize=12)
        plt.title(f'{name}\nTest R²={test_r2:.3f}, Pearson={test_pearson:.3f}', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_path = os.path.join(save_dir, f'pred_scatter_{name.split()[0].lower()}.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"   ✅ Prediction plot saved to: {plot_path}")

    res_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("📊 Model Baseline Results Summary:")
    print("="*60)
    print(res_df.to_string(index=False))
    
    baseline_path = os.path.join(save_dir, 'model_baselines.csv')
    res_df.to_csv(baseline_path, index=False)
    print(f"\n✅ Baseline results saved to: {baseline_path}")

    # === 生成诊断报告摘要 ===
    print("\n" + "="*60)
    print("📋 DIAGNOSIS SUMMARY")
    print("="*60)
    
    max_corr = corr_df['abs_pearson'].max()
    rf_test_pearson = res_df[res_df['Model'] == 'RandomForest (Non-linear)']['Test Pearson'].values[0]
    rf_train_r2 = res_df[res_df['Model'] == 'RandomForest (Non-linear)']['Train R2'].values[0]
    
    print(f"\n1. Feature-Label Correlation:")
    print(f"   - Maximum |Pearson| correlation: {max_corr:.3f}")
    if max_corr >= 0.3:
        print(f"   ✅ Strong signal detected (|r| >= 0.3)")
    elif max_corr >= 0.1:
        print(f"   ⚠️  Weak signal detected (0.1 <= |r| < 0.3)")
    else:
        print(f"   ❌ Very weak signal (|r| < 0.1) - data representation may be insufficient")
    
    print(f"\n2. Upper Bound Test (Random Forest):")
    print(f"   - Train R²: {rf_train_r2:.3f}")
    print(f"   - Test Pearson: {rf_test_pearson:.3f}")
    if rf_test_pearson >= 0.5:
        print(f"   ✅ Good upper bound - data contains sufficient information")
    elif rf_test_pearson >= 0.3:
        print(f"   ⚠️  Moderate upper bound - data has some signal but may be noisy")
    else:
        print(f"   ❌ Low upper bound - current pocket representation may lack critical information")
    
    print(f"\n✅ Full diagnosis completed. All results saved to: {save_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze feature-label relationships in dataset (data-level diagnosis)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python src/analyze_feature_label_relation.py --dataset data/processed/kcat_full.pt
  
  # Custom output directory
  python src/analyze_feature_label_relation.py --dataset data/processed/kcat_full.pt --save_dir results/diagnosis/run_01
        """
    )
    parser.add_argument('--dataset', type=str, default='data/processed/kcat_full.pt',
                       help='Path to .pt dataset file')
    parser.add_argument('--save_dir', type=str, default='results/diagnosis/feature_analysis',
                       help='Output directory for analysis results')
    args = parser.parse_args()
    
    analyze_dataset(args.dataset, args.save_dir)

