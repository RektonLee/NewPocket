#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于序列相似度的 KNN 基准测试脚本 (Check KNN Baseline)

目的:
    验证仅通过序列相似度（"查表"或"近邻加权"）能否达到深度学习模型的预测精度。
    如果此 Baseline 效果远低于 PocketGNN，说明 PocketGNN 确实学到了结构信息，而不仅仅是记忆了同源序列。

逻辑:
    1. 读取 MMseqs2 的搜索结果 (test -> train 的相似度映射)。
    2. 对于每个测试样本，找到其在训练集中的所有"近邻" (neighbors)。
    3. 预测值 = 近邻 kcat 的加权平均 (权重 = sequence identity)。
    4. 如果没有近邻，使用训练集全局平均值 (Global Mean) 作为兜底。

使用方法:
    python scripts/check_knn_baseline.py \
        --train_csv data/processed/kcat_full_1213.csv \
        --test_csv data/processed/kcat_test_new.csv \
        --search_result results/leakage_check/search_results_40.tsv \
        --output_dir results/knn_baseline
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def load_kcat_data(csv_path):
    """读取 CSV 加载 kcat 数据"""
    print(f"加载数据: {csv_path}")
    df = pd.read_csv(csv_path)
    # 确保有 sample_id 和 kcat 相关列
    # 训练集可能有 'log10_kcat' 或需要从 kcat 计算
    if 'log10_kcat' not in df.columns:
        if 'kcat' in df.columns:
            df['log10_kcat'] = np.log10(df['kcat'] + 1e-9)
        elif 'kcat_value' in df.columns:
            # 适配 kcat_value 列
            df['log10_kcat'] = np.log10(pd.to_numeric(df['kcat_value'], errors='coerce') + 1e-9)
        elif 'experimental value[log10]' in df.columns:
             # 适配测试集的 log10 列
            df['log10_kcat'] = df['experimental value[log10]']
        else:
            print(f"列名: {df.columns.tolist()}")
            raise ValueError(f"无法在 {csv_path} 中找到 log10_kcat 或 kcat 或 kcat_value 列")
    
    # 移除无效值
    df = df.dropna(subset=['log10_kcat'])
    
    return df.set_index('sample_id')['log10_kcat'].to_dict()

def load_mmseqs_hits(tsv_path):
    """
    读取 MMseqs2 结果
    格式: query_id, target_id, identity, ...
    返回: query_id -> list of (target_id, identity)
    """
    print(f"加载 MMseqs2 搜索结果: {tsv_path}")
    hits = {}
    with open(tsv_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            query_id = parts[0]
            target_id = parts[1]
            try:
                identity = float(parts[2])
            except ValueError:
                continue
            
            if query_id not in hits:
                hits[query_id] = []
            hits[query_id].append((target_id, identity))
    
    print(f"  包含 {len(hits)} 个测试样本的匹配信息")
    return hits

def predict_knn(test_ids, hits_map, train_kcat_map, global_mean, top_k=10):
    """
    对测试集进行 KNN 预测
    """
    predictions = []
    
    # 统计覆盖情况
    hit_counts = []
    
    for q_id in test_ids:
        preds = []
        weights = []
        
        # 1. 获取该样本的 hits
        if q_id in hits_map:
            # 按相似度排序，取 top_k
            sample_hits = sorted(hits_map[q_id], key=lambda x: x[1], reverse=True)[:top_k]
            
            for t_id, identity in sample_hits:
                if t_id in train_kcat_map:
                    preds.append(train_kcat_map[t_id])
                    # 权重可以是 identity 本身，或者 identity 的平方/指数
                    # 这里直接用 identity 作为权重
                    weights.append(identity)
            
        hit_counts.append(len(preds))
        
        # 2. 计算加权平均
        if len(preds) > 0:
            # 加权平均
            weighted_sum = sum(p * w for p, w in zip(preds, weights))
            total_weight = sum(weights)
            y_pred = weighted_sum / total_weight
            status = 'hit'
        else:
            # 兜底：使用全局平均值
            y_pred = global_mean
            status = 'miss'
            
        predictions.append({
            'sample_id': q_id,
            'y_pred_knn': y_pred,
            'status': status,
            'num_neighbors': len(preds),
            'max_identity': max(weights) if weights else 0.0
        })
        
    return pd.DataFrame(predictions)

def evaluate_predictions(df, true_vals_map, output_dir, label):
    """评估预测结果"""
    # 关联真实值
    y_true = []
    y_pred = []
    
    for _, row in df.iterrows():
        sid = row['sample_id']
        if sid in true_vals_map:
            y_true.append(true_vals_map[sid])
            y_pred.append(row['y_pred_knn'])
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) == 0:
        print(f"  {label}: 无有效样本")
        return
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    pearson = pearsonr(y_true, y_pred)[0]
    
    print(f"\n[{label}] 评估结果 (N={len(y_true)}):")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2:   {r2:.4f}")
    print(f"  Pearson r: {pearson:.4f}")
    
    # 绘图
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, s=15)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title(f"{label} KNN Baseline\nR2={r2:.3f}, r={pearson:.3f}")
    plt.xlabel("True log10(kcat)")
    plt.ylabel("KNN Predicted log10(kcat)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"scatter_{label.lower().replace(' ', '_')}.png"))
    plt.close()
    
    return {
        'rmse': rmse,
        'r2': r2,
        'pearson': pearson,
        'n': len(y_true)
    }

def main():
    parser = argparse.ArgumentParser(description="KNN Baseline for kcat prediction")
    parser.add_argument("--train_csv", required=True, help="训练集 CSV")
    parser.add_argument("--test_csv", required=True, help="测试集 CSV")
    parser.add_argument("--search_result", required=True, help="MMseqs2 搜索结果 TSV")
    parser.add_argument("--output_dir", default="results/knn_baseline")
    parser.add_argument("--top_k", type=int, default=5, help="取前 K 个近邻 (默认: 5)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 加载数据
    train_kcat = load_kcat_data(args.train_csv)
    test_kcat = load_kcat_data(args.test_csv)
    
    # 计算训练集均值 (兜底用)
    train_vals = list(train_kcat.values())
    global_mean = sum(train_vals) / len(train_vals)
    print(f"训练集全局均值: {global_mean:.4f}")
    
    # 2. 加载 Hits
    hits_map = load_mmseqs_hits(args.search_result)
    
    # 3. 预测
    test_ids = list(test_kcat.keys())
    df_pred = predict_knn(test_ids, hits_map, train_kcat, global_mean, top_k=args.top_k)
    
    # 保存预测结果
    df_pred['y_true'] = df_pred['sample_id'].map(test_kcat)
    df_pred.to_csv(os.path.join(args.output_dir, "knn_predictions.csv"), index=False)
    
    # 4. 整体评估
    print("\n=== 整体评估 (All Test Samples) ===")
    evaluate_predictions(df_pred, test_kcat, args.output_dir, "Overall")
    
    # 5. 分组评估：Leaked (Hit) vs Safe (Miss)
    print("\n=== 分组评估: Leaked Samples (With Homologs) ===")
    df_leaked = df_pred[df_pred['status'] == 'hit']
    evaluate_predictions(df_leaked, test_kcat, args.output_dir, "Leaked_Only")
    
    print("\n=== 分组评估: Safe Samples (No Homologs) ===")
    df_safe = df_pred[df_pred['status'] == 'miss']
    evaluate_predictions(df_safe, test_kcat, args.output_dir, "Safe_Only")
    
    # 6. 对比分析
    print("\n" + "="*50)
    print("结论分析:")
    print("如果 KNN Baseline 的性能 (特别是 Overall) 远低于 PocketGNN (R2 > 0.9)，")
    print("则说明 PocketGNN 成功利用了结构信息进行了超越同源推断的预测。")
    print("如果 KNN Baseline 也很高，则说明数据泄露确实导致了任务变得简单。")
    print("="*50)

if __name__ == "__main__":
    main()

