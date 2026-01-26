#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 kcat_test_new.csv 上测试 UniKP
需要从 kcat_test_results.csv 获取 SMILES
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# 添加路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR))

from run_unikp import UniKPPredictor, evaluate_predictions
import pickle


def main():
    # 加载数据
    print("📊 加载数据...")
    df_new = pd.read_csv(os.path.join(PROJECT_ROOT, 'data/processed/kcat_test_new.csv'))
    df_results = pd.read_csv(os.path.join(PROJECT_ROOT, 'data/processed/kcat_test_results.csv'))
    
    print(f"   kcat_test_new.csv 样本数: {len(df_new)}")
    print(f"   kcat_test_results.csv 样本数: {len(df_results)}")
    
    # 合并获取 smiles
    df_merged = df_new.merge(
        df_results[['sample_id', 'smiles']], 
        on='sample_id', 
        how='inner'
    )
    
    print(f"   合并后样本数: {len(df_merged)}")
    
    # 清理数据
    df_clean = df_merged.dropna(subset=['sequence', 'smiles', 'log10_kcat']).copy()
    df_clean = df_clean[~df_clean['smiles'].str.contains(r'\.', regex=True, na=False)]
    print(f"   清理后样本数: {len(df_clean)}")
    
    # 准备数据
    sequences = df_clean['sequence'].tolist()
    smiles_list = df_clean['smiles'].tolist()
    y_true = df_clean['log10_kcat'].values  # 已经是 log10 值
    
    print(f"   真实值范围: [{y_true.min():.2f}, {y_true.max():.2f}]")
    
    # 初始化预测器
    print("\n🔬 初始化 UniKP...")
    predictor = UniKPPredictor(device='cuda:1')
    
    # 提取特征
    print("\n📊 提取特征...")
    features = predictor.extract_features(sequences, smiles_list)
    
    # 交叉验证预测
    print("\n🔄 进行交叉验证预测...")
    y_pred = predictor.cross_validate(features, y_true, n_splits=5)
    
    # 评估
    print("\n📈 评估结果 (kcat_test_new):")
    metrics = evaluate_predictions(y_true, y_pred, prefix="unikp_")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(PROJECT_ROOT, "results", f"unikp_test_new_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    results_df = pd.DataFrame({
        'sample_id': df_clean['sample_id'].values,
        'sequence': sequences,
        'smiles': smiles_list,
        'log_kcat_true': y_true,
        'log_kcat_pred': y_pred,
    })
    results_file = os.path.join(save_dir, "unikp_predictions.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\n✅ 预测结果已保存: {results_file}")
    
    # 保存特征
    features_file = os.path.join(save_dir, "unikp_features.pkl")
    with open(features_file, 'wb') as f:
        pickle.dump(features, f)
    print(f"✅ 特征已保存: {features_file}")
    
    # 保存评估指标
    metrics_df = pd.DataFrame([metrics])
    metrics_file = os.path.join(save_dir, "unikp_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print(f"✅ 评估指标已保存: {metrics_file}")
    
    return metrics


if __name__ == "__main__":
    main()







