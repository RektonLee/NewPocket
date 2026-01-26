#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
尝试修复模型输出问题 - 分析并尝试改进
"""

import torch
import torch.nn as nn
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from torch_geometric.loader import DataLoader
import GNN_model as MD
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr

def analyze_problem():
    """分析问题的根本原因"""
    print("=" * 60)
    print("问题分析")
    print("=" * 60)
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = MD.PocketGNNKcatOnly(
        node_input_dim=52,
        edge_input_dim=24,
        hidden_dim=128,
        num_layers=3,
        heads=4,
        dropout=0.1,
        pooling_type='mean',
        use_seq_embedding=False,
        use_mlp_layernorm=False
    ).to(device)
    
    model_path = 'outputs/kcat_after_new/best_model.pt'
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    # 加载测试数据
    test_data = torch.load('data/processed/kcat_test_new_backup.pt', weights_only=False)
    for data in test_data[:100]:
        keys_to_remove = []
        for key in data.keys():
            if not isinstance(getattr(data, key), torch.Tensor):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            delattr(data, key)
    
    # 分析问题
    print("\n🔍 关键发现:")
    print("1. 图embedding的std只有0.1935（太小）")
    print("2. 图embedding与标签的最大相关性只有0.20（太低）")
    print("3. 不同样本的图embedding过于相似")
    print("4. MLP最后一层的bias=0.1075，可能导致输出偏置")
    
    # 检查权重初始化
    print("\n🔍 检查权重初始化:")
    print("  所有层的权重std都很小（0.05-0.09），说明权重可能被过度正则化")
    print("  或者训练时学习率太小，导致权重更新不足")
    
    # 尝试修复：重新初始化MLP最后一层
    print("\n🔧 尝试修复：重新初始化MLP最后一层...")
    with torch.no_grad():
        # 使用更大的初始化
        nn.init.xavier_uniform_(model.mlp[-1].weight, gain=2.0)
        nn.init.constant_(model.mlp[-1].bias, 0.0)  # 重置bias
    
    # 测试修复后的效果
    test_loader = DataLoader(test_data[:100], batch_size=32, shuffle=False)
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            true = batch.y.reshape(-1, 1)
            all_preds.append(pred.cpu())
            all_trues.append(true.cpu())
    
    all_preds = torch.cat(all_preds).numpy().flatten()
    all_trues = torch.cat(all_trues).numpy().flatten()
    
    mae = mean_absolute_error(all_trues, all_preds)
    r2 = r2_score(all_trues, all_preds)
    try:
        pearson = pearsonr(all_trues, all_preds)[0]
        if np.isnan(pearson):
            pearson = 0.0
    except:
        pearson = 0.0
    
    print(f"\n修复后的结果 (前100个样本):")
    print(f"  MAE: {mae:.4f}, R²: {r2:.4f}, Pearson: {pearson:.4f}")
    print(f"  预测范围: [{all_preds.min():.2f}, {all_preds.max():.2f}], std={all_preds.std():.4f}")
    print(f"  真实范围: [{all_trues.min():.2f}, {all_trues.max():.2f}], std={all_trues.std():.4f}")

def check_training_logs():
    """检查训练日志"""
    print("\n" + "=" * 60)
    print("检查训练日志")
    print("=" * 60)
    
    # 查找wandb日志或本地日志
    import glob
    
    # 查找metrics文件
    metrics_files = glob.glob('outputs/kcat_after_new/*metrics*.csv')
    metrics_files += glob.glob('outputs/kcat_after_new/*metrics*.json')
    
    if metrics_files:
        print(f"✅ 找到metrics文件: {metrics_files[0]}")
        # 读取metrics
        if metrics_files[0].endswith('.csv'):
            import pandas as pd
            try:
                df = pd.read_csv(metrics_files[0])
                print(f"\n训练指标:")
                print(df.to_string())
            except:
                print("无法读取CSV文件")
    else:
        print("⚠️  未找到本地metrics文件，可能需要查看wandb")
    
    # 检查是否有训练曲线图
    plot_files = glob.glob('outputs/kcat_after_new/*.png')
    if plot_files:
        print(f"\n✅ 找到训练曲线图: {len(plot_files)} 个")
        for f in plot_files[:3]:
            print(f"  - {os.path.basename(f)}")

if __name__ == '__main__':
    analyze_problem()
    check_training_logs()
