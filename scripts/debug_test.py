#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试测试脚本 - 深入分析模型预测问题
"""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from torch_geometric.loader import DataLoader
import GNN_model as MD
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def main():
    # 加载测试数据
    test_data = torch.load('data/processed/kcat_test_new_backup.pt', weights_only=False)
    print(f"📥 测试集: {len(test_data)} 个样本")
    
    # 检查标签分布
    all_y = torch.stack([d.y for d in test_data])
    print(f"\n📊 标签统计:")
    print(f"  min: {all_y.min():.4f}, max: {all_y.max():.4f}")
    print(f"  mean: {all_y.mean():.4f}, std: {all_y.std():.4f}")
    print(f"  中位数: {all_y.median():.4f}")
    
    # 清理数据（移除非tensor属性）
    print("\n🧹 清理数据对象...")
    for data in test_data:
        keys_to_remove = []
        for key in data.keys():  # keys()是方法，需要调用
            try:
                value = getattr(data, key)
                if not isinstance(value, torch.Tensor):
                    keys_to_remove.append(key)
            except:
                pass
        for key in keys_to_remove:
            try:
                delattr(data, key)
            except:
                pass
    
    # 创建模型（使用正确的配置）
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 初始化模型 (device: {device})...")
    
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
    
    # 加载权重
    model_path = 'outputs/kcat_after_new/best_model.pt'
    print(f"📥 加载模型权重: {model_path}")
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("✅ 模型加载成功")
    
    # 测试前几个样本
    print("\n🔍 测试前10个样本:")
    test_loader = DataLoader(test_data[:10], batch_size=5, shuffle=False)
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = batch.to(device)
            pred = model(batch)
            true = batch.y.reshape(-1, 1)
            
            all_preds.append(pred.cpu())
            all_trues.append(true.cpu())
            
            print(f"\nBatch {batch_idx + 1}:")
            for i in range(len(true)):
                true_val = true[i,0].item()
                pred_val = pred[i,0].item()
                error = abs(pred_val - true_val)
                print(f"  样本 {i+1}: True={true_val:.4f}, Pred={pred_val:.4f}, Error={error:.4f}")
    
    all_preds = torch.cat(all_preds)
    all_trues = torch.cat(all_trues)
    print(f"\n前10个样本统计:")
    print(f"  True range: [{all_trues.min():.2f}, {all_trues.max():.2f}]")
    print(f"  Pred range: [{all_preds.min():.2f}, {all_preds.max():.2f}]")
    print(f"  Mean error: {(all_preds - all_trues).mean():.4f}")
    print(f"  MAE: {(all_preds - all_trues).abs().mean():.4f}")
    
    # 全量测试
    print(f"\n🚀 开始全量推理 ({len(test_data)} 个样本)...")
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = batch.to(device)
            pred = model(batch)
            true = batch.y.reshape(-1, 1)
            
            all_preds.append(pred.cpu())
            all_trues.append(true.cpu())
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  已处理 {batch_idx + 1}/{len(test_loader)} 个批次")
    
    all_preds = torch.cat(all_preds).numpy().flatten()
    all_trues = torch.cat(all_trues).numpy().flatten()
    
    # 计算指标
    mae = mean_absolute_error(all_trues, all_preds)
    rmse = np.sqrt(mean_squared_error(all_trues, all_preds))
    r2 = r2_score(all_trues, all_preds)
    try:
        pearson = pearsonr(all_trues, all_preds)[0]
        if np.isnan(pearson):
            pearson = 0.0
    except:
        pearson = 0.0
    
    print(f"\n📊 完整测试集结果:")
    print(f"  MAE:     {mae:.4f}")
    print(f"  RMSE:    {rmse:.4f}")
    print(f"  R²:      {r2:.4f}")
    print(f"  Pearson: {pearson:.4f}")
    
    print(f"\n📈 数据分布:")
    print(f"  True: mean={all_trues.mean():.4f}, std={all_trues.std():.4f}, range=[{all_trues.min():.2f}, {all_trues.max():.2f}]")
    print(f"  Pred: mean={all_preds.mean():.4f}, std={all_preds.std():.4f}, range=[{all_preds.min():.2f}, {all_preds.max():.2f}]")
    
    # 检查是否有异常值
    print(f"\n🔍 异常值检查:")
    errors = np.abs(all_preds - all_trues)
    large_error = errors > 2.0
    print(f"  误差>2.0的样本数: {large_error.sum()} ({large_error.sum()/len(all_trues)*100:.1f}%)")
    if large_error.sum() > 0:
        print(f"  最大误差: {errors.max():.4f}")
        worst_idx = np.argmax(errors)
        print(f"  最差样本: True={all_trues[worst_idx]:.4f}, Pred={all_preds[worst_idx]:.4f}, Error={errors[worst_idx]:.4f}")
    
    # 检查预测是否集中在某个值
    print(f"\n🔍 预测分布检查:")
    unique_preds = len(np.unique(all_preds))
    print(f"  唯一预测值数量: {unique_preds} / {len(all_preds)}")
    if unique_preds < len(all_preds) * 0.1:
        print(f"  ⚠️  警告: 预测值过于集中，可能模型输出常数")
    
    # 检查预测方差
    pred_std = all_preds.std()
    true_std = all_trues.std()
    print(f"  预测标准差: {pred_std:.4f}")
    print(f"  真实标准差: {true_std:.4f}")
    if pred_std < true_std * 0.1:
        print(f"  ⚠️  警告: 预测方差过小，模型可能输出接近常数")

if __name__ == '__main__':
    main()
