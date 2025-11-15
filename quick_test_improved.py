#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试改进训练效果
"""

import torch
import numpy as np
from torch_geometric.loader import DataLoader
import GNN_model as MD
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
import os
import argparse

def test_model_performance(dataset_path, model_path, model_config=None):
    """测试模型性能"""
    print(f"🔍 测试数据集: {dataset_path}")
    print(f"🤖 模型路径: {model_path}")
    
    # 加载数据
    data_list = torch.load(dataset_path, weights_only=False)
    print(f"📊 加载了 {len(data_list)} 个样本")
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    val_loader = DataLoader(data_list, batch_size=32)
    
    # 加载模型
    node_input_dim = data_list[0].x.shape[1]
    edge_input_dim = data_list[0].edge_attr.shape[1]
    
    # 使用默认配置或提供的配置
    if model_config is None:
        model_config = {
            'hidden_dim': 128,
            'num_layers': 3,
            'heads': 4,
            'dropout': 0.1
        }
    
    model = MD.PocketGNNKcatOnly(
        node_input_dim=node_input_dim,
        edge_input_dim=edge_input_dim,
        **model_config
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 推理
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            actual_batch_size = batch.num_graphs
            
            if batch.y.shape[0] == actual_batch_size:
                log_y = batch.y.reshape(actual_batch_size, 1)
            else:
                y_reshaped = batch.y.reshape(actual_batch_size, 2)
                log_y = y_reshaped[:, 0:1]
            
            out = model(batch)
            all_y_true.append(log_y.cpu())
            all_y_pred.append(out.cpu())
    
    all_y_true = torch.cat(all_y_true, dim=0).numpy().flatten()
    all_y_pred = torch.cat(all_y_pred, dim=0).numpy().flatten()
    
    # 计算指标
    r2 = r2_score(all_y_true, all_y_pred)
    mae = mean_absolute_error(all_y_true, all_y_pred)
    pearson = pearsonr(all_y_true, all_y_pred)[0]
    
    print(f"📈 性能指标:")
    print(f"  R²: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Pearson: {pearson:.4f}")
    
    return r2, mae, pearson

def compare_models():
    """比较不同模型的性能"""
    print("🔍 比较不同模型的性能...")
    
    # 原始模型
    print("\n" + "="*50)
    print("📊 原始模型性能:")
    try:
        r2_orig, mae_orig, pearson_orig = test_model_performance(
            "kcat_test_new.pt", 
            "outputs/kcat_after_new/best_model.pt"
        )
    except Exception as e:
        print(f"❌ 原始模型测试失败: {e}")
        r2_orig, mae_orig, pearson_orig = 0, 0, 0
    
    # 改进模型（如果存在）
    print("\n" + "="*50)
    print("📊 改进模型性能:")
    try:
        r2_improved, mae_improved, pearson_improved = test_model_performance(
            "kcat_test_new.pt", 
            "outputs/improved_training/best_model.pt",
            model_config={'hidden_dim': 256, 'num_layers': 4, 'heads': 8, 'dropout': 0.2}
        )
    except Exception as e:
        print(f"❌ 改进模型测试失败: {e}")
        r2_improved, mae_improved, pearson_improved = 0, 0, 0
    
    # 高级模型（如果存在）
    print("\n" + "="*50)
    print("📊 高级模型性能:")
    try:
        r2_advanced, mae_advanced, pearson_advanced = test_model_performance(
            "kcat_test_new.pt", 
            "outputs/advanced_training/best_model.pt",
            model_config={'hidden_dim': 256, 'num_layers': 4, 'heads': 8, 'dropout': 0.2}
        )
    except Exception as e:
        print(f"❌ 高级模型测试失败: {e}")
        r2_advanced, mae_advanced, pearson_advanced = 0, 0, 0
    
    # 总结对比
    print("\n" + "="*50)
    print("📊 性能对比总结:")
    print(f"{'模型':<15} {'R²':<8} {'MAE':<8} {'Pearson':<8}")
    print("-" * 50)
    print(f"{'原始模型':<15} {r2_orig:<8.4f} {mae_orig:<8.4f} {pearson_orig:<8.4f}")
    print(f"{'改进模型':<15} {r2_improved:<8.4f} {mae_improved:<8.4f} {pearson_improved:<8.4f}")
    print(f"{'高级模型':<15} {r2_advanced:<8.4f} {mae_advanced:<8.4f} {pearson_advanced:<8.4f}")
    
    # 计算改进幅度
    if r2_orig > 0:
        print(f"\n📈 改进幅度:")
        if r2_improved > 0:
            print(f"  改进模型 R² 提升: {((r2_improved - r2_orig) / r2_orig * 100):+.1f}%")
        if r2_advanced > 0:
            print(f"  高级模型 R² 提升: {((r2_advanced - r2_orig) / r2_orig * 100):+.1f}%")

def quick_improvement_test():
    """快速改进测试"""
    print("🚀 开始快速改进测试...")
    
    # 检查是否存在改进的模型
    improved_model_path = "outputs/improved_training/best_model.pt"
    advanced_model_path = "outputs/advanced_training/best_model.pt"
    
    if os.path.exists(improved_model_path):
        print("✅ 找到改进模型，开始测试...")
        test_model_performance("kcat_test_new.pt", improved_model_path)
    elif os.path.exists(advanced_model_path):
        print("✅ 找到高级模型，开始测试...")
        test_model_performance("kcat_test_new.pt", advanced_model_path)
    else:
        print("❌ 未找到改进的模型，请先运行训练脚本")
        print("💡 建议运行:")
        print("   python train_improved.py --dataset kcat_train_after_new_clean.pt --save_dir outputs/improved_training --use_mixup --use_label_smoothing")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quick test of improved models')
    parser.add_argument('--compare', action='store_true', help='Compare all available models')
    parser.add_argument('--quick', action='store_true', help='Quick improvement test')
    args = parser.parse_args()
    
    if args.compare:
        compare_models()
    elif args.quick:
        quick_improvement_test()
    else:
        print("🔍 选择测试模式:")
        print("  --compare: 比较所有可用模型")
        print("  --quick: 快速改进测试")
        print("\n💡 示例:")
        print("  python quick_test_improved.py --compare")
        print("  python quick_test_improved.py --quick")



