#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超参数搜索脚本
通过网格搜索找到最佳的超参数组合
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from torch_geometric.loader import DataLoader
import GNN_model as MD
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import itertools
import json
from datetime import datetime
import argparse

def compute_metrics(y_true_log, y_pred_log):
    y_true_log = y_true_log.numpy()
    y_pred_log = y_pred_log.numpy()
    return {
        'MAE': mean_absolute_error(y_true_log, y_pred_log),
        'RMSE': np.sqrt(mean_squared_error(y_true_log, y_pred_log)),
        'R2': r2_score(y_true_log, y_pred_log),
        'Pearson': pearsonr(y_true_log.flatten(), y_pred_log.flatten())[0]
    }

def train_single_config(dataset_path, config, max_epochs=100, patience=20):
    """训练单个配置"""
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    data_list = torch.load(dataset_path, weights_only=False)
    np.random.shuffle(data_list)
    split = int(0.8 * len(data_list))
    train_loader = DataLoader(data_list[:split], batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(data_list[split:], batch_size=config['batch_size'])
    
    # 初始化模型
    node_input_dim = data_list[0].x.shape[1]
    edge_input_dim = data_list[0].edge_attr.shape[1]
    
    model = MD.PocketGNNKcatOnly(
        node_input_dim=node_input_dim,
        edge_input_dim=edge_input_dim,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        heads=config['heads'],
        dropout=config['dropout']
    ).to(device)
    
    # 优化器
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    else:  # SGD
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], momentum=0.9)
    
    # 损失函数
    if config['loss'] == 'MSE':
        criterion = nn.MSELoss()
    elif config['loss'] == 'Huber':
        criterion = nn.HuberLoss()
    else:  # SmoothL1
        criterion = nn.SmoothL1Loss()
    
    # 学习率调度器
    if config['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    elif config['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    elif config['scheduler'] == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    else:
        scheduler = None
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_r2 = -float('inf')
    best_pearson = -float('inf')
    
    for epoch in range(max_epochs):
        # 训练
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            actual_batch_size = batch.num_graphs
            if batch.y.shape[0] == actual_batch_size:
                log_y = batch.y.reshape(actual_batch_size, 1)
            else:
                y_reshaped = batch.y.reshape(actual_batch_size, 2)
                log_y = y_reshaped[:, 0:1]
            
            out = model(batch)
            loss = criterion(out, log_y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['gradient_clip'])
            optimizer.step()
            train_losses.append(loss.item())
        
        # 验证
        model.eval()
        val_losses = []
        y_true_log, y_pred_log = [], []
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
                loss = criterion(out, log_y)
                val_losses.append(loss.item())
                y_true_log.append(log_y.cpu())
                y_pred_log.append(out.cpu())
        
        val_loss = np.mean(val_losses)
        y_true_log = torch.cat(y_true_log, dim=0)
        y_pred_log = torch.cat(y_pred_log, dim=0)
        metrics = compute_metrics(y_true_log, y_pred_log)
        
        # 学习率调度
        if config['scheduler'] == 'plateau':
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_r2 = metrics['R2']
            best_pearson = metrics['Pearson']
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
    
    return {
        'best_val_loss': best_val_loss,
        'best_r2': best_r2,
        'best_pearson': best_pearson,
        'epochs_trained': epoch + 1
    }

def grid_search(dataset_path, save_dir="outputs/hyperparameter_search"):
    """网格搜索最佳超参数"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 定义搜索空间
    param_grid = {
        'hidden_dim': [128, 256, 512],
        'num_layers': [3, 4, 5],
        'heads': [4, 8, 16],
        'dropout': [0.1, 0.2, 0.3],
        'lr': [1e-4, 5e-4, 1e-3, 2e-3],
        'batch_size': [16, 32, 64],
        'weight_decay': [1e-5, 1e-4, 1e-3],
        'optimizer': ['Adam', 'AdamW', 'SGD'],
        'loss': ['MSE', 'Huber', 'SmoothL1'],
        'scheduler': ['cosine', 'step', 'plateau', 'none'],
        'gradient_clip': [0.5, 1.0, 2.0]
    }
    
    # 生成所有组合（限制数量）
    keys, values = zip(*param_grid.items())
    all_combinations = list(itertools.product(*values))
    
    # 随机采样减少搜索空间
    if len(all_combinations) > 50:
        np.random.seed(42)
        selected_indices = np.random.choice(len(all_combinations), 50, replace=False)
        all_combinations = [all_combinations[i] for i in selected_indices]
    
    print(f"🔍 开始网格搜索，共 {len(all_combinations)} 个配置")
    
    results = []
    best_score = -float('inf')
    best_config = None
    
    for i, combination in enumerate(all_combinations):
        config = dict(zip(keys, combination))
        print(f"\n📊 配置 {i+1}/{len(all_combinations)}: {config}")
        
        try:
            result = train_single_config(dataset_path, config)
            result['config'] = config
            results.append(result)
            
            # 使用R²作为主要评估指标
            score = result['best_r2']
            if score > best_score:
                best_score = score
                best_config = config
                print(f"✅ 新的最佳配置！R² = {score:.4f}")
            else:
                print(f"📈 当前结果: R² = {score:.4f}")
                
        except Exception as e:
            print(f"❌ 配置失败: {e}")
            continue
    
    # 保存结果
    results_file = os.path.join(save_dir, f"hyperparameter_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 排序结果
    results.sort(key=lambda x: x['best_r2'], reverse=True)
    
    print(f"\n🏆 最佳配置:")
    print(f"R²: {best_score:.4f}")
    print(f"配置: {best_config}")
    
    # 保存最佳配置
    best_config_file = os.path.join(save_dir, "best_config.json")
    with open(best_config_file, 'w') as f:
        json.dump({
            'best_config': best_config,
            'best_score': best_score,
            'all_results': results[:10]  # 保存前10个结果
        }, f, indent=2)
    
    # 创建结果摘要
    create_results_summary(results, save_dir)
    
    return best_config, results

def create_results_summary(results, save_dir):
    """创建结果摘要"""
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # 转换为DataFrame
    df_results = []
    for result in results:
        row = result['config'].copy()
        row.update({
            'best_r2': result['best_r2'],
            'best_pearson': result['best_pearson'],
            'best_val_loss': result['best_val_loss'],
            'epochs_trained': result['epochs_trained']
        })
        df_results.append(row)
    
    df = pd.DataFrame(df_results)
    df = df.sort_values('best_r2', ascending=False)
    
    # 保存CSV
    df.to_csv(os.path.join(save_dir, 'hyperparameter_search_results.csv'), index=False)
    
    # 创建可视化
    plt.figure(figsize=(15, 10))
    
    # R²分布
    plt.subplot(2, 3, 1)
    plt.hist(df['best_r2'], bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Best R²')
    plt.ylabel('Frequency')
    plt.title('Distribution of R² Scores')
    plt.grid(True, alpha=0.3)
    
    # 学习率 vs R²
    plt.subplot(2, 3, 2)
    plt.scatter(df['lr'], df['best_r2'], alpha=0.6)
    plt.xlabel('Learning Rate')
    plt.ylabel('Best R²')
    plt.title('Learning Rate vs R²')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    # 隐藏维度 vs R²
    plt.subplot(2, 3, 3)
    plt.scatter(df['hidden_dim'], df['best_r2'], alpha=0.6)
    plt.xlabel('Hidden Dimension')
    plt.ylabel('Best R²')
    plt.title('Hidden Dimension vs R²')
    plt.grid(True, alpha=0.3)
    
    # Dropout vs R²
    plt.subplot(2, 3, 4)
    plt.scatter(df['dropout'], df['best_r2'], alpha=0.6)
    plt.xlabel('Dropout')
    plt.ylabel('Best R²')
    plt.title('Dropout vs R²')
    plt.grid(True, alpha=0.3)
    
    # 优化器 vs R²
    plt.subplot(2, 3, 5)
    df.boxplot(column='best_r2', by='optimizer', ax=plt.gca())
    plt.title('Optimizer vs R²')
    plt.suptitle('')  # 移除自动标题
    
    # 损失函数 vs R²
    plt.subplot(2, 3, 6)
    df.boxplot(column='best_r2', by='loss', ax=plt.gca())
    plt.title('Loss Function vs R²')
    plt.suptitle('')  # 移除自动标题
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'hyperparameter_search_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 结果已保存到 {save_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter search for GNN model')
    parser.add_argument('--dataset', type=str, default="kcat_train_after_new_clean.pt", help='Path to .pt dataset')
    parser.add_argument('--save_dir', type=str, default='outputs/hyperparameter_search', help='Output directory')
    args = parser.parse_args()
    
    print("🚀 开始超参数搜索...")
    best_config, results = grid_search(args.dataset, args.save_dir)
    
    print(f"\n🎯 搜索完成！最佳配置已保存到 {args.save_dir}")
    print("💡 建议使用最佳配置重新训练模型")



