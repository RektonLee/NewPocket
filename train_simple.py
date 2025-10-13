#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版训练脚本 - 用于调试
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_metrics(y_true_log, y_pred_log):
    y_true_log = y_true_log.numpy()
    y_pred_log = y_pred_log.numpy()
    return {
        'MAE': mean_absolute_error(y_true_log, y_pred_log),
        'RMSE': np.sqrt(mean_squared_error(y_true_log, y_pred_log)),
        'R2': r2_score(y_true_log, y_pred_log),
        'Pearson': pearsonr(y_true_log.flatten(), y_pred_log.flatten())[0]
    }

def train_simple(dataset_path, save_dir="outputs_simple", batch_size=16, lr=1e-3, max_epochs=100):
    # 加载数据
    data_list = torch.load(dataset_path, weights_only=False)
    print(f'数据集大小: {len(data_list)}')
    print(f'第一个样本的y值: {data_list[0].y}')
    print(f'y值形状: {data_list[0].y.shape}')
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)

    # 数据分割
    np.random.shuffle(data_list)
    split = int(0.8 * len(data_list))
    train_loader = DataLoader(data_list[:split], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data_list[split:], batch_size=batch_size)

    # 模型参数
    node_input_dim = data_list[0].x.shape[1]
    edge_input_dim = data_list[0].edge_attr.shape[1]
    print(f"Node input dim: {node_input_dim}, Edge input dim: {edge_input_dim}")
    
    # 使用更简单的模型
    model = MD.PocketGNNKcatOnly(
        node_input_dim=node_input_dim, 
        edge_input_dim=edge_input_dim,
        hidden_dim=128,  # 减小隐藏层
        num_layers=3,     # 减少层数
        heads=4,          # 减少注意力头
        dropout=0.1
    ).to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    print("开始训练...")
    for epoch in range(1, max_epochs + 1):
        # 训练
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # 只使用kcat标签
            actual_batch_size = batch.num_graphs
            y_reshaped = batch.y.reshape(actual_batch_size, 2)
            log_y = y_reshaped[:, 0:1]  # 只取kcat列
            
            out = model(batch)
            loss = criterion(out, log_y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        
        # 验证
        model.eval()
        val_losses = []
        y_true_log, y_pred_log = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                actual_batch_size = batch.num_graphs
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
        
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | R2: {metrics['R2']:.3f}")
        
        # 保存模型
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch}.pt"))
    
    print("训练完成!")
    return model

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="kcat_dataset_enhanced1.pt")
    parser.add_argument('--save_dir', type=str, default='outputs_simple')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    args = parser.parse_args()
    
    train_simple(args.dataset, args.save_dir, args.batch_size, args.lr, args.max_epochs)

