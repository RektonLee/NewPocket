#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建修复后的数据集
解决训练集和测试集数据分布不匹配的问题
"""

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    print("🔍 分析数据分布问题...")
    
    # 加载数据
    train_data = torch.load('kcat_train_after_new_clean.pt', weights_only=False)
    test_data = torch.load('kcat_test_new.pt', weights_only=False)
    
    print(f"训练集: {len(train_data)} 样本")
    print(f"测试集: {len(test_data)} 样本")
    
    # 提取y值
    train_y = torch.cat([data.y for data in train_data]).numpy().flatten()
    test_y = torch.cat([data.y for data in test_data]).numpy().flatten()
    
    print(f"训练集y范围: [{train_y.min():.3f}, {train_y.max():.3f}]")
    print(f"测试集y范围: [{test_y.min():.3f}, {test_y.max():.3f}]")
    
    # 计算重叠度
    overlap_min = max(train_y.min(), test_y.min())
    overlap_max = min(train_y.max(), test_y.max())
    overlap_range = max(0, overlap_max - overlap_min)
    total_range = max(train_y.max(), test_y.max()) - min(train_y.min(), test_y.min())
    overlap_ratio = overlap_range / total_range if total_range > 0 else 0
    
    print(f"重叠比例: {overlap_ratio:.1%}")
    
    if overlap_ratio < 0.3:
        print("❌ 检测到严重的数据分布不匹配问题！")
        print("🔧 开始修复...")
        
        # 合并所有数据
        all_data = train_data + test_data
        all_y = torch.cat([data.y for data in all_data]).numpy().flatten()
        
        print(f"合并后总数据量: {len(all_data)}")
        
        # 创建分层标签（基于y值的分位数）
        n_bins = 10
        y_bins = pd.cut(all_y, bins=n_bins, labels=False)
        
        # 分层划分
        indices = np.arange(len(all_data))
        train_indices, test_indices = train_test_split(
            indices, 
            test_size=0.2, 
            random_state=42,
            stratify=y_bins
        )
        
        # 创建新的训练集和测试集
        new_train_data = [all_data[i] for i in train_indices]
        new_test_data = [all_data[i] for i in test_indices]
        
        # 验证新分布
        new_train_y = torch.cat([data.y for data in new_train_data]).numpy().flatten()
        new_test_y = torch.cat([data.y for data in new_test_data]).numpy().flatten()
        
        print(f"新训练集: {len(new_train_data)} 样本, y范围: [{new_train_y.min():.3f}, {new_train_y.max():.3f}]")
        print(f"新测试集: {len(new_test_data)} 样本, y范围: [{new_test_y.min():.3f}, {new_test_y.max():.3f}]")
        
        # 计算新的重叠度
        overlap_min_new = max(new_train_y.min(), new_test_y.min())
        overlap_max_new = min(new_train_y.max(), new_test_y.max())
        overlap_range_new = max(0, overlap_max_new - overlap_min_new)
        total_range_new = max(new_train_y.max(), new_test_y.max()) - min(new_train_y.min(), new_test_y.min())
        new_overlap_ratio = overlap_range_new / total_range_new if total_range_new > 0 else 0
        
        print(f"新数据重叠度: {new_overlap_ratio:.1%}")
        
        # 保存新数据集
        torch.save(new_train_data, 'kcat_train_fixed.pt')
        torch.save(new_test_data, 'kcat_test_fixed.pt')
        print("✅ 新数据集已保存: kcat_train_fixed.pt, kcat_test_fixed.pt")
        
        # 创建训练脚本
        create_training_script()
        
    else:
        print("✅ 数据分布匹配度较好")

def create_training_script():
    """创建使用修复数据的训练脚本"""
    script_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用修复后数据集的训练脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import GNN_model as MD
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.metadata_utils import update_training_results

def compute_metrics(y_true_log, y_pred_log):
    y_true_log = y_true_log.numpy()
    y_pred_log = y_pred_log.numpy()
    return {
        'MAE': mean_absolute_error(y_true_log, y_pred_log),
        'RMSE': np.sqrt(mean_squared_error(y_true_log, y_pred_log)),
        'R2': r2_score(y_true_log, y_pred_log),
        'Pearson': pearsonr(y_true_log.flatten(), y_pred_log.flatten())[0]
    }

def train_fixed_data(dataset_path, save_dir="outputs", batch_size=32, lr=1e-3, max_epochs=500):
    """使用修复后的数据集进行训练"""
    dataset = torch.load(dataset_path, weights_only=False)
    
    # 检查数据集是否包含 NaN
    for data in dataset:
        if torch.isnan(data.x).any() or torch.isnan(data.y).any():
            raise ValueError("数据集中包含 NaN 值")
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(save_dir)

    # 加载数据集
    data_list = torch.load(dataset_path, weights_only=False)
    print(f"加载了 {len(data_list)} 个样本")
    
    # 分析数据分布
    all_y = torch.cat([data.y for data in data_list]).numpy().flatten()
    print(f"数据y值范围: [{all_y.min():.3f}, {all_y.max():.3f}]")
    print(f"数据y值均值: {all_y.mean():.3f}, 标准差: {all_y.std():.3f}")
    
    # 随机划分训练集和验证集
    np.random.shuffle(data_list)
    split = int(0.8 * len(data_list))
    train_loader = DataLoader(data_list[:split], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data_list[split:], batch_size=batch_size)

    # 初始化模型
    node_input_dim = data_list[0].x.shape[1]
    edge_input_dim = data_list[0].edge_attr.shape[1]
    print(f"Node input dim: {node_input_dim}, Edge input dim: {edge_input_dim}")
    
    model = MD.PocketGNNKcatOnly(
        node_input_dim=node_input_dim,
        edge_input_dim=edge_input_dim,
        hidden_dim=128,
        num_layers=3,
        heads=4,
        dropout=0.1
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')

    # 训练循环
    for epoch in range(1, max_epochs + 1):
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

        # 记录
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("R2/val", metrics['R2'], epoch)
        writer.add_scalar("Pearson/val", metrics['Pearson'], epoch)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | R2: {metrics['R2']:.3f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

    writer.close()
    print("✅ 训练完成！")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="kcat_train_fixed.pt", help='Path to .pt dataset')
    parser.add_argument('--save_dir', type=str, default='outputs/fixed_data_training', help='Output directory')
    args = parser.parse_args()
    
    train_fixed_data(args.dataset, args.save_dir)
'''
    
    with open('train_fixed_data.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("✅ 训练脚本已创建: train_fixed_data.py")

if __name__ == "__main__":
    main()



