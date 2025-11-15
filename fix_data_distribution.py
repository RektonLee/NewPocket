#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据分布修复脚本
解决训练集和测试集数据分布不匹配的问题
"""

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import hashlib

def analyze_current_distribution():
    """分析当前数据分布"""
    print("🔍 分析当前数据分布...")
    
    train_data = torch.load('kcat_train_after_new_clean.pt', weights_only=False)
    test_data = torch.load('kcat_test_new.pt', weights_only=False)
    
    train_y = torch.cat([data.y for data in train_data]).numpy().flatten()
    test_y = torch.cat([data.y for data in test_data]).numpy().flatten()
    
    print(f"训练集: {len(train_data)} 样本, y范围: [{train_y.min():.3f}, {train_y.max():.3f}]")
    print(f"测试集: {len(test_data)} 样本, y范围: [{test_y.min():.3f}, {test_y.max():.3f}]")
    
    # 计算重叠度
    overlap_min = max(train_y.min(), test_y.min())
    overlap_max = min(train_y.max(), test_y.max())
    overlap_range = max(0, overlap_max - overlap_min)
    total_range = max(train_y.max(), test_y.max()) - min(train_y.min(), test_y.min())
    overlap_ratio = overlap_range / total_range if total_range > 0 else 0
    
    print(f"数据重叠度: {overlap_ratio:.1%}")
    
    return train_data, test_data, train_y, test_y, overlap_ratio

def solution1_merge_and_resplit(train_data, test_data):
    """解决方案1: 合并数据集并重新划分"""
    print("\n🔧 解决方案1: 合并数据集并重新划分")
    
    # 合并所有数据
    all_data = train_data + test_data
    print(f"合并后总数据量: {len(all_data)}")
    
    # 提取y值用于分层抽样
    all_y = torch.cat([data.y for data in all_data]).numpy().flatten()
    
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
    overlap_min = max(new_train_y.min(), new_test_y.min())
    overlap_max = min(new_train_y.max(), new_test_y.max())
    overlap_range = max(0, overlap_max - overlap_min)
    total_range = max(new_train_y.max(), new_test_y.max()) - min(new_train_y.min(), new_test_y.min())
    new_overlap_ratio = overlap_range / total_range if total_range > 0 else 0
    
    print(f"新数据重叠度: {new_overlap_ratio:.1%}")
    
    # 保存新数据集
    torch.save(new_train_data, 'kcat_train_fixed.pt')
    torch.save(new_test_data, 'kcat_test_fixed.pt')
    print("✅ 新数据集已保存: kcat_train_fixed.pt, kcat_test_fixed.pt")
    
    return new_train_data, new_test_data

def solution2_filter_by_overlap_range(train_data, test_data):
    """解决方案2: 只保留重叠范围内的数据"""
    print("\n🔧 解决方案2: 过滤重叠范围外的数据")
    
    train_y = torch.cat([data.y for data in train_data]).numpy().flatten()
    test_y = torch.cat([data.y for data in test_data]).numpy().flatten()
    
    # 计算重叠范围
    overlap_min = max(train_y.min(), test_y.min())
    overlap_max = min(train_y.max(), test_y.max())
    
    print(f"重叠范围: [{overlap_min:.3f}, {overlap_max:.3f}]")
    
    # 过滤训练集
    train_filtered = []
    for data in train_data:
        if overlap_min <= data.y.item() <= overlap_max:
            train_filtered.append(data)
    
    # 过滤测试集
    test_filtered = []
    for data in test_data:
        if overlap_min <= data.y.item() <= overlap_max:
            test_filtered.append(data)
    
    print(f"过滤后训练集: {len(train_filtered)} 样本 (原 {len(train_data)})")
    print(f"过滤后测试集: {len(test_filtered)} 样本 (原 {len(test_data)})")
    
    if len(train_filtered) < 100 or len(test_filtered) < 50:
        print("⚠️ 过滤后数据量太少，不建议使用此方案")
        return None, None
    
    # 保存过滤后的数据
    torch.save(train_filtered, 'kcat_train_overlap_only.pt')
    torch.save(test_filtered, 'kcat_test_overlap_only.pt')
    print("✅ 过滤后数据集已保存: kcat_train_overlap_only.pt, kcat_test_overlap_only.pt")
    
    return train_filtered, test_filtered

def solution3_create_balanced_test_set(train_data, test_data):
    """解决方案3: 从训练集中创建平衡的测试集"""
    print("\n🔧 解决方案3: 从训练集创建平衡测试集")
    
    train_y = torch.cat([data.y for data in train_data]).numpy().flatten()
    test_y = torch.cat([data.y for data in test_data]).numpy().flatten()
    
    # 计算测试集的y值范围
    test_min, test_max = test_y.min(), test_y.max()
    print(f"目标测试集范围: [{test_min:.3f}, {test_max:.3f}]")
    
    # 从训练集中选择在测试集范围内的样本
    train_in_test_range = []
    for data in train_data:
        if test_min <= data.y.item() <= test_max:
            train_in_test_range.append(data)
    
    print(f"训练集中在测试范围内的样本: {len(train_in_test_range)}")
    
    if len(train_in_test_range) < 100:
        print("⚠️ 训练集中符合测试范围的样本太少")
        return None, None
    
    # 从这些样本中随机选择作为新的测试集
    np.random.seed(42)
    test_size = min(len(train_in_test_range) // 4, 500)  # 最多500个测试样本
    test_indices = np.random.choice(len(train_in_test_range), test_size, replace=False)
    
    new_test_data = [train_in_test_range[i] for i in test_indices]
    remaining_train_data = [data for i, data in enumerate(train_in_test_range) if i not in test_indices]
    
    # 合并剩余的原始训练集
    train_outside_range = [data for data in train_data if not (test_min <= data.y.item() <= test_max)]
    new_train_data = remaining_train_data + train_outside_range
    
    print(f"新训练集: {len(new_train_data)} 样本")
    print(f"新测试集: {len(new_test_data)} 样本")
    
    # 验证新分布
    new_train_y = torch.cat([data.y for data in new_train_data]).numpy().flatten()
    new_test_y = torch.cat([data.y for data in new_test_data]).numpy().flatten()
    
    print(f"新训练集y范围: [{new_train_y.min():.3f}, {new_train_y.max():.3f}]")
    print(f"新测试集y范围: [{new_test_y.min():.3f}, {new_test_y.max():.3f}]")
    
    # 保存新数据集
    torch.save(new_train_data, 'kcat_train_balanced.pt')
    torch.save(new_test_data, 'kcat_test_balanced.pt')
    print("✅ 平衡数据集已保存: kcat_train_balanced.pt, kcat_test_balanced.pt")
    
    return new_train_data, new_test_data

def create_training_script_for_fixed_data():
    """为修复后的数据创建训练脚本"""
    print("\n📝 创建修复数据的训练脚本...")
    
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

def main():
    """主函数"""
    print("🚀 开始修复数据分布问题...")
    
    # 分析当前分布
    train_data, test_data, train_y, test_y, overlap_ratio = analyze_current_distribution()
    
    if overlap_ratio < 0.3:
        print(f"\n❌ 检测到严重的数据分布不匹配问题 (重叠度: {overlap_ratio:.1%})")
        print("🔧 提供多种解决方案...")
        
        # 解决方案1: 合并重新划分
        print("\n" + "="*50)
        new_train1, new_test1 = solution1_merge_and_resplit(train_data, test_data)
        
        # 解决方案2: 过滤重叠范围
        print("\n" + "="*50)
        new_train2, new_test2 = solution2_filter_by_overlap_range(train_data, test_data)
        
        # 解决方案3: 创建平衡测试集
        print("\n" + "="*50)
        new_train3, new_test3 = solution3_create_balanced_test_set(train_data, test_data)
        
        # 创建训练脚本
        print("\n" + "="*50)
        create_training_script_for_fixed_data()
        
        print("\n✅ 数据修复完成！")
        print("\n📋 可用的修复后数据集:")
        print("  1. kcat_train_fixed.pt / kcat_test_fixed.pt (合并重新划分)")
        if new_train2 is not None:
            print("  2. kcat_train_overlap_only.pt / kcat_test_overlap_only.pt (重叠范围)")
        if new_train3 is not None:
            print("  3. kcat_train_balanced.pt / kcat_test_balanced.pt (平衡测试集)")
        
        print("\n🚀 建议使用方案1的数据集重新训练模型:")
        print("  python train_fixed_data.py --dataset kcat_train_fixed.pt --save_dir outputs/fixed_training")
        
    else:
        print(f"✅ 数据分布匹配度较好 (重叠度: {overlap_ratio:.1%})")
        print("问题可能在其他方面，如特征质量、模型复杂度等")

if __name__ == "__main__":
    main()



