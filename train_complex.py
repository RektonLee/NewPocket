#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复杂版本的训练脚本 - 增强表达能力，参考AM-GCN思路
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

def compute_metrics(y_true_log, y_pred_log):
    y_true_log = y_true_log.numpy()
    y_pred_log = y_pred_log.numpy()
    return {
        'MAE': mean_absolute_error(y_true_log, y_pred_log),
        'RMSE': np.sqrt(mean_squared_error(y_true_log, y_pred_log)),
        'R2': r2_score(y_true_log, y_pred_log),
        'Pearson': pearsonr(y_true_log.flatten(), y_pred_log.flatten())[0]
    }

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=30, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

def train_complex(dataset_path, save_dir="outputs", batch_size=32, lr=1e-3, max_epochs=400):
    """复杂版本的训练函数 - 增强表达能力"""
    dataset = torch.load(dataset_path, weights_only=False)
    
    # 检查数据集是否包含 NaN
    for data in dataset:
        if torch.isnan(data.x).any() or torch.isnan(data.y).any():
            raise ValueError("数据集中包含 NaN 值")
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(save_dir)

    # === Load dataset ===
    data_list = torch.load(dataset_path, weights_only=False)
    print(f"数据集大小: {len(data_list)}")
    print(f"第一个样本: {data_list[0]}")
    
    actual_num_atom_types = data_list[0].x.shape[1]
    np.random.shuffle(data_list)
    split = int(0.8 * len(data_list))
    train_loader = DataLoader(data_list[:split], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data_list[split:], batch_size=batch_size)

    # === Initialize complex model ===
    node_input_dim = data_list[0].x.shape[1]
    edge_input_dim = data_list[0].edge_attr.shape[1]
    print(f"Node input dim: {node_input_dim}, Edge input dim: {edge_input_dim}")
    
    # 复杂模型参数 - 增强表达能力
    model = MD.PocketGNNKcatOnly(
        node_input_dim=node_input_dim, 
        edge_input_dim=edge_input_dim,
        hidden_dim=120,       # 120能被heads整除
        num_layers=4,         # 增加到4层
        heads=6,              # 6个注意力头，每个头20维
        dropout=0.15          # 适中的dropout
    ).to(device)
    
    # 优化器添加轻微权重衰减
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )
    
    # 早停机制 - 更宽松
    early_stopping = EarlyStopping(patience=30, min_delta=0.001)
    
    best_val_loss = float('inf')
    best_epoch = 0
    best_r2 = -float('inf')

    # 添加损失记录列表
    train_loss_history = []
    val_loss_history = []
    r2_history = []
    pearson_history = []
    lr_history = []
    
    print("开始训练复杂模型...")
    print(f"模型参数: hidden_dim=120, num_layers=4, heads=6, dropout=0.15")
    print(f"优化器: Adam with weight_decay=1e-5")
    print(f"学习率调度: ReduceLROnPlateau")
    print(f"早停: patience=30, min_delta=0.001")
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # 只使用kcat标签（第一列）
            actual_batch_size = batch.num_graphs
            y_reshaped = batch.y.reshape(actual_batch_size, 2)
            log_y = y_reshaped[:, 0:1]  # 只取kcat列
            
            out = model(batch)
            loss = criterion(out, log_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)

        # === Validation ===
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
        
        # 学习率调度
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        r2_history.append(metrics['R2'])
        pearson_history.append(metrics['Pearson'])
        lr_history.append(current_lr)
        
        # === Logging ===
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("R2/val", metrics['R2'], epoch)
        writer.add_scalar("Pearson/val", metrics['Pearson'], epoch)
        writer.add_scalar("Learning_Rate", current_lr, epoch)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | R2: {metrics['R2']:.3f} | Pearson: {metrics['Pearson']:.3f} | LR: {current_lr:.2e}")

        # === Save best model (based on R2) ===
        if metrics['R2'] > best_r2:
            best_r2 = metrics['R2']
            best_epoch = epoch
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

        # === Early stopping ===
        if early_stopping(val_loss):
            print(f"早停触发！在第 {epoch} 轮停止训练")
            print(f"最佳验证损失: {best_val_loss:.4f} (第 {best_epoch} 轮)")
            print(f"最佳R²: {best_r2:.3f}")
            break

    writer.close()
    print(f"✅ 训练完成！最佳模型保存在第 {best_epoch} 轮，R²: {best_r2:.3f}")

    # 绘制训练曲线
    actual_epochs = len(train_loss_history)
    plt.figure(figsize=(18, 12))
    
    # 损失曲线
    plt.subplot(3, 3, 1)
    plt.plot(range(1, actual_epochs + 1), train_loss_history, label='Train Loss', color='blue')
    plt.plot(range(1, actual_epochs + 1), val_loss_history, label='Validation Loss', color='orange')
    plt.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (Complex)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # R2和Pearson曲线
    plt.subplot(3, 3, 2)
    plt.plot(range(1, actual_epochs + 1), r2_history, label='R² Score', color='green')
    plt.plot(range(1, actual_epochs + 1), pearson_history, label='Pearson Correlation', color='purple')
    plt.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('R² and Pearson Correlation (Complex)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 学习率变化
    plt.subplot(3, 3, 3)
    plt.plot(range(1, actual_epochs + 1), lr_history, color='brown', alpha=0.7)
    plt.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 预测vs真实值散点图
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt")))
    model.eval()
    
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            batch_size = batch.num_graphs
            
            y_reshaped = batch.y.reshape(batch_size, 2)
            log_y = y_reshaped[:, 0:1]
            
            out = model(batch)
            all_y_true.append(log_y.cpu())
            all_y_pred.append(out.cpu())
    
    all_y_true = torch.cat(all_y_true, dim=0).numpy()
    all_y_pred = torch.cat(all_y_pred, dim=0).numpy()
    
    plt.subplot(3, 3, 4)
    plt.scatter(all_y_true.flatten(), all_y_pred.flatten(), alpha=0.6, color='blue')
    plt.plot([all_y_true.min(), all_y_true.max()], 
             [all_y_true.min(), all_y_true.max()], 'r--', alpha=0.8)
    plt.xlabel('True kcat (log10)')
    plt.ylabel('Predicted kcat (log10)')
    r2_kcat = r2_score(all_y_true.flatten(), all_y_pred.flatten())
    plt.title(f'kcat: True vs Predicted (R² = {r2_kcat:.3f})')
    plt.grid(True, alpha=0.3)
    
    # 残差图
    plt.subplot(3, 3, 5)
    residuals = all_y_true.flatten() - all_y_pred.flatten()
    plt.scatter(all_y_pred.flatten(), residuals, alpha=0.6, color='red')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    plt.xlabel('Predicted kcat (log10)')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    # 损失差距分析
    plt.subplot(3, 3, 6)
    loss_gap = [val_loss_history[i] - train_loss_history[i] for i in range(len(train_loss_history))]
    plt.plot(range(1, actual_epochs + 1), loss_gap, color='red', alpha=0.7)
    plt.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Val Loss - Train Loss')
    plt.title('Overfitting Gap Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 学习曲线
    plt.subplot(3, 3, 7)
    plt.plot(range(1, actual_epochs + 1), train_loss_history, label='Train Loss', color='blue', alpha=0.7)
    plt.plot(range(1, actual_epochs + 1), val_loss_history, label='Val Loss', color='orange', alpha=0.7)
    plt.fill_between(range(1, actual_epochs + 1), train_loss_history, val_loss_history, alpha=0.2, color='gray')
    plt.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve with Gap')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 性能指标趋势
    plt.subplot(3, 3, 8)
    plt.plot(range(1, actual_epochs + 1), r2_history, label='R²', color='green', alpha=0.7)
    plt.plot(range(1, actual_epochs + 1), pearson_history, label='Pearson', color='purple', alpha=0.7)
    plt.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Performance Metrics Trend')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 模型复杂度对比
    plt.subplot(3, 3, 9)
    models = ['Original', 'Simplified', 'Balanced', 'Complex']
    hidden_dims = [128, 64, 96, 120]
    layers = [3, 2, 3, 4]
    heads = [4, 2, 3, 6]
    
    x = np.arange(len(models))
    width = 0.25
    
    plt.bar(x - width, hidden_dims, width, label='Hidden Dim', alpha=0.7)
    plt.bar(x, [l*20 for l in layers], width, label='Layers (×20)', alpha=0.7)
    plt.bar(x + width, heads, width, label='Heads', alpha=0.7)
    
    plt.xlabel('Model Version')
    plt.ylabel('Parameter Value')
    plt.title('Model Complexity Comparison')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'complex_training_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存训练指标
    metrics_df = pd.DataFrame({
        'Epoch': range(1, actual_epochs + 1),
        'Train_Loss': train_loss_history,
        'Val_Loss': val_loss_history,
        'R2': r2_history,
        'Pearson': pearson_history,
        'Learning_Rate': lr_history
    })
    metrics_df.to_csv(os.path.join(save_dir, 'complex_training_metrics.csv'), index=False)
    
    # 保存最终评估结果
    final_metrics = {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'best_r2': best_r2,
        'final_r2': r2_history[-1],
        'final_pearson': pearson_history[-1],
        'model_params': {
            'hidden_dim': 120,
            'num_layers': 4,
            'heads': 6,
            'dropout': 0.15,
            'weight_decay': 1e-5
        }
    }
    
    with open(os.path.join(save_dir, 'complex_summary.txt'), 'w') as f:
        f.write("复杂模型训练总结\n")
        f.write("="*50 + "\n")
        f.write(f"最佳轮次: {best_epoch}\n")
        f.write(f"最佳验证损失: {best_val_loss:.4f}\n")
        f.write(f"最佳R²: {best_r2:.3f}\n")
        f.write(f"最终R²: {r2_history[-1]:.3f}\n")
        f.write(f"最终Pearson: {pearson_history[-1]:.3f}\n")
        f.write(f"\n复杂参数:\n")
        f.write(f"  hidden_dim: 120 (增强表达能力)\n")
        f.write(f"  num_layers: 4 (增加深度)\n")
        f.write(f"  heads: 6 (增加注意力头)\n")
        f.write(f"  dropout: 0.15 (轻微正则化)\n")
        f.write(f"  weight_decay: 1e-5 (轻微权重衰减)\n")
        f.write(f"  学习率调度: ReduceLROnPlateau\n")
        f.write(f"  早停: patience=30\n")
        f.write(f"\n模型版本对比:\n")
        f.write(f"  原始版本: hidden_dim=128, dropout=0.1, 严重过拟合\n")
        f.write(f"  简化版本: hidden_dim=64, dropout=0.3, R²≈0.38\n")
        f.write(f"  平衡版本: hidden_dim=96, dropout=0.2, 目标R²>0.5\n")
        f.write(f"  复杂版本: hidden_dim=120, dropout=0.15, 目标R²>0.6\n")
    
    print(f"✅ 复杂训练完成！结果保存在 {save_dir}")
    print(f"最佳R²: {best_r2:.3f} (第 {best_epoch} 轮)")
    
    return final_metrics

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="kcat_full.pt", help='Path to .pt dataset')
    parser.add_argument('--save_dir', type=str, default='outputs/kcat_complex', help='Save directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=400, help='Maximum epochs')
    args = parser.parse_args()
    
    from utils.metadata_utils import save_metadata

    # 保存元数据
    save_metadata(
        save_dir=args.save_dir,
        dataset_path=args.dataset,
        graph_builder_version='enhanced_builder',
        gnn_model_version='PocketGNNKcatOnly_Complex',
        comments='Complex model: enhanced capacity, adaptive learning rate, reference AM-GCN'
    )

    final_metrics = train_complex(
        dataset_path=args.dataset,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs
    )
