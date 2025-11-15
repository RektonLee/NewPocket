#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的训练脚本 - 提升模型泛化能力
通过调整超参数、添加正则化、改进训练策略来解决数据分布不匹配问题
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
import math
import json

def compute_metrics(y_true_log, y_pred_log):
    y_true_log = y_true_log.numpy()
    y_pred_log = y_pred_log.numpy()
    return {
        'MAE': mean_absolute_error(y_true_log, y_pred_log),
        'RMSE': np.sqrt(mean_squared_error(y_true_log, y_pred_log)),
        'R2': r2_score(y_true_log, y_pred_log),
        'Pearson': pearsonr(y_true_log.flatten(), y_pred_log.flatten())[0]
    }

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance and hard examples"""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        mse_loss = F.mse_loss(inputs, targets, reduction='none')
        pt = torch.exp(-mse_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * mse_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """Label Smoothing for better generalization"""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        mse_loss = F.mse_loss(pred, target, reduction='none')
        # 添加标签平滑
        smooth_loss = mse_loss * (1 - self.smoothing) + self.smoothing * mse_loss.mean()
        return smooth_loss.mean()

def get_learning_rate_scheduler(optimizer, scheduler_type='cosine', **kwargs):
    """获取学习率调度器"""
    if scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=kwargs.get('T_max', 500))
    elif scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=kwargs.get('step_size', 100), gamma=kwargs.get('gamma', 0.5))
    elif scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    elif scheduler_type == 'warmup_cosine':
        from torch.optim.lr_scheduler import LambdaLR
        def lr_lambda(epoch):
            if epoch < kwargs.get('warmup_epochs', 50):
                return epoch / kwargs.get('warmup_epochs', 50)
            else:
                return 0.5 * (1 + math.cos(math.pi * (epoch - kwargs.get('warmup_epochs', 50)) / (kwargs.get('T_max', 500) - kwargs.get('warmup_epochs', 50))))
        return LambdaLR(optimizer, lr_lambda)
    else:
        return None

def mixup_data(x, y, alpha=1.0):
    """Mixup数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup损失计算"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_improved(dataset_path, save_dir="outputs", batch_size=32, lr=1e-3, max_epochs=500, 
                  use_mixup=True, use_label_smoothing=True, use_focal_loss=False,
                  scheduler_type='warmup_cosine', weight_decay=1e-4, gradient_clip=1.0):
    """
    改进的训练函数
    """
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
    print(f"📊 加载了 {len(data_list)} 个样本")
    
    # 分析数据分布
    all_y = torch.cat([data.y for data in data_list]).numpy().flatten()
    print(f"📈 数据y值范围: [{all_y.min():.3f}, {all_y.max():.3f}]")
    print(f"📈 数据y值均值: {all_y.mean():.3f}, 标准差: {all_y.std():.3f}")
    
    np.random.shuffle(data_list)
    split = int(0.8 * len(data_list))
    train_loader = DataLoader(data_list[:split], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data_list[split:], batch_size=batch_size)

    # === Initialize model with improved architecture ===
    node_input_dim = data_list[0].x.shape[1]
    edge_input_dim = data_list[0].edge_attr.shape[1]
    print(f"🔧 Node input dim: {node_input_dim}, Edge input dim: {edge_input_dim}")
    
    # 改进的模型参数
    hidden_dim = 256      # 增加隐藏维度
    num_layers = 4        # 增加层数
    heads = 8             # 增加注意力头
    dropout = 0.2         # 增加dropout

    model = MD.PocketGNNKcatOnly(
        node_input_dim=node_input_dim,
        edge_input_dim=edge_input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
        dropout=dropout
    ).to(device)
    
    # 改进的优化器设置
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    
    # 选择损失函数
    if use_focal_loss:
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
    elif use_label_smoothing:
        criterion = LabelSmoothingLoss(smoothing=0.1)
    else:
        criterion = nn.MSELoss()
    
    # 学习率调度器
    scheduler = get_learning_rate_scheduler(optimizer, scheduler_type, T_max=max_epochs, warmup_epochs=50)
    
    best_val_loss = float('inf')
    patience = 50
    patience_counter = 0

    # 添加损失记录列表
    train_loss_history = []
    val_loss_history = []
    r2_history = []
    pearson_history = []
    lr_history = []
    
    # 记录训练信息
    try:
        update_training_results(
            save_dir=save_dir,
            status="running",
            model_hidden_dim=hidden_dim,
            model_num_layers=num_layers,
            model_heads=heads,
            model_dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            device=str(device)
        )
    except Exception as _:
        pass

    print(f"🚀 开始改进训练...")
    print(f"📊 模型参数: hidden_dim={hidden_dim}, num_layers={num_layers}, heads={heads}, dropout={dropout}")
    print(f"🔧 优化器: AdamW, lr={lr}, weight_decay={weight_decay}")
    print(f"📚 损失函数: {criterion.__class__.__name__}")
    print(f"🔄 学习率调度: {scheduler_type}")
    print(f"🎯 数据增强: Mixup={use_mixup}, LabelSmoothing={use_label_smoothing}")

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
            
            # 应用Mixup数据增强
            if use_mixup and np.random.random() < 0.5:
                # 对图特征进行mixup（简化版本）
                mixed_out, y_a, y_b, lam = mixup_data(out, log_y, alpha=0.2)
                loss = mixup_criterion(criterion, mixed_out, y_a, y_b, lam)
            else:
                loss = criterion(out, log_y)
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            
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
        
        # 记录历史
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        r2_history.append(metrics['R2'])
        pearson_history.append(metrics['Pearson'])
        lr_history.append(optimizer.param_groups[0]['lr'])

        # 学习率调度
        if scheduler_type == 'plateau':
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()

        # === Logging ===
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("R2/val", metrics['R2'], epoch)
        writer.add_scalar("Pearson/val", metrics['Pearson'], epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | R2: {metrics['R2']:.3f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # === Early Stopping ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"🛑 Early stopping at epoch {epoch}")
            break

    writer.close()
    print("✅ 改进训练完成！")

    # 保存模型配置
    model_config = {
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'heads': heads,
        'dropout': dropout,
        'node_input_dim': node_input_dim,
        'edge_input_dim': edge_input_dim,
        'training_params': {
            'lr': lr,
            'batch_size': batch_size,
            'max_epochs': max_epochs,
            'weight_decay': weight_decay,
            'gradient_clip': gradient_clip,
            'use_mixup': use_mixup,
            'use_label_smoothing': use_label_smoothing,
            'use_focal_loss': use_focal_loss,
            'scheduler_type': scheduler_type
        }
    }
    
    config_file = os.path.join(save_dir, "model_config.json")
    with open(config_file, 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"💾 模型配置已保存到: {config_file}")

    # 训练结束后绘制图像
    create_improved_plots(train_loss_history, val_loss_history, r2_history, pearson_history, lr_history, save_dir, max_epochs)
    
    return model, best_val_loss

def create_improved_plots(train_loss_history, val_loss_history, r2_history, pearson_history, lr_history, save_dir, max_epochs):
    """创建改进训练的可视化图表"""
    print("🎨 生成改进训练可视化图表...")
    
    # 1. 损失曲线
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Train Loss', color='blue')
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (Improved)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. R2和Pearson相关系数曲线
    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(r2_history) + 1), r2_history, label='R² Score', color='green')
    plt.plot(range(1, len(pearson_history) + 1), pearson_history, label='Pearson Correlation', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('R² and Pearson Correlation (Improved)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 学习率变化
    plt.subplot(2, 2, 3)
    plt.plot(range(1, len(lr_history) + 1), lr_history, color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # 4. 训练稳定性
    plt.subplot(2, 2, 4)
    if len(val_loss_history) > 10:
        val_loss_smooth = np.convolve(val_loss_history, np.ones(10)/10, mode='valid')
        plt.plot(range(10, len(val_loss_history) + 1), val_loss_smooth, color='red', label='Smoothed Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Smoothed Validation Loss')
    plt.title('Training Stability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'improved_training_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 改进训练图表已保存")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Improved training script for better generalization')
    parser.add_argument('--dataset', type=str, default="kcat_train_after_new_clean.pt", help='Path to .pt dataset')
    parser.add_argument('--save_dir', type=str, default='outputs/improved_training', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='Max epochs')
    parser.add_argument('--use_mixup', action='store_true', help='Use Mixup data augmentation')
    parser.add_argument('--use_label_smoothing', action='store_true', help='Use label smoothing')
    parser.add_argument('--use_focal_loss', action='store_true', help='Use focal loss')
    parser.add_argument('--scheduler', type=str, default='warmup_cosine', choices=['cosine', 'step', 'plateau', 'warmup_cosine'], help='Learning rate scheduler')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping')
    
    args = parser.parse_args()
    
    print("🚀 开始改进训练...")
    print(f"📁 数据集: {args.dataset}")
    print(f"💾 保存目录: {args.save_dir}")
    print(f"🔧 批次大小: {args.batch_size}")
    print(f"📚 学习率: {args.lr}")
    print(f"🔄 最大轮数: {args.epochs}")
    print(f"🎯 数据增强: Mixup={args.use_mixup}, LabelSmoothing={args.use_label_smoothing}")
    print(f"📊 损失函数: FocalLoss={args.use_focal_loss}")
    print(f"🔄 学习率调度: {args.scheduler}")
    
    train_improved(
        args.dataset, 
        args.save_dir, 
        args.batch_size, 
        args.lr, 
        args.epochs,
        use_mixup=args.use_mixup,
        use_label_smoothing=args.use_label_smoothing,
        use_focal_loss=args.use_focal_loss,
        scheduler_type=args.scheduler,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip
    )
