#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级训练策略脚本
使用多种技术提升模型泛化能力，解决数据分布不匹配问题
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
import random

def set_seed(seed=42):
    """设置随机种子确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AdaptiveLoss(nn.Module):
    """自适应损失函数，根据数据分布调整"""
    def __init__(self, alpha=0.5, beta=0.5):
        super(AdaptiveLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
        self.huber = nn.HuberLoss()
        
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        huber_loss = self.huber(pred, target)
        return self.alpha * mse_loss + self.beta * huber_loss

class FocalMSE(nn.Module):
    """Focal MSE Loss for handling hard examples"""
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalMSE, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        mse_loss = F.mse_loss(pred, target, reduction='none')
        pt = torch.exp(-mse_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * mse_loss
        return focal_loss.mean()

class DomainAdaptationLoss(nn.Module):
    """域适应损失，帮助模型处理分布不匹配"""
    def __init__(self, lambda_domain=0.1):
        super(DomainAdaptationLoss, self).__init__()
        self.lambda_domain = lambda_domain
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target, domain_features=None):
        # 主要回归损失
        main_loss = self.mse(pred, target)
        
        # 域适应损失（简化版本）
        if domain_features is not None:
            # 计算域间差异
            domain_loss = torch.var(domain_features, dim=0).mean()
            return main_loss + self.lambda_domain * domain_loss
        
        return main_loss

class EMA:
    """指数移动平均，用于模型权重平滑"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
                
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
                
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def compute_metrics(y_true_log, y_pred_log):
    y_true_log = y_true_log.numpy()
    y_pred_log = y_pred_log.numpy()
    return {
        'MAE': mean_absolute_error(y_true_log, y_pred_log),
        'RMSE': np.sqrt(mean_squared_error(y_true_log, y_pred_log)),
        'R2': r2_score(y_true_log, y_pred_log),
        'Pearson': pearsonr(y_true_log.flatten(), y_pred_log.flatten())[0]
    }

def get_learning_rate_scheduler(optimizer, scheduler_type='cosine', **kwargs):
    """获取学习率调度器"""
    if scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=kwargs.get('T_max', 500))
    elif scheduler_type == 'cosine_warmup':
        from torch.optim.lr_scheduler import LambdaLR
        def lr_lambda(epoch):
            if epoch < kwargs.get('warmup_epochs', 50):
                return epoch / kwargs.get('warmup_epochs', 50)
            else:
                return 0.5 * (1 + math.cos(math.pi * (epoch - kwargs.get('warmup_epochs', 50)) / (kwargs.get('T_max', 500) - kwargs.get('warmup_epochs', 50))))
        return LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == 'one_cycle':
        return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=kwargs.get('max_lr', 1e-2), 
                                           total_steps=kwargs.get('total_steps', 500))
    elif scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
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

def cutmix_data(x, y, alpha=1.0):
    """CutMix数据增强（简化版本）"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    # 随机选择混合比例
    lam = max(lam, 1 - lam)
    
    # 简单的特征混合
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def train_advanced(dataset_path, save_dir="outputs", batch_size=32, lr=1e-3, max_epochs=500,
                  use_ema=True, use_mixup=True, use_cutmix=True, use_adaptive_loss=True,
                  scheduler_type='cosine_warmup', weight_decay=1e-4, gradient_clip=1.0,
                  ema_decay=0.999, mixup_alpha=0.2, cutmix_alpha=1.0):
    """
    高级训练函数
    """
    set_seed(42)  # 设置随机种子
    
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
    
    # 计算数据分布统计
    y_mean, y_std = all_y.mean(), all_y.std()
    y_min, y_max = all_y.min(), all_y.max()
    
    np.random.shuffle(data_list)
    split = int(0.8 * len(data_list))
    train_loader = DataLoader(data_list[:split], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data_list[split:], batch_size=batch_size)

    # === Initialize model with advanced architecture ===
    node_input_dim = data_list[0].x.shape[1]
    edge_input_dim = data_list[0].edge_attr.shape[1]
    print(f"🔧 Node input dim: {node_input_dim}, Edge input dim: {edge_input_dim}")
    
    # 改进的模型参数
    hidden_dim = 256
    num_layers = 4
    heads = 8
    dropout = 0.2

    model = MD.PocketGNNKcatOnly(
        node_input_dim=node_input_dim,
        edge_input_dim=edge_input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
        dropout=dropout
    ).to(device)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    
    # 损失函数
    if use_adaptive_loss:
        criterion = AdaptiveLoss(alpha=0.7, beta=0.3)
    else:
        criterion = nn.MSELoss()
    
    # 学习率调度器
    scheduler = get_learning_rate_scheduler(optimizer, scheduler_type, T_max=max_epochs, warmup_epochs=50)
    
    # EMA
    if use_ema:
        ema = EMA(model, decay=ema_decay)
        ema.register()
    
    best_val_loss = float('inf')
    best_r2 = -float('inf')
    patience = 50
    patience_counter = 0

    # 记录列表
    train_loss_history = []
    val_loss_history = []
    r2_history = []
    pearson_history = []
    lr_history = []
    
    print(f"🚀 开始高级训练...")
    print(f"📊 模型参数: hidden_dim={hidden_dim}, num_layers={num_layers}, heads={heads}, dropout={dropout}")
    print(f"🔧 优化器: AdamW, lr={lr}, weight_decay={weight_decay}")
    print(f"📚 损失函数: {criterion.__class__.__name__}")
    print(f"🔄 学习率调度: {scheduler_type}")
    print(f"🎯 数据增强: Mixup={use_mixup}, CutMix={use_cutmix}")
    print(f"📈 EMA: {use_ema}")

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
            
            # 数据增强
            augmentation_applied = False
            if use_mixup and np.random.random() < 0.5:
                mixed_out, y_a, y_b, lam = mixup_data(out, log_y, alpha=mixup_alpha)
                loss = mixup_criterion(criterion, mixed_out, y_a, y_b, lam)
                augmentation_applied = True
            elif use_cutmix and np.random.random() < 0.3:
                mixed_out, y_a, y_b, lam = cutmix_data(out, log_y, alpha=cutmix_alpha)
                loss = mixup_criterion(criterion, mixed_out, y_a, y_b, lam)
                augmentation_applied = True
            
            if not augmentation_applied:
                loss = criterion(out, log_y)
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            
            optimizer.step()
            
            # 更新EMA
            if use_ema:
                ema.update()
            
            train_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)

        # === Validation ===
        if use_ema:
            ema.apply_shadow()
        
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
        
        if use_ema:
            ema.restore()
        
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
            best_r2 = metrics['R2']
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"🛑 Early stopping at epoch {epoch}")
            break

    writer.close()
    print("✅ 高级训练完成！")

    # 训练结束后绘制图像
    create_advanced_plots(train_loss_history, val_loss_history, r2_history, pearson_history, lr_history, save_dir, max_epochs)
    
    return model, best_val_loss, best_r2

def create_advanced_plots(train_loss_history, val_loss_history, r2_history, pearson_history, lr_history, save_dir, max_epochs):
    """创建高级训练的可视化图表"""
    print("🎨 生成高级训练可视化图表...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 损失曲线
    axes[0, 0].plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Train Loss', color='blue')
    axes[0, 0].plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Validation Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss (Advanced)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. R²和Pearson相关系数曲线
    axes[0, 1].plot(range(1, len(r2_history) + 1), r2_history, label='R² Score', color='green')
    axes[0, 1].plot(range(1, len(pearson_history) + 1), pearson_history, label='Pearson Correlation', color='orange')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('R² and Pearson Correlation (Advanced)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 学习率变化
    axes[0, 2].plot(range(1, len(lr_history) + 1), lr_history, color='purple')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Learning Rate')
    axes[0, 2].set_title('Learning Rate Schedule')
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 训练稳定性
    if len(val_loss_history) > 10:
        val_loss_smooth = np.convolve(val_loss_history, np.ones(10)/10, mode='valid')
        axes[1, 0].plot(range(10, len(val_loss_history) + 1), val_loss_smooth, color='red', label='Smoothed Val Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Smoothed Validation Loss')
    axes[1, 0].set_title('Training Stability')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 损失分布
    axes[1, 1].hist(train_loss_history, bins=30, alpha=0.7, label='Train Loss', color='blue')
    axes[1, 1].hist(val_loss_history, bins=30, alpha=0.7, label='Val Loss', color='red')
    axes[1, 1].set_xlabel('Loss Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Loss Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 性能指标对比
    axes[1, 2].plot(range(1, len(r2_history) + 1), r2_history, label='R²', color='green')
    axes[1, 2].plot(range(1, len(pearson_history) + 1), pearson_history, label='Pearson', color='orange')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].set_title('Performance Metrics')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'advanced_training_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 高级训练图表已保存")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Advanced training script for better generalization')
    parser.add_argument('--dataset', type=str, default="kcat_train_after_new_clean.pt", help='Path to .pt dataset')
    parser.add_argument('--save_dir', type=str, default='outputs/advanced_training', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='Max epochs')
    parser.add_argument('--use_ema', action='store_true', help='Use Exponential Moving Average')
    parser.add_argument('--use_mixup', action='store_true', help='Use Mixup data augmentation')
    parser.add_argument('--use_cutmix', action='store_true', help='Use CutMix data augmentation')
    parser.add_argument('--use_adaptive_loss', action='store_true', help='Use adaptive loss function')
    parser.add_argument('--scheduler', type=str, default='cosine_warmup', choices=['cosine', 'cosine_warmup', 'one_cycle', 'plateau'], help='Learning rate scheduler')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup alpha parameter')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0, help='CutMix alpha parameter')
    
    args = parser.parse_args()
    
    print("🚀 开始高级训练...")
    print(f"📁 数据集: {args.dataset}")
    print(f"💾 保存目录: {args.save_dir}")
    print(f"🔧 批次大小: {args.batch_size}")
    print(f"📚 学习率: {args.lr}")
    print(f"🔄 最大轮数: {args.epochs}")
    print(f"🎯 数据增强: Mixup={args.use_mixup}, CutMix={args.use_cutmix}")
    print(f"📊 损失函数: AdaptiveLoss={args.use_adaptive_loss}")
    print(f"🔄 学习率调度: {args.scheduler}")
    print(f"📈 EMA: {args.use_ema}")
    
    train_advanced(
        args.dataset, 
        args.save_dir, 
        args.batch_size, 
        args.lr, 
        args.epochs,
        use_ema=args.use_ema,
        use_mixup=args.use_mixup,
        use_cutmix=args.use_cutmix,
        use_adaptive_loss=args.use_adaptive_loss,
        scheduler_type=args.scheduler,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        ema_decay=args.ema_decay,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha
    )



