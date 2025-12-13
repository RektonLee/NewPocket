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
matplotlib.use('Agg')  # 确保在没有GUI的环境中使用
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

def train(dataset_path, save_dir="outputs", batch_size=32, lr=1e-3, max_epochs=500, 
          weight_decay=0.0, dropout=0.1, loss_type='mse', scheduler_type=None, 
          pooling_type='mean', use_seq_embedding=False, seq_embedding_path=None):
    """
    增强的训练函数，支持多种正则化和模型配置选项
    
    参数:
        dataset_path: 数据集路径
        save_dir: 输出目录
        batch_size: 批次大小
        lr: 学习率
        max_epochs: 最大训练轮数
        weight_decay: L2正则化系数 (推荐: 1e-4)
        dropout: Dropout概率 (推荐: 0.3-0.4)
        loss_type: 损失函数类型 ('mse' 或 'huber')
        scheduler_type: 学习率调度器 (None, 'plateau', 或 'cosine')
        pooling_type: 池化方式 ('mean' 或 'attention')
        use_seq_embedding: 是否使用序列嵌入
        seq_embedding_path: 序列嵌入文件路径
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
    data_list = torch.load(dataset_path, weights_only=False)  # List[Data]
    print(data_list[0])  # 打印第一个图数据
    actual_num_atom_types = data_list[0].x.shape[1]
    np.random.shuffle(data_list)
    split = int(0.8 * len(data_list))
    train_loader = DataLoader(data_list[:split], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data_list[split:], batch_size=batch_size)

    print(data_list[0].temperature)
    # === Initialize model ===
    node_input_dim = data_list[0].x.shape[1] #default 52
    edge_input_dim = data_list[0].edge_attr.shape[1]  # 现在应该是24维
    print(f"Node input dim: {node_input_dim}, Edge input dim: {edge_input_dim}")
    
    # 使用新的kcat专用模型
    hidden_dim = 128  # 模型隐藏维度
    num_layers = 3    # 层数
    heads = 4         # 注意力头数

    model = MD.PocketGNNKcatOnly(
        node_input_dim=node_input_dim,
        edge_input_dim=edge_input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
        dropout=dropout,
        pooling_type=pooling_type,
        use_seq_embedding=use_seq_embedding
    ).to(device)
    
    # === Optimizer with weight decay ===
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # === Loss function ===
    if loss_type == 'huber':
        criterion = nn.HuberLoss(delta=1.0)
        print("✅ 使用 Huber Loss（对离群点更鲁棒）")
    else:
        criterion = nn.MSELoss()
        print("✅ 使用 MSE Loss")
    
    # === Learning rate scheduler ===
    scheduler = None
    if scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        print("✅ 使用 ReduceLROnPlateau 调度器")
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-6
        )
        print("✅ 使用 CosineAnnealingWarmRestarts 调度器")
    
    best_val_loss = float('inf')
    
    # 加载序列嵌入（如果使用）
    seq_embeddings = None
    if use_seq_embedding and seq_embedding_path:
        seq_embeddings = torch.load(seq_embedding_path)
        print(f"✅ 加载序列嵌入: {seq_embedding_path}")
    
    print(f"\n=== 训练配置 ===")
    print(f"Weight Decay: {weight_decay}")
    print(f"Dropout: {dropout}")
    print(f"Loss Type: {loss_type}")
    print(f"Scheduler: {scheduler_type}")
    print(f"Pooling Type: {pooling_type}")
    print(f"Use Seq Embedding: {use_seq_embedding}")
    print(f"================\n")

    # 添加损失记录列表
    train_loss_history = []
    val_loss_history = []
    r2_history = []
    pearson_history = []
    
    # 将本次训练的关键超参数与环境信息记录到共享表中（保持运行中状态）
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

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            # print("✅ batch.y.shape:", batch.y.shape)
            # print("✅ batch.batch.shape:", batch.batch.shape)
            # print("✅ batch_size:", batch_size)
            # print("❓ batch.y:", batch.y)

            batch = batch.to(device)
            optimizer.zero_grad()
            
            # 只使用kcat标签（第一列）
            actual_batch_size = batch.num_graphs  # 使用实际的 batch size

            # 如果batch.y只有kcat值，应该这样处理：
            if batch.y.shape[0] == actual_batch_size:
                # 只有kcat值
                log_y = batch.y.reshape(actual_batch_size, 1)
            else:
                # 有kcat和km值
                y_reshaped = batch.y.reshape(actual_batch_size, 2)
                log_y = y_reshaped[:, 0:1]  # 只取kcat列
            
            if torch.isnan(batch.x).any():
                print("❌ batch.x 中含有 NaN")
            if torch.isnan(batch.edge_attr).any():
                print("❌ batch.edge_attr 中含有 NaN")
            if torch.isnan(batch.y).any():
                print("❌ batch.y 中含有 NaN")
            # print("batch.x max:", batch.x.max().item(), "min:", batch.x.min().item())
            # print("batch.edge_attr max:", batch.edge_attr.max().item(), "min:", batch.edge_attr.min().item())
            # print("batch.y:", batch.y[:5])
            out = model(batch)
           
            loss = criterion(out, log_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
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
                
                # 处理标签 - 适配不同的数据格式
                actual_batch_size = batch.num_graphs  # 使用实际的 batch size
                if batch.y.shape[0] == actual_batch_size:
                    # 只有kcat值
                    log_y = batch.y.reshape(actual_batch_size, 1)
                else:
                    # 有kcat和km值
                    y_reshaped = batch.y.reshape(actual_batch_size, 2)
                    log_y = y_reshaped[:, 0:1]  # 只取kcat列 [batch_size, 1]
                
                try:
                    out = model(batch)
                except ValueError as e:
                    print(f"❌ NaN 输出，batch中数据文件: {[d.pdb_id for d in batch]}")
                    raise e
                loss = criterion(out, log_y)
                val_losses.append(loss.item())
                y_true_log.append(log_y.cpu())
                y_pred_log.append(out.cpu())
        val_loss = np.mean(val_losses)
        y_true_log = torch.cat(y_true_log, dim=0)
        y_pred_log = torch.cat(y_pred_log, dim=0)
        metrics = compute_metrics(y_true_log, y_pred_log)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        r2_history.append(metrics['R2'])
        pearson_history.append(metrics['Pearson'])
        
        # === Learning rate scheduling ===
        if scheduler is not None:
            if scheduler_type == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # === Logging ===
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("R2/val", metrics['R2'], epoch)
        writer.add_scalar("Pearson/val", metrics['Pearson'], epoch)
        if scheduler is not None:
            writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | R2: {metrics['R2']:.3f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # === Save best model ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

    writer.close()
    print("✅ Training finished. Best model saved.")

    # 训练结束后绘制图像
    # 1. 损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_epochs + 1), train_loss_history, label='Train Loss')
    plt.plot(range(1, max_epochs + 1), val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()
    
    # 2. R2和Pearson相关系数曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_epochs + 1), r2_history, label='R² Score')
    plt.plot(range(1, max_epochs + 1), pearson_history, label='Pearson Correlation')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('R² and Pearson Correlation During Training')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'metrics_curve.png'))
    plt.close()
    
    # 3. 预测vs真实值散点图 (使用最好的模型)
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt")))
    model.eval()
    
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            batch_size = batch.num_graphs
            
            # 处理标签 - 适配不同的数据格式，与训练时保持一致
            if batch.y.shape[0] == batch_size:
                # 只有kcat值
                log_y = batch.y.reshape(batch_size, 1)
            else:
                # 有kcat和km值
                y_reshaped = batch.y.reshape(batch_size, 2)
                log_y = y_reshaped[:, 0:1]  # 只取kcat列
            
            try:
                out = model(batch)
            except ValueError as e:
                print(f"❌ NaN 输出，batch中数据文件: {[d.pdb_id for d in batch]}")
                raise e
            all_y_true.append(log_y.cpu())
            all_y_pred.append(out.cpu())
    
    all_y_true = torch.cat(all_y_true, dim=0).numpy()
    all_y_pred = torch.cat(all_y_pred, dim=0).numpy()
    
    # 绘制kcat散点图（只有一个图）
    plt.figure(figsize=(8, 6))
    
    # kcat散点图
    plt.scatter(all_y_true.flatten(), all_y_pred.flatten(), alpha=0.6)
    plt.plot([all_y_true.min(), all_y_true.max()], 
             [all_y_true.min(), all_y_true.max()], 'r--')
    plt.xlabel('True kcat (log10)')
    plt.ylabel('Predicted kcat (log10)')
    r2_kcat = r2_score(all_y_true.flatten(), all_y_pred.flatten())
    pearson_end = pearsonr(all_y_true.flatten(), all_y_pred.flatten())[0]
    plt.title(f'kcat: True vs Predicted (R² = {r2_kcat:.3f})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'kcat_prediction_scatter.png'))
    plt.close()
    
    # 4. kcat密度图
    plt.figure(figsize=(8, 7))
    true_vals = all_y_true.flatten()
    pred_vals = all_y_pred.flatten()
    sns.kdeplot(x=true_vals, y=pred_vals, cmap="viridis", fill=True, thresh=0.05)
    plt.plot([true_vals.min(), true_vals.max()], 
             [true_vals.min(), true_vals.max()], 'r--')
    plt.xlabel('True kcat (log10)')
    plt.ylabel('Predicted kcat (log10)')
    plt.title('kcat: Density Plot')
    plt.savefig(os.path.join(save_dir, 'kcat_density.png'))
    plt.close()
    
    # 保存最终指标到文件
    metrics_df = pd.DataFrame({
        'Epoch': range(1, max_epochs + 1),
        'Train_Loss': train_loss_history,
        'Val_Loss': val_loss_history,
        'R2': r2_history,
        'Pearson': pearson_history
    })
    metrics_df.to_csv(os.path.join(save_dir, 'training_metrics.csv'), index=False)
    
    # 记录最终结果到共享表
    try:
        update_training_results(
            save_dir=save_dir,
            status="completed",
            best_val_loss=best_val_loss,
            final_r2=r2_kcat,
            final_pearson=pearson_end
        )
    except Exception as _:
        pass

    print("✅ Training finished. Best model and plots saved to", save_dir)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='训练增强版 GNN 模型用于 kcat 预测')
    
    # 基础参数
    parser.add_argument('--dataset', type=str, default="kcat_train_after_new_clean.pt", 
                        help='Path to .pt dataset')
    parser.add_argument('--save_dir', type=str, default='outputs/kcat_after_new',
                        help='输出目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--max_epochs', type=int, default=500, help='最大训练轮数')
    
    # 正则化参数（阶段一）
    parser.add_argument('--weight_decay', type=float, default=0.0, 
                        help='L2正则化系数 (推荐: 1e-4)')
    parser.add_argument('--dropout', type=float, default=0.1, 
                        help='Dropout概率 (推荐: 0.3-0.4)')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'huber'],
                        help='损失函数类型')
    parser.add_argument('--scheduler', type=str, default=None, 
                        choices=[None, 'plateau', 'cosine'],
                        help='学习率调度器类型')
    
    # 模型结构参数（阶段二）
    parser.add_argument('--pooling_type', type=str, default='mean', 
                        choices=['mean', 'attention'],
                        help='图池化方式')
    
    # 序列嵌入参数（阶段三）
    parser.add_argument('--use_seq_embedding', action='store_true',
                        help='是否使用序列嵌入（ESM-2）')
    parser.add_argument('--seq_embedding_path', type=str, default=None,
                        help='序列嵌入文件路径 (.pt)')
    
    args = parser.parse_args()
    from utils.metadata_utils import save_metadata

    # 训练开始时，保存metadata
    save_metadata(
        save_dir=args.save_dir,
        dataset_path=args.dataset,
        graph_builder_version='enhanced_builder',
        gnn_model_version='PocketGNNKcatOnly_Enhanced',
        comments=f'Enhanced training: weight_decay={args.weight_decay}, dropout={args.dropout}, '
                f'loss={args.loss}, scheduler={args.scheduler}, pooling={args.pooling_type}, '
                f'seq_emb={args.use_seq_embedding}'
    )

    train(
        dataset_path=args.dataset, 
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        loss_type=args.loss,
        scheduler_type=args.scheduler,
        pooling_type=args.pooling_type,
        use_seq_embedding=args.use_seq_embedding,
        seq_embedding_path=args.seq_embedding_path
    )
