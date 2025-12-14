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
from metadata_utils import update_training_results

def compute_metrics(y_true_log, y_pred_log):
    y_true_log = y_true_log.numpy()
    y_pred_log = y_pred_log.numpy()
    return {
        'MAE': mean_absolute_error(y_true_log, y_pred_log),
        'RMSE': np.sqrt(mean_squared_error(y_true_log, y_pred_log)),
        'R2': r2_score(y_true_log, y_pred_log),
        'Pearson': pearsonr(y_true_log.flatten(), y_pred_log.flatten())[0]
    }

def train(args):
    dataset_path = args.dataset
    save_dir = args.save_dir
    batch_size = 32 # defaults if not in args, but usually controlled by loop or constant
    lr = 1e-3
    max_epochs = 500
    
    # Check if we should override defaults with args
    if hasattr(args, 'batch_size'): batch_size = args.batch_size
    if hasattr(args, 'lr'): lr = args.lr
    if hasattr(args, 'epochs'): max_epochs = args.epochs

    print(f"Loading dataset from {dataset_path}...")
    try:
        data_list = torch.load(dataset_path, weights_only=False)  # List[Data]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 注意：Data 对象可能包含字符串类型的元数据属性（如 ec, pdb_id, sample_id）
    # 这些属性在匹配 embeddings 时需要，但在 DataLoader collate 时会出错
    # 因此我们在匹配完 embeddings 后再删除这些属性
    
    # === Phase 3: Load Sequence Embeddings (Late Fusion) ===
    seq_embedding_dim = 0
    if args.use_seq_embedding:
        print(f"Loading sequence embeddings from {args.seq_embedding_path}...")
        if not os.path.exists(args.seq_embedding_path):
            raise FileNotFoundError(f"Sequence embedding file not found: {args.seq_embedding_path}")
        
        # Assume embeddings is a dict: {identifier: tensor}
        # Identifier could be uniprot_id or pdb_id or sample_id
        # We need to match what's in data_list (data.pdb_id or data.uniprot_id)
        embeddings_map = torch.load(args.seq_embedding_path, weights_only=False) # or specific loading logic
        
        # Check first key to see format if needed, or just try to match
        print(f"Loaded {len(embeddings_map)} embeddings.")
        
        # Check what keys are available in embeddings (for debugging)
        sample_keys = list(embeddings_map.keys())[:5]
        print(f"Sample embedding keys: {sample_keys}")
        
        # Check what IDs are available in data (for debugging)
        sample_data_ids = []
        for i, data in enumerate(data_list[:5]):
            ids = {}
            if hasattr(data, 'sample_id'):
                ids['sample_id'] = data.sample_id
            if hasattr(data, 'pdb_id'):
                ids['pdb_id'] = data.pdb_id
            if hasattr(data, 'uniprot_id'):
                ids['uniprot_id'] = data.uniprot_id
            sample_data_ids.append(ids)
        print(f"Sample data IDs: {sample_data_ids}")
        
        matched_count = 0
        unmatched_samples = []
        for data in data_list:
            # Key matching logic: try sample_id first (as per instructions "sample_id 对齐")
            # Then try pdb_id, uniprot_id as fallback
            key = None
            embedding = None
            key_type = None
            
            # Priority: sample_id > pdb_id > uniprot_id
            if hasattr(data, 'sample_id') and data.sample_id is not None:
                key = str(data.sample_id)
                key_type = 'sample_id'
            elif hasattr(data, 'pdb_id') and data.pdb_id is not None:
                key = str(data.pdb_id)
                key_type = 'pdb_id'
                # pdb_id might have format like "kcat_000002_61151_10A.pdb", try without extension
                if key not in embeddings_map and '.' in key:
                    key = key.split('.')[0]
            elif hasattr(data, 'uniprot_id') and data.uniprot_id is not None:
                key = str(data.uniprot_id)
                key_type = 'uniprot_id'
            
            if key and key in embeddings_map:
                embedding = embeddings_map[key]
            elif key:
                # Record unmatched for debugging (only first few)
                if len(unmatched_samples) < 5:
                    unmatched_samples.append((key, key_type))

            if embedding is not None:
                # Ensure it's a tensor
                if not isinstance(embedding, torch.Tensor):
                    embedding = torch.tensor(embedding, dtype=torch.float)
                
                # Check dim
                if seq_embedding_dim == 0:
                    seq_embedding_dim = embedding.shape[0]
                    print(f"Sequence embedding dimension: {seq_embedding_dim}")
                
                data.seq_embedding = embedding.unsqueeze(0) # [1, dim] for batching
                matched_count += 1

        print(f"Matched embeddings for {matched_count}/{len(data_list)} samples.")
        if unmatched_samples:
            print(f"Sample unmatched keys: {unmatched_samples}")
        
        # If we didn't find any, we can't proceed with seq embedding
        if matched_count == 0:
            print("Warning: No embeddings matched! Disabling sequence embedding.")
            seq_embedding_dim = 0
    
    # 移除字符串类型的元数据属性，避免 DataLoader collate 时出错
    # PyTorch Geometric 的 Batch.from_data_list 无法处理字符串属性
    print("Removing string metadata attributes to avoid collate errors...")
    for data in data_list:
        # 移除字符串属性（这些无法转换为 tensor）
        if hasattr(data, 'ec'):
            delattr(data, 'ec')
        if hasattr(data, 'pdb_id'):
            delattr(data, 'pdb_id')
        if hasattr(data, 'sample_id'):
            delattr(data, 'sample_id')
        if hasattr(data, 'uniprot_id'):
            delattr(data, 'uniprot_id')
            args.use_seq_embedding = False
        else:
            # Fill missing with zeros
            for data in data_list:
                if not hasattr(data, 'seq_embedding'):
                    data.seq_embedding = torch.zeros((1, seq_embedding_dim), dtype=torch.float)

    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(save_dir)

    # === Load dataset ===
    # data_list already loaded
    print(data_list[0])  # 打印第一个图数据
    
    np.random.shuffle(data_list)
    split = int(0.8 * len(data_list))
    train_loader = DataLoader(data_list[:split], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data_list[split:], batch_size=batch_size)

    # === Initialize model ===
    node_input_dim = data_list[0].x.shape[1] #default 52
    edge_input_dim = data_list[0].edge_attr.shape[1]  # 现在应该是24维
    print(f"Node input dim: {node_input_dim}, Edge input dim: {edge_input_dim}")
    
    # 使用新的kcat专用模型
    hidden_dim = 128  # 模型隐藏维度
    num_layers = 3    # 层数
    heads = 4         # 注意力头数
    dropout = args.dropout # Configurable dropout

    model = MD.PocketGNNKcatOnly(
        node_input_dim=node_input_dim,
        edge_input_dim=edge_input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
        dropout=dropout,
        pooling_type=args.pooling_type,
        seq_embedding_dim=seq_embedding_dim
    ).to(device)
    
    # === Phase 1: Optimizer & Loss & Scheduler ===
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    
    if args.loss == 'huber':
        criterion = nn.HuberLoss(delta=1.0)
    else:
        criterion = nn.MSELoss()
        
    scheduler = None
    if args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    elif args.scheduler == 'cosine':
        # T_0 could be max_epochs
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    best_val_loss = float('inf')
    
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
            device=str(device),
            # New params
            weight_decay=args.weight_decay,
            pooling_type=args.pooling_type,
            loss_type=args.loss,
            use_seq_embedding=args.use_seq_embedding
        )
    except Exception as _:
        pass

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
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
                
            out = model(batch)
           
            loss = criterion(out, log_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            train_losses.append(loss.item())
        
        # Step scheduler for cosine (batch level is better but epoch is fine for restart)
        # if args.scheduler == 'cosine': scheduler.step()
        
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
        
        # === Scheduler Step ===
        if args.scheduler == 'plateau':
            scheduler.step(val_loss)
        elif args.scheduler == 'cosine':
            scheduler.step()

        # === Logging ===
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("R2/val", metrics['R2'], epoch)
        writer.add_scalar("Pearson/val", metrics['Pearson'], epoch)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | R2: {metrics['R2']:.3f}")

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="kcat_train_after_new_clean.pt", help='Path to .pt dataset')
    parser.add_argument('--save_dir', type=str, default='outputs/kcat_after_new')
    
    # Phase 1: Training & Regularization
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 regularization')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'huber'], help='Loss function')
    parser.add_argument('--scheduler', type=str, default='none', choices=['none', 'plateau', 'cosine'], help='LR Scheduler')
    
    # Phase 2: Pooling
    parser.add_argument('--pooling_type', type=str, default='mean', choices=['mean', 'global_attention'], help='Graph pooling type')
    
    # Phase 3: ESM Sequence Embedding
    parser.add_argument('--use_seq_embedding', action='store_true', help='Enable ESM sequence embedding (Late Fusion)')
    parser.add_argument('--seq_embedding_path', type=str, default='data/esm_embeddings.pt', help='Path to ESM embeddings dictionary')
    
    args = parser.parse_args()
    
    from metadata_utils import save_metadata

    # 训练开始时，加上这行保存metadata
    save_metadata(
        save_dir=args.save_dir,
        dataset_path=args.dataset,
        graph_builder_version='enhanced_builder',
        gnn_model_version='PocketGNNKcatOnly',
        comments=f'Enhanced: pooling={args.pooling_type}, seq_emb={args.use_seq_embedding}, loss={args.loss}, wd={args.weight_decay}'
    )

    train(args)
