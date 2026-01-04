import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import GNN_model as MD
from quantile_loss import quantile_loss, compute_quantile_metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')  # 确保在没有GUI的环境中使用
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from metadata_utils import update_training_results, save_metadata
import wandb

def compute_metrics(y_true_log, y_pred_log):
    """
    计算评估指标，包含异常值处理和诊断信息
    """
    # 转换为numpy并展平
    y_true_log = y_true_log.numpy().flatten()
    y_pred_log = y_pred_log.numpy().flatten()
    
    # 检查NaN和Inf值
    true_has_nan = np.isnan(y_true_log).any()
    pred_has_nan = np.isnan(y_pred_log).any()
    true_has_inf = np.isinf(y_true_log).any()
    pred_has_inf = np.isinf(y_pred_log).any()
    
    if true_has_nan or pred_has_nan or true_has_inf or pred_has_inf:
        print(f"⚠️  警告: 检测到异常值 - true_nan:{true_has_nan}, pred_nan:{pred_has_nan}, "
              f"true_inf:{true_has_inf}, pred_inf:{pred_has_inf}")
        # 移除NaN和Inf值
        valid_mask = ~(np.isnan(y_true_log) | np.isnan(y_pred_log) | 
                      np.isinf(y_true_log) | np.isinf(y_pred_log))
        if valid_mask.sum() == 0:
            print("❌ 所有值都是NaN或Inf，无法计算指标")
            return {'MAE': np.nan, 'RMSE': np.nan, 'R2': -np.inf, 'Pearson': 0.0}
        y_true_log = y_true_log[valid_mask]
        y_pred_log = y_pred_log[valid_mask]
        print(f"   保留 {len(y_true_log)}/{len(valid_mask)} 个有效值")
    
    # ⚡ 裁剪极端预测值（防止 R² 计算异常）
    # log10 kcat 的合理范围通常是 [-10, 10]，超出范围的可能是异常值
    y_pred_log = np.clip(y_pred_log, -10.0, 10.0)
    
    # 检查预测值的范围（诊断信息）
    pred_min, pred_max = y_pred_log.min(), y_pred_log.max()
    pred_mean, pred_std = y_pred_log.mean(), y_pred_log.std()
    true_mean, true_std = y_true_log.mean(), y_true_log.std()
    
    # 如果预测值或真实值的标准差为0，R²计算会有问题
    if pred_std == 0 or true_std == 0:
        print(f"⚠️  警告: 预测值或真实值标准差为0 - pred_std:{pred_std:.6f}, true_std:{true_std:.6f}")
        # 如果预测值都是常数，R²为负是正常的
        r2_val = -np.inf if pred_std == 0 else 0.0
    else:
        try:
            r2_val = r2_score(y_true_log, y_pred_log)
            # 如果R²异常负值，记录诊断信息
            if r2_val < -100:
                print(f"⚠️  异常R²值: {r2_val:.2f}")
                print(f"   预测值范围: [{pred_min:.2f}, {pred_max:.2f}], mean={pred_mean:.2f}, std={pred_std:.2f}")
                print(f"   真实值范围: [{y_true_log.min():.2f}, {y_true_log.max():.2f}], mean={true_mean:.2f}, std={true_std:.2f}")
        except Exception as e:
            print(f"❌ R²计算失败: {e}")
            r2_val = -np.inf
    
    # 计算Pearson相关系数（处理异常情况）
    try:
        if pred_std == 0 or true_std == 0:
            pearson_val = 0.0
        else:
            pearson_val = pearsonr(y_true_log, y_pred_log)[0]
            if np.isnan(pearson_val):
                pearson_val = 0.0
    except Exception as e:
        print(f"⚠️  Pearson计算失败: {e}")
        pearson_val = 0.0
    
    return {
        'MAE': mean_absolute_error(y_true_log, y_pred_log),
        'RMSE': np.sqrt(mean_squared_error(y_true_log, y_pred_log)),
        'R2': r2_val,
        'Pearson': pearson_val
    }

def train(args):
    dataset_path = args.dataset
    save_dir = args.save_dir
    batch_size = 32 # defaults if not in args, but usually controlled by loop or constant
    lr = 5e-4  # ⚡ 对于大模型，使用更小的学习率
    max_epochs = 500
    
    # Check if we should override defaults with args
    if hasattr(args, 'batch_size'): batch_size = args.batch_size
    if args.lr is not None:
        lr = args.lr
    else:
        # ⚡ 如果没有指定 lr，根据模型大小自动调整
        # 大模型（hidden_dim >= 256）使用更小的默认学习率
        if hasattr(args, 'hidden_dim') and args.hidden_dim >= 256:
            lr = 5e-4
        # 否则使用默认的 1e-3（已经在上面定义了）
    if hasattr(args, 'epochs'): max_epochs = args.epochs
    
    # ========== 实验命名和目录管理 ==========
    # 获取实验名称和 run_id（通过 save_metadata，它会自动管理 experiments/ 目录）
    # 如果用户没有指定 exp_name，使用默认值
    exp_name = getattr(args, 'exp_name', 'kcat_default')
    exp_name, run_id = save_metadata(
        save_dir=save_dir,
        dataset_path=dataset_path,
        exp_name=exp_name,
        graph_builder_version='enhanced_builder',
        gnn_model_version='PocketGNNKcatOnly',
        comments=f'Enhanced: pooling={args.pooling_type}, seq_emb={args.use_seq_embedding}, loss={args.loss}, wd={args.weight_decay}'
    )
    
    # 统一命名规则：确保 outputs/、wandb/、experiments/ 目录中的内容对应
    # wandb run name 格式: {exp_name}_{run_id}，例如 "kcat_attn_v1_run_01"
    wandb_run_name = f"{exp_name}_{run_id}"
    
    # 定义 device（在 wandb_config 之前）
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # ========== 初始化 WandB ==========
    # 登录 wandb（使用提供的 API key）
    wandb.login(key="46dbe55e52d029976ffa0e29c90f0d32410e1504")
    
    # 创建 wandb config 字典
    wandb_config = {
        "dataset": os.path.basename(dataset_path),
        "batch_size": batch_size,
        "lr": lr,
        "max_epochs": max_epochs,
        "device": str(device),
        "loss_type": args.loss,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "scheduler": args.scheduler,
        "pooling_type": args.pooling_type,
        "use_seq_embedding": args.use_seq_embedding,
        "exp_name": exp_name,
        "run_id": run_id,
        "save_dir": save_dir,
    }
    
    # 创建 tags 用于在 wandb 界面快速筛选
    # 基于关键参数生成 tags，方便在 wandb 中快速找到对应的 run
    tags = [
        f"pooling_{args.pooling_type}",
        f"scheduler_{args.scheduler}",
        f"loss_{args.loss}",
    ]
    if args.use_seq_embedding:
        tags.append("with_seq_emb")
    else:
        tags.append("no_seq_emb")
    tags.append(exp_name)  # 添加 exp_name 作为 tag
    
    # 创建 notes（描述信息），包含关键参数摘要
    # 这样在 wandb 的 run 列表中就能看到关键信息，不需要点进去
    notes = f"""Key Parameters:
- Pooling: {args.pooling_type}
- Scheduler: {args.scheduler}
- Loss: {args.loss}
- Dropout: {args.dropout}
- Weight Decay: {args.weight_decay}
- Seq Embedding: {'Yes' if args.use_seq_embedding else 'No'}
- Dataset: {os.path.basename(dataset_path)}
- Save Dir: {save_dir}
"""
    
    # 初始化 wandb run
    # 注意：run name 在这里指定，确保与 experiments/ 目录中的命名对应
    wandb.init(
        project="enzyme_kcat_prediction",  # wandb 项目名称
        name=wandb_run_name,  # ⭐ 在这里指定 run 名称：格式为 {exp_name}_{run_id}
        config=wandb_config,
        tags=tags,  # 添加 tags 用于快速筛选
        notes=notes,  # 添加 notes 显示关键参数摘要
        dir=os.path.dirname(save_dir) if save_dir else ".",  # wandb 日志目录
    )
    
    # 在本地保存参数配置文件，方便快速查看
    os.makedirs(save_dir, exist_ok=True)
    config_file = os.path.join(save_dir, "training_config.txt")
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Training Configuration\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Experiment Name: {exp_name}\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"WandB Run Name: {wandb_run_name}\n")
        f.write(f"\n--- Dataset ---\n")
        f.write(f"Dataset Path: {dataset_path}\n")
        f.write(f"Dataset Name: {os.path.basename(dataset_path)}\n")
        f.write(f"\n--- Model Architecture ---\n")
        f.write(f"Pooling Type: {args.pooling_type}\n")
        f.write(f"Use Seq Embedding: {args.use_seq_embedding}\n")
        if args.use_seq_embedding:
            f.write(f"Seq Embedding Path: {args.seq_embedding_path}\n")
        f.write(f"\n--- Training Hyperparameters ---\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Max Epochs: {max_epochs}\n")
        f.write(f"Loss Type: {args.loss}\n")
        f.write(f"Dropout: {args.dropout}\n")
        f.write(f"Weight Decay: {args.weight_decay}\n")
        f.write(f"Scheduler: {args.scheduler}\n")
        f.write(f"\n--- Environment ---\n")
        f.write(f"Device: {device}\n")
        f.write(f"Save Directory: {save_dir}\n")
        f.write(f"\n--- Command Line Arguments ---\n")
        # 保存完整的命令行参数（便于复现）
        import sys
        f.write(f"Command: {' '.join(sys.argv)}\n")
        f.write("\n" + "=" * 60 + "\n")
    print(f"✅ Training configuration saved to: {config_file}")

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
    
    # 如果启用了seq_embedding但某些样本没有匹配到，用零向量填充
    if args.use_seq_embedding and seq_embedding_dim > 0:
        missing_count = 0
        for data in data_list:
            if not hasattr(data, 'seq_embedding'):
                data.seq_embedding = torch.zeros((1, seq_embedding_dim), dtype=torch.float)
                missing_count += 1
        if missing_count > 0:
            print(f"Warning: {missing_count} samples missing seq_embedding, filled with zeros.")

    # device 已在前面定义（用于 wandb_config）
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
    hidden_dim = args.hidden_dim  # 模型隐藏维度
    num_layers = args.num_layers    # 层数
    heads = args.heads         # 注意力头数
    dropout = args.dropout # Configurable dropout

    # 解析quantile levels
    use_quantile = (args.loss == 'quantile')
    if use_quantile:
        quantiles = [float(q) for q in args.quantiles.split(',')]
        print(f"Using Quantile Regression with quantiles: {quantiles}")
    else:
        quantiles = None
    
    model = MD.PocketGNNKcatOnly(
        node_input_dim=node_input_dim,
        edge_input_dim=edge_input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
        dropout=dropout,
        pooling_type=args.pooling_type,
        use_seq_embedding=args.use_seq_embedding,
        seq_embedding_dim=seq_embedding_dim if args.use_seq_embedding else 0,
        output_quantiles=use_quantile
    ).to(device)
    
    # === Phase 1: Optimizer & Loss & Scheduler ===
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    
    if args.loss == 'huber':
        criterion = nn.HuberLoss(delta=1.0)
    elif args.loss == 'quantile':
        criterion = None  # 使用自定义的quantile_loss函数
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
            
            # 裁剪极端预测值（防止训练不稳定）
            # log10 kcat 的合理范围是 [-10, 10]，超出范围会导致 R² 异常
            out = torch.clamp(out, min=-10.0, max=10.0)
            
            # 检查输出是否包含异常值
            if torch.isnan(out).any() or torch.isinf(out).any():
                print(f"⚠️  Epoch {epoch} 训练阶段: 模型输出包含异常值")
                # 跳过这个batch，不更新参数
                continue
            
            # 计算loss
            if use_quantile:
                loss = quantile_loss(out, log_y, quantiles=quantiles)
            else:
                loss = criterion(out, log_y)
            
            # 检查loss是否异常
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e6:
                print(f"⚠️  Epoch {epoch} 训练阶段: 异常loss值 {loss.item()}")
                continue
            
            loss.backward()
            # 增强梯度裁剪：更严格的限制，防止梯度爆炸
            # ⚡ 对于大模型，使用更小的 max_norm
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            if grad_norm > 5.0:  # 如果梯度范数很大，记录警告
                print(f"⚠️  Epoch {epoch}: 梯度范数较大 {grad_norm:.2f} (已裁剪到 0.5)")
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
                    # ⚡ 检查预测值是否在合理范围内（防止极端值导致 R² 异常）
                    if torch.isnan(out).any() or torch.isinf(out).any():
                        print(f"⚠️  Epoch {epoch}: 检测到 NaN/Inf 预测值，跳过该 batch")
                        continue
                    # 裁剪极端预测值到合理范围（log10 kcat 通常在 [-10, 10]）
                    out = torch.clamp(out, min=-10.0, max=10.0)
                except ValueError as e:
                    print(f"❌ Epoch {epoch}: NaN 输出，跳过该 batch")
                    continue
                except Exception as e:
                    print(f"❌ Epoch {epoch}: 模型前向传播异常: {e}，跳过该 batch")
                    continue
                    
                # 计算loss
                if use_quantile:
                    loss = quantile_loss(out, log_y, quantiles=quantiles)
                else:
                    loss = criterion(out, log_y)
                    
                # 如果 loss 异常大，也跳过
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1000:
                    print(f"⚠️  Epoch {epoch}: Loss 异常 ({loss.item():.2f})，跳过该 batch")
                    continue
                    
                val_losses.append(loss.item())
                y_true_log.append(log_y.cpu())
                y_pred_log.append(out.cpu())
        val_loss = np.mean(val_losses) if val_losses else float('inf')
        y_true_log = torch.cat(y_true_log, dim=0) if y_true_log else torch.tensor([])
        y_pred_log = torch.cat(y_pred_log, dim=0) if y_pred_log else torch.tensor([])
        
        # 如果验证集为空（所有batch都被跳过），使用默认值
        if len(y_true_log) == 0:
            print(f"⚠️  Epoch {epoch}: 验证集所有batch都被跳过，使用默认指标")
            if use_quantile:
                metrics = {'MAE': np.nan, 'RMSE': np.nan, 'R2': -np.inf, 'Pearson': 0.0, 
                          'coverage': 0.0, 'interval_width': np.nan}
            else:
                metrics = {'MAE': np.nan, 'RMSE': np.nan, 'R2': -np.inf, 'Pearson': 0.0}
        else:
            if use_quantile:
                # 使用quantile metrics
                quantile_metrics = compute_quantile_metrics(y_pred_log, y_true_log, quantiles=quantiles)
                # 对于quantile，使用中位数预测来计算传统指标
                y_pred_median = y_pred_log[:, 1:2]  # 取中位数（第2列）
                standard_metrics = compute_metrics(y_true_log, y_pred_median)
                metrics = {**standard_metrics, **quantile_metrics}
            else:
                metrics = compute_metrics(y_true_log, y_pred_log)
        
        # 如果R²异常负值，记录更详细的诊断信息
        if metrics['R2'] < -100:
            pred_stats = {
                'min': float(y_pred_log.min().item()) if len(y_pred_log) > 0 else 0.0,
                'max': float(y_pred_log.max().item()) if len(y_pred_log) > 0 else 0.0,
                'mean': float(y_pred_log.mean().item()) if len(y_pred_log) > 0 else 0.0,
                'std': float(y_pred_log.std().item()) if len(y_pred_log) > 0 else 0.0
            }
            true_stats = {
                'min': float(y_true_log.min().item()) if len(y_true_log) > 0 else 0.0,
                'max': float(y_true_log.max().item()) if len(y_true_log) > 0 else 0.0,
                'mean': float(y_true_log.mean().item()) if len(y_true_log) > 0 else 0.0,
                'std': float(y_true_log.std().item()) if len(y_true_log) > 0 else 0.0
            }
            print(f"⚠️  Epoch {epoch}: R²异常负值 ({metrics['R2']:.2f})")
            print(f"   预测值统计: {pred_stats}")
            print(f"   真实值统计: {true_stats}")
            # 记录到wandb以便后续分析
            wandb.log({
                "debug/pred_min": pred_stats['min'],
                "debug/pred_max": pred_stats['max'],
                "debug/pred_mean": pred_stats['mean'],
                "debug/pred_std": pred_stats['std'],
                "debug/true_std": true_stats['std'],
                "epoch": epoch
            })
        
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
        
        # 记录到 wandb
        log_dict = {
            "Loss/train": train_loss,
            "Loss/val": val_loss,
            "R2/val": metrics['R2'],
            "Pearson/val": metrics['Pearson'],
            "MAE/val": metrics['MAE'],
            "RMSE/val": metrics['RMSE'],
            "epoch": epoch
        }
        # 如果是quantile模式，添加额外的指标
        if use_quantile:
            log_dict.update({
                "Coverage/val": metrics.get('coverage', 0.0),
                "IntervalWidth/val": metrics.get('interval_width', np.nan)
            })
        wandb.log(log_dict)

        # 如果 R² 异常，打印更多诊断信息
        if metrics['R2'] < -10.0:
            print(f"⚠️  Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | R2: {metrics['R2']:.3f} (异常!)")
        else:
            if use_quantile:
                print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | R2: {metrics['R2']:.3f} | Pearson: {metrics['Pearson']:.3f} | Coverage: {metrics.get('coverage', 0.0):.3f}")
            else:
                print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | R2: {metrics['R2']:.3f} | Pearson: {metrics['Pearson']:.3f}")

        # === Save best model ===
        # ⚡ 只有当 R² 合理（> -10）且 loss 正常时才保存最佳模型
        if val_loss < best_val_loss and metrics['R2'] > -10.0:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            # 记录最佳模型到 wandb
            wandb.log({"best_val_loss": best_val_loss, "best_epoch": epoch})
        elif metrics['R2'] <= -10.0:
            print(f"⚠️  Epoch {epoch}: R² 异常负值 ({metrics['R2']:.2f})，跳过保存最佳模型")

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
    
    # 计算最终指标（在绘制图像之前）
    if use_quantile:
        # 使用中位数预测计算传统指标
        y_pred_median = all_y_pred[:, 1:2].flatten()  # 中位数（第2列）
        r2_kcat = r2_score(all_y_true.flatten(), y_pred_median)
        pearson_end = pearsonr(all_y_true.flatten(), y_pred_median)[0]
        
        # Quantile metrics
        quantile_metrics_end = compute_quantile_metrics(
            torch.tensor(all_y_pred), 
            torch.tensor(all_y_true), 
            quantiles=quantiles
        )
        
        # 绘制带置信区间的散点图
        plt.figure(figsize=(10, 8))
        y_true_flat = all_y_true.flatten()
        y_pred_low = all_y_pred[:, 0]   # 5% 分位数
        y_pred_median = all_y_pred[:, 1]  # 50% 分位数
        y_pred_high = all_y_pred[:, 2]  # 95% 分位数
        
        # 按真实值排序以便绘制区间
        sort_idx = np.argsort(y_true_flat)
        y_true_sorted = y_true_flat[sort_idx]
        y_pred_low_sorted = y_pred_low[sort_idx]
        y_pred_median_sorted = y_pred_median[sort_idx]
        y_pred_high_sorted = y_pred_high[sort_idx]
        
        # 绘制置信区间（阴影区域）
        plt.fill_between(y_true_sorted, y_pred_low_sorted, y_pred_high_sorted, 
                        alpha=0.3, color='blue', label=f'95% Confidence Interval')
        
        # 绘制中位数预测
        plt.scatter(y_true_flat, y_pred_median, alpha=0.6, s=15, label='Median Prediction')
        
        # 绘制完美预测线
        plt.plot([y_true_flat.min(), y_true_flat.max()], 
                 [y_true_flat.min(), y_true_flat.max()], 'r--', lw=2, label='Perfect')
        
        plt.xlabel('True kcat (log10)')
        plt.ylabel('Predicted kcat (log10)')
        plt.title(f'kcat Prediction with 95% CI\nR²={r2_kcat:.3f}, Coverage={quantile_metrics_end["coverage"]:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'kcat_prediction_with_ci.png'), dpi=150)
        plt.close()
        
        # 也保存传统散点图（仅中位数）
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true_flat, y_pred_median, alpha=0.6)
        plt.plot([y_true_flat.min(), y_true_flat.max()], 
                 [y_true_flat.min(), y_true_flat.max()], 'r--')
        plt.xlabel('True kcat (log10)')
        plt.ylabel('Predicted kcat (log10)')
        plt.title(f'kcat: True vs Predicted (Median, R² = {r2_kcat:.3f})')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'kcat_prediction_scatter.png'))
        plt.close()
    else:
        r2_kcat = r2_score(all_y_true.flatten(), all_y_pred.flatten())
        pearson_end = pearsonr(all_y_true.flatten(), all_y_pred.flatten())[0]
        
        # 绘制kcat散点图（只有一个图）
        plt.figure(figsize=(8, 6))
        plt.scatter(all_y_true.flatten(), all_y_pred.flatten(), alpha=0.6)
        plt.plot([all_y_true.min(), all_y_true.max()], 
                 [all_y_true.min(), all_y_true.max()], 'r--')
        plt.xlabel('True kcat (log10)')
        plt.ylabel('Predicted kcat (log10)')
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
    
    # 记录最终指标到 wandb（在计算完 r2_kcat 和 pearson_end 之后）
    final_log = {
        "final/best_val_loss": best_val_loss,
        "final/r2": r2_kcat,
        "final/pearson": pearson_end
    }
    if use_quantile:
        final_log.update({
            "final/coverage": quantile_metrics_end['coverage'],
            "final/interval_width": quantile_metrics_end['interval_width']
        })
    wandb.log(final_log)
    
    # 上传最终图像到 wandb（可选）
    if os.path.exists(os.path.join(save_dir, 'kcat_prediction_scatter.png')):
        wandb.log({"prediction_scatter": wandb.Image(os.path.join(save_dir, 'kcat_prediction_scatter.png'))})
    if os.path.exists(os.path.join(save_dir, 'loss_curve.png')):
        wandb.log({"loss_curve": wandb.Image(os.path.join(save_dir, 'loss_curve.png'))})
    
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
    
    wandb.finish()
    print("✅ Training finished. Best model and plots saved to", save_dir)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="kcat_train_after_new_clean.pt", help='Path to .pt dataset')
    parser.add_argument('--save_dir', type=str, default='outputs/kcat_after_new')
    
    # ========== 实验命名参数 ==========
    # ⭐ 在这里指定实验名称（exp_name）
    # 格式建议：描述性名称，如 "kcat_attn_v1", "kcat_seq_emb_v2" 等
    # 如果不指定，将使用默认值 "kcat_default"
    # run_id 会自动由 metadata_utils 生成（如 run_01, run_02...）
    # wandb run name 会自动设置为 {exp_name}_{run_id}，确保与 experiments/ 目录对应
    parser.add_argument('--exp_name', type=str, default='kcat_esm_full', 
                       help='Experiment name (e.g., "kcat_attn_v1"). Used for experiments/ and wandb run naming.')
    
    # Phase 1: Training & Regularization
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (default: auto-adjusted based on model size)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 regularization')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'huber', 'quantile'], help='Loss function (quantile for uncertainty quantification)')
    parser.add_argument('--quantiles', type=str, default='0.05,0.5,0.95', help='Quantile levels for quantile loss (comma-separated, e.g., 0.05,0.5,0.95)')
    parser.add_argument('--scheduler', type=str, default='none', choices=['none', 'plateau', 'cosine'], help='LR Scheduler')
    
    # Phase 2: Pooling
    parser.add_argument('--pooling_type', type=str, default='mean', choices=['mean', 'global_attention', 'set2set'], help='Graph pooling type: mean, global_attention, or set2set')
    
    # Phase 3: ESM Sequence Embedding
    parser.add_argument('--use_seq_embedding', action='store_true', help='Enable ESM sequence embedding (Late Fusion)')
    parser.add_argument('--seq_embedding_path', type=str, default='data/esm_embeddings.pt', help='Path to ESM embeddings dictionary')
    
    # Model Architecture Hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
    args = parser.parse_args()

    train(args)
