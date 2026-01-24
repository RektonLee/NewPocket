import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import json
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

def parse_split_ratios(ratio_str):
    parts = [p.strip() for p in ratio_str.split(",") if p.strip()]
    ratios = [float(p) for p in parts]
    if len(ratios) not in (2, 3):
        raise ValueError(f"split_ratios must have 2 or 3 values, got: {ratio_str}")
    total = sum(ratios)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split_ratios must sum to 1.0, got sum={total:.4f}")
    return ratios

def read_mmseqs_tsv(tsv_path):
    member_to_rep = {}
    with open(tsv_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                rep = parts[0]
                mem = parts[1]
                member_to_rep[mem] = rep
    return member_to_rep

def make_cluster_ids(member_to_rep):
    reps = sorted(set(member_to_rep.values()))
    rep_to_cid = {rep: i for i, rep in enumerate(reps)}
    member_to_cid = {m: rep_to_cid[rep] for m, rep in member_to_rep.items()}
    return member_to_cid, rep_to_cid

def split_clusters(rep_to_cid, seed=42, ratios=(0.8, 0.1, 0.1)):
    assert abs(sum(ratios) - 1.0) < 1e-6
    cids = list(rep_to_cid.values())
    rng = np.random.default_rng(seed)
    rng.shuffle(cids)
    n = len(cids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_c = set(cids[:n_train])
    val_c = set(cids[n_train:n_train + n_val])
    test_c = set(cids[n_train + n_val:])
    return train_c, val_c, test_c

def split_dataset_by_clusters(dataset, member_to_cid, train_c, val_c, test_c):
    train, val, test, missing = [], [], [], 0
    for d in dataset:
        sid = getattr(d, "sample_id", None)
        if sid is None:
            raise ValueError("Data object missing sample_id; cannot do homology split.")
        if sid not in member_to_cid:
            missing += 1
            continue
        cid = member_to_cid[sid]
        if cid in train_c:
            train.append(d)
        elif cid in val_c:
            val.append(d)
        else:
            test.append(d)
    return train, val, test, missing

def load_dataset_list(path, label):
    if not path:
        return []
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} dataset not found: {path}")
    print(f"Loading {label} dataset from {path}...")
    return torch.load(path, weights_only=False)

def assign_seq_embeddings(data_list, embeddings_map, seq_embedding_dim):
    matched_count = 0
    unmatched_samples = []
    for data in data_list:
        key = None
        embedding = None
        key_type = None

        if hasattr(data, 'sample_id') and data.sample_id is not None:
            key = str(data.sample_id)
            key_type = 'sample_id'
        elif hasattr(data, 'pdb_id') and data.pdb_id is not None:
            key = str(data.pdb_id)
            key_type = 'pdb_id'
            if key not in embeddings_map and '.' in key:
                key = key.split('.')[0]
        elif hasattr(data, 'uniprot_id') and data.uniprot_id is not None:
            key = str(data.uniprot_id)
            key_type = 'uniprot_id'

        if key and key in embeddings_map:
            embedding = embeddings_map[key]
        elif key:
            if len(unmatched_samples) < 5:
                unmatched_samples.append((key, key_type))

        if embedding is not None:
            if not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding, dtype=torch.float)
            if seq_embedding_dim == 0:
                seq_embedding_dim = embedding.shape[0]
                print(f"Sequence embedding dimension: {seq_embedding_dim}")
            data.seq_embedding = embedding.unsqueeze(0)
            matched_count += 1

    return seq_embedding_dim, matched_count, unmatched_samples

def remove_string_metadata(data_list):
    for data in data_list:
        if hasattr(data, 'ec'):
            delattr(data, 'ec')
        if hasattr(data, 'pdb_id'):
            delattr(data, 'pdb_id')
        if hasattr(data, 'sample_id'):
            delattr(data, 'sample_id')
        if hasattr(data, 'uniprot_id'):
            delattr(data, 'uniprot_id')

def fill_missing_seq_embeddings(data_list, seq_embedding_dim):
    if seq_embedding_dim <= 0:
        return 0
    missing_count = 0
    for data in data_list:
        if not hasattr(data, 'seq_embedding'):
            data.seq_embedding = torch.zeros((1, seq_embedding_dim), dtype=torch.float)
            missing_count += 1
    return missing_count

def train(args):
    dataset_path = args.dataset
    if args.train_dataset:
        dataset_path = args.train_dataset
    save_dir = args.save_dir
    batch_size = 32 # defaults if not in args, but usually controlled by loop or constant
    lr = 5e-4  # ⚡ 对于大模型，使用更小的学习率
    max_epochs = 500  # 默认训练轮数

    split_strategy = "random"
    if args.train_dataset:
        split_strategy = "pre_split"
    elif args.cluster_tsv:
        split_strategy = "cluster"
    
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
    
    # 处理 epochs 参数（支持 --max_epochs 和 --epochs）
    if args.max_epochs is not None:
        max_epochs = args.max_epochs
    elif args.epochs is not None:
        max_epochs = args.epochs
    
    # 处理 loss_type 参数（向后兼容）
    if args.loss_type is not None:
        print(f"⚠️  警告: --loss_type 已弃用，请使用 --loss")
        args.loss = args.loss_type
    
    # ========== 实验命名和目录管理 ==========
    # 获取实验名称和 run_id（通过 save_metadata，它会自动管理 experiments/ 目录）
    # 如果用户没有指定 exp_name，使用默认值
    exp_name = getattr(args, 'exp_name', 'kcat_default')
    model_type = getattr(args, 'model_type', 'PocketGNNKcatOnly')  # 向后兼容

    # ⚡ 简化目录结构：统一使用 experiments/exp_name/run_id 作为 save_dir
    # 这样所有文件（模型、图像、元数据）都在一个地方，不需要 outputs/ 和 experiments/ 两个目录
    # 先调用 save_metadata 创建实验记录
    exp_name, run_id = save_metadata(
        save_dir=save_dir,  # 临时传入，稍后会更新
        dataset_path=dataset_path,
        exp_name=exp_name,
        graph_builder_version='enhanced_builder',
        gnn_model_version=model_type,  # 使用实际的模型类型
        comments=f'Model={model_type}, pooling={args.pooling_type}, seq_emb={args.use_seq_embedding}, loss={args.loss}, wd={args.weight_decay}'
    )
    
    # 从 experiments 目录读取实际的 run_dir 路径
    from pathlib import Path
    from metadata_utils import ExperimentTracker
    tracker = ExperimentTracker()
    run_dir = tracker.base_dir / exp_name / run_id
    
    # 更新 save_dir 为统一的 experiments 目录（替代原来的 outputs/xxx）
    save_dir = str(run_dir)
    print(f"📁 统一目录: {save_dir} (包含模型、图像、元数据，替代 outputs/ 和 experiments/ 分离)")
    
    # 统一命名规则：wandb run name 格式: {exp_name}_{run_id}，例如 "kcat_attn_v1_run_01"
    wandb_run_name = f"{exp_name}_{run_id}"
    
    # 定义 device（在 wandb_config 之前）
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # ========== 初始化 WandB ==========
    # 登录 wandb（使用提供的 API key）
    wandb.login(key="46dbe55e52d029976ffa0e29c90f0d32410e1504")
    
    # 创建 wandb config 字典
    dataset_name = os.path.basename(dataset_path) if dataset_path else "custom_split"
    wandb_config = {
        "dataset": dataset_name,
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
        "model_type": model_type,  # 记录模型类型
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "heads": heads,
        "split_strategy": split_strategy,
        "train_dataset": args.train_dataset,
        "val_dataset": args.val_dataset,
        "test_dataset": args.test_dataset,
        "cluster_tsv": args.cluster_tsv,
        "split_ratios": args.split_ratios,
        "split_seed": args.split_seed,
        "exp_name": exp_name,
        "run_id": run_id,
        "save_dir": save_dir,
    }

    # 创建 tags 用于在 wandb 界面快速筛选
    # 基于关键参数生成 tags，方便在 wandb 中快速找到对应的 run
    tags = [
        f"model_{model_type}",
        f"pooling_{args.pooling_type}",
        f"scheduler_{args.scheduler}",
        f"loss_{args.loss}",
    ]
    tags.append(f"split_{split_strategy}")
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
        if dataset_path:
            f.write(f"Dataset Name: {os.path.basename(dataset_path)}\n")
        f.write(f"Split Strategy: {split_strategy}\n")
        f.write(f"Train Dataset: {args.train_dataset}\n")
        f.write(f"Val Dataset: {args.val_dataset}\n")
        f.write(f"Test Dataset: {args.test_dataset}\n")
        f.write(f"Cluster TSV: {args.cluster_tsv}\n")
        f.write(f"Split Ratios: {args.split_ratios}\n")
        f.write(f"Split Seed: {args.split_seed}\n")
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

    train_list = []
    val_list = []
    test_list = []

    if args.train_dataset:
        train_list = load_dataset_list(args.train_dataset, "train")
        val_list = load_dataset_list(args.val_dataset, "val") if args.val_dataset else []
        test_list = load_dataset_list(args.test_dataset, "test") if args.test_dataset else []
    else:
        if not dataset_path:
            raise ValueError("No dataset provided. Use --dataset or --train_dataset.")
        try:
            data_list = load_dataset_list(dataset_path, "full")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return

        ratios = parse_split_ratios(args.split_ratios)
        if args.cluster_tsv:
            if len(ratios) == 2:
                ratios = (ratios[0], ratios[1] / 2.0, ratios[1] / 2.0)
                print(f"Cluster split: expanding ratios to train/val/test = {ratios}")
            member_to_rep = read_mmseqs_tsv(args.cluster_tsv)
            member_to_cid, rep_to_cid = make_cluster_ids(member_to_rep)
            train_c, val_c, test_c = split_clusters(rep_to_cid, seed=args.split_seed, ratios=ratios)
            train_list, val_list, test_list, missing = split_dataset_by_clusters(
                data_list, member_to_cid, train_c, val_c, test_c
            )
            if missing > 0:
                print(f"Warning: {missing} samples missing in cluster table; dropped to avoid leakage.")
        else:
            rng = np.random.default_rng(args.split_seed)
            rng.shuffle(data_list)
            train_ratio = ratios[0]
            val_ratio = ratios[1]
            test_ratio = ratios[2] if len(ratios) == 3 else 0.0
            n = len(data_list)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            train_list = data_list[:train_end]
            val_list = data_list[train_end:val_end]
            test_list = data_list[val_end:] if test_ratio > 0 else []

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
        sample_pool = (train_list + val_list + test_list)
        for data in sample_pool[:5]:
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
        for label, data_list in (("train", train_list), ("val", val_list), ("test", test_list)):
            if not data_list:
                continue
            seq_embedding_dim, count, unmatched = assign_seq_embeddings(
                data_list, embeddings_map, seq_embedding_dim
            )
            matched_count += count
            if unmatched:
                unmatched_samples.extend(unmatched)

        total_samples = len(train_list) + len(val_list) + len(test_list)
        print(f"Matched embeddings for {matched_count}/{total_samples} samples.")
        if unmatched_samples:
            print(f"Sample unmatched keys: {unmatched_samples[:5]}")

        if matched_count == 0:
            print("Warning: No embeddings matched! Disabling sequence embedding.")
            args.use_seq_embedding = False
            seq_embedding_dim = 0
    
    # 移除字符串类型的元数据属性，避免 DataLoader collate 时出错
    # PyTorch Geometric 的 Batch.from_data_list 无法处理字符串属性
    print("Removing string metadata attributes to avoid collate errors...")
    remove_string_metadata(train_list)
    remove_string_metadata(val_list)
    remove_string_metadata(test_list)
    
    # 如果启用了seq_embedding但某些样本没有匹配到，用零向量填充
    if args.use_seq_embedding and seq_embedding_dim > 0:
        missing_count = 0
        missing_count += fill_missing_seq_embeddings(train_list, seq_embedding_dim)
        missing_count += fill_missing_seq_embeddings(val_list, seq_embedding_dim)
        missing_count += fill_missing_seq_embeddings(test_list, seq_embedding_dim)
        if missing_count > 0:
            print(f"Warning: {missing_count} samples missing seq_embedding, filled with zeros.")

    # device 已在前面定义（用于 wandb_config）
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(save_dir)

    # === Load dataset ===
    if not train_list or not val_list:
        raise ValueError("Train/val split resulted in empty set. Check split ratios and data.")
    print(train_list[0])  # 打印第一个图数据

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size)
    test_loader = DataLoader(test_list, batch_size=batch_size) if test_list else None

    # === Initialize model ===
    node_input_dim = train_list[0].x.shape[1] #default 52
    edge_input_dim = train_list[0].edge_attr.shape[1]  # 现在应该是24维
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
        # 解析quantile权重（如果提供）
        if hasattr(args, 'quantile_weights') and args.quantile_weights is not None:
            quantile_weights = [float(w) for w in args.quantile_weights.split(',')]
            assert len(quantile_weights) == len(quantiles), "Quantile weights must match quantiles length"
            print(f"Using quantile weights: {quantile_weights}")
        else:
            quantile_weights = None
    else:
        quantiles = None
        quantile_weights = None

    # ========== 模型选择与初始化 ==========
    model_type = getattr(args, 'model_type', 'PocketGNNKcatOnly')  # 向后兼容

    if model_type == 'PHPTransformer':
        print(f"🔬 Using PHPTransformer (Physics-informed Hierarchical Pocket Transformer)")
        print(f"   - Dual-stream GNN (Geometry + Electronic)")
        print(f"   - Cross-Attention Fusion")
        print(f"   - Residual Graph Transformer Blocks")
        model = MD.PHPTransformer(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            use_seq_embedding=args.use_seq_embedding,
            seq_embedding_dim=seq_embedding_dim if args.use_seq_embedding else 0,
            pooling_type='mean'  # PHPTransformer目前只支持mean pooling
        ).to(device)
    else:
        print(f"📊 Using {model_type} (Baseline single-stream GAT)")
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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience, verbose=True)
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
                loss = quantile_loss(out, log_y, quantiles=quantiles, weights=quantile_weights)
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
                    loss = quantile_loss(out, log_y, quantiles=quantiles, weights=quantile_weights)
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
        
        # ⚡ 添加详细的置信区间分析可视化
        print("\n" + "=" * 60)
        print("📊 置信区间质量分析")
        print("=" * 60)
        
        # 计算详细统计
        in_interval = (y_true_flat >= y_pred_low) & (y_true_flat <= y_pred_high)
        out_below = y_true_flat < y_pred_low  # 真实值低于下界
        out_above = y_true_flat > y_pred_high  # 真实值高于上界
        interval_widths = y_pred_high - y_pred_low
        
        print(f"✅ Coverage: {quantile_metrics_end['coverage']:.1%} (理想值: {(quantiles[2] - quantiles[0])*100:.1f}%)")
        print(f"   落在区间内: {in_interval.sum()}/{len(y_true_flat)} 个样本")
        print(f"   真实值 < 预测下界 (模型过于乐观，预测偏高): {out_below.sum()} 个样本 ({out_below.sum()/len(y_true_flat):.1%})")
        print(f"   真实值 > 预测上界 (模型过于保守，预测偏低): {out_above.sum()} 个样本 ({out_above.sum()/len(y_true_flat):.1%})")
        print(f"\n📏 区间宽度统计:")
        print(f"   平均宽度: {interval_widths.mean():.3f}")
        print(f"   中位数宽度: {np.median(interval_widths):.3f}")
        print(f"   最小宽度: {interval_widths.min():.3f}")
        print(f"   最大宽度: {interval_widths.max():.3f}")
        print(f"   标准差: {interval_widths.std():.3f}")
        
        # 按真实值范围分组分析 coverage
        print(f"\n📈 按真实值范围分组的 Coverage:")
        true_ranges = [
            (y_true_flat.min(), np.percentile(y_true_flat, 25), "低值 (0-25%)"),
            (np.percentile(y_true_flat, 25), np.percentile(y_true_flat, 75), "中值 (25-75%)"),
            (np.percentile(y_true_flat, 75), y_true_flat.max(), "高值 (75-100%)")
        ]
        for low, high, label in true_ranges:
            mask = (y_true_flat >= low) & (y_true_flat < high)
            if mask.sum() > 0:
                group_coverage = in_interval[mask].mean()
                print(f"   {label}: {group_coverage:.1%} ({in_interval[mask].sum()}/{mask.sum()})")
        
        print(f"\n💡 图例说明:")
        print(f"   - 红色向下三角形 (左上角): 真实值很小，但模型预测值偏大 → 模型过于乐观")
        print(f"   - 橙色向上三角形 (右下角): 真实值很大，但模型预测值偏小 → 模型过于保守")
        print(f"   - 蓝色圆点: 真实值落在预测区间内的样本")
        
        # 1. 区间宽度分布直方图
        plt.figure(figsize=(10, 6))
        plt.hist(interval_widths, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(interval_widths.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {interval_widths.mean():.3f}')
        plt.xlabel('Interval Width (log10 kcat)')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Prediction Interval Widths\nMean: {interval_widths.mean():.3f}, Median: {np.median(interval_widths):.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'interval_width_distribution.png'), dpi=150)
        plt.close()
        print(f"   ✅ 区间宽度分布图已保存")
        
        # 2. Coverage vs Interval Width 散点图（按样本）
        plt.figure(figsize=(10, 6))
        colors = ['green' if in_int else 'red' for in_int in in_interval]
        plt.scatter(interval_widths, np.abs(y_true_flat - y_pred_median), 
                   c=colors, alpha=0.5, s=20)
        plt.xlabel('Interval Width')
        plt.ylabel('|True - Median Prediction|')
        plt.title('Coverage Analysis: Interval Width vs Prediction Error\n(Green: in interval, Red: out of interval)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'coverage_analysis.png'), dpi=150)
        plt.close()
        print(f"   ✅ Coverage 分析图已保存")
        
        # 3. 改进的置信区间图：标记超出区间的样本
        plt.figure(figsize=(12, 8))
        # 绘制置信区间（阴影区域）
        plt.fill_between(y_true_sorted, y_pred_low_sorted, y_pred_high_sorted, 
                        alpha=0.3, color='blue', label=f'90% Confidence Interval')
        
        # 绘制中位数预测
        plt.scatter(y_true_flat, y_pred_median, alpha=0.4, s=15, 
                   c='blue', label='Median Prediction (in interval)')
        
        # 标记超出区间的样本
        # 注意：在散点图中，X轴是真实值，Y轴是预测中位数
        # 蓝色阴影区域：对于每个真实值(X)，从预测下界(Y)到预测上界(Y)
        # 红色点：真实值 < 预测下界，意味着真实值在蓝色区间下方
        # 但红色点的Y坐标是预测中位数，所以如果预测中位数在蓝色区间中上方，红色点也会在那里
        if out_below.sum() > 0:
            # 真实值 < 预测下界：说明模型预测的下界太高了（过于乐观）
            # 红色点的位置：(真实值, 预测中位数)
            # 如果预测中位数在蓝色区间中上方，红色点也会在那里
            plt.scatter(y_true_flat[out_below], y_pred_median[out_below], 
                       alpha=0.8, s=30, c='red', marker='v', 
                       label=f'True < Lower Bound (真实值在区间下方, {out_below.sum()})')
            # 用垂直线标记真实值的位置（在X轴上）
            for i in range(min(50, out_below.sum())):  # 只显示前50个
                idx = np.where(out_below)[0][i]
                # 从真实值位置（完美预测线）到预测中位数
                plt.plot([y_true_flat[idx], y_true_flat[idx]], 
                        [y_true_flat[idx], y_pred_median[idx]], 
                        'r-', alpha=0.2, linewidth=1)
        
        if out_above.sum() > 0:
            # 真实值 > 预测上界：说明模型预测的上界太低了（过于保守）
            plt.scatter(y_true_flat[out_above], y_pred_median[out_above], 
                       alpha=0.8, s=30, c='orange', marker='^', 
                       label=f'True > Upper Bound (真实值在区间上方, {out_above.sum()})')
            # 用垂直线标记真实值的位置
            for i in range(min(50, out_above.sum())):
                idx = np.where(out_above)[0][i]
                plt.plot([y_true_flat[idx], y_true_flat[idx]], 
                        [y_true_flat[idx], y_pred_median[idx]], 
                        'orange', alpha=0.2, linewidth=1)
        
        # 绘制完美预测线
        plt.plot([y_true_flat.min(), y_true_flat.max()], 
                 [y_true_flat.min(), y_true_flat.max()], 'k--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('True kcat (log10)')
        plt.ylabel('Predicted kcat (log10)')
        plt.title(f'Confidence Interval Quality Analysis\nR²={r2_kcat:.3f}, Coverage={quantile_metrics_end["coverage"]:.1%} (Target: {(quantiles[2]-quantiles[0])*100:.1f}%)')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'kcat_prediction_with_ci_detailed.png'), dpi=150)
        plt.close()
        print(f"   ✅ 详细置信区间图已保存")
        
        # 4. 区间宽度 vs 真实值的关系
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true_flat, interval_widths, alpha=0.5, s=20, c=in_interval, cmap='RdYlGn')
        plt.xlabel('True kcat (log10)')
        plt.ylabel('Interval Width')
        plt.title('Interval Width vs True Value\n(Green: in interval, Red: out of interval)')
        plt.colorbar(label='In Interval')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'interval_width_vs_true.png'), dpi=150)
        plt.close()
        print(f"   ✅ 区间宽度 vs 真实值图已保存")
        
        print("=" * 60)
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
    if use_quantile:
        pred_vals = all_y_pred[:, 1].flatten()  # 使用中位数
    else:
        pred_vals = all_y_pred.flatten()
    sns.kdeplot(x=true_vals, y=pred_vals, cmap="viridis", fill=True, thresh=0.05)
    plt.plot([true_vals.min(), true_vals.max()], 
             [true_vals.min(), true_vals.max()], 'r--')
    plt.xlabel('True kcat (log10)')
    plt.ylabel('Predicted kcat (log10)')
    plt.title('kcat: Density Plot')
    plt.savefig(os.path.join(save_dir, 'kcat_density.png'))
    plt.close()

    # 5. Test set evaluation (optional)
    test_metrics = None
    if test_loader is not None:
        model.eval()
        test_true = []
        test_pred = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                batch_size = batch.num_graphs
                if batch.y.shape[0] == batch_size:
                    log_y = batch.y.reshape(batch_size, 1)
                else:
                    y_reshaped = batch.y.reshape(batch_size, 2)
                    log_y = y_reshaped[:, 0:1]
                out = model(batch)
                out = torch.clamp(out, min=-10.0, max=10.0)
                test_true.append(log_y.cpu())
                test_pred.append(out.cpu())

        if test_true and test_pred:
            test_true = torch.cat(test_true, dim=0)
            test_pred = torch.cat(test_pred, dim=0)
            if use_quantile:
                test_pred_median = test_pred[:, 1:2]
                test_metrics = compute_metrics(test_true, test_pred_median)
                test_metrics.update(compute_quantile_metrics(test_pred, test_true, quantiles=quantiles))
                test_pred_plot = test_pred_median.numpy().flatten()
            else:
                test_metrics = compute_metrics(test_true, test_pred)
                test_pred_plot = test_pred.numpy().flatten()

            test_true_plot = test_true.numpy().flatten()
            plt.figure(figsize=(8, 6))
            plt.scatter(test_true_plot, test_pred_plot, alpha=0.6)
            plt.plot([test_true_plot.min(), test_true_plot.max()],
                     [test_true_plot.min(), test_true_plot.max()], 'r--')
            plt.xlabel('True kcat (log10)')
            plt.ylabel('Predicted kcat (log10)')
            plt.title(f'Test: True vs Predicted (R² = {test_metrics["R2"]:.3f})')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'kcat_prediction_scatter_test.png'))
            plt.close()

            with open(os.path.join(save_dir, 'test_metrics.json'), "w") as f:
                json.dump(test_metrics, f, indent=2)

            wandb.log({
                "test/R2": test_metrics["R2"],
                "test/Pearson": test_metrics["Pearson"],
                "test/MAE": test_metrics["MAE"],
                "test/RMSE": test_metrics["RMSE"]
            })
    
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
    if test_metrics:
        final_log.update({
            "final/test_r2": test_metrics.get("R2"),
            "final/test_pearson": test_metrics.get("Pearson"),
            "final/test_mae": test_metrics.get("MAE"),
            "final/test_rmse": test_metrics.get("RMSE")
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
    parser.add_argument('--train_dataset', type=str, default=None, help='Optional pre-split train .pt dataset')
    parser.add_argument('--val_dataset', type=str, default=None, help='Optional pre-split val .pt dataset')
    parser.add_argument('--test_dataset', type=str, default=None, help='Optional pre-split test .pt dataset')
    parser.add_argument('--cluster_tsv', type=str, default=None, help='MMseqs2 cluster TSV for homology-aware split')
    parser.add_argument('--split_ratios', type=str, default="0.8,0.2", help='Split ratios (train,val[,test]), must sum to 1.0')
    parser.add_argument('--split_seed', type=int, default=42, help='Random seed for data splitting')
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
    parser.add_argument('--loss_type', type=str, default=None, help='Deprecated: use --loss instead')
    parser.add_argument('--quantiles', type=str, default='0.05,0.5,0.95', help='Quantile levels for quantile loss (comma-separated, e.g., 0.05,0.5,0.95)')
    parser.add_argument('--quantile_weights', type=str, default=None, help='Weights for each quantile (comma-separated, e.g., 0.2,0.6,0.2). Higher weight on median reduces conservatism.')
    parser.add_argument('--scheduler', type=str, default='none', choices=['none', 'plateau', 'cosine'], help='LR Scheduler')
    parser.add_argument('--patience', type=int, default=10, help='Patience for ReduceLROnPlateau scheduler')
    parser.add_argument('--max_epochs', type=int, default=None, help='Maximum number of training epochs')
    parser.add_argument('--epochs', type=int, default=None, help='Alias for --max_epochs')
    
    # Phase 2: Pooling
    parser.add_argument('--pooling_type', type=str, default='mean', choices=['mean', 'global_attention', 'set2set'], help='Graph pooling type: mean, global_attention, or set2set')
    
    # Phase 3: ESM Sequence Embedding
    parser.add_argument('--use_seq_embedding', action='store_true', help='Enable ESM sequence embedding (Late Fusion)')
    parser.add_argument('--seq_embedding_path', type=str, default='data/esm_embeddings.pt', help='Path to ESM embeddings dictionary')
    
    # Model Architecture Hyperparameters
    parser.add_argument('--model_type', type=str, default='PocketGNNKcatOnly',
                       choices=['PocketGNNKcatOnly', 'PHPTransformer'],
                       help='Model architecture: PocketGNNKcatOnly (baseline) or PHPTransformer (dual-stream hierarchical)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
    args = parser.parse_args()

    train(args)
