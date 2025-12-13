import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 添加项目根目录到 Python 路径，支持从根目录运行 python src/train.py
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 也添加 src 目录，兼容两种运行方式

import GNN_model as MD
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')  # 确保在没有GUI的环境中使用
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from metadata_utils import update_training_results
import wandb

def compute_metrics(y_true_log, y_pred_log):
    y_true_log = y_true_log.numpy()
    y_pred_log = y_pred_log.numpy()
    
    # 计算 Pearson 相关系数，处理常数输入的情况
    try:
        pearson_val = pearsonr(y_true_log.flatten(), y_pred_log.flatten())[0]
        if np.isnan(pearson_val):
            pearson_val = 0.0  # 如果预测值或真实值是常数，相关系数为 0
    except (ValueError, RuntimeWarning):
        pearson_val = 0.0
    
    return {
        'MAE': mean_absolute_error(y_true_log, y_pred_log),
        'RMSE': np.sqrt(mean_squared_error(y_true_log, y_pred_log)),
        'R2': r2_score(y_true_log, y_pred_log),
        'Pearson': pearson_val
    }

def train(dataset_path, save_dir="outputs", batch_size=32, lr=1e-3, max_epochs=500, label_permutation=False, frozen_encoder=False, load_checkpoint=None, exp_name="kcat_attn_v1"):
    from datetime import datetime
    dataset = torch.load(dataset_path, weights_only=False)
    
    # 检查数据集是否包含 NaN
    for data in dataset:
        if torch.isnan(data.x).any() or torch.isnan(data.y).any():
            raise ValueError("数据集中包含 NaN 值")
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(save_dir)
    
    # === 构建 W&B 元数据 ===
    # A. Group: 用 exp_name 把同一条实验主线串起来（最重要）
    group = exp_name
    
    # B. Tags: 标记关键信息（便于在 W&B 界面筛选）
    tags = []
    if label_permutation:
        tags.append("diag")
        tags.append("label_permutation")
    if frozen_encoder:
        tags.append("diag")
        tags.append("frozen_encoder")
    if load_checkpoint:
        tags.append("from_checkpoint")
        # 从 checkpoint 路径提取关键信息
        ckpt_basename = os.path.basename(load_checkpoint)
        # 尝试提取 epoch 号或其他标识
        if "epoch" in ckpt_basename.lower():
            tags.append(f"ckpt={ckpt_basename}")
        else:
            tags.append(f"ckpt={ckpt_basename[:20]}")  # 截断过长的路径
    
    # 添加超参数标签（便于筛选）
    tags.append(f"lr={lr}")
    tags.append(f"bs={batch_size}")
    
    # C. Name: 语义-超参数-时间戳（格式：diag/frozen_encoder-lr3e-3-ckpt130530-20251213_140046）
    base_name = os.path.basename(save_dir)
    
    # 提取时间戳（假设格式为 prefix_YYYYMMDD_HHMMSS 或 YYYYMMDD_HHMMSS）
    timestamp = None
    if "_" in base_name:
        parts = base_name.split("_")
        # 检查最后两部分是否是时间戳格式（YYYYMMDD_HHMMSS）
        if len(parts) >= 2:
            last_two = "_".join(parts[-2:])
            if len(last_two) == 15 and last_two.replace("_", "").isdigit():  # YYYYMMDD_HHMMSS
                timestamp = last_two
                semantic_prefix = "_".join(parts[:-2]) if len(parts) > 2 else exp_name
            else:
                timestamp = parts[-1] if parts[-1] else datetime.now().strftime("%Y%m%d_%H%M%S")
                semantic_prefix = "_".join(parts[:-1]) if len(parts) > 1 else exp_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            semantic_prefix = exp_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        semantic_prefix = exp_name
    
    # 构建语义前缀（基于 exp_name 和诊断标志）
    name_parts = []
    if label_permutation:
        name_parts.append("perm")
    if frozen_encoder:
        name_parts.append("frozen")
    if load_checkpoint:
        # 从 checkpoint 路径提取简短标识
        ckpt_short = os.path.basename(load_checkpoint).replace(".pt", "").replace("best_model", "ckpt")
        if len(ckpt_short) > 15:
            ckpt_short = ckpt_short[:15]
        name_parts.append(ckpt_short)
    
    # 如果没有任何特殊标志，使用 exp_name
    if not name_parts:
        semantic_str = semantic_prefix
    else:
        semantic_str = f"{semantic_prefix}/" + "-".join(name_parts) if semantic_prefix else "-".join(name_parts)
    
    # 格式化学习率（使其更易读）
    if lr >= 1:
        lr_str = f"{lr:.0f}"
    elif lr >= 0.01:
        lr_str = f"{lr:.2f}".rstrip("0").rstrip(".")
    else:
        # 科学计数法，但格式化为更易读的形式
        lr_str = f"{lr:.0e}".replace("e-0", "e-").replace("e+", "e")
    
    # 最终 name: 语义-超参数-时间戳
    wandb_name = f"{semantic_str}-lr{lr_str}-{timestamp}"
    
    # D. Notes: 写一句人话（W&B 页面里一眼就懂）
    notes_parts = []
    if label_permutation:
        notes_parts.append("Label Permutation Test (diagnostic)")
    if frozen_encoder:
        notes_parts.append("Frozen Encoder Test (diagnostic)")
        if load_checkpoint:
            notes_parts.append(f"from {os.path.basename(load_checkpoint)}")
    if not notes_parts:
        notes_parts.append(f"Baseline: {exp_name}")
    notes = " | ".join(notes_parts)
    
    # 初始化 wandb
    wandb.login(key="46dbe55e52d029976ffa0e29c90f0d32410e1504")
    wandb_config = {
        "dataset": os.path.basename(dataset_path),
        "batch_size": batch_size,
        "lr": lr,
        "max_epochs": max_epochs,
        "device": str(device),
        "label_permutation": label_permutation,
        "frozen_encoder": frozen_encoder,
    }
    if load_checkpoint is not None:
        wandb_config["load_checkpoint"] = load_checkpoint
    
    # 打印 W&B 配置信息（便于调试和确认）
    print(f"📊 W&B 配置:")
    print(f"   Group: {group}")
    print(f"   Name: {wandb_name}")
    print(f"   Tags: {tags}")
    print(f"   Notes: {notes}")
    
    wandb.init(
        project="enzyme-kcat-prediction",
        name=wandb_name,
        group=group,
        tags=tags,
        notes=notes,
        config=wandb_config
    )

    # === Load dataset ===
    data_list = torch.load(dataset_path, weights_only=False)  # List[Data]
    print(data_list[0])  # 打印第一个图数据
    actual_num_atom_types = data_list[0].x.shape[1]
    
    # === Label Permutation Test (诊断测试) ===
    # 在数据加载后、数据集划分前进行 label 随机置换
    # 用于诊断模型是否真正使用图表示，而不是 pipeline bug
    if label_permutation:
        print("⚠️  启用 Label Permutation Test - 将对标签进行随机置换")
        print("   这是诊断测试，用于验证模型是否真正使用图表示")
        # 收集所有标签（保持原始形状）
        all_y_list = [d.y.clone() for d in data_list]
        # 生成随机置换索引（对样本进行置换，而不是对标签值）
        perm = torch.randperm(len(data_list))
        # 对每个数据样本的标签进行置换
        for i, d in enumerate(data_list):
            d.y = all_y_list[perm[i]].clone()
        print(f"✅ 标签置换完成，共 {len(data_list)} 个样本")
        print("   预期结果：如果模型真正使用图表示，val_pearson ≈ 0, val_r2 ≈ 0")
    
    # === 数据集划分（使用固定随机种子确保一致性） ===
    # 设置随机种子，确保每次运行的数据集划分一致
    # 这对于 Frozen Encoder Test 很重要，需要与 baseline 使用相同的验证集
    np.random.seed(42)
    np.random.shuffle(data_list)
    
    # === 清理 Data 对象：移除字符串属性（PyG DataLoader 无法 collate 字符串） ===
    # PyG 的 DataLoader 会尝试将所有属性 collate 成 tensor，但字符串无法转换
    # 需要保留的属性：x, edge_index, edge_attr, pos, y, batch, temperature (如果是 tensor)
    # 需要移除的属性：pdb_id, sample_id, ec (字符串或非 tensor 类型)
    print("🧹 清理 Data 对象：移除字符串属性以兼容 DataLoader...")
    total_removed = 0
    sample_keys_removed = set()
    for data in data_list:
        # 获取所有属性名（keys 是方法，需要调用）
        keys_to_remove = []
        for key in data.keys():
            value = getattr(data, key)
            # 如果不是 tensor 类型，需要移除（字符串、整数等）
            if not isinstance(value, torch.Tensor):
                keys_to_remove.append(key)
                sample_keys_removed.add(key)
        
        # 移除非 tensor 属性
        for key in keys_to_remove:
            delattr(data, key)
            total_removed += 1
    
    if total_removed > 0:
        print(f"✅ 清理完成，共移除了 {total_removed} 个非 tensor 属性")
        print(f"   移除的属性包括: {', '.join(sorted(sample_keys_removed))}")
    else:
        print("✅ 数据已清理，无需移除属性")
    
    split = int(0.8 * len(data_list))
    train_loader = DataLoader(data_list[:split], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data_list[split:], batch_size=batch_size)
    print(f"📊 数据集划分：训练集 {len(data_list[:split])} 个样本，验证集 {len(data_list[split:])} 个样本")

    print(data_list[0].temperature)
    # === Initialize model ===
    node_input_dim = data_list[0].x.shape[1] #default 52
    edge_input_dim = data_list[0].edge_attr.shape[1]  # 现在应该是24维
    print(f"Node input dim: {node_input_dim}, Edge input dim: {edge_input_dim}")
    
    # 使用新的kcat专用模型
    hidden_dim = 128  # 模型隐藏维度
    num_layers = 3    # 层数
    heads = 4         # 注意力头数
    dropout = 0.1     # Dropout 概率

    # 更新 wandb config 包含模型参数
    wandb.config.update({
        "node_input_dim": node_input_dim,
        "edge_input_dim": edge_input_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "heads": heads,
        "dropout": dropout,
    })

    model = MD.PocketGNNKcatOnly(
        node_input_dim=node_input_dim,
        edge_input_dim=edge_input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
        dropout=dropout
    ).to(device)
    
    # === 加载预训练权重（如果指定） ===
    if load_checkpoint is not None:
        if not os.path.exists(load_checkpoint):
            raise FileNotFoundError(f"❌ 找不到 checkpoint 文件: {load_checkpoint}")
        print(f"📥 加载预训练权重: {load_checkpoint}")
        model.load_state_dict(torch.load(load_checkpoint, map_location=device, weights_only=False))
        print("✅ 预训练权重加载完成")
    
    # === Frozen Encoder Test (诊断测试) ===
    # 冻结 GNN encoder，只训练最后的 MLP regression head
    # 用于诊断 encoder 是否已经饱和
    # 注意：如果启用 frozen_encoder，必须先加载预训练权重
    if frozen_encoder:
        if load_checkpoint is None:
            raise ValueError("❌ 错误：Frozen Encoder Test 需要先加载预训练权重！请使用 --load_checkpoint 参数")
        
        print("⚠️  启用 Frozen Encoder Test - 将冻结 GNN encoder，只训练 MLP regression head")
        print("   这是诊断测试，用于验证 encoder 是否已经饱和")
        
        # === 关键步骤：重置 MLP head 参数 ===
        # 为了正确测试 encoder 表示是否 linearly-usable，需要从头训练 head
        # 而不是继续使用已经训练好的 head 权重
        print("🔄 重置 MLP head 参数（从头开始训练）...")
        for name, module in model.named_modules():
            if name.startswith("mlp"):
                if isinstance(module, nn.Linear):
                    # 重新初始化 Linear 层的权重和偏置
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
        print("✅ MLP head 参数已重置为随机初始化")
        
        # 冻结 encoder，只训练 MLP head
        frozen_params = 0
        trainable_params = 0
        for name, param in model.named_parameters():
            # 使用 startswith 而不是 in，更精确地匹配 mlp 参数
            if not name.startswith("mlp"):
                param.requires_grad = False
                frozen_params += param.numel()
            else:
                trainable_params += param.numel()
        print(f"✅ 参数冻结完成：冻结 {frozen_params:,} 个参数，可训练 {trainable_params:,} 个参数（MLP head）")
        print("   预期结果：")
        print("   - 如果性能 ≈ 原模型（Pearson ≈ 0.60）→ encoder 表示已饱和（linearly-usable）")
        print("   - 如果性能明显下降（Pearson < 0.3）→ encoder 仍需端到端协同优化")
        print("   - 如果性能 ≈ 0 → 实现有 bug（如没正确加载权重）")
        if lr <= 1e-3:
            print(f"   💡 建议：frozen encoder 时可以使用稍大的学习率（如 3e-3），当前 lr={lr}")
    
    # 创建 optimizer：如果冻结了 encoder，只优化可训练的参数
    if frozen_encoder:
        trainable_params_list = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(trainable_params_list, lr=lr)
        print(f"✅ Optimizer 已创建，只优化可训练参数（MLP head）")
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
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
                    # 注意：pdb_id 等字符串属性已在数据清理时移除，无法访问
                    print(f"❌ NaN 输出，batch 索引: {batch.batch.unique()}")
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
        # === Logging ===
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("R2/val", metrics['R2'], epoch)
        writer.add_scalar("Pearson/val", metrics['Pearson'], epoch)
        
        # wandb logging
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_r2": metrics['R2'],
            "val_pearson": metrics['Pearson'],
            "val_mae": metrics['MAE'],
            "val_rmse": metrics['RMSE'],
        })

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | R2: {metrics['R2']:.3f}")

        # === Save best model ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(save_dir, "best_model.pt")
            torch.save(model.state_dict(), model_path)
            # 保存最佳模型到 wandb
            wandb.save(model_path)

    writer.close()
    wandb.log({"best_val_loss": best_val_loss})
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
                # 注意：pdb_id 等字符串属性已在数据清理时移除，无法访问
                print(f"❌ NaN 输出，batch 索引: {batch.batch.unique()}")
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
    scatter_path = os.path.join(save_dir, 'kcat_prediction_scatter.png')
    plt.savefig(scatter_path)
    plt.close()
    # 上传散点图到 wandb
    wandb.log({"kcat_scatter": wandb.Image(scatter_path)})
    
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
    density_path = os.path.join(save_dir, 'kcat_density.png')
    plt.savefig(density_path)
    plt.close()
    # 上传密度图到 wandb
    wandb.log({"kcat_density": wandb.Image(density_path)})
    
    # 保存最终指标到文件
    metrics_df = pd.DataFrame({
        'Epoch': range(1, max_epochs + 1),
        'Train_Loss': train_loss_history,
        'Val_Loss': val_loss_history,
        'R2': r2_history,
        'Pearson': pearson_history
    })
    metrics_csv_path = os.path.join(save_dir, 'training_metrics.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    # 上传指标表格到 wandb
    wandb.log({"training_metrics": wandb.Table(dataframe=metrics_df)})
    
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
    
    # 记录最终指标到 wandb
    wandb.log({
        "final_r2": r2_kcat,
        "final_pearson": pearson_end,
        "best_val_loss": best_val_loss,
    })
    
    # 上传损失曲线和指标曲线到 wandb
    loss_curve_path = os.path.join(save_dir, 'loss_curve.png')
    metrics_curve_path = os.path.join(save_dir, 'metrics_curve.png')
    if os.path.exists(loss_curve_path):
        wandb.log({"loss_curve": wandb.Image(loss_curve_path)})
    if os.path.exists(metrics_curve_path):
        wandb.log({"metrics_curve": wandb.Image(metrics_curve_path)})
    
    wandb.finish()
    print("✅ Training finished. Best model and plots saved to", save_dir)

if __name__ == '__main__':
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="data/processed/kcat_full_1213.pt", help='Path to .pt dataset')
    parser.add_argument('--save_dir', type=str, default=None, help='Output directory (if not specified, will auto-generate with timestamp)')
    parser.add_argument('--exp_name', type=str, default='kcat_attn_v1', help='Experiment name (semantic, e.g., kcat_attn_v1)')
    parser.add_argument('--no_timestamp', action='store_true', help='Disable automatic timestamp in save_dir (use fixed path, may overwrite previous results)')
    parser.add_argument('--label_permutation', action='store_true', help='Enable Label Permutation Test (diagnostic test to verify model uses graph representation)')
    parser.add_argument('--frozen_encoder', action='store_true', help='Enable Frozen Encoder Test (freeze GNN encoder, only train MLP regression head to diagnose encoder saturation). Requires --load_checkpoint')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to pretrained model checkpoint (.pt file) to load before training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=500, help='Maximum number of training epochs')
    args = parser.parse_args()
    
    # 如果没有指定 save_dir，自动生成带时间戳的路径
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = f'outputs/kcat_{timestamp}'
    # 如果指定了 save_dir 且没有禁用时间戳，则在路径末尾添加时间戳（避免覆盖）
    elif not args.no_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = args.save_dir.rstrip('/')
        args.save_dir = f"{base_dir}_{timestamp}"
    
    from metadata_utils import save_metadata

    # 训练开始时，加上这行保存metadata
    # exp_name 可以从参数传入，或使用默认值
    exp_name = getattr(args, 'exp_name', 'kcat_attn_fulldata1213')  # 默认实验名称
    
    save_metadata(
        save_dir=args.save_dir,
        dataset_path=args.dataset,
        exp_name=exp_name,
        graph_builder_version='enhanced_builder',
        gnn_model_version='PocketGNNKcatOnly',
        comments='Enhanced features + kcat-only prediction + angle features,9124 items ,simplist model'
    )

    train(
        args.dataset, 
        args.save_dir, 
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
        label_permutation=args.label_permutation,
        frozen_encoder=args.frozen_encoder,
        load_checkpoint=args.load_checkpoint,
        exp_name=exp_name
    )
