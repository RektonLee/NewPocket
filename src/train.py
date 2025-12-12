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
import wandb

try:
    from utils.metadata_utils import update_training_results, save_metadata  # type: ignore
except Exception:
    update_training_results = None
    save_metadata = None

def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(y_true_log, y_pred_log):
    if torch.is_tensor(y_true_log):
        y_true_log = y_true_log.detach().cpu().numpy()
    if torch.is_tensor(y_pred_log):
        y_pred_log = y_pred_log.detach().cpu().numpy()
    return {
        'MAE': mean_absolute_error(y_true_log, y_pred_log),
        'RMSE': np.sqrt(mean_squared_error(y_true_log, y_pred_log)),
        'R2': r2_score(y_true_log, y_pred_log),
        'Pearson': pearsonr(y_true_log.flatten(), y_pred_log.flatten())[0]
    }

def train(dataset_path, save_dir="outputs", batch_size=32, lr=1e-3, max_epochs=500, seed: int = 42, device: str | None = None, use_wandb: bool = True):
    _set_seed(seed)

    data_list = torch.load(dataset_path, weights_only=False)
    
    # 检查数据集是否包含 NaN
    for data in data_list:
        if torch.isnan(data.x).any() or torch.isnan(data.y).any():
            raise ValueError("数据集中包含 NaN 值")
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(save_dir)
    
    # 初始化 wandb
    run = None
    if use_wandb:
        # 不要在代码里硬编码密钥：使用环境变量 WANDB_API_KEY 或者提前 `wandb login`
        try:
            if os.environ.get("WANDB_API_KEY"):
                wandb.login(key=os.environ["WANDB_API_KEY"])
            run = wandb.init(
                project="enzyme-kcat-prediction",
                name=os.path.basename(save_dir),
                config={
                    "dataset": os.path.basename(dataset_path),
                    "batch_size": batch_size,
                    "lr": lr,
                    "max_epochs": max_epochs,
                    "seed": seed,
                    "device": str(device),
                }
            )
        except Exception:
            run = None

    # === Load dataset ===
    print(data_list[0])  # 打印第一个图数据
    rng = np.random.default_rng(seed)
    rng.shuffle(data_list)
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
    dropout = 0.1     # Dropout 概率

    # 更新 wandb config 包含模型参数
    if run is not None:
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
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')

    # 添加损失记录列表
    train_loss_history = []
    val_loss_history = []
    r2_history = []
    pearson_history = []
    
    # 将本次训练的关键超参数与环境信息记录到共享表中（保持运行中状态）
    if update_training_results is not None:
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
        except Exception:
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
        # === Logging ===
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("R2/val", metrics['R2'], epoch)
        writer.add_scalar("Pearson/val", metrics['Pearson'], epoch)
        
        # wandb logging
        if run is not None:
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
            if run is not None:
                wandb.save(model_path)

    writer.close()
    if run is not None:
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
    scatter_path = os.path.join(save_dir, 'kcat_prediction_scatter.png')
    plt.savefig(scatter_path)
    plt.close()
    # 上传散点图到 wandb
    if run is not None:
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
    if run is not None:
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
    if run is not None:
        wandb.log({"training_metrics": wandb.Table(dataframe=metrics_df)})
    
    # 记录最终结果到共享表
    if update_training_results is not None:
        try:
            update_training_results(
                save_dir=save_dir,
                status="completed",
                best_val_loss=best_val_loss,
                final_r2=r2_kcat,
                final_pearson=pearson_end
            )
        except Exception:
            pass
    
    # 记录最终指标到 wandb
    if run is not None:
        wandb.log({
            "final_r2": r2_kcat,
            "final_pearson": pearson_end,
            "best_val_loss": best_val_loss,
        })
    
    # 上传损失曲线和指标曲线到 wandb
    loss_curve_path = os.path.join(save_dir, 'loss_curve.png')
    metrics_curve_path = os.path.join(save_dir, 'metrics_curve.png')
    if run is not None:
        if os.path.exists(loss_curve_path):
            wandb.log({"loss_curve": wandb.Image(loss_curve_path)})
        if os.path.exists(metrics_curve_path):
            wandb.log({"metrics_curve": wandb.Image(metrics_curve_path)})
    
    if run is not None:
        wandb.finish()
    print("✅ Training finished. Best model and plots saved to", save_dir)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="kcat_train_after_new_clean.pt", help='Path to .pt dataset')
    parser.add_argument('--save_dir', type=str, default='outputs/kcat_after_new')
    parser.add_argument('--device', type=str, default=None, help='e.g. "cuda", "cuda:0", "cpu"')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--no_wandb', action='store_true')
    args = parser.parse_args()

    # 训练开始时，加上这行保存metadata
    if save_metadata is not None:
        try:
            save_metadata(
                save_dir=args.save_dir,
                dataset_path=args.dataset,
                graph_builder_version='enhanced_builder',
                gnn_model_version='PocketGNNKcatOnly',
                comments='Enhanced features + kcat-only prediction + angle features,9124 items ,simplist model'
            )
        except Exception:
            pass

    train(
        dataset_path=args.dataset,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
        seed=args.seed,
        device=args.device,
        use_wandb=(not args.no_wandb),
    )
