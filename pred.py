import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
import GNN_model as MD
import os
import argparse
import matplotlib
matplotlib.use('Agg')  # 确保在没有GUI的环境中使用
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import json

# 配置参数 - 与train.py保持一致
def predict(dataset_path, model_path, save_dir="outputs", batch_size=32, model_config=None):
    """
    预测函数，与train.py的配置保持一致
    """
    # 加载数据
    data_list = torch.load(dataset_path, weights_only=False)
    print(f"📊 加载了 {len(data_list)} 个样本")
    
    # 检查数据集是否包含 NaN
    for data in data_list:
        if torch.isnan(data.x).any() or torch.isnan(data.y).any():
            raise ValueError("数据集中包含 NaN 值")
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    val_loader = DataLoader(data_list, batch_size=batch_size)
    
    # 加载模型 - 使用与train.py相同的模型配置
    node_input_dim = data_list[0].x.shape[1]
    edge_input_dim = data_list[0].edge_attr.shape[1]
    print(f"Node input dim: {node_input_dim}, Edge input dim: {edge_input_dim}")
    
    # 使用模型配置（如果提供）或默认配置
    if model_config is None:
        # 默认配置（与原始train.py一致）
        model_config = {
            'hidden_dim': 128,
            'num_layers': 3,
            'heads': 4,
            'dropout': 0.1
        }
    
    print(f"🔧 使用模型配置: {model_config}")
    
    model = MD.PocketGNNKcatOnly(
        node_input_dim=node_input_dim, 
        edge_input_dim=edge_input_dim,
        **model_config
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设置为评估模式，这会自动禁用dropout和其他训练时的行为

    print("🔧 模型已设置为评估模式（dropout已自动禁用）")

    # 推理 - 与train.py的处理方式保持一致
    all_y_true = []
    all_y_pred = []
    sample_ids = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            actual_batch_size = batch.num_graphs
            
            # 处理标签 - 适配不同的数据格式
            if len(batch.y.shape) == 1 or batch.y.shape[1] == 1:
                # 如果y只是kcat格式 [batch_size] 或 [batch_size, 1]
                log_y = batch.y.reshape(actual_batch_size, 1)  # [batch_size, 1]
            else:
                # 如果y是[kcat, km]格式 [batch_size, 2]
                y_reshaped = batch.y.reshape(actual_batch_size, 2)  # [kcat, km]
                log_y = y_reshaped[:, 0:1]  # 只取kcat列 [batch_size, 1]
            
            out = model(batch)
            all_y_true.append(log_y.cpu())
            all_y_pred.append(out.cpu())
            
            # 收集sample_id信息
            for i in range(actual_batch_size):
                if hasattr(batch, 'sample_id'):
                    sample_ids.append(batch.sample_id[i])
                else:
                    sample_ids.append(f"sample_{len(sample_ids)}")
    
    all_y_true = torch.cat(all_y_true, dim=0).numpy()
    all_y_pred = torch.cat(all_y_pred, dim=0).numpy()
    
    print(f"🔍 预测结果统计:")
    print(f"真实值范围 (log10): {all_y_true.min():.3f} 到 {all_y_true.max():.3f}")
    print(f"预测值范围 (log10): {all_y_pred.min():.3f} 到 {all_y_pred.max():.3f}")
    
    # 反log10变换，获取原始尺度的值
    all_y_true_original = np.power(10, all_y_true.flatten())
    all_y_pred_original = np.power(10, all_y_pred.flatten())
    
    # 计算kcat的误差
    kcat_error_relative = np.abs(all_y_pred_original - all_y_true_original) / all_y_true_original
    kcat_error_absolute = np.abs(all_y_pred.flatten() - all_y_true.flatten())
    
    # 创建结果DataFrame
    df = pd.DataFrame({
        "sample_id": sample_ids,
        "kcat_true_log10": all_y_true.flatten(),
        "kcat_pred_log10": all_y_pred.flatten(),
        "kcat_true_original": all_y_true_original,
        "kcat_pred_original": all_y_pred_original,
        "kcat_error_log10": kcat_error_absolute,
        "kcat_error_relative": kcat_error_relative
    })
    
    # 保存结果
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, "prediction_results.csv")
    df.to_csv(output_path, index=False)
    
    print(f"✅ 预测完成，结果保存到: {output_path}")
    print(f"📊 平均绝对误差 (log10): {kcat_error_absolute.mean():.3f}")
    print(f"📊 平均相对误差: {kcat_error_relative.mean():.3f}")
    
    # 计算评估指标
    mae = mean_absolute_error(all_y_true.flatten(), all_y_pred.flatten())
    rmse = np.sqrt(mean_squared_error(all_y_true.flatten(), all_y_pred.flatten()))
    r2 = r2_score(all_y_true.flatten(), all_y_pred.flatten())
    pearson = pearsonr(all_y_true.flatten(), all_y_pred.flatten())[0]
    
    print(f"📈 评估指标:")
    print(f"  MAE (log10): {mae:.3f}")
    print(f"  RMSE (log10): {rmse:.3f}")
    print(f"  R²: {r2:.3f}")
    print(f"  Pearson: {pearson:.3f}")
    
    # 生成可视化图表
    create_plots(df, all_y_true.flatten(), all_y_pred.flatten(), save_dir)
    
    return df

def predict_improved(dataset_path, model_path, save_dir="outputs", batch_size=32, 
                     model_config=None, load_config_from_file=False):
    """
    改进模型的预测函数，支持从配置文件加载模型参数
    """
    # 如果指定从文件加载配置
    if load_config_from_file and model_config is None:
        config_file = os.path.join(os.path.dirname(model_path), "model_config.json")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                model_config = json.load(f)
            print(f"📁 从文件加载模型配置: {config_file}")
        else:
            print(f"⚠️ 配置文件不存在: {config_file}，使用默认配置")
    
    # 如果没有提供配置，使用改进模型的默认配置
    if model_config is None:
        model_config = {
            'hidden_dim': 256,  # 改进模型的默认配置
            'num_layers': 4,
            'heads': 8,
            'dropout': 0.2
        }
        print(f"🔧 使用改进模型默认配置: {model_config}")
    
    # 调用原始预测函数
    return predict(dataset_path, model_path, save_dir, batch_size, model_config)

def auto_detect_model_config(model_path):
    """
    自动检测模型配置类型
    """
    model_dir = os.path.dirname(model_path)
    
    # 检查是否存在配置文件
    config_file = os.path.join(model_dir, "model_config.json")
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config, "从配置文件加载"
    
    # 根据目录名推断配置类型
    if "improved" in model_dir.lower():
        return {
            'hidden_dim': 256,
            'num_layers': 4,
            'heads': 8,
            'dropout': 0.2
        }, "改进模型配置"
    elif "advanced" in model_dir.lower():
        return {
            'hidden_dim': 256,
            'num_layers': 4,
            'heads': 8,
            'dropout': 0.2
        }, "高级模型配置"
    else:
        return {
            'hidden_dim': 128,
            'num_layers': 3,
            'heads': 4,
            'dropout': 0.1
        }, "原始模型配置"

def predict_auto(dataset_path, model_path, save_dir="outputs", batch_size=32):
    """
    自动检测模型类型并进行预测
    """
    print(f"🔍 自动检测模型配置...")
    
    # 自动检测配置
    model_config, config_source = auto_detect_model_config(model_path)
    print(f"📊 检测到配置类型: {config_source}")
    print(f"🔧 模型配置: {model_config}")
    
    # 调用预测函数
    return predict(dataset_path, model_path, save_dir, batch_size, model_config)

def create_plots(df, y_true, y_pred, save_dir):
    """
    创建预测结果的可视化图表，类似train.py中的绘图功能
    """
    print("🎨 生成可视化图表...")
    
    # 1. 预测vs真实值散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, s=20)
    plt.plot([y_true.min(), y_true.max()], 
             [y_true.min(), y_true.max()], 'r--', linewidth=2)
    plt.xlabel('True kcat (log10)')
    plt.ylabel('Predicted kcat (log10)')
    r2 = r2_score(y_true, y_pred)
    plt.title(f'kcat: True vs Predicted (R² = {r2:.3f})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'kcat_prediction_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 密度图
    plt.figure(figsize=(8, 7))
    sns.kdeplot(x=y_true, y=y_pred, cmap="viridis", fill=True, thresh=0.05)
    plt.plot([y_true.min(), y_true.max()], 
             [y_true.min(), y_true.max()], 'r--', linewidth=2)
    plt.xlabel('True kcat (log10)')
    plt.ylabel('Predicted kcat (log10)')
    plt.title('kcat: Density Plot')
    plt.savefig(os.path.join(save_dir, 'kcat_density.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 残差图
    residuals = y_pred - y_true
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted kcat (log10)')
    plt.ylabel('Residuals (log10)')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'kcat_residuals.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 误差分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Line')
    plt.xlabel('Residuals (log10)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'kcat_residuals_histogram.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 综合评估图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 散点图
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=20)
    axes[0, 0].plot([y_true.min(), y_true.max()], 
                    [y_true.min(), y_true.max()], 'r--', linewidth=2)
    axes[0, 0].set_xlabel('True kcat (log10)')
    axes[0, 0].set_ylabel('Predicted kcat (log10)')
    axes[0, 0].set_title(f'Prediction vs Truth (R² = {r2:.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 残差图
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=20)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Predicted kcat (log10)')
    axes[0, 1].set_ylabel('Residuals (log10)')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 残差分布
    axes[1, 0].hist(residuals, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Line')
    axes[1, 0].set_xlabel('Residuals (log10)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Residuals')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 误差统计
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pearson = pearsonr(y_true, y_pred)[0]
    
    metrics_text = f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r2:.3f}\nPearson: {pearson:.3f}'
    axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes, 
                    fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Evaluation Metrics')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'kcat_comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 图表已保存到 {save_dir}/")
    print(f"  📊 kcat_prediction_scatter.png - 预测vs真实值散点图")
    print(f"  📊 kcat_density.png - 密度图")
    print(f"  📊 kcat_residuals.png - 残差图")
    print(f"  📊 kcat_residuals_histogram.png - 残差分布直方图")
    print(f"  📊 kcat_comprehensive_analysis.png - 综合分析图")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prediction script with support for improved models')
    parser.add_argument('--dataset', type=str, default="kcat_test_new.pt", help='Path to .pt dataset')
    parser.add_argument('--model', type=str, default='outputs/kcat_after_new/best_model.pt', help='Path to model')
    parser.add_argument('--save_dir', type=str, default='outputs/predictions/test_on_after_newmodel', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    # 新增参数：预测模式
    parser.add_argument('--mode', type=str, default='original', 
                       choices=['original', 'improved', 'auto', 'custom'],
                       help='Prediction mode: original, improved, auto, or custom')
    
    # 自定义模型配置参数
    parser.add_argument('--hidden_dim', type=int, default=None, help='Hidden dimension (for custom mode)')
    parser.add_argument('--num_layers', type=int, default=None, help='Number of layers (for custom mode)')
    parser.add_argument('--heads', type=int, default=None, help='Number of attention heads (for custom mode)')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate (for custom mode)')
    
    # 配置文件相关参数
    parser.add_argument('--config_file', type=str, default=None, help='Path to model config JSON file')
    parser.add_argument('--load_config', action='store_true', help='Load config from model directory')
    
    args = parser.parse_args()
    
    print("🚀 开始预测...")
    print(f"📁 数据集: {args.dataset}")
    print(f"🤖 模型: {args.model}")
    print(f"💾 保存目录: {args.save_dir}")
    print(f"🔧 批次大小: {args.batch_size}")
    print(f"📊 预测模式: {args.mode}")
    
    # 根据模式选择预测函数
    if args.mode == 'original':
        print("📊 使用原始模型配置进行预测")
        predict(args.dataset, args.model, args.save_dir, args.batch_size)
        
    elif args.mode == 'improved':
        print("📊 使用改进模型配置进行预测")
        model_config = None
        if args.load_config:
            model_config = None  # 让函数自动从文件加载
        elif args.config_file:
            with open(args.config_file, 'r') as f:
                model_config = json.load(f)
        predict_improved(args.dataset, args.model, args.save_dir, args.batch_size, 
                        model_config, args.load_config)
        
    elif args.mode == 'custom':
        print("📊 使用自定义模型配置进行预测")
        if not all([args.hidden_dim, args.num_layers, args.heads, args.dropout is not None]):
            print("❌ 自定义模式需要提供所有模型参数: --hidden_dim, --num_layers, --heads, --dropout")
            exit(1)
        
        model_config = {
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'heads': args.heads,
            'dropout': args.dropout
        }
        print(f"🔧 自定义配置: {model_config}")
        predict(args.dataset, args.model, args.save_dir, args.batch_size, model_config)
        
    else:  # auto mode
        print("📊 自动检测模型配置进行预测")
        predict_auto(args.dataset, args.model, args.save_dir, args.batch_size)