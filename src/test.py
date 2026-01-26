import torch
import torch.nn as nn
import os
import sys
import numpy as np
from torch_geometric.loader import DataLoader

# 添加项目根目录到 Python 路径，支持从根目录运行 python src/test.py
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
import json

def compute_metrics(y_true_log, y_pred_log):
    """
    计算评估指标（与 train.py 中的函数保持一致）
    """
    # 安全地转换为 numpy：先 detach 再移到 CPU
    y_true_log = y_true_log.detach().cpu().numpy()
    y_pred_log = y_pred_log.detach().cpu().numpy()
    
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

def clean_data_objects(data_list):
    """
    清理 Data 对象：移除字符串属性（PyG DataLoader 无法 collate 字符串）
    与 train.py 中的逻辑保持一致
    """
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

def test(test_dataset_path, model_path, save_dir="outputs/test_results", batch_size=32, 
         hidden_dim=128, num_layers=3, heads=4, dropout=0.1, pooling_type='mean',
         use_seq_embedding=False, seq_embedding_path=None, use_mlp_layernorm=None):
    """
    在测试集上评估模型
    
    Args:
        test_dataset_path: 测试数据集路径（.pt 文件）
        model_path: 训练好的模型权重路径（.pt 文件）
        save_dir: 结果保存目录
        batch_size: 批次大小
        hidden_dim: 模型隐藏维度（需要与训练时一致）
        num_layers: 模型层数（需要与训练时一致）
        heads: 注意力头数（需要与训练时一致）
        dropout: Dropout 概率（需要与训练时一致）
        pooling_type: 池化类型（需要与训练时一致，默认 'mean'）
        use_seq_embedding: 是否使用序列嵌入（需要与训练时一致）
        seq_embedding_path: 序列嵌入文件路径（如果启用序列嵌入）
    """
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("🧪 测试集评估")
    print("=" * 60)
    print(f"📁 测试数据集: {test_dataset_path}")
    print(f"🤖 模型权重: {model_path}")
    print(f"💾 结果保存目录: {save_dir}")
    print(f"🖥️  设备: {device}")
    print()
    
    # === 加载测试数据集 ===
    print("📥 加载测试数据集...")
    if not os.path.exists(test_dataset_path):
        raise FileNotFoundError(f"❌ 找不到测试数据集文件: {test_dataset_path}")
    
    test_data_list = torch.load(test_dataset_path, weights_only=False)
    print(f"✅ 测试集加载完成，共 {len(test_data_list)} 个样本")
    
    # 检查数据集是否包含 NaN
    for i, data in enumerate(test_data_list):
        if torch.isnan(data.x).any() or torch.isnan(data.y).any():
            raise ValueError(f"❌ 测试数据集中第 {i} 个样本包含 NaN 值")
    
    # === 加载序列嵌入（如果启用） ===
    seq_embedding_dim = 0
    if use_seq_embedding:
        if seq_embedding_path is None:
            raise ValueError("❌ 启用序列嵌入但未提供 seq_embedding_path")
        
        print(f"\n📥 加载序列嵌入: {seq_embedding_path}")
        if not os.path.exists(seq_embedding_path):
            raise FileNotFoundError(f"❌ 找不到序列嵌入文件: {seq_embedding_path}")
        
        embeddings_map = torch.load(seq_embedding_path, weights_only=False)
        print(f"✅ 加载了 {len(embeddings_map)} 个序列嵌入")
        
        # 匹配序列嵌入（与 train.py 逻辑一致）
        matched_count = 0
        unmatched_samples = []
        for data in test_data_list:
            key = None
            embedding = None
            key_type = None
            
            # Priority: sample_id > pdb_id > uniprot_id
            # 尝试多个key，直到找到匹配的
            if hasattr(data, 'sample_id') and data.sample_id is not None:
                key = str(data.sample_id)
                key_type = 'sample_id'
                if key in embeddings_map:
                    embedding = embeddings_map[key]
            
            # 如果 sample_id 不匹配，尝试 pdb_id
            if embedding is None and hasattr(data, 'pdb_id') and data.pdb_id is not None:
                key = str(data.pdb_id)
                key_type = 'pdb_id'
                if key in embeddings_map:
                    embedding = embeddings_map[key]
                # 如果完整 pdb_id 不匹配，尝试去掉扩展名
                elif '.' in key:
                    key_no_ext = key.split('.')[0]
                    if key_no_ext in embeddings_map:
                        embedding = embeddings_map[key_no_ext]
                        key = key_no_ext
            
            # 如果 pdb_id 也不匹配，尝试 uniprot_id
            if embedding is None and hasattr(data, 'uniprot_id') and data.uniprot_id is not None:
                key = str(data.uniprot_id)
                key_type = 'uniprot_id'
                if key in embeddings_map:
                    embedding = embeddings_map[key]
            
            # 记录未匹配的样本（用于调试）
            if embedding is None:
                if len(unmatched_samples) < 5:
                    unmatched_samples.append((key, key_type))
            
            if embedding is not None:
                if not isinstance(embedding, torch.Tensor):
                    embedding = torch.tensor(embedding, dtype=torch.float)
                
                if seq_embedding_dim == 0:
                    seq_embedding_dim = embedding.shape[0]
                    print(f"✅ 序列嵌入维度: {seq_embedding_dim}")
                
                data.seq_embedding = embedding.unsqueeze(0)  # [1, dim]
                matched_count += 1
        
        print(f"✅ 匹配了 {matched_count}/{len(test_data_list)} 个序列嵌入")
        if unmatched_samples:
            print(f"   示例未匹配的keys: {unmatched_samples}")
        
        # 重要：即使匹配失败，如果训练时使用了序列嵌入，测试时也必须使用
        # 否则模型结构不匹配会导致加载失败
        # 如果匹配失败，会用零向量填充（在后续代码中处理）
        if matched_count == 0:
            print("⚠️  警告: 没有匹配到任何序列嵌入")
            print("   如果训练时使用了序列嵌入，将用零向量填充以保持模型结构一致")
            # 尝试从第一个embedding推断维度
            if len(embeddings_map) > 0:
                first_emb = list(embeddings_map.values())[0]
                if isinstance(first_emb, torch.Tensor):
                    seq_embedding_dim = first_emb.shape[0]
                else:
                    seq_embedding_dim = len(first_emb)
                print(f"   从embedding文件推断维度: {seq_embedding_dim}")
            else:
                raise ValueError("无法推断序列嵌入维度，且没有匹配到任何嵌入")
    
    # === 清理数据对象（移除字符串属性） ===
    # 与 train.py 保持一致：先匹配序列嵌入，再删除字符串属性
    print("\n🧹 清理数据对象：移除字符串属性...")
    for data in test_data_list:
        if hasattr(data, 'ec'):
            delattr(data, 'ec')
        if hasattr(data, 'pdb_id'):
            delattr(data, 'pdb_id')
        if hasattr(data, 'sample_id'):
            delattr(data, 'sample_id')
        if hasattr(data, 'uniprot_id'):
            delattr(data, 'uniprot_id')
    
    # 如果启用了序列嵌入但某些样本没有匹配到，用零向量填充
    if use_seq_embedding and seq_embedding_dim > 0:
        missing_count = 0
        for data in test_data_list:
            if not hasattr(data, 'seq_embedding'):
                data.seq_embedding = torch.zeros((1, seq_embedding_dim), dtype=torch.float)
                missing_count += 1
        if missing_count > 0:
            print(f"⚠️  警告: {missing_count} 个样本缺少序列嵌入，已用零向量填充")
    
    # === 创建 DataLoader ===
    test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=False)
    print(f"📊 测试集 DataLoader 创建完成，批次大小: {batch_size}")
    
    # === 初始化模型 ===
    print("\n🔧 初始化模型...")
    sample_data = test_data_list[0]
    node_input_dim = sample_data.x.shape[1]  # 默认 52
    edge_input_dim = sample_data.edge_attr.shape[1]  # 应该是 24 维
    print(f"   Node input dim: {node_input_dim}, Edge input dim: {edge_input_dim}")
    
    # 自动检测是否使用LayerNorm（通过检查模型权重）
    if use_mlp_layernorm is None:
        print("   🔍 自动检测模型结构...")
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        # 检查mlp.0是否是LayerNorm（LayerNorm的weight是1D，Linear的weight是2D）
        if 'mlp.0.weight' in state_dict:
            mlp_0_weight = state_dict['mlp.0.weight']
            if mlp_0_weight.dim() == 1:
                # 1D权重 -> LayerNorm
                use_mlp_layernorm = True
                print("   ✅ 检测到模型使用LayerNorm")
            else:
                # 2D权重 -> Linear（旧模型）
                use_mlp_layernorm = False
                print("   ✅ 检测到模型不使用LayerNorm（旧版本）")
        else:
            # 默认使用LayerNorm（新版本）
            use_mlp_layernorm = True
            print("   ⚠️  无法检测，默认使用LayerNorm")
    
    model = MD.PocketGNNKcatOnly(
        node_input_dim=node_input_dim,
        edge_input_dim=edge_input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
        dropout=dropout,
        pooling_type=pooling_type,
        use_seq_embedding=use_seq_embedding,
        seq_embedding_dim=seq_embedding_dim if use_seq_embedding else 0,
        use_mlp_layernorm=use_mlp_layernorm
    ).to(device)
    
    # === 加载模型权重 ===
    print(f"\n📥 加载模型权重: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 找不到模型权重文件: {model_path}")
    
    # 加载模型权重
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✅ 模型权重加载完成（严格匹配）")
    except RuntimeError as e:
        print("⚠️  严格匹配失败，尝试使用strict=False加载...")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"   ⚠️  缺失的keys: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"   ⚠️  缺失的keys: {missing_keys}")
        if unexpected_keys:
            print(f"   ⚠️  多余的keys: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"   ⚠️  多余的keys: {unexpected_keys}")
        print("✅ 模型权重加载完成（非严格匹配）")
    model.eval()
    
    # === 在测试集上进行推理 ===
    print("\n🔮 开始测试集推理...")
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = batch.to(device)
            actual_batch_size = batch.num_graphs
            
            # 处理标签 - 适配不同的数据格式（与 train.py 保持一致）
            if batch.y.shape[0] == actual_batch_size:
                # 只有 kcat 值
                log_y = batch.y.reshape(actual_batch_size, 1)
            else:
                # 有 kcat 和 km 值，只取 kcat 列
                y_reshaped = batch.y.reshape(actual_batch_size, 2)
                log_y = y_reshaped[:, 0:1]  # 只取 kcat 列 [batch_size, 1]
            
            # 检查输入数据是否包含 NaN
            if torch.isnan(batch.x).any():
                print(f"⚠️  警告: batch {batch_idx} 的 batch.x 中含有 NaN")
            if torch.isnan(batch.edge_attr).any():
                print(f"⚠️  警告: batch {batch_idx} 的 batch.edge_attr 中含有 NaN")
            if torch.isnan(batch.y).any():
                print(f"⚠️  警告: batch {batch_idx} 的 batch.y 中含有 NaN")
            
            # 模型推理
            try:
                out = model(batch)
            except Exception as e:
                print(f"❌ 推理失败，batch 索引: {batch.batch.unique()}")
                raise e
            
            # 检查输出是否包含 NaN
            if torch.isnan(out).any():
                print(f"⚠️  警告: batch {batch_idx} 的模型输出中含有 NaN")
            
            all_y_true.append(log_y.cpu())
            all_y_pred.append(out.cpu())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"   已处理 {batch_idx + 1}/{len(test_loader)} 个批次")
    
    # === 合并所有预测结果 ===
    all_y_true = torch.cat(all_y_true, dim=0)
    all_y_pred = torch.cat(all_y_pred, dim=0)
    print(f"✅ 推理完成，共处理 {len(test_data_list)} 个样本")
    
    # === 计算评估指标 ===
    print("\n📊 计算评估指标...")
    metrics = compute_metrics(all_y_true, all_y_pred)
    
    print("\n" + "=" * 60)
    print("📈 测试集评估结果")
    print("=" * 60)
    print(f"   MAE:     {metrics['MAE']:.4f}")
    print(f"   RMSE:    {metrics['RMSE']:.4f}")
    print(f"   R²:      {metrics['R2']:.4f}")
    print(f"   Pearson: {metrics['Pearson']:.4f}")
    print("=" * 60)
    
    # === 保存指标到文件 ===
    # 将 numpy 类型转换为 Python 原生类型，以便 JSON 序列化
    metrics_serializable = {
        'MAE': float(metrics['MAE']),
        'RMSE': float(metrics['RMSE']),
        'R2': float(metrics['R2']),
        'Pearson': float(metrics['Pearson'])
    }
    
    metrics_dict = {
        'test_metrics': metrics_serializable,
        'test_samples': len(test_data_list),
        'model_path': model_path,
        'test_dataset_path': test_dataset_path,
        'model_config': {
            'node_input_dim': int(node_input_dim),
            'edge_input_dim': int(edge_input_dim),
            'hidden_dim': int(hidden_dim),
            'num_layers': int(num_layers),
            'heads': int(heads),
            'dropout': float(dropout),
            'pooling_type': str(pooling_type),
            'use_seq_embedding': bool(use_seq_embedding),
            'seq_embedding_dim': int(seq_embedding_dim) if use_seq_embedding else 0
        }
    }
    
    metrics_json_path = os.path.join(save_dir, 'test_metrics.json')
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
    print(f"\n💾 指标已保存到: {metrics_json_path}")
    
    # 同时保存为 CSV 格式（便于查看）
    metrics_df = pd.DataFrame([metrics])
    metrics_csv_path = os.path.join(save_dir, 'test_metrics.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"💾 指标已保存到: {metrics_csv_path}")
    
    # === 生成可视化图表 ===
    print("\n📊 生成可视化图表...")
    
    all_y_true_np = all_y_true.numpy()
    all_y_pred_np = all_y_pred.numpy()
    
    # 1. 散点图：真实值 vs 预测值
    plt.figure(figsize=(8, 6))
    plt.scatter(all_y_true_np.flatten(), all_y_pred_np.flatten(), alpha=0.6, s=20)
    plt.plot([all_y_true_np.min(), all_y_true_np.max()], 
             [all_y_true_np.min(), all_y_true_np.max()], 'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('True kcat (log10)', fontsize=12)
    plt.ylabel('Predicted kcat (log10)', fontsize=12)
    plt.title(f'Test Set: True vs Predicted kcat (R² = {metrics["R2"]:.3f}, Pearson = {metrics["Pearson"]:.3f})', fontsize=13)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    scatter_path = os.path.join(save_dir, 'test_kcat_prediction_scatter.png')
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 散点图已保存: {scatter_path}")
    
    # 2. 密度图
    plt.figure(figsize=(8, 7))
    true_vals = all_y_true_np.flatten()
    pred_vals = all_y_pred_np.flatten()
    sns.kdeplot(x=true_vals, y=pred_vals, cmap="viridis", fill=True, thresh=0.05, alpha=0.8)
    plt.plot([true_vals.min(), true_vals.max()], 
             [true_vals.min(), true_vals.max()], 'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('True kcat (log10)', fontsize=12)
    plt.ylabel('Predicted kcat (log10)', fontsize=12)
    plt.title('Test Set: kcat Prediction Density Plot', fontsize=13)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    density_path = os.path.join(save_dir, 'test_kcat_density.png')
    plt.savefig(density_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 密度图已保存: {density_path}")
    
    # 3. 残差图（预测误差分布）
    residuals = (all_y_pred_np - all_y_true_np).flatten()
    plt.figure(figsize=(8, 6))
    plt.scatter(all_y_true_np.flatten(), residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('True kcat (log10)', fontsize=12)
    plt.ylabel('Residual (Predicted - True)', fontsize=12)
    plt.title('Test Set: Prediction Residuals', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    residual_path = os.path.join(save_dir, 'test_residuals.png')
    plt.savefig(residual_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 残差图已保存: {residual_path}")
    
    # 4. 残差直方图
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Residual (Predicted - True)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Test Set: Residual Distribution (Mean = {residuals.mean():.4f}, Std = {residuals.std():.4f})', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    residual_hist_path = os.path.join(save_dir, 'test_residual_histogram.png')
    plt.savefig(residual_hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 残差直方图已保存: {residual_hist_path}")
    
    # === 保存详细的预测结果 ===
    results_df = pd.DataFrame({
        'true_kcat_log10': all_y_true_np.flatten(),
        'predicted_kcat_log10': all_y_pred_np.flatten(),
        'residual': residuals,
        'absolute_error': np.abs(residuals)
    })
    results_csv_path = os.path.join(save_dir, 'test_predictions.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"\n💾 详细预测结果已保存到: {results_csv_path}")
    
    print("\n" + "=" * 60)
    print("✅ 测试集评估完成！")
    print(f"📁 所有结果已保存到: {save_dir}")
    print("=" * 60)
    
    return metrics

if __name__ == '__main__':
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='在测试集上评估训练好的模型')
    parser.add_argument('--test_dataset', type=str, required=True, 
                       help='测试数据集路径（.pt 文件）')
    parser.add_argument('--model', type=str, required=True,
                       help='训练好的模型权重路径（.pt 文件，通常是 best_model.pt）')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='结果保存目录（如果未指定，将自动生成带时间戳的路径）')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小（默认: 32）')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='模型隐藏维度（需要与训练时一致，默认: 128）')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='模型层数（需要与训练时一致，默认: 3）')
    parser.add_argument('--heads', type=int, default=4,
                       help='注意力头数（需要与训练时一致，默认: 4）')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout 概率（需要与训练时一致，默认: 0.1）')
    parser.add_argument('--pooling_type', type=str, default='set2set',
                       choices=['mean', 'global_attention', 'set2set'],
                       help='池化类型（需要与训练时一致，默认: mean）')
    parser.add_argument('--use_seq_embedding', action='store_true',
                       help='启用序列嵌入（需要与训练时一致）')
    parser.add_argument('--seq_embedding_path', type=str, default=None,
                       help='序列嵌入文件路径（如果启用序列嵌入）')
    parser.add_argument('--use_mlp_layernorm', type=lambda x: (str(x).lower() == 'true'), default=None,
                       help='是否使用MLP LayerNorm（None=自动检测，True/False=手动指定）')
    
    args = parser.parse_args()
    
    # 如果没有指定 save_dir，自动生成带时间戳的路径
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_basename = os.path.basename(args.model).replace('.pt', '')
        args.save_dir = f'outputs/test_{model_basename}_{timestamp}'
    
    test(
        test_dataset_path=args.test_dataset,
        model_path=args.model,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        pooling_type=args.pooling_type,
        use_seq_embedding=args.use_seq_embedding,
        seq_embedding_path=args.seq_embedding_path,
        use_mlp_layernorm=args.use_mlp_layernorm
    )

