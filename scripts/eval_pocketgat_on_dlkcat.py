#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 DLKcat 官方测试集上评估 Pocket-GAT 方法

重要：DLKcat 测试集与用户训练数据有部分重叠，需要排除这些样本进行公平评估
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from tqdm import tqdm
import hashlib

# 项目路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import GNN_model as MD


def shuffle_dataset(dataset, seed=1234):
    np.random.seed(seed)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    return [dataset[i] for i in indices]


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]


def load_dlkcat_test_set():
    """加载 DLKcat 官方测试集"""
    dlkcat_path = os.path.join(
        PROJECT_ROOT, 
        'benchmark_tools/DLKcat/DeeplearningApproach/Data/database/Kcat_combination_0918_wildtype_mutant.json'
    )
    
    with open(dlkcat_path, 'r') as f:
        data = json.load(f)
    
    # 官方划分
    dataset = shuffle_dataset(data, seed=1234)
    train_set, rest = split_dataset(dataset, 0.8)
    dev_set, test_set = split_dataset(rest, 0.5)
    
    return test_set


def load_user_training_sequences():
    """加载用户训练数据的序列"""
    csv_path = os.path.join(PROJECT_ROOT, 'data/processed/kcat_full_1213.csv')
    df = pd.read_csv(csv_path)
    return set(df['sequence'].dropna())


def filter_non_overlapping(test_set, train_seqs):
    """过滤掉与训练数据重叠的样本"""
    non_overlapping = []
    overlapping = []
    
    for item in test_set:
        if item['Sequence'] not in train_seqs:
            non_overlapping.append(item)
        else:
            overlapping.append(item)
    
    return non_overlapping, overlapping


def check_existing_pockets(test_samples, pocket_dir):
    """检查哪些样本已有 pocket 数据"""
    existing = []
    missing = []
    
    for item in test_samples:
        smiles = item['Smiles']
        seq = item['Sequence']
        
        # 计算 pocket 文件名的 hash
        pocket_hash = int(hashlib.sha256(smiles.encode()).hexdigest(), 16) & 0xffff
        
        # 检查各种可能的 pocket 路径
        found = False
        for sample_dir in os.listdir(pocket_dir) if os.path.exists(pocket_dir) else []:
            pocket_path = os.path.join(pocket_dir, sample_dir, f'{sample_dir}_{pocket_hash}_10A.pdb')
            if os.path.exists(pocket_path):
                item['pocket_path'] = pocket_path
                existing.append(item)
                found = True
                break
        
        if not found:
            missing.append(item)
    
    return existing, missing


def build_graph_from_pocket(pocket_path, kcat_value, temperature=303.15):
    """从 pocket PDB 文件构建图数据"""
    from graph_builder_rbf import build_graph, parse_pocket
    from build_graph_dataset import compute_angle_features, compute_dihedral_features
    
    try:
        atoms = parse_pocket(pocket_path)
        if len(atoms) < 3:
            return None
        
        data = build_graph(atoms, temperature)
        
        # 添加角度和二面角特征
        edge_index = data.edge_index
        pos = data.pos
        
        angle_features = compute_angle_features(edge_index, pos)
        dihedral_features = compute_dihedral_features(edge_index, pos)
        
        original_edge_attr = data.edge_attr
        enhanced_edge_attr = torch.cat([
            original_edge_attr,
            angle_features,
            dihedral_features
        ], dim=1)
        
        data.edge_attr = enhanced_edge_attr
        data.y = torch.log10(torch.tensor([kcat_value], dtype=torch.float))
        
        return data
    except Exception as e:
        print(f"构建图失败: {e}")
        return None


def evaluate_with_model(model, data_list, device, batch_size=32):
    """使用模型评估"""
    model.eval()
    
    # 清理数据对象
    for data in data_list:
        keys_to_remove = []
        for key in data.keys():
            value = getattr(data, key)
            if not isinstance(value, torch.Tensor):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            delattr(data, key)
    
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    
    y_true_list = []
    y_pred_list = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="推理中"):
            batch = batch.to(device)
            output = model(batch)
            
            if output.dim() == 2:
                output = output.squeeze(-1)
            
            y_true_list.append(batch.y.cpu().numpy())
            y_pred_list.append(output.cpu().numpy())
    
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    
    return y_true.flatten(), y_pred.flatten()


def compute_metrics(y_true, y_pred):
    """计算评估指标"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    return {
        'n_valid': len(y_true),
        'r2': r2_score(y_true, y_pred),
        'pearson_r': pearsonr(y_true, y_pred)[0],
        'spearman_r': spearmanr(y_true, y_pred)[0],
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
    }


def plot_scatter(y_true, y_pred, model_name, save_path, metrics):
    """创建散点图"""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, s=15)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('True log10(kcat)', fontsize=12)
    plt.ylabel('Predicted log10(kcat)', fontsize=12)
    plt.title(f'{model_name} on DLKcat Test Set (Non-overlapping)\nR² = {metrics["r2"]:.3f}, Pearson r = {metrics["pearson_r"]:.3f}', fontsize=13)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, 
                        default='outputs/kcat_20251213_204244/best_model.pt',
                        help='训练好的模型路径')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--node_input_dim', type=int, default=52)
    parser.add_argument('--edge_input_dim', type=int, default=24)
    parser.add_argument('--max_samples', type=int, default=None, help='最大样本数（测试用）')
    args = parser.parse_args()
    
    print("=" * 60)
    print("在 DLKcat 官方测试集上评估 Pocket-GAT")
    print("=" * 60)
    
    # 1. 加载 DLKcat 测试集
    print("\n📊 加载 DLKcat 官方测试集...")
    test_set = load_dlkcat_test_set()
    print(f"   测试集样本数: {len(test_set)}")
    
    # 2. 加载用户训练序列
    print("\n📊 加载用户训练数据序列...")
    train_seqs = load_user_training_sequences()
    print(f"   训练数据唯一序列数: {len(train_seqs)}")
    
    # 3. 过滤重叠样本
    print("\n🔍 过滤与训练数据重叠的样本...")
    non_overlapping, overlapping = filter_non_overlapping(test_set, train_seqs)
    print(f"   非重叠样本数: {len(non_overlapping)}")
    print(f"   重叠样本数: {len(overlapping)}")
    print(f"   ⚠️ 将只在非重叠样本上进行公平评估")
    
    if args.max_samples:
        non_overlapping = non_overlapping[:args.max_samples]
        print(f"   限制为前 {args.max_samples} 个样本")
    
    # 4. 检查已有的 pocket 数据
    # 尝试多个可能的 pocket 目录
    pocket_dirs = [
        os.path.join(PROJECT_ROOT, 'data/processed/pockets'),
        '/home/lizihao/Work/enzyme_prediction/PGNN/sample_data/samples',
        os.path.join(PROJECT_ROOT, 'output/pdb_files'),
    ]
    
    print("\n🔍 检查已有的 pocket 数据...")
    
    # 由于 DLKcat 测试集的数据没有 UniProt ID，我们需要通过序列来匹配
    # 首先检查用户已有的图数据中是否有匹配的序列
    
    # 加载用户已有的图数据
    existing_data_path = os.path.join(PROJECT_ROOT, 'data/processed/kcat_full_1213.pt')
    if os.path.exists(existing_data_path):
        print(f"   加载已有图数据: {existing_data_path}")
        existing_data = torch.load(existing_data_path, weights_only=False)
        print(f"   已有图数据样本数: {len(existing_data)}")
        
        # 加载对应的 CSV 来获取序列
        csv_path = os.path.join(PROJECT_ROOT, 'data/processed/kcat_full_1213.csv')
        df = pd.read_csv(csv_path)
        
        # 创建序列到图数据的映射
        seq_to_data = {}
        for i, row in df.iterrows():
            if i < len(existing_data):
                seq_to_data[row['sequence']] = existing_data[i]
        
        print(f"   序列到图数据映射: {len(seq_to_data)} 条")
    else:
        seq_to_data = {}
        print("   未找到已有图数据")
    
    # 5. 匹配 DLKcat 测试样本与已有数据
    print("\n📊 匹配 DLKcat 测试样本与已有图数据...")
    matched_data = []
    unmatched_samples = []
    
    for item in non_overlapping:
        seq = item['Sequence']
        if seq in seq_to_data:
            # 复制数据并更新 y 值
            data = seq_to_data[seq]
            # 创建新的数据对象，使用 DLKcat 的 kcat 值
            new_data = Data(
                x=data.x.clone(),
                edge_index=data.edge_index.clone(),
                edge_attr=data.edge_attr.clone(),
                pos=data.pos.clone() if hasattr(data, 'pos') else None,
            )
            new_data.y = torch.log10(torch.tensor([float(item['Value'])], dtype=torch.float))
            if hasattr(data, 'temperature'):
                new_data.temperature = data.temperature.clone()
            matched_data.append(new_data)
        else:
            unmatched_samples.append(item)
    
    print(f"   匹配到的样本数: {len(matched_data)}")
    print(f"   未匹配的样本数: {len(unmatched_samples)}")
    
    if len(matched_data) == 0:
        print("\n❌ 没有匹配到任何样本，无法评估")
        print("   需要先为 DLKcat 测试集生成 pocket 数据")
        return
    
    # 6. 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(PROJECT_ROOT, 'results', f'pocketgat_dlkcat_eval_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 7. 加载模型
    print(f"\n🧠 加载模型: {args.model_path}")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    model = MD.PocketGNNKcatOnly(
        node_input_dim=args.node_input_dim,
        edge_input_dim=args.edge_input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        pooling_type='mean',
        use_seq_embedding=False
    ).to(device)
    
    model_path = os.path.join(PROJECT_ROOT, args.model_path)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("✅ 模型加载完成!")
    
    # 8. 评估
    print(f"\n📈 开始评估 ({len(matched_data)} 个样本)...")
    y_true, y_pred = evaluate_with_model(model, matched_data, device, args.batch_size)
    
    # 9. 计算指标
    metrics = compute_metrics(y_true, y_pred)
    
    print("\n" + "=" * 60)
    print("📊 Pocket-GAT 评估结果 (DLKcat 非重叠测试集)")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")
    
    # 10. 保存结果
    results_df = pd.DataFrame({
        'log_kcat_true': y_true,
        'log_kcat_pred': y_pred,
    })
    results_path = os.path.join(output_dir, 'pocketgat_predictions.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ 预测结果已保存: {results_path}")
    
    # 11. 绘图
    plot_path = os.path.join(output_dir, 'pocketgat_scatter.png')
    plot_scatter(y_true, y_pred, 'Pocket-GAT', plot_path, metrics)
    print(f"✅ 散点图已保存: {plot_path}")
    
    # 12. 保存指标
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, 'pocketgat_metrics.csv'), index=False)
    
    # 13. 保存评估信息
    info = {
        'total_dlkcat_test': len(test_set),
        'overlapping_with_train': len(overlapping),
        'non_overlapping': len(non_overlapping),
        'matched_with_existing_data': len(matched_data),
        'unmatched': len(unmatched_samples),
        'model_path': args.model_path,
        **metrics
    }
    with open(os.path.join(output_dir, 'eval_info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\n✅ 所有结果已保存到: {output_dir}")
    
    return metrics


if __name__ == "__main__":
    main()






