#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度调试脚本 - 系统检查训练数据、图特征、模型权重
"""

import torch
import torch.nn.functional as F
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from torch_geometric.loader import DataLoader
import GNN_model as MD

def check_training_data_quality():
    """检查训练数据质量"""
    print("=" * 60)
    print("1. 检查训练数据质量")
    print("=" * 60)
    
    train_data_path = 'data/processed/kcat_train_after_new_clean.pt'
    if not os.path.exists(train_data_path):
        print(f"❌ 找不到训练数据: {train_data_path}")
        return None
    
    train_data = torch.load(train_data_path, weights_only=False)
    print(f"✅ 加载训练数据: {len(train_data)} 个样本")
    
    # 检查标签分布
    all_y = torch.stack([d.y for d in train_data])
    print(f"\n📊 标签统计:")
    print(f"  min: {all_y.min():.4f}, max: {all_y.max():.4f}")
    print(f"  mean: {all_y.mean():.4f}, std: {all_y.std():.4f}")
    print(f"  中位数: {all_y.median():.4f}")
    
    # 检查图特征
    print(f"\n🔍 检查图特征（前100个样本）...")
    node_dims = []
    edge_dims = []
    edge_attr_none_count = 0
    
    for i, data in enumerate(train_data[:100]):
        node_dims.append(data.x.shape[1])
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_dims.append(data.edge_attr.shape[1])
        else:
            edge_attr_none_count += 1
    
    print(f"  节点特征维度: {set(node_dims)} (应该是52)")
    if edge_dims:
        print(f"  边特征维度: {set(edge_dims)} (应该是24)")
    if edge_attr_none_count > 0:
        print(f"  ⚠️  警告: {edge_attr_none_count} 个样本缺少边特征!")
    
    return train_data

def check_model_weights():
    """检查模型权重"""
    print("\n" + "=" * 60)
    print("2. 检查模型权重")
    print("=" * 60)
    
    model_path = 'outputs/kcat_after_new/best_model.pt'
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型: {model_path}")
        return None
    
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    
    print(f"✅ 加载模型权重")
    
    # 检查各层权重的统计
    print(f"\n📊 权重统计:")
    
    # Node encoder
    node_enc_weight = state_dict['node_encoder.weight']
    print(f"  node_encoder.weight: mean={node_enc_weight.mean():.4f}, std={node_enc_weight.std():.4f}, range=[{node_enc_weight.min():.4f}, {node_enc_weight.max():.4f}]")
    
    # Attention layers
    att_layers = [k for k in state_dict.keys() if 'att_layers' in k and 'lin.weight' in k]
    print(f"\n  Attention层权重 ({len(att_layers)} 个):")
    for i, key in enumerate(sorted(att_layers)[:3]):  # 只显示前3个
        w = state_dict[key]
        print(f"    {key}: mean={w.mean():.4f}, std={w.std():.4f}, range=[{w.min():.4f}, {w.max():.4f}]")
    
    # MLP layers
    mlp_layers = [k for k in state_dict.keys() if 'mlp' in k and 'weight' in k]
    print(f"\n  MLP层权重:")
    for key in sorted(mlp_layers):
        w = state_dict[key]
        print(f"    {key}: mean={w.mean():.4f}, std={w.std():.4f}, range=[{w.min():.4f}, {w.max():.4f}]")
        if 'bias' in key.replace('weight', 'bias'):
            bias_key = key.replace('weight', 'bias')
            if bias_key in state_dict:
                b = state_dict[bias_key]
                if b.numel() == 1:
                    print(f"      bias: {b.item():.4f}")
                else:
                    print(f"      bias: mean={b.mean():.4f}, std={b.std():.4f}")
    
    return state_dict

def check_forward_pass():
    """检查前向传播过程"""
    print("\n" + "=" * 60)
    print("3. 检查前向传播过程")
    print("=" * 60)
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = MD.PocketGNNKcatOnly(
        node_input_dim=52,
        edge_input_dim=24,
        hidden_dim=128,
        num_layers=3,
        heads=4,
        dropout=0.1,
        pooling_type='mean',
        use_seq_embedding=False,
        use_mlp_layernorm=False
    ).to(device)
    
    model_path = 'outputs/kcat_after_new/best_model.pt'
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    # 加载测试数据
    test_data = torch.load('data/processed/kcat_test_new_backup.pt', weights_only=False)
    
    # 清理数据
    for data in test_data[:10]:
        keys_to_remove = []
        for key in data.keys():
            if not isinstance(getattr(data, key), torch.Tensor):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            delattr(data, key)
    
    # 检查几个样本的前向传播
    print(f"\n🔍 检查5个样本的前向传播:")
    
    for idx, sample in enumerate(test_data[:5]):
        batch = DataLoader([sample], batch_size=1)
        sample_batch = next(iter(batch)).to(device)
        
        with torch.no_grad():
            # 节点编码
            x = sample_batch.x
            x_encoded = model.node_encoder(x)
            x_encoded = torch.relu(x_encoded)
            
            # GAT层
            x_gat = x_encoded.clone()
            for i, layer in enumerate(model.att_layers):
                if i == 0:
                    x_gat = layer(x_gat, sample_batch.edge_index, edge_attr=sample_batch.edge_attr)
                else:
                    x_gat = layer(x_gat, sample_batch.edge_index)
                x_gat = F.elu(x_gat)
            
            # 池化
            from torch_geometric.nn import global_mean_pool
            graph_emb = global_mean_pool(x_gat, sample_batch.batch)
            
            # MLP
            x_mlp = graph_emb
            for i, layer in enumerate(model.mlp):
                if isinstance(layer, torch.nn.Linear):
                    x_mlp_before = x_mlp.clone()
                    x_mlp = layer(x_mlp)
                    if i == len(model.mlp) - 1:  # 最后一层
                        final_pred = x_mlp
                    else:
                        x_mlp = torch.relu(x_mlp)
                elif isinstance(layer, torch.nn.Dropout):
                    pass
                else:
                    x_mlp = layer(x_mlp)
            
            print(f"\n  样本 {idx+1}:")
            print(f"    节点特征: mean={x.mean():.4f}, std={x.std():.4f}")
            print(f"    编码后: mean={x_encoded.mean():.4f}, std={x_encoded.std():.4f}")
            print(f"    GAT后: mean={x_gat.mean():.4f}, std={x_gat.std():.4f}")
            print(f"    图embedding: mean={graph_emb.mean():.4f}, std={graph_emb.std():.4f}, range=[{graph_emb.min():.4f}, {graph_emb.max():.4f}]")
            print(f"    最终预测: {final_pred.item():.4f}, 真实值: {sample.y.item():.4f}")

def check_graph_embedding_diversity():
    """检查图embedding的多样性"""
    print("\n" + "=" * 60)
    print("4. 检查图embedding的多样性")
    print("=" * 60)
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = MD.PocketGNNKcatOnly(
        node_input_dim=52,
        edge_input_dim=24,
        hidden_dim=128,
        num_layers=3,
        heads=4,
        dropout=0.1,
        pooling_type='mean',
        use_seq_embedding=False,
        use_mlp_layernorm=False
    ).to(device)
    
    model_path = 'outputs/kcat_after_new/best_model.pt'
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    # 加载测试数据
    test_data = torch.load('data/processed/kcat_test_new_backup.pt', weights_only=False)
    
    # 清理数据
    for data in test_data[:100]:
        keys_to_remove = []
        for key in data.keys():
            if not isinstance(getattr(data, key), torch.Tensor):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            delattr(data, key)
    
    # 获取图embedding
    test_loader = DataLoader(test_data[:100], batch_size=32, shuffle=False)
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            emb = model.get_graph_embedding(batch)
            all_embeddings.append(emb.cpu())
            all_labels.append(batch.y.cpu())
    
    all_embeddings = torch.cat(all_embeddings).numpy()
    all_labels = torch.cat(all_labels).numpy().flatten()
    
    print(f"\n📊 图embedding统计 (100个样本):")
    print(f"  mean: {all_embeddings.mean():.4f}, std: {all_embeddings.std():.4f}")
    print(f"  range: [{all_embeddings.min():.4f}, {all_embeddings.max():.4f}]")
    
    # 检查embedding与标签的相关性
    print(f"\n🔍 Embedding与标签的相关性:")
    from scipy.stats import pearsonr
    
    # 对每个维度计算与标签的相关性
    correlations = []
    for dim in range(all_embeddings.shape[1]):
        try:
            corr = pearsonr(all_embeddings[:, dim], all_labels)[0]
            if not np.isnan(corr):
                correlations.append((dim, corr))
        except:
            pass
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"  前10个与标签最相关的维度:")
    for dim, corr in correlations[:10]:
        print(f"    维度 {dim}: Pearson={corr:.4f}")
    
    # 检查embedding的多样性（不同样本的embedding是否相似）
    print(f"\n🔍 Embedding多样性检查:")
    # 计算样本间embedding的相似度（余弦相似度）
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(all_embeddings[:20])  # 只检查前20个样本
    # 去除对角线（自己与自己的相似度）
    mask = ~np.eye(similarities.shape[0], dtype=bool)
    similarities_off_diag = similarities[mask]
    print(f"  前20个样本间的平均余弦相似度: {similarities_off_diag.mean():.4f}")
    print(f"  如果相似度接近1.0，说明embedding过于相似，缺乏区分度")

def main():
    print("🔬 深度调试分析")
    print("=" * 60)
    
    # 1. 检查训练数据
    train_data = check_training_data_quality()
    
    # 2. 检查模型权重
    state_dict = check_model_weights()
    
    # 3. 检查前向传播
    check_forward_pass()
    
    # 4. 检查图embedding多样性
    check_graph_embedding_diversity()
    
    print("\n" + "=" * 60)
    print("✅ 深度调试完成")
    print("=" * 60)

if __name__ == '__main__':
    main()
