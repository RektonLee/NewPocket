#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析模型输出问题 - 深入诊断
"""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from torch_geometric.loader import DataLoader
import GNN_model as MD
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def analyze_model_output():
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
    for data in test_data:
        keys_to_remove = []
        for key in data.keys():
            if not isinstance(getattr(data, key), torch.Tensor):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            delattr(data, key)
    
    # 分析模型内部激活
    print("🔍 分析模型内部激活值...")
    test_loader = DataLoader(test_data[:100], batch_size=10, shuffle=False)
    
    graph_embs = []
    mlp_inputs = []
    predictions = []
    true_values = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            true = batch.y.reshape(-1, 1)
            
            # 获取图embedding
            graph_emb = model.get_graph_embedding(batch)
            graph_embs.append(graph_emb.cpu())
            
            # 获取MLP输入（就是graph_emb，因为没有序列嵌入）
            mlp_inputs.append(graph_emb.cpu())
            
            # 获取预测
            pred = model(batch)
            predictions.append(pred.cpu())
            true_values.append(true.cpu())
    
    graph_embs = torch.cat(graph_embs).numpy()
    predictions = torch.cat(predictions).numpy().flatten()
    true_values = torch.cat(true_values).numpy().flatten()
    
    print(f"\n📊 激活值统计 (100个样本):")
    print(f"  图embedding:")
    print(f"    mean: {graph_embs.mean():.4f}, std: {graph_embs.std():.4f}")
    print(f"    range: [{graph_embs.min():.4f}, {graph_embs.max():.4f}]")
    print(f"  预测值:")
    print(f"    mean: {predictions.mean():.4f}, std: {predictions.std():.4f}")
    print(f"    range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"  真实值:")
    print(f"    mean: {true_values.mean():.4f}, std: {true_values.std():.4f}")
    print(f"    range: [{true_values.min():.4f}, {true_values.max():.4f}]")
    
    # 检查MLP各层的输出范围
    print(f"\n🔍 检查MLP各层输出范围...")
    sample = test_data[0]
    batch = DataLoader([sample], batch_size=1)
    sample_batch = next(iter(batch)).to(device)
    
    with torch.no_grad():
        graph_emb = model.get_graph_embedding(sample_batch)
        x = graph_emb
        
        print(f"  MLP输入 (graph_emb): mean={x.mean():.4f}, std={x.std():.4f}, range=[{x.min():.4f}, {x.max():.4f}]")
        
        for i, layer in enumerate(model.mlp):
            if isinstance(layer, torch.nn.Linear):
                x_before = x.clone()
                x = layer(x)
                print(f"  MLP[{i}] Linear({layer.in_features}, {layer.out_features}):")
                print(f"    输出: mean={x.mean():.4f}, std={x.std():.4f}, range=[{x.min():.4f}, {x.max():.4f}]")
                print(f"    权重: mean={layer.weight.data.mean():.4f}, std={layer.weight.data.std():.4f}")
                if layer.bias is not None:
                    if layer.bias.numel() == 1:
                        print(f"    偏置: {layer.bias.data.item():.4f}")
                    else:
                        print(f"    偏置: mean={layer.bias.data.mean():.4f}, std={layer.bias.data.std():.4f}")
            elif isinstance(layer, (torch.nn.ReLU, torch.nn.ELU)):
                x = layer(x)
                print(f"  MLP[{i}] {type(layer).__name__}: mean={x.mean():.4f}, range=[{x.min():.4f}, {x.max():.4f}]")
            elif isinstance(layer, torch.nn.Dropout):
                pass
            else:
                x = layer(x)
                if hasattr(x, 'mean'):
                    print(f"  MLP[{i}] {type(layer).__name__}: mean={x.mean():.4f}, range=[{x.min():.4f}, {x.max():.4f}]")
    
    # 绘制分布对比图
    print(f"\n📊 生成分布对比图...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(graph_embs.flatten(), bins=50, alpha=0.7, label='Graph Embedding')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Graph Embedding Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(predictions, bins=50, alpha=0.7, label='Predictions', color='orange')
    axes[1].hist(true_values, bins=50, alpha=0.7, label='True Values', color='green')
    axes[1].set_xlabel('log10(kcat)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Predictions vs True Values')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].scatter(true_values, predictions, alpha=0.5, s=10)
    axes[2].plot([true_values.min(), true_values.max()], 
                 [true_values.min(), true_values.max()], 'r--', linewidth=2)
    axes[2].set_xlabel('True log10(kcat)')
    axes[2].set_ylabel('Predicted log10(kcat)')
    axes[2].set_title('True vs Predicted (100 samples)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = 'outputs/model_analysis.png'
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ 已保存到: {output_path}")
    plt.close()

if __name__ == '__main__':
    analyze_model_output()
