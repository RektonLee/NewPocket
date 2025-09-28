#!/usr/bin/env python3
"""
测试注意力权重可视化功能
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualize_attention import AttentionVisualizer
from GNN_model import PocketGNNWithAttentionNoTemp
from torch_geometric.data import Data

def create_sample_data():
    """创建示例数据用于测试"""
    # 创建简单的图数据
    num_nodes = 20
    num_edges = 30
    
    # 节点特征（one-hot编码）
    x = torch.eye(num_nodes, 10)  # 10种原子类型
    
    # 边索引
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # 创建图数据
    data = Data(x=x, edge_index=edge_index)
    
    return data

def test_attention_visualization():
    """测试注意力可视化功能"""
    print("开始测试注意力可视化功能...")
    
    # 创建示例数据
    sample_data = create_sample_data()
    print(f"创建了包含 {sample_data.x.size(0)} 个节点的示例数据")
    
    # 创建模型
    model = PocketGNNWithAttentionNoTemp(
        node_input_dim=10,
        edge_input_dim=1,
        hidden_dim=64,
        num_layers=2,
        heads=2,
        dropout=0.1
    )
    
    # 创建虚拟的注意力权重用于测试
    num_edges = sample_data.edge_index.size(1)
    attention_weights = []
    
    # 模拟多层注意力权重
    for layer in range(2):
        # 创建随机注意力权重
        attn = torch.rand(1, 2, num_edges)  # [batch, heads, edges]
        attention_weights.append((None, attn))
    
    print(f"创建了 {len(attention_weights)} 层注意力权重")
    
    # 创建可视化器
    visualizer = AttentionVisualizer.__new__(AttentionVisualizer)
    visualizer.device = 'cpu'
    visualizer.model = model
    
    # 创建输出目录
    output_dir = "test_attention_output"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 测试注意力热力图
        print("生成注意力热力图...")
        visualizer.visualize_attention_heatmap(
            sample_data, attention_weights,
            os.path.join(output_dir, "test_heatmap.png")
        )
        
        # 测试注意力网络图
        print("生成注意力网络图...")
        visualizer.visualize_attention_network(
            sample_data, attention_weights,
            os.path.join(output_dir, "test_network.png")
        )
        
        # 测试注意力统计图
        print("生成注意力统计图...")
        visualizer.visualize_attention_statistics(
            attention_weights,
            os.path.join(output_dir, "test_stats.png")
        )
        
        # 测试注意力模式分析
        print("分析注意力模式...")
        analysis = visualizer.analyze_attention_patterns(sample_data, attention_weights)
        print("注意力模式分析结果:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")
        
        print(f"\n所有测试可视化结果已保存到: {output_dir}")
        print("测试完成！")
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_attention_visualization()
