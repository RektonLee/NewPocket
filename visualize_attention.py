#!/usr/bin/env python3
"""
注意力权重可视化脚本
用于可视化GNN模型中的注意力权重，展示模型关注的重点区域
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from torch_geometric.data import Data, DataLoader
import pandas as pd
import os
import sys
from typing import List, Tuple, Dict, Optional
import argparse

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from GNN_model import PocketGNNWithAttentionNoTemp
# from data_loader import EnzymeDataset  # 暂时注释掉，直接使用Data类

class AttentionVisualizer:
    """注意力权重可视化器"""
    
    def __init__(self, model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化可视化器
        
        Args:
            model_path: 模型文件路径
            device: 计算设备
        """
        self.device = device
        self.model = self._load_model(model_path)
        self.model.eval()
        
    def _load_model(self, model_path: str):
        """加载训练好的模型"""
        try:
            # 假设模型使用PocketGNNWithAttentionNoTemp
            model = PocketGNNWithAttentionNoTemp(
                num_atom_types=100,  # 根据实际数据调整
                hidden_dim=128,
                num_layers=3,
                heads=4,
                dropout=0.1
            )
            
            # 加载模型权重
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            return model
        except Exception as e:
            print(f"加载模型失败: {e}")
            return None
    
    def visualize_attention_heatmap(self, data: Data, attention_weights: List[torch.Tensor], 
                                  output_path: str = "attention_heatmap.png"):
        """
        可视化注意力权重热力图
        
        Args:
            data: 图数据
            attention_weights: 注意力权重列表
            output_path: 输出文件路径
        """
        if not attention_weights:
            print("没有注意力权重数据")
            return
        
        # 获取节点数量
        num_nodes = data.x.size(0)
        
        # 创建注意力权重矩阵
        attention_matrix = np.zeros((num_nodes, num_nodes))
        
        # 处理注意力权重（取最后一层的权重）
        if len(attention_weights) > 0:
            last_attention = attention_weights[-1]  # 最后一层
            if isinstance(last_attention, tuple):
                last_attention = last_attention[0]  # 取注意力权重部分
            
            # 将注意力权重转换为矩阵
            if last_attention.dim() == 3:  # [batch, heads, edges]
                attention_matrix = last_attention[0].mean(dim=0).cpu().numpy()
            elif last_attention.dim() == 2:  # [heads, edges]
                attention_matrix = last_attention.mean(dim=0).cpu().numpy()
        
        # 创建热力图
        plt.figure(figsize=(12, 10))
        
        # 使用自定义颜色映射
        cmap = LinearSegmentedColormap.from_list('attention', ['white', 'yellow', 'orange', 'red'])
        
        # 绘制热力图
        sns.heatmap(attention_matrix, 
                   cmap=cmap, 
                   annot=False, 
                   fmt='.3f',
                   cbar=True,
                   square=True,
                   linewidths=0.5)
        
        plt.title('Attention Weight Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Target Nodes', fontsize=12)
        plt.ylabel('Source Nodes', fontsize=12)
        
        # 保存图片
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"注意力热力图已保存到: {output_path}")
    
    def visualize_attention_network(self, data: Data, attention_weights: List[torch.Tensor],
                                  output_path: str = "attention_network.png"):
        """
        可视化注意力网络图
        
        Args:
            data: 图数据
            attention_weights: 注意力权重列表
            output_path: 输出文件路径
        """
        if not attention_weights:
            print("没有注意力权重数据")
            return
        
        # 获取节点和边信息
        edge_index = data.edge_index.cpu().numpy()
        num_nodes = data.x.size(0)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # 计算节点位置（使用圆形布局）
        angles = np.linspace(0, 2*np.pi, num_nodes, endpoint=False)
        radius = 5
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        
        # 绘制节点
        node_colors = ['lightblue'] * num_nodes
        for i, (xi, yi) in enumerate(zip(x, y)):
            circle = plt.Circle((xi, yi), 0.3, color=node_colors[i], alpha=0.8)
            ax.add_patch(circle)
            ax.text(xi, yi, str(i), ha='center', va='center', fontsize=8, fontweight='bold')
        
        # 绘制注意力边
        if len(attention_weights) > 0:
            last_attention = attention_weights[-1]
            if isinstance(last_attention, tuple):
                last_attention = last_attention[0]
            
            # 获取注意力权重
            if last_attention.dim() == 3:
                attn_weights = last_attention[0].mean(dim=0).cpu().numpy()
            else:
                attn_weights = last_attention.mean(dim=0).cpu().numpy()
            
            # 绘制边
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[:, i]
                if src < num_nodes and dst < num_nodes:
                    x_src, y_src = x[src], y[src]
                    x_dst, y_dst = x[dst], y[dst]
                    
                    # 获取注意力权重
                    if i < attn_weights.shape[0]:
                        weight = attn_weights[i]
                    else:
                        weight = 0.0
                    
                    # 根据权重设置颜色和透明度
                    alpha = min(0.8, max(0.1, weight))
                    color = 'red' if weight > 0.5 else 'blue'
                    
                    ax.plot([x_src, x_dst], [y_src, y_dst], 
                           color=color, alpha=alpha, linewidth=2)
        
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_aspect('equal')
        ax.set_title('Attention Network Visualization', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], color='red', lw=2, label='High Attention'),
            plt.Line2D([0], [0], color='blue', lw=2, label='Low Attention'),
            plt.Circle((0, 0), 0.3, color='lightblue', label='Nodes')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"注意力网络图已保存到: {output_path}")
    
    def visualize_attention_statistics(self, attention_weights: List[torch.Tensor],
                                     output_path: str = "attention_stats.png"):
        """
        可视化注意力权重统计信息
        
        Args:
            attention_weights: 注意力权重列表
            output_path: 输出文件路径
        """
        if not attention_weights:
            print("没有注意力权重数据")
            return
        
        # 收集所有注意力权重
        all_weights = []
        for layer_idx, layer_weights in enumerate(attention_weights):
            if isinstance(layer_weights, tuple):
                layer_weights = layer_weights[0]
            
            if layer_weights.dim() == 3:
                weights = layer_weights[0].flatten().cpu().numpy()
            else:
                weights = layer_weights.flatten().cpu().numpy()
            
            all_weights.extend(weights)
        
        all_weights = np.array(all_weights)
        
        # 创建统计图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 注意力权重分布直方图
        axes[0, 0].hist(all_weights, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Attention Weight Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Attention Weight')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 每层注意力权重箱线图
        layer_weights = []
        layer_labels = []
        for layer_idx, layer_weights in enumerate(attention_weights):
            if isinstance(layer_weights, tuple):
                layer_weights = layer_weights[0]
            
            if layer_weights.dim() == 3:
                weights = layer_weights[0].flatten().cpu().numpy()
            else:
                weights = layer_weights.flatten().cpu().numpy()
            
            layer_weights.append(weights)
            layer_labels.append(f'Layer {layer_idx + 1}')
        
        axes[0, 1].boxplot(layer_weights, labels=layer_labels)
        axes[0, 1].set_title('Attention Weights by Layer', fontweight='bold')
        axes[0, 1].set_ylabel('Attention Weight')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 注意力权重累积分布
        sorted_weights = np.sort(all_weights)
        cumulative = np.arange(1, len(sorted_weights) + 1) / len(sorted_weights)
        axes[1, 0].plot(sorted_weights, cumulative, linewidth=2, color='green')
        axes[1, 0].set_title('Cumulative Distribution of Attention Weights', fontweight='bold')
        axes[1, 0].set_xlabel('Attention Weight')
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 注意力权重统计信息
        stats_text = f"""
        Attention Weight Statistics:
        
        Mean: {np.mean(all_weights):.4f}
        Median: {np.median(all_weights):.4f}
        Std: {np.std(all_weights):.4f}
        Min: {np.min(all_weights):.4f}
        Max: {np.max(all_weights):.4f}
        
        Percentiles:
        25%: {np.percentile(all_weights, 25):.4f}
        75%: {np.percentile(all_weights, 75):.4f}
        90%: {np.percentile(all_weights, 90):.4f}
        95%: {np.percentile(all_weights, 95):.4f}
        """
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Statistical Summary', fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"注意力统计图已保存到: {output_path}")
    
    def analyze_attention_patterns(self, data: Data, attention_weights: List[torch.Tensor]) -> Dict:
        """
        分析注意力模式
        
        Args:
            data: 图数据
            attention_weights: 注意力权重列表
            
        Returns:
            注意力模式分析结果
        """
        if not attention_weights:
            return {}
        
        # 获取最后一层注意力权重
        last_attention = attention_weights[-1]
        if isinstance(last_attention, tuple):
            last_attention = last_attention[0]
        
        if last_attention.dim() == 3:
            attn_weights = last_attention[0].mean(dim=0).cpu().numpy()
        else:
            attn_weights = last_attention.mean(dim=0).cpu().numpy()
        
        # 分析注意力模式
        analysis = {
            'total_edges': len(attn_weights),
            'high_attention_edges': np.sum(attn_weights > 0.5),
            'low_attention_edges': np.sum(attn_weights < 0.1),
            'mean_attention': np.mean(attn_weights),
            'std_attention': np.std(attn_weights),
            'max_attention': np.max(attn_weights),
            'min_attention': np.min(attn_weights)
        }
        
        return analysis

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

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='注意力权重可视化')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--data_path', type=str, required=True, help='数据文件路径')
    parser.add_argument('--output_dir', type=str, default='attention_visualization', help='输出目录')
    parser.add_argument('--sample_idx', type=int, default=0, help='要可视化的样本索引')
    parser.add_argument('--device', type=str, default='auto', help='计算设备')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化可视化器
    visualizer = AttentionVisualizer(args.model_path, device)
    
    if visualizer.model is None:
        print("无法加载模型，退出")
        return
    
    # 加载数据 - 简化版本，直接创建示例数据
    try:
        # 创建示例数据用于测试
        print("创建示例数据用于测试...")
        sample_data = create_sample_data()
        sample_data = sample_data.to(device)
        
        # 获取注意力权重
        with torch.no_grad():
            output, attention_weights = visualizer.model(sample_data, return_attention_weights=True)
        
        print(f"样本 {args.sample_idx} 的预测结果: {output.cpu().numpy()}")
        print(f"注意力权重层数: {len(attention_weights)}")
        
        # 生成可视化
        base_name = f"sample_{args.sample_idx}"
        
        # 1. 注意力热力图
        visualizer.visualize_attention_heatmap(
            sample_data, attention_weights,
            os.path.join(args.output_dir, f"{base_name}_heatmap.png")
        )
        
        # 2. 注意力网络图
        visualizer.visualize_attention_network(
            sample_data, attention_weights,
            os.path.join(args.output_dir, f"{base_name}_network.png")
        )
        
        # 3. 注意力统计图
        visualizer.visualize_attention_statistics(
            attention_weights,
            os.path.join(args.output_dir, f"{base_name}_stats.png")
        )
        
        # 4. 分析注意力模式
        analysis = visualizer.analyze_attention_patterns(sample_data, attention_weights)
        print("\n注意力模式分析:")
        for key, value in analysis.items():
            print(f"{key}: {value}")
        
        print(f"\n所有可视化结果已保存到: {args.output_dir}")
        
    except Exception as e:
        print(f"处理数据时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
