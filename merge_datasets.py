#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并两个pt数据集的脚本
用于合并kcat_train_before.pt和kcat_train_after.pt
"""

import torch
import os
from torch_geometric.data import Data

def merge_datasets(pt_file1, pt_file2, output_file):
    """
    合并两个pt数据集
    根据build_graph_dataset.py的格式，pt文件是Data对象列表
    可以直接用列表拼接合并
    
    Args:
        pt_file1: 第一个pt文件路径
        pt_file2: 第二个pt文件路径  
        output_file: 输出合并后的pt文件路径
    """
    print(f"正在加载 {pt_file1}...")
    dataset1 = torch.load(pt_file1, weights_only=False)
    print(f"数据集1包含 {len(dataset1)} 个样本")
    
    print(f"正在加载 {pt_file2}...")
    dataset2 = torch.load(pt_file2, weights_only=False)
    print(f"数据集2包含 {len(dataset2)} 个样本")
    
    # 简单列表拼接合并
    print(f"\n正在合并数据集...")
    merged_dataset = dataset1 + dataset2
    print(f"合并后数据集包含 {len(merged_dataset)} 个样本")
    
    # 保存合并后的数据集
    print(f"正在保存到 {output_file}...")
    torch.save(merged_dataset, output_file)
    print(f"✅ 成功保存合并后的数据集到 {output_file}")
    
    # 显示格式信息
    if len(merged_dataset) > 0:
        sample = merged_dataset[0]
        print(f"\n📊 合并后数据集格式:")
        print(f"  节点特征维度: {sample.x.shape[1] if hasattr(sample, 'x') else 'N/A'}")
        print(f"  边特征维度: {sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') else 'N/A'}")
        print(f"  标签维度: {sample.y.shape if hasattr(sample, 'y') else 'N/A'}")
    
    return merged_dataset

def main():
    # 定义文件路径
    pt_file1 = 'kcat_train_before.pt'  # 第一个pt文件
    pt_file2 = 'kcat_train_after.pt'    # 第二个pt文件
    output_file = 'kcat_train_merged.pt'  # 合并后的输出文件
    
    # 检查文件是否存在
    if not os.path.exists(pt_file1):
        print(f"❌ 文件不存在: {pt_file1}")
        return
    
    if not os.path.exists(pt_file2):
        print(f"❌ 文件不存在: {pt_file2}")
        return
    
    # 执行合并
    merged_dataset = merge_datasets(pt_file1, pt_file2, output_file)
    
    # 显示合并后的统计信息
    print(f"\n📊 合并结果统计:")
    print(f"  总样本数: {len(merged_dataset)}")
    
    if len(merged_dataset) > 0:
        sample = merged_dataset[0]
        print(f"  节点特征维度: {sample.x.shape[1] if hasattr(sample, 'x') and sample.x is not None else 'N/A'}")
        print(f"  边特征维度: {sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else 'N/A'}")
        print(f"  标签维度: {sample.y.shape if hasattr(sample, 'y') and sample.y is not None else 'N/A'}")

if __name__ == '__main__':
    main()
