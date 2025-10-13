#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
防止数据泄露的合理数据集划分
"""

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os

def analyze_dataset_similarity(dataset_path):
    """分析数据集中可能的相似性"""
    dataset = torch.load(dataset_path, weights_only=False)
    
    # 提取元数据
    ec_numbers = []
    pdb_ids = []
    sample_ids = []
    
    for data in dataset:
        if hasattr(data, 'ec'):
            ec_numbers.append(data.ec)
        if hasattr(data, 'pdb_id'):
            pdb_ids.append(data.pdb_id)
        if hasattr(data, 'sample_id'):
            sample_ids.append(data.sample_id)
    
    print(f"数据集大小: {len(dataset)}")
    print(f"唯一EC号数量: {len(set(ec_numbers))}")
    print(f"唯一PDB ID数量: {len(set(pdb_ids))}")
    print(f"唯一样本ID数量: {len(set(sample_ids))}")
    
    # 分析EC号分布
    ec_counts = defaultdict(int)
    for ec in ec_numbers:
        ec_counts[ec] += 1
    
    print(f"\nEC号分布 (前10个):")
    for ec, count in sorted(ec_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {ec}: {count} 个样本")
    
    return ec_numbers, pdb_ids, sample_ids

def split_by_ec_number(dataset_path, test_size=0.2, random_state=42):
    """按EC号进行分层划分，确保同一EC号的酶不会同时出现在训练集和测试集"""
    dataset = torch.load(dataset_path, weights_only=False)
    
    # 提取EC号
    ec_numbers = []
    for data in dataset:
        if hasattr(data, 'ec'):
            ec_numbers.append(data.ec)
        else:
            ec_numbers.append('unknown')
    
    # 按EC号分组
    ec_to_indices = defaultdict(list)
    for i, ec in enumerate(ec_numbers):
        ec_to_indices[ec].append(i)
    
    # 获取所有EC号
    unique_ecs = list(ec_to_indices.keys())
    print(f"发现 {len(unique_ecs)} 个不同的EC号")
    
    # 按EC号数量排序，确保大类别优先分配
    ec_sizes = [(ec, len(indices)) for ec, indices in ec_to_indices.items()]
    ec_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # 分层划分EC号
    train_ecs = []
    val_ecs = []
    
    for ec, size in ec_sizes:
        if len(train_ecs) == 0 or len(val_ecs) == 0:
            # 第一个EC号分配给训练集
            train_ecs.append(ec)
        else:
            # 根据当前比例决定分配
            current_train_ratio = len(train_ecs) / (len(train_ecs) + len(val_ecs))
            if current_train_ratio < 0.8:
                train_ecs.append(ec)
            else:
                val_ecs.append(ec)
    
    # 根据EC号分配样本索引
    train_indices = []
    val_indices = []
    
    for ec in train_ecs:
        train_indices.extend(ec_to_indices[ec])
    
    for ec in val_ecs:
        val_indices.extend(ec_to_indices[ec])
    
    print(f"训练集: {len(train_indices)} 个样本 ({len(train_ecs)} 个EC号)")
    print(f"验证集: {len(val_indices)} 个样本 ({len(val_ecs)} 个EC号)")
    
    return train_indices, val_indices

def split_by_protein_similarity(dataset_path, test_size=0.2, random_state=42):
    """按蛋白质相似性进行划分"""
    dataset = torch.load(dataset_path, weights_only=True)
    
    # 提取蛋白质信息
    protein_groups = defaultdict(list)
    
    for i, data in enumerate(dataset):
        if hasattr(data, 'pdb_id'):
            # 使用PDB ID的前4位作为蛋白质家族标识
            family_id = data.pdb_id[:4] if len(data.pdb_id) >= 4 else data.pdb_id
            protein_groups[family_id].append(i)
        else:
            protein_groups['unknown'].append(i)
    
    # 按蛋白质家族划分
    families = list(protein_groups.keys())
    train_families, val_families = train_test_split(
        families, test_size=test_size, random_state=random_state
    )
    
    train_indices = []
    val_indices = []
    
    for family in train_families:
        train_indices.extend(protein_groups[family])
    
    for family in val_families:
        val_indices.extend(protein_groups[family])
    
    print(f"按蛋白质家族划分:")
    print(f"训练集: {len(train_indices)} 个样本 ({len(train_families)} 个家族)")
    print(f"验证集: {len(val_indices)} 个样本 ({len(val_families)} 个家族)")
    
    return train_indices, val_indices

def create_proper_split(dataset_path, split_method='ec', test_size=0.2, random_state=42):
    """创建防止数据泄露的数据集划分"""
    
    print("🔍 分析数据集相似性...")
    analyze_dataset_similarity(dataset_path)
    
    if split_method == 'ec':
        print("\n📊 按EC号进行分层划分...")
        train_indices, val_indices = split_by_ec_number(dataset_path, test_size, random_state)
    elif split_method == 'protein':
        print("\n🧬 按蛋白质家族进行划分...")
        train_indices, val_indices = split_by_protein_similarity(dataset_path, test_size, random_state)
    else:
        raise ValueError("split_method 必须是 'ec' 或 'protein'")
    
    # 保存划分结果
    split_info = {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'split_method': split_method,
        'test_size': test_size,
        'random_state': random_state
    }
    
    output_path = dataset_path.replace('.pt', f'_split_{split_method}.pt')
    torch.save(split_info, output_path)
    
    print(f"\n✅ 划分结果已保存到: {output_path}")
    return train_indices, val_indices

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="kcat_dataset_enhanced1.pt")
    parser.add_argument('--method', type=str, choices=['ec', 'protein'], default='ec')
    parser.add_argument('--test_size', type=float, default=0.2)
    
    args = parser.parse_args()
    
    create_proper_split(args.dataset, args.method, args.test_size)
