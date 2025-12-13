#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 .pt 文件中提取基本信息并导出为 CSV

使用方法:
    python scripts/export_pt_to_csv.py \
        --pt_file data/processed/kcat_test_fixed.pt \
        --output data/processed/kcat_test_fixed.csv
"""

import argparse
import torch
import pandas as pd
import numpy as np


def extract_data_info(pt_path):
    """从 .pt 文件中提取基本信息"""
    print(f"加载数据集: {pt_path}")
    dataset = torch.load(pt_path, weights_only=False)
    
    records = []
    for i, data in enumerate(dataset):
        record = {}
        
        # 基本标识信息
        record['index'] = i
        record['sample_id'] = getattr(data, 'sample_id', None)
        record['pdb_id'] = getattr(data, 'pdb_id', None)
        record['ec'] = getattr(data, 'ec', None)
        
        # 标签值 (log10(kcat))
        y = getattr(data, 'y', None)
        if y is not None:
            if isinstance(y, torch.Tensor):
                y_val = y.item() if y.numel() == 1 else y.tolist()
            else:
                y_val = y
            record['log10_kcat'] = y_val
            # 如果 y 是 log10，计算原始 kcat
            # if isinstance(y_val, (int, float)):
            record['kcat'] = 10 ** y_val
            # else:
            #     record['kcat'] = None
        else:
            record['log10_kcat'] = None
            record['kcat'] = None
        
        # 图统计信息（可选，但不包含实际嵌入）
        x = getattr(data, 'x', None)
        if x is not None and isinstance(x, torch.Tensor):
            record['num_nodes'] = x.shape[0]
            record['num_node_features'] = x.shape[1] if len(x.shape) > 1 else None
        else:
            record['num_nodes'] = None
            record['num_node_features'] = None
        
        edge_index = getattr(data, 'edge_index', None)
        if edge_index is not None and isinstance(edge_index, torch.Tensor):
            record['num_edges'] = edge_index.shape[1] if len(edge_index.shape) > 1 else 0
        else:
            record['num_edges'] = None
        
        # 其他可能存在的属性
        record['temperature'] = getattr(data, 'temperature', None)
        if record['temperature'] is not None and isinstance(record['temperature'], torch.Tensor):
            record['temperature'] = record['temperature'].item() if record['temperature'].numel() == 1 else record['temperature'].tolist()
        
        pos = getattr(data, 'pos', None)
        if pos is not None and isinstance(pos, torch.Tensor):
            record['has_pos'] = True
        else:
            record['has_pos'] = False
        
        records.append(record)
    
    print(f"  提取了 {len(records)} 条记录")
    return records


def main():
    parser = argparse.ArgumentParser(
        description="从 .pt 文件中提取基本信息并导出为 CSV"
    )
    parser.add_argument(
        "--pt_file", required=True,
        help="输入的 .pt 文件路径"
    )
    parser.add_argument(
        "--output", required=True,
        help="输出的 CSV 文件路径"
    )
    
    args = parser.parse_args()
    
    # 提取数据
    records = extract_data_info(args.pt_file)
    
    # 转换为 DataFrame
    df = pd.DataFrame(records)
    
    # 重新排列列的顺序，让重要信息在前面
    preferred_order = ['index', 'sample_id', 'pdb_id', 'ec', 'log10_kcat', 'kcat', 
                       'num_nodes', 'num_edges', 'num_node_features', 'temperature', 'has_pos']
    # 只保留存在的列
    columns = [col for col in preferred_order if col in df.columns]
    # 添加其他列
    other_columns = [col for col in df.columns if col not in columns]
    df = df[columns + other_columns]
    
    # 保存为 CSV
    df.to_csv(args.output, index=False)
    print(f"\n✅ 已保存到: {args.output}")
    print(f"   总记录数: {len(df)}")
    print(f"   列数: {len(df.columns)}")
    print(f"\n前5行预览:")
    print(df.head().to_string())
    
    # 统计信息
    print(f"\n统计信息:")
    print(f"  唯一 sample_id 数: {df['sample_id'].nunique()}")
    print(f"  唯一 pdb_id 数: {df['pdb_id'].nunique()}")
    if 'log10_kcat' in df.columns:
        print(f"  log10_kcat 范围: [{df['log10_kcat'].min():.4f}, {df['log10_kcat'].max():.4f}]")
        print(f"  log10_kcat 均值: {df['log10_kcat'].mean():.4f}")
        print(f"  log10_kcat 缺失数: {df['log10_kcat'].isna().sum()}")


if __name__ == "__main__":
    main()

