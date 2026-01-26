#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建测试数据集脚本
从kcat_test_new.csv和kcat_test_results.csv合并数据，构建.pt测试数据集
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import torch
from tqdm import tqdm
from build_graph_dataset import build_graphs_for_sample, PoseSetData

def main():
    # 读取测试数据
    print("📥 读取测试数据...")
    test_new = pd.read_csv('data/processed/kcat_test_new.csv')
    test_results = pd.read_csv('data/processed/kcat_test_results.csv')
    
    print(f"   kcat_test_new.csv: {len(test_new)} 个样本")
    print(f"   kcat_test_results.csv: {len(test_results)} 个样本")
    
    # 合并数据
    merged = test_new.merge(test_results[['sample_id', 'smiles']], on='sample_id', how='inner')
    print(f"✅ 合并后: {len(merged)} 个样本")
    
    # 检查必要的列
    required_cols = ['sample_id', 'kcat', 'smiles']
    missing_cols = [col for col in required_cols if col not in merged.columns]
    if missing_cols:
        raise ValueError(f"缺少必要的列: {missing_cols}")
    
    # 设置pocket目录
    POCKET_BASE_DIR = 'sample_data/samples'
    SAVE_PATH = 'data/processed/kcat_test_new.pt'
    
    dataset = []
    successful_count = 0
    failed_count = 0
    successful_indices = []
    
    print(f"\n🔨 开始构建图数据集...")
    print(f"   Pocket目录: {POCKET_BASE_DIR}")
    print(f"   输出文件: {SAVE_PATH}")
    
    for idx, row in tqdm(merged.iterrows(), total=len(merged), desc="处理样本"):
        sample_id = row['sample_id']
        smiles = row['smiles']
        kcat_value = row['kcat']
        temperature = row.get('temperature', 303.15)
        ec = row.get('ec', 1)
        
        try:
            # 构建图（支持多pose）
            data = build_graphs_for_sample(
                sample_id=sample_id,
                smiles=smiles,
                kcat_value=kcat_value,
                pocket_base_dir=POCKET_BASE_DIR,
                temperature=temperature,
                use_multipose=True,  # 启用多pose模式
                verbose=False
            )
            
            if data is None:
                failed_count += 1
                continue
            
            # 添加ec信息
            if isinstance(data, PoseSetData):
                for g in data.graphs:
                    g.ec = ec
            else:
                data.ec = ec
            
            dataset.append(data)
            successful_indices.append(idx)
            successful_count += 1
            
        except Exception as e:
            print(f'\n❌ Error processing {sample_id}: {e}')
            failed_count += 1
            continue
    
    # 保存pt文件
    print(f"\n💾 保存数据集...")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(dataset, SAVE_PATH)
    print(f'✅ 保存了 {len(dataset)} 个样本到 {SAVE_PATH}')
    
    # 统计多pose样本数量
    multipose_count = sum(1 for d in dataset if isinstance(d, PoseSetData))
    if multipose_count > 0:
        print(f'   - {multipose_count} 个样本包含多个pose')
        print(f'   - {len(dataset) - multipose_count} 个样本只有单个pose')
    
    # 保存成功处理的CSV文件
    successful_df = merged.loc[successful_indices].copy()
    successful_csv_path = SAVE_PATH.replace('.pt', '_successful.csv')
    successful_df.to_csv(successful_csv_path, index=False)
    print(f'✅ 保存了 {len(successful_df)} 个成功样本到 {successful_csv_path}')
    
    print(f'\n📊 统计:')
    print(f'   成功: {successful_count}, 失败: {failed_count}')
    if successful_count + failed_count > 0:
        print(f'   成功率: {successful_count/(successful_count+failed_count)*100:.1f}%')

if __name__ == '__main__':
    main()
