#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 .pt 文件中的 kcat 值，从 CSV 文件中读取正确的 experimental value[log10] 并更新
"""

import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path


def fix_kcat_values(pt_path, csv_path, output_path=None, backup=True):
    """
    从 CSV 文件中读取正确的 kcat 值并更新 .pt 文件
    
    Args:
        pt_path: 输入的 .pt 文件路径
        csv_path: CSV 文件路径，包含 sample_id 和 experimental value[log10]
        output_path: 输出文件路径（如果为 None，则覆盖原文件）
        backup: 是否创建备份
    """
    print(f"加载 .pt 文件: {pt_path}")
    dataset = torch.load(pt_path, weights_only=False)
    print(f"  数据集大小: {len(dataset)}")
    
    print(f"\n加载 CSV 文件: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  CSV 记录数: {len(df)}")
    
    # 检查必要的列
    if 'sample_id' not in df.columns:
        raise ValueError(f"CSV 文件缺少 'sample_id' 列")
    
    exp_col = 'experimental value[log10]'
    if exp_col not in df.columns:
        raise ValueError(f"CSV 文件缺少 '{exp_col}' 列")
    
    # 创建 sample_id -> log10_kcat 的映射
    id_to_kcat = dict(zip(df['sample_id'], df[exp_col]))
    print(f"  CSV 中有效的 sample_id 数: {len(id_to_kcat)}")
    
    # 统计信息
    updated_count = 0
    missing_count = 0
    unchanged_count = 0
    
    # 更新数据集
    print(f"\n开始更新 kcat 值...")
    for i, data in enumerate(dataset):
        sample_id = getattr(data, 'sample_id', None)
        
        if sample_id is None:
            print(f"  警告: 第 {i} 个数据没有 sample_id，跳过")
            missing_count += 1
            continue
        
        if sample_id not in id_to_kcat:
            print(f"  警告: sample_id '{sample_id}' 在 CSV 中未找到")
            missing_count += 1
            continue
        
        # 获取新的 kcat 值
        new_kcat_log10 = id_to_kcat[sample_id]
        
        # 检查是否为 NaN
        if pd.isna(new_kcat_log10):
            print(f"  警告: sample_id '{sample_id}' 的 kcat 值为 NaN，跳过")
            missing_count += 1
            continue
        
        # 获取旧值
        old_y = getattr(data, 'y', None)
        if old_y is not None:
            if isinstance(old_y, torch.Tensor):
                old_val = old_y.item() if old_y.numel() == 1 else old_y[0].item()
            else:
                old_val = old_y[0] if isinstance(old_y, (list, np.ndarray)) else old_y
        else:
            old_val = None
        
        # 更新 y 值
        # 根据 DEV_GUIDE.md，y 应该是 [1] 维度的 tensor，值为 log10(kcat)
        new_y = torch.tensor([new_kcat_log10], dtype=torch.float32)
        data.y = new_y
        
        # 统计
        if old_val is not None and abs(old_val - new_kcat_log10) < 1e-6:
            unchanged_count += 1
        else:
            updated_count += 1
            if updated_count <= 10:  # 只显示前10个更新
                print(f"  [{i}] {sample_id}: {old_val:.4f} -> {new_kcat_log10:.4f}")
    
    print(f"\n更新统计:")
    print(f"  已更新: {updated_count}")
    print(f"  未变化: {unchanged_count}")
    print(f"  缺失/跳过: {missing_count}")
    print(f"  总计: {len(dataset)}")
    
    # 创建备份
    if backup and output_path is None:
        backup_path = pt_path.replace('.pt', '_backup.pt')
        print(f"\n创建备份: {backup_path}")
        torch.save(torch.load(pt_path, weights_only=False), backup_path)
    
    # 保存更新后的文件
    if output_path is None:
        output_path = pt_path
    
    print(f"\n保存到: {output_path}")
    torch.save(dataset, output_path)
    print(f"✅ 完成！")


def main():
    parser = argparse.ArgumentParser(
        description="修复 .pt 文件中的 kcat 值，从 CSV 文件中读取正确的 experimental value[log10] 并更新"
    )
    parser.add_argument(
        "--pt_file", required=True,
        help="输入的 .pt 文件路径"
    )
    parser.add_argument(
        "--csv_file", required=True,
        help="CSV 文件路径，包含 sample_id 和 experimental value[log10]"
    )
    parser.add_argument(
        "--output", default=None,
        help="输出文件路径（默认覆盖原文件）"
    )
    parser.add_argument(
        "--no-backup", action="store_true",
        help="不创建备份文件"
    )
    
    args = parser.parse_args()
    
    fix_kcat_values(
        args.pt_file,
        args.csv_file,
        output_path=args.output,
        backup=not args.no_backup
    )


if __name__ == "__main__":
    main()

