!/usr/bin/env python3
"""
筛选成功预测的条目（有pdb文件的条目）
"""

import pandas as pd
import os
import re
from pathlib import Path

def extract_sample_id_from_pdb_path(pdb_path):
    """从pdb文件路径中提取sample_id"""
    # 例如: /home/lizihao/Work/enzyme_prediction/PGNN/kcat_full_after/samples/kcat_017576/kcat_017576_protein.pdb
    # 提取: kcat_017576
    match = re.search(r'kcat_(\d+)', pdb_path)
    if match:
        return f"kcat_{match.group(1)}"
    return None

def main():
    # 读取原始CSV文件
    csv_file = "/home/lizihao/Work/enzyme_prediction/PGNN/kcat_data_after_17574.csv"
    print(f"读取CSV文件: {csv_file}")
    
    df = pd.read_csv(csv_file)
    print(f"原始数据条目数: {len(df)}")
    
    # 读取成功预测的pdb文件列表
    pdb_files_path = "/home/lizihao/Work/enzyme_prediction/PGNN/successful_pdb_files.txt"
    print(f"读取成功预测的pdb文件列表: {pdb_files_path}")
    
    with open(pdb_files_path, 'r') as f:
        pdb_files = [line.strip() for line in f.readlines()]
    
    print(f"成功预测的pdb文件数量: {len(pdb_files)}")
    
    # 从pdb文件路径中提取sample_id
    successful_sample_ids = set()
    for pdb_file in pdb_files:
        sample_id = extract_sample_id_from_pdb_path(pdb_file)
        if sample_id:
            successful_sample_ids.add(sample_id)
    
    print(f"提取到的成功预测sample_id数量: {len(successful_sample_ids)}")
    
    # 筛选CSV数据
    successful_df = df[df['sample_id'].isin(successful_sample_ids)]
    
    print(f"筛选后的数据条目数: {len(successful_df)}")
    
    # 保存筛选后的CSV文件
    output_file = "/home/lizihao/Work/enzyme_prediction/PGNN/kcat_data_successful_pdb.csv"
    successful_df.to_csv(output_file, index=False)
    print(f"筛选后的数据已保存到: {output_file}")
    
    # 显示一些统计信息
    print("\n统计信息:")
    print(f"原始数据: {len(df)} 条")
    print(f"成功预测: {len(successful_df)} 条")
    print(f"成功率: {len(successful_df)/len(df)*100:.2f}%")
    
    # 显示前几个成功预测的sample_id
    print(f"\n前10个成功预测的sample_id:")
    for i, sample_id in enumerate(list(successful_sample_ids)[:10]):
        print(f"  {i+1}. {sample_id}")

if __name__ == "__main__":
    main()
