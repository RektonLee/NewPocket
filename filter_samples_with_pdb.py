#!/usr/bin/env python3
import pandas as pd
import os
import glob

def main():
    # 读取CSV文件
    print("正在读取 kcat_data_successful_pdb2.csv...")
    df = pd.read_csv('kcat_data_successful_pdb2.csv')
    print(f"原始数据行数: {len(df)}")
    
    # 获取所有有_10A.pdb文件的sample_id
    print("正在查找所有_10A.pdb文件...")
    pdb_files = glob.glob('kcat_full_after/samples/*/*_10A.pdb')
    print(f"找到 {len(pdb_files)} 个_10A.pdb文件")
    
    # 从文件路径中提取sample_id
    sample_ids_with_pdb = set()
    for pdb_file in pdb_files:
        # 从路径中提取sample_id，例如从 kcat_full_after/samples/kcat_019683/xxx_10A.pdb 提取 kcat_019683
        path_parts = pdb_file.split('/')
        if len(path_parts) >= 3:
            sample_id = path_parts[-2]  # 倒数第二个部分就是sample_id
            sample_ids_with_pdb.add(sample_id)
    
    print(f"找到 {len(sample_ids_with_pdb)} 个不同的sample_id有对应的_10A.pdb文件")
    
    # 过滤CSV数据，只保留有对应_10A.pdb文件的sample_id
    print("正在过滤数据...")
    filtered_df = df[df['sample_id'].isin(sample_ids_with_pdb)]
    print(f"过滤后数据行数: {len(filtered_df)}")
    
    # 保存到新的CSV文件
    output_file = 'kcat_after_model_new.csv'
    filtered_df.to_csv(output_file, index=False)
    print(f"已保存到 {output_file}")
    
    # 显示一些统计信息
    print(f"\n统计信息:")
    print(f"- 原始数据: {len(df)} 行")
    print(f"- 有_10A.pdb文件的sample_id: {len(sample_ids_with_pdb)} 个")
    print(f"- 过滤后数据: {len(filtered_df)} 行")
    print(f"- 覆盖率: {len(filtered_df)/len(df)*100:.2f}%")

if __name__ == "__main__":
    main()
