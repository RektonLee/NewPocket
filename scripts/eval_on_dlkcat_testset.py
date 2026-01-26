#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 DLKcat/UniKP 官方数据集上评估我们的 Pocket-GAT 方法

DLKcat 数据集:
- 来源: benchmark_tools/DLKcat/DeeplearningApproach/Data/database/Kcat_combination_0918_wildtype_mutant.json
- 总样本: 17010 (9529 wildtype + 7481 mutant)
- 官方划分: 80% train, 10% dev, 10% test (seed=1234)

步骤:
1. 加载 DLKcat 官方数据集
2. 使用相同的划分方式获取测试集
3. 在测试集上运行我们的方法（需要先生成 pocket 图数据）
4. 评估并与 DLKcat/UniKP 对比
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

# 项目路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

# 数据路径
DLKCAT_DATA_PATH = os.path.join(
    PROJECT_ROOT, 
    'benchmark_tools/DLKcat/DeeplearningApproach/Data/database/Kcat_combination_0918_wildtype_mutant.json'
)


def shuffle_dataset(dataset, seed=1234):
    """与 DLKcat 相同的 shuffle 方式"""
    np.random.seed(seed)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    return [dataset[i] for i in indices]


def split_dataset(dataset, ratio):
    """与 DLKcat 相同的 split 方式"""
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]


def load_dlkcat_dataset():
    """加载 DLKcat 官方数据集"""
    print(f"📊 加载 DLKcat 数据集: {DLKCAT_DATA_PATH}")
    
    with open(DLKCAT_DATA_PATH, 'r') as f:
        data = json.load(f)
    
    print(f"   总样本数: {len(data)}")
    
    # 统计
    wildtype_count = sum(1 for d in data if d.get('Type') == 'wildtype')
    mutant_count = sum(1 for d in data if d.get('Type') == 'mutant')
    print(f"   Wildtype: {wildtype_count}, Mutant: {mutant_count}")
    
    return data


def get_official_test_set(data, seed=1234):
    """
    获取与 DLKcat 官方相同的测试集
    官方划分: 80% train, 10% dev, 10% test
    """
    dataset = shuffle_dataset(data, seed)
    dataset_train, dataset_rest = split_dataset(dataset, 0.8)
    dataset_dev, dataset_test = split_dataset(dataset_rest, 0.5)
    
    print(f"\n📊 数据集划分 (与 DLKcat 官方相同):")
    print(f"   Train: {len(dataset_train)}")
    print(f"   Dev: {len(dataset_dev)}")
    print(f"   Test: {len(dataset_test)}")
    
    return dataset_train, dataset_dev, dataset_test


def prepare_test_data(test_set, output_csv):
    """
    将测试集转换为 CSV 格式，用于我们的方法处理
    """
    records = []
    for i, item in enumerate(test_set):
        records.append({
            'sample_id': f'dlkcat_test_{i:05d}',
            'sequence': item['Sequence'],
            'substrate_smiles': item['Smiles'],
            'kcat_value': float(item['Value']),
            'ec_number': item.get('ECNumber', ''),
            'organism': item.get('Organism', ''),
            'substrate_name': item.get('Substrate', ''),
            'type': item.get('Type', 'wildtype'),
        })
    
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ 测试集已保存: {output_csv}")
    print(f"   样本数: {len(df)}")
    
    return df


def evaluate_predictions(y_true, y_pred, prefix=""):
    """计算评估指标"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]
    
    if len(y_true_valid) < 2:
        return {f"{prefix}n_valid": 0}
    
    pearson_r, _ = pearsonr(y_true_valid, y_pred_valid)
    spearman_r, _ = spearmanr(y_true_valid, y_pred_valid)
    
    metrics = {
        f"{prefix}n_valid": len(y_true_valid),
        f"{prefix}r2": r2_score(y_true_valid, y_pred_valid),
        f"{prefix}pearson_r": pearson_r,
        f"{prefix}spearman_r": spearman_r,
        f"{prefix}rmse": np.sqrt(mean_squared_error(y_true_valid, y_pred_valid)),
        f"{prefix}mae": mean_absolute_error(y_true_valid, y_pred_valid),
    }
    return metrics


def main():
    """主函数"""
    print("=" * 60)
    print("在 DLKcat 官方测试集上评估")
    print("=" * 60)
    
    # 1. 加载数据集
    data = load_dlkcat_dataset()
    
    # 2. 获取官方测试集
    train_set, dev_set, test_set = get_official_test_set(data)
    
    # 3. 准备测试数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(PROJECT_ROOT, 'results', f'dlkcat_official_test_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    test_csv = os.path.join(output_dir, 'dlkcat_test_set.csv')
    test_df = prepare_test_data(test_set, test_csv)
    
    # 4. 显示 kcat 分布
    kcat_values = test_df['kcat_value'].values
    log_kcat = np.log10(kcat_values + 1e-10)
    
    print(f"\n📊 测试集 kcat 分布:")
    print(f"   kcat 范围: [{kcat_values.min():.4f}, {kcat_values.max():.4f}]")
    print(f"   log10(kcat) 范围: [{log_kcat.min():.2f}, {log_kcat.max():.2f}]")
    print(f"   log10(kcat) 中位数: {np.median(log_kcat):.2f}")
    
    # 5. 按类型分析
    wildtype_df = test_df[test_df['type'] == 'wildtype']
    mutant_df = test_df[test_df['type'] == 'mutant']
    print(f"\n📊 测试集类型分布:")
    print(f"   Wildtype: {len(wildtype_df)}")
    print(f"   Mutant: {len(mutant_df)}")
    
    # 6. 提示后续步骤
    print("\n" + "=" * 60)
    print("📋 后续步骤:")
    print("=" * 60)
    print(f"""
要在此测试集上评估 Pocket-GAT 方法，需要:

1. 生成蛋白质结构 (ESMFold/AlphaFold):
   - 共 {len(test_df)} 个蛋白质序列需要结构预测
   
2. 分子对接 (DiffDock):
   - 获取蛋白质-底物复合物结构
   
3. 口袋提取:
   - 提取 10Å 结合口袋
   
4. 构建图数据集:
   - 运行 build_graph_dataset.py
   
5. 模型推理:
   - 加载训练好的模型进行预测

由于结构预测和对接步骤耗时较长，建议:
- 先在小批量 (100-500) 样本上测试
- 使用已有的结构缓存

测试集 CSV 已保存: {test_csv}
""")
    
    # 保存数据集信息
    info = {
        'total_samples': len(data),
        'train_samples': len(train_set),
        'dev_samples': len(dev_set),
        'test_samples': len(test_set),
        'wildtype_in_test': len(wildtype_df),
        'mutant_in_test': len(mutant_df),
        'seed': 1234,
        'split_ratio': '80:10:10'
    }
    
    info_path = os.path.join(output_dir, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"✅ 数据集信息已保存: {info_path}")
    
    return test_df, output_dir


if __name__ == "__main__":
    test_df, output_dir = main()






