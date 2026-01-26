#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理 DLKcat 测试集：生成蛋白质结构、分子对接、口袋提取、构建图数据

步骤:
1. 加载 DLKcat 官方测试集
2. 使用 ESMFold 预测蛋白质结构
3. 使用 DiffDock 进行分子对接
4. 提取 10Å 结合口袋
5. 构建图数据
6. 保存为 .pt 文件供模型使用
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm
import hashlib
import traceback

# 项目路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from torch_geometric.data import Data


def shuffle_dataset(dataset, seed=1234):
    np.random.seed(seed)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    return [dataset[i] for i in indices]


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]


def load_dlkcat_test_set():
    """加载 DLKcat 官方测试集"""
    dlkcat_path = os.path.join(
        PROJECT_ROOT, 
        'benchmark_tools/DLKcat/DeeplearningApproach/Data/database/Kcat_combination_0918_wildtype_mutant.json'
    )
    
    with open(dlkcat_path, 'r') as f:
        data = json.load(f)
    
    # 官方划分
    dataset = shuffle_dataset(data, seed=1234)
    train_set, rest = split_dataset(dataset, 0.8)
    dev_set, test_set = split_dataset(rest, 0.5)
    
    return test_set


def predict_structure_esmfold(sequence, output_path):
    """使用 ESMFold 预测蛋白质结构"""
    from transformers import AutoTokenizer, EsmForProteinFolding
    from data_loader import convert_outputs_to_pdb
    
    # 序列过长限制
    if len(sequence) > 800:
        sequence = sequence[:400] + sequence[-400:]
        print(f"   序列截断为 {len(sequence)} 氨基酸")
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.cuda()
    
    tokenized_input = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)['input_ids']
    tokenized_input = tokenized_input.cuda()
    
    with torch.no_grad():
        output = model(tokenized_input)
    
    pdb_content = convert_outputs_to_pdb(output)[0]
    
    with open(output_path, 'w') as f:
        f.write(pdb_content)
    
    # 清理显存
    del model, tokenizer, output
    torch.cuda.empty_cache()
    
    return output_path


def process_single_sample(sample_id, sequence, smiles, kcat_value, output_dir, pdb_cache_dir):
    """处理单个样本：结构预测 → 对接 → 口袋提取 → 构图"""
    from docking import run_preprocess, clean_altlocs
    from graph_builder_rbf import build_graph, parse_pocket
    from build_graph_dataset import compute_angle_features, compute_dihedral_features
    
    # 1. 生成蛋白质 PDB
    pdb_path = os.path.join(pdb_cache_dir, f"{sample_id}.pdb")
    
    if not os.path.exists(pdb_path):
        print(f"   🔬 预测结构: {sample_id}")
        try:
            predict_structure_esmfold(sequence, pdb_path)
        except Exception as e:
            print(f"   ❌ 结构预测失败: {e}")
            return None
    else:
        print(f"   ✅ 使用缓存结构: {pdb_path}")
    
    # 2. 计算 pocket 文件名
    pocket_hash = int(hashlib.sha256(smiles.encode()).hexdigest(), 16) & 0xffff
    pocket_path = os.path.join(output_dir, f"{sample_id}_{pocket_hash}_10A.pdb")
    
    if os.path.exists(pocket_path):
        print(f"   ✅ 使用缓存口袋: {pocket_path}")
    else:
        # 3. 运行分子对接和口袋提取
        print(f"   🧪 运行对接和口袋提取...")
        try:
            success = run_preprocess(
                uniprot_id=sample_id,
                smiles=smiles,
                prot_pdb_path=pdb_path,
                output_pocket_path=pocket_path,
                index=0
            )
            if not success or not os.path.exists(pocket_path):
                print(f"   ❌ 对接失败")
                return None
        except Exception as e:
            print(f"   ❌ 对接异常: {e}")
            return None
    
    # 4. 构建图数据
    try:
        atoms = parse_pocket(pocket_path)
        if len(atoms) < 3:
            print(f"   ❌ 原子数太少: {len(atoms)}")
            return None
        
        data = build_graph(atoms, temperature=303.15)
        
        # 添加角度和二面角特征
        edge_index = data.edge_index
        pos = data.pos
        
        angle_features = compute_angle_features(edge_index, pos)
        dihedral_features = compute_dihedral_features(edge_index, pos)
        
        original_edge_attr = data.edge_attr
        enhanced_edge_attr = torch.cat([
            original_edge_attr,
            angle_features,
            dihedral_features
        ], dim=1)
        
        data.edge_attr = enhanced_edge_attr
        data.y = torch.log10(torch.tensor([kcat_value], dtype=torch.float))
        data.sample_id = sample_id
        data.pdb_id = f"{sample_id}_{pocket_hash}_10A.pdb"
        
        return data
        
    except Exception as e:
        print(f"   ❌ 构图失败: {e}")
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='data/processed/dlkcat_test_pockets')
    parser.add_argument('--pdb_cache_dir', type=str, default='output/dlkcat_pdb_cache')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=None)
    parser.add_argument('--save_interval', type=int, default=50, help='每处理多少个样本保存一次')
    args = parser.parse_args()
    
    print("=" * 60)
    print("处理 DLKcat 测试集")
    print("=" * 60)
    
    # 创建目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.pdb_cache_dir, exist_ok=True)
    
    # 加载测试集
    print("\n📊 加载 DLKcat 官方测试集...")
    test_set = load_dlkcat_test_set()
    print(f"   测试集样本数: {len(test_set)}")
    
    # 处理范围
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx else len(test_set)
    samples_to_process = test_set[start_idx:end_idx]
    print(f"   处理范围: [{start_idx}, {end_idx})")
    print(f"   待处理样本数: {len(samples_to_process)}")
    
    # 处理样本
    dataset = []
    failed_samples = []
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(PROJECT_ROOT, 'data/processed', f'dlkcat_test_{timestamp}.pt')
    
    for i, item in enumerate(tqdm(samples_to_process, desc="处理中")):
        global_idx = start_idx + i
        sample_id = f"dlkcat_test_{global_idx:05d}"
        
        sequence = item['Sequence']
        smiles = item['Smiles']
        kcat_value = float(item['Value'])
        
        # 跳过无效 SMILES
        if '.' in smiles:
            print(f"   ⚠️ 跳过多分子 SMILES: {sample_id}")
            failed_samples.append({'idx': global_idx, 'reason': 'multi-molecule SMILES'})
            continue
        
        if kcat_value <= 0:
            print(f"   ⚠️ 跳过无效 kcat: {sample_id}")
            failed_samples.append({'idx': global_idx, 'reason': 'invalid kcat'})
            continue
        
        print(f"\n[{i+1}/{len(samples_to_process)}] 处理 {sample_id}")
        
        data = process_single_sample(
            sample_id=sample_id,
            sequence=sequence,
            smiles=smiles,
            kcat_value=kcat_value,
            output_dir=args.output_dir,
            pdb_cache_dir=args.pdb_cache_dir
        )
        
        if data is not None:
            dataset.append(data)
            print(f"   ✅ 成功! 当前数据集大小: {len(dataset)}")
        else:
            failed_samples.append({'idx': global_idx, 'reason': 'processing failed'})
        
        # 定期保存
        if (i + 1) % args.save_interval == 0 and len(dataset) > 0:
            torch.save(dataset, save_path)
            print(f"\n💾 中间保存: {save_path} ({len(dataset)} 样本)")
    
    # 最终保存
    if len(dataset) > 0:
        torch.save(dataset, save_path)
        print(f"\n✅ 最终保存: {save_path}")
        print(f"   成功样本数: {len(dataset)}")
        print(f"   失败样本数: {len(failed_samples)}")
    else:
        print("\n❌ 没有成功处理任何样本")
    
    # 保存失败记录
    if failed_samples:
        failed_path = os.path.join(PROJECT_ROOT, 'results', f'dlkcat_test_failed_{timestamp}.json')
        os.makedirs(os.path.dirname(failed_path), exist_ok=True)
        with open(failed_path, 'w') as f:
            json.dump(failed_samples, f, indent=2)
        print(f"   失败记录: {failed_path}")


if __name__ == "__main__":
    main()






