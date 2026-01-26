#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版脚本：将CSV数据转换为PyTorch Geometric Data对象
增加了角度特征和几何增强
用于构建训练数据集
"""

import pandas as pd
import os
import hashlib
import torch
import numpy as np
from torch_geometric.data import Data
from Bio.PDB import PDBParser
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import logging
import math
import json
import glob

# 导入graph_builder_rbf中的函数
from graph_builder_rbf import build_graph, parse_pocket, gaussian_rbf


class PoseSetData:
    """
    多pose数据集结构
    存储一个样本的多个pose图
    
    Attributes:
        graphs: list[Data] - K个图的列表
        y: torch.Tensor - 标签值 [1]
        pose_scores: torch.Tensor - pose置信度分数 [K]（可选）
        sample_id: str - 样本ID
        metadata: dict - 其他元数据
    """
    def __init__(self, graphs, y, sample_id=None, pose_scores=None, metadata=None):
        self.graphs = graphs  # list[Data]
        self.y = y  # [1]
        self.sample_id = sample_id
        self.pose_scores = pose_scores  # [K] 或 None
        self.metadata = metadata or {}
    
    def __len__(self):
        return len(self.graphs)
    
    def __repr__(self):
        return f"PoseSetData(n_poses={len(self.graphs)}, sample_id={self.sample_id})"

def compute_angle_features(edge_index, pos, max_neighbors=10):
    """
    计算键角特征 (Bond Angles)
    对每条边，计算其与相邻边的夹角
    
    Args:
        edge_index: [2, E] 边索引
        pos: [N, 3] 原子坐标
        max_neighbors: 最大邻居数，用于控制计算复杂度
    
    Returns:
        angle_features: [E, angle_dim] 角度特征
    """
    row, col = edge_index
    num_edges = edge_index.shape[1]
    
    # 计算边向量
    edge_vec = pos[col] - pos[row]  # [E, 3]
    edge_length = torch.norm(edge_vec, dim=1, keepdim=True)  # [E, 1]
    edge_vec_norm = edge_vec / (edge_length + 1e-8)  # [E, 3] 归一化
    
    angle_features = []
    
    for i in range(num_edges):
        center_atom = row[i]  # 中心原子
        neighbor_atom = col[i]  # 邻居原子
        
        # 找到中心原子的所有邻居（除了当前邻居）
        center_neighbors = edge_index[1][edge_index[0] == center_atom]
        other_neighbors = center_neighbors[center_neighbors != neighbor_atom]
        
        if len(other_neighbors) == 0:
            # 如果没有其他邻居，使用零向量
            angle_features.append(torch.zeros(4))  # [cos_min, cos_max, cos_mean, num_angles]
            continue
        
        # 限制邻居数量以控制计算复杂度
        if len(other_neighbors) > max_neighbors:
            other_neighbors = other_neighbors[:max_neighbors]
        
        # 计算当前边与其他边的夹角余弦值
        current_vec = edge_vec_norm[i]  # [3]
        cos_angles = []
        
        for other_neighbor in other_neighbors:
            # 找到对应的边索引
            other_edge_idx = ((edge_index[0] == center_atom) & (edge_index[1] == other_neighbor)).nonzero(as_tuple=True)[0]
            if len(other_edge_idx) > 0:
                other_vec = edge_vec_norm[other_edge_idx[0]]  # [3]
                cos_angle = torch.dot(current_vec, other_vec).clamp(-1, 1)
                cos_angles.append(cos_angle)
        
        if len(cos_angles) > 0:
            cos_angles = torch.stack(cos_angles)
            # 统计特征：最小值、最大值、均值、角度数量
            angle_stats = torch.tensor([
                cos_angles.min(),
                cos_angles.max(), 
                cos_angles.mean(),
                len(cos_angles) / max_neighbors  # 归一化的角度数量
            ])
        else:
            angle_stats = torch.zeros(4)
        
        angle_features.append(angle_stats)
    
    return torch.stack(angle_features)  # [E, 4]

def compute_dihedral_features(edge_index, pos, max_dihedrals=5):
    """
    计算二面角特征 (Dihedral Angles)
    对每条边，计算涉及该边的二面角
    
    Args:
        edge_index: [2, E] 边索引
        pos: [N, 3] 原子坐标
        max_dihedrals: 最大二面角数量
    
    Returns:
        dihedral_features: [E, dihedral_dim] 二面角特征
    """
    row, col = edge_index
    num_edges = edge_index.shape[1]
    
    dihedral_features = []
    
    for i in range(num_edges):
        atom_b = row[i]  # 中心边的起点
        atom_c = col[i]  # 中心边的终点
        
        # 找到atom_b和atom_c的邻居
        b_neighbors = edge_index[1][edge_index[0] == atom_b]
        c_neighbors = edge_index[1][edge_index[0] == atom_c]
        
        # 去除中心边上的原子
        b_others = b_neighbors[b_neighbors != atom_c]
        c_others = c_neighbors[c_neighbors != atom_b]
        
        dihedrals = []
        count = 0
        
        # 计算二面角 A-B-C-D
        for atom_a in b_others:
            for atom_d in c_others:
                if count >= max_dihedrals:
                    break
                
                # 计算二面角
                vec_ba = pos[atom_a] - pos[atom_b]
                vec_bc = pos[atom_c] - pos[atom_b]
                vec_cb = pos[atom_b] - pos[atom_c]
                vec_cd = pos[atom_d] - pos[atom_c]
                
                # 计算法向量
                n1 = torch.cross(vec_ba, vec_bc)
                n2 = torch.cross(vec_cb, vec_cd)
                
                # 归一化
                n1_norm = n1 / (torch.norm(n1) + 1e-8)
                n2_norm = n2 / (torch.norm(n2) + 1e-8)
                
                # 计算二面角余弦值
                cos_dihedral = torch.dot(n1_norm, n2_norm).clamp(-1, 1)
                dihedrals.append(cos_dihedral)
                count += 1
            
            if count >= max_dihedrals:
                break
        
        if len(dihedrals) > 0:
            dihedrals = torch.stack(dihedrals)
            # 统计特征
            dihedral_stats = torch.tensor([
                dihedrals.min(),
                dihedrals.max(),
                dihedrals.mean(),
                len(dihedrals) / max_dihedrals  # 归一化的二面角数量
            ])
        else:
            dihedral_stats = torch.zeros(4)
        
        dihedral_features.append(dihedral_stats)
    
    return torch.stack(dihedral_features)  # [E, 4]

def enhanced_build_graph(atoms, temperature, verbose=False):
    """
    增强版图构建函数，添加角度和二面角特征
    
    Args:
        atoms: 原子列表
        temperature: 温度
        verbose: 是否打印详细信息
    
    Returns:
        Data对象，包含24维边特征
    """
    # 使用原始的build_graph函数
    data = build_graph(atoms, temperature)
    
    # 提取边信息
    edge_index = data.edge_index
    pos = data.pos
    
    # 计算角度特征
    if verbose:
        print("计算键角特征...")
    angle_features = compute_angle_features(edge_index, pos)
    
    # 计算二面角特征  
    if verbose:
        print("计算二面角特征...")
    dihedral_features = compute_dihedral_features(edge_index, pos)
    
    # 将新特征添加到边特征中
    original_edge_attr = data.edge_attr  # [E, 16] (RBF features)
    enhanced_edge_attr = torch.cat([
        original_edge_attr,      # [E, 16] 原始RBF特征
        angle_features,          # [E, 4]  键角特征
        dihedral_features        # [E, 4]  二面角特征
    ], dim=1)  # [E, 24]
    
    # 更新数据对象
    data.edge_attr = enhanced_edge_attr
    
    if verbose:
        print(f"边特征维度从 {original_edge_attr.shape[1]} 增加到 {enhanced_edge_attr.shape[1]}")
    
    return data


def find_pose_files(base_path, sample_id, pocket_hash):
    """
    查找所有pose文件（支持多pose）
    
    Args:
        base_path: pocket基础目录
        sample_id: 样本ID
        pocket_hash: pocket哈希值
    
    Returns:
        list: pose文件路径列表，按pose索引排序
    """
    base_name = f'{sample_id}_{pocket_hash}'
    
    # 首先尝试查找多pose文件（pose_0.pdb, pose_1.pdb等）
    pose_files = []
    pose_pattern = os.path.join(base_path, sample_id, f'{base_name}_pose*.pdb')
    found_files = glob.glob(pose_pattern)
    
    if found_files:
        # 提取pose索引并排序
        import re
        pose_files_with_idx = []
        for f in found_files:
            m = re.search(r'pose(\d+)\.pdb$', f)
            if m:
                idx = int(m.group(1))
                pose_files_with_idx.append((idx, f))
        pose_files_with_idx.sort(key=lambda x: x[0])
        pose_files = [f for _, f in pose_files_with_idx]
    else:
        # 回退到单pose文件
        single_pose = os.path.join(base_path, sample_id, f'{base_name}_10A.pdb')
        if os.path.exists(single_pose):
            pose_files = [single_pose]
    
    return pose_files


def load_pose_metadata(sample_dir):
    """
    从docking_meta.json加载pose元数据（confidence等）
    
    Args:
        sample_dir: 样本目录（包含diffdock_output）
    
    Returns:
        dict: pose元数据，包含confidence_map等
    """
    meta_paths = [
        os.path.join(sample_dir, 'diffdock_output', 'docking_meta.json'),
        os.path.join(sample_dir, 'docking_meta.json'),
    ]
    
    for meta_path in meta_paths:
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load metadata from {meta_path}: {e}")
    
    return None

def build_graphs_for_sample(sample_id, smiles, kcat_value, pocket_base_dir, 
                            temperature=303.15, use_multipose=True, verbose=False):
    """
    为单个样本构建图（支持多pose）
    
    Args:
        sample_id: 样本ID
        smiles: SMILES字符串
        kcat_value: kcat值
        pocket_base_dir: pocket基础目录
        temperature: 温度
        use_multipose: 是否使用多pose（如果可用）
        verbose: 是否打印详细信息
    
    Returns:
        PoseSetData或Data对象（如果只有一个pose）
    """
    pocket_hash = int(hashlib.sha256(smiles.encode()).hexdigest(), 16) & 0xffff
    
    # 查找所有pose文件
    pose_files = find_pose_files(pocket_base_dir, sample_id, pocket_hash)
    
    if not pose_files:
        return None
    
    # 加载pose元数据（confidence等）
    sample_dir = os.path.join(pocket_base_dir, sample_id)
    metadata = load_pose_metadata(sample_dir)
    
    graphs = []
    pose_scores = []
    
    for pose_idx, pose_file in enumerate(pose_files):
        try:
            atoms = parse_pocket(pose_file)
            if len(atoms) < 3:
                if verbose:
                    print(f"Warning: {pose_file} has too few atoms ({len(atoms)})")
                continue
            
            # 构建图
            data = enhanced_build_graph(atoms, temperature, verbose=verbose)
            data.pdb_id = os.path.basename(pose_file)
            data.sample_id = sample_id
            data.pose_id = pose_idx
            
            graphs.append(data)
            
            # 尝试从元数据获取confidence
            if metadata and 'confidence_map' in metadata:
                conf = metadata['confidence_map'].get(os.path.basename(pose_file))
                if conf is not None:
                    pose_scores.append(conf)
                else:
                    pose_scores.append(0.0)  # 默认分数
            else:
                pose_scores.append(0.0)
                
        except Exception as e:
            if verbose:
                print(f'Error processing pose {pose_idx} for {sample_id}: {e}')
            continue
    
    if not graphs:
        return None
    
    # 创建标签
    y = torch.log10(torch.tensor([kcat_value], dtype=torch.float))
    
    # 如果只有一个pose且不使用多pose模式，返回单个Data对象（向后兼容）
    if len(graphs) == 1 and not use_multipose:
        data = graphs[0]
        data.y = y
        return data
    
    # 否则返回PoseSetData
    pose_scores_tensor = torch.tensor(pose_scores, dtype=torch.float) if pose_scores else None
    
    return PoseSetData(
        graphs=graphs,
        y=y,
        sample_id=sample_id,
        pose_scores=pose_scores_tensor,
        metadata=metadata or {}
    )


def main():
    print("开始执行main函数...")
    import argparse
    
    parser = argparse.ArgumentParser(description='构建图数据集（支持多pose）')
    parser.add_argument('--input', type=str, default='kcat_data_with_ids.csv', help='输入CSV文件')
    parser.add_argument('--output', type=str, default='kcat_train_full_1213.pt', help='输出.pt文件')
    parser.add_argument('--pocket_dir', type=str, 
                       default='/home/lizihao/Work/enzyme_prediction/PGNN/sample_data/samples',
                       help='Pocket基础目录')
    parser.add_argument('--use_multipose', action='store_true', 
                       help='使用多pose模式（如果可用）')
    parser.add_argument('--temperature', type=float, default=303.15, help='温度')
    
    args = parser.parse_args()
    
    # 读取训练数据
    df = pd.read_csv(args.input)
    print(f'Loading {len(df)} samples from CSV')

    dataset = []
    successful_count = 0
    failed_count = 0
    successful_indices = []  # 记录成功处理的样本索引

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        sample_id = row['sample_id']
        smiles = row['substrate_smiles']
        kcat_value = row['kcat_value']
        ec = row.get('ec', 1)  # 如果有ec列则使用，否则默认为1
        
        try:
            data = build_graphs_for_sample(
                sample_id=sample_id,
                smiles=smiles,
                kcat_value=kcat_value,
                pocket_base_dir=args.pocket_dir,
                temperature=args.temperature,
                use_multipose=args.use_multipose,
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
            print(f'Error processing {sample_id}: {e}')
            failed_count += 1
            continue

    # 保存pt文件
    torch.save(dataset, args.output)
    print(f'✅ Saved {len(dataset)} samples to {args.output}')
    
    # 统计多pose样本数量
    multipose_count = sum(1 for d in dataset if isinstance(d, PoseSetData))
    if multipose_count > 0:
        print(f'   - {multipose_count} samples with multiple poses')
        print(f'   - {len(dataset) - multipose_count} samples with single pose')
    
    # 保存成功处理的CSV文件
    successful_df = df.loc[successful_indices].copy()
    successful_csv_path = args.output.replace('.pt', '_successful.csv')
    successful_df.to_csv(successful_csv_path, index=False)
    print(f'✅ Saved {len(successful_df)} successful samples to {successful_csv_path}')
    
    print(f'Successful: {successful_count}, Failed: {failed_count}')
    if successful_count + failed_count > 0:
        print(f'Success rate: {successful_count/(successful_count+failed_count)*100:.1f}%')

if __name__ == '__main__':
    main()
