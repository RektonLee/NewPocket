#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行 DLKcat 预测
基于 DLKcat: https://github.com/SysBioChalmers/DLKcat

DLKcat 运行逻辑:
1. 底物处理: SMILES → RDKit分子图 → 指纹 + 邻接矩阵
2. 蛋白质处理: 序列 → 3-gram分词 → 词典映射
3. 模型: GNN(分子) + CNN(序列) → 回归预测 log2(kcat)
4. 输出: 2^预测值 = kcat (1/s)
"""

import os
import sys
import math
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

# 路径设置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DLKCAT_PATH = os.path.join(PROJECT_ROOT, "benchmark_tools", "DLKcat")
DLKCAT_DL_PATH = os.path.join(DLKCAT_PATH, "DeeplearningApproach")
DLKCAT_CODE_PATH = os.path.join(DLKCAT_DL_PATH, "Code")
DLKCAT_DATA_PATH = os.path.join(DLKCAT_DL_PATH, "Data", "input")
DLKCAT_MODEL_PATH = os.path.join(DLKCAT_DL_PATH, "Results", "output")

# 添加路径
sys.path.insert(0, DLKCAT_CODE_PATH)
sys.path.insert(0, os.path.join(DLKCAT_CODE_PATH, "model"))
sys.path.insert(0, os.path.join(DLKCAT_CODE_PATH, "example"))


def evaluate_predictions(y_true, y_pred, prefix=""):
    """计算评估指标"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]
    
    if len(y_true_valid) < 2:
        return {f"{prefix}n_valid": 0}
    
    pearson_r, _ = stats.pearsonr(y_true_valid, y_pred_valid)
    spearman_r, _ = stats.spearmanr(y_true_valid, y_pred_valid)
    
    metrics = {
        f"{prefix}n_valid": len(y_true_valid),
        f"{prefix}r2": r2_score(y_true_valid, y_pred_valid),
        f"{prefix}pearson_r": pearson_r,
        f"{prefix}spearman_r": spearman_r,
        f"{prefix}rmse": np.sqrt(mean_squared_error(y_true_valid, y_pred_valid)),
        f"{prefix}mae": mean_absolute_error(y_true_valid, y_pred_valid),
    }
    return metrics


class DLKcatPredictor:
    """DLKcat 预测器封装"""
    
    def __init__(self, device='cuda:1'):
        import torch
        import pickle
        from rdkit import Chem
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🔧 使用设备: {self.device}")
        
        # 加载词典
        print("📚 加载词典文件...")
        self.fingerprint_dict = self._load_pickle(os.path.join(DLKCAT_DATA_PATH, 'fingerprint_dict.pickle'))
        self.atom_dict = self._load_pickle(os.path.join(DLKCAT_DATA_PATH, 'atom_dict.pickle'))
        self.bond_dict = self._load_pickle(os.path.join(DLKCAT_DATA_PATH, 'bond_dict.pickle'))
        self.edge_dict = self._load_pickle(os.path.join(DLKCAT_DATA_PATH, 'edge_dict.pickle'))
        self.word_dict = self._load_pickle(os.path.join(DLKCAT_DATA_PATH, 'sequence_dict.pickle'))
        
        n_fingerprint = len(self.fingerprint_dict)
        n_word = len(self.word_dict)
        print(f"   指纹词典大小: {n_fingerprint}, 序列词典大小: {n_word}")
        
        # 模型参数
        self.radius = 2
        self.ngram = 3
        dim = 10
        layer_gnn = 3
        window = 11
        layer_cnn = 3
        layer_output = 3
        
        # 加载模型
        print("🧠 加载 DLKcat 模型...")
        import model as dlkcat_model
        self.model = dlkcat_model.KcatPrediction(
            self.device, n_fingerprint, n_word, 2*dim, 
            layer_gnn, window, layer_cnn, layer_output
        ).to(self.device)
        
        # 模型权重文件
        model_file = os.path.join(
            DLKCAT_MODEL_PATH,
            'all--radius2--ngram3--dim20--layer_gnn3--window11--layer_cnn3--layer_output3--lr1e-3--lr_decay0.5--decay_interval10--weight_decay1e-6--iteration50'
        )
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"找不到模型文件: {model_file}")
        
        self.model.load_state_dict(torch.load(model_file, map_location=self.device))
        self.model.eval()
        print("✅ 模型加载完成!")
    
    def _load_pickle(self, path):
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def _split_sequence(self, sequence):
        """将蛋白质序列转换为 n-gram 整数序列"""
        sequence = '-' + sequence + '='
        words = []
        for i in range(len(sequence) - self.ngram + 1):
            ngram = sequence[i:i+self.ngram]
            if ngram in self.word_dict:
                words.append(self.word_dict[ngram])
            else:
                words.append(0)  # 未知 n-gram
        return np.array(words)
    
    def _create_atoms(self, mol):
        """创建原子 ID 列表"""
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        for a in mol.GetAromaticAtoms():
            i = a.GetIdx()
            atoms[i] = (atoms[i], 'aromatic')
        
        result = []
        for a in atoms:
            if a in self.atom_dict:
                result.append(self.atom_dict[a])
            else:
                result.append(0)
        return np.array(result)
    
    def _create_ijbonddict(self, mol):
        """创建键字典"""
        i_jbond_dict = defaultdict(lambda: [])
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bond_type = str(b.GetBondType())
            if bond_type in self.bond_dict:
                bond = self.bond_dict[bond_type]
            else:
                bond = 0
            i_jbond_dict[i].append((j, bond))
            i_jbond_dict[j].append((i, bond))
        return i_jbond_dict
    
    def _extract_fingerprints(self, atoms, i_jbond_dict):
        """使用 Weisfeiler-Lehman 算法提取分子指纹"""
        if len(atoms) == 1 or self.radius == 0:
            fingerprints = []
            for a in atoms:
                if a in self.fingerprint_dict:
                    fingerprints.append(self.fingerprint_dict[a])
                else:
                    fingerprints.append(0)
            return np.array(fingerprints)
        
        nodes = atoms
        i_jedge_dict = i_jbond_dict
        
        for _ in range(self.radius):
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                if fingerprint in self.fingerprint_dict:
                    fingerprints.append(self.fingerprint_dict[fingerprint])
                else:
                    fingerprints.append(0)
            
            nodes = fingerprints
            
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    key = (both_side, edge)
                    if key in self.edge_dict:
                        edge = self.edge_dict[key]
                    else:
                        edge = 0
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict
        
        return np.array(fingerprints)
    
    def _create_adjacency(self, mol):
        """创建邻接矩阵"""
        from rdkit import Chem
        return np.array(Chem.GetAdjacencyMatrix(mol))
    
    def predict_single(self, sequence, smiles):
        """预测单个样本"""
        import torch
        from rdkit import Chem
        
        # 检查 SMILES 是否包含多个分子（用 . 分隔）
        if "." in smiles:
            return np.nan
        
        try:
            # 解析 SMILES
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            if mol is None:
                return np.nan
            
            # 提取特征
            atoms = self._create_atoms(mol)
            i_jbond_dict = self._create_ijbonddict(mol)
            fingerprints = self._extract_fingerprints(atoms, i_jbond_dict)
            adjacency = self._create_adjacency(mol)
            words = self._split_sequence(sequence)
            
            # 转为 tensor
            fingerprints = torch.LongTensor(fingerprints).to(self.device)
            adjacency = torch.FloatTensor(adjacency).to(self.device)
            words = torch.LongTensor(words).to(self.device)
            
            # 预测
            with torch.no_grad():
                inputs = [fingerprints, adjacency, words]
                prediction = self.model.forward(inputs)
                kcat_log2 = prediction.item()
            
            # DLKcat 输出是 log2(kcat)，转换为 log10(kcat) 以便与其他方法比较
            kcat = math.pow(2, kcat_log2)
            kcat_log10 = math.log10(kcat) if kcat > 0 else np.nan
            
            return kcat_log10
            
        except Exception as e:
            # print(f"预测失败: {e}")
            return np.nan
    
    def predict_batch(self, sequences, smiles_list):
        """批量预测"""
        predictions = []
        for seq, smi in tqdm(zip(sequences, smiles_list), total=len(sequences), desc="DLKcat预测"):
            pred = self.predict_single(seq, smi)
            predictions.append(pred)
        return np.array(predictions)


def main():
    parser = argparse.ArgumentParser(description="运行 DLKcat 预测")
    parser.add_argument("--dataset", type=str, required=True, help="数据集 CSV 文件路径")
    parser.add_argument("--save_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--device", type=str, default="cuda:1", help="计算设备")
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数（用于测试）")
    
    args = parser.parse_args()
    
    # 创建输出目录 [[memory:7023581]]
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = os.path.join(PROJECT_ROOT, "results", f"dlkcat_benchmark_{timestamp}")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载数据
    print(f"📊 加载数据集: {args.dataset}")
    df = pd.read_csv(args.dataset)
    print(f"   样本数: {len(df)}")
    
    # 检查必要的列
    required_cols = ['sequence', 'substrate_smiles', 'kcat_value']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要的列: {col}")
    
    # 清理数据
    df_clean = df.dropna(subset=required_cols).copy()
    
    # 转换 kcat_value 为数值类型
    df_clean['kcat_value'] = pd.to_numeric(df_clean['kcat_value'], errors='coerce')
    df_clean = df_clean.dropna(subset=['kcat_value'])
    df_clean = df_clean[df_clean['kcat_value'] > 0]  # 过滤非正值
    print(f"   清理后样本数: {len(df_clean)}")
    
    # 限制样本数（用于测试）
    if args.max_samples:
        df_clean = df_clean.head(args.max_samples)
        print(f"   使用样本数: {len(df_clean)}")
    
    # 准备数据
    sequences = df_clean['sequence'].tolist()
    smiles_list = df_clean['substrate_smiles'].tolist()
    y_true = np.log10(df_clean['kcat_value'].values + 1e-10)
    
    # 初始化预测器并运行预测
    print(f"\n🔬 初始化 DLKcat...")
    predictor = DLKcatPredictor(device=args.device)
    
    print(f"\n🔬 运行 DLKcat 预测...")
    y_pred = predictor.predict_batch(sequences, smiles_list)
    
    # 评估
    print(f"\n📈 评估结果:")
    metrics = evaluate_predictions(y_true, y_pred, prefix="dlkcat_")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")
    
    # 保存结果
    results_df = pd.DataFrame({
        'sequence': sequences,
        'substrate_smiles': smiles_list,
        'kcat_true': df_clean['kcat_value'].values,
        'log_kcat_true': y_true,
        'log_kcat_pred': y_pred,
    })
    results_file = os.path.join(args.save_dir, "dlkcat_predictions.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\n✅ 预测结果已保存: {results_file}")
    
    # 保存评估指标
    metrics_df = pd.DataFrame([metrics])
    metrics_file = os.path.join(args.save_dir, "dlkcat_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print(f"✅ 评估指标已保存: {metrics_file}")
    
    return metrics


if __name__ == "__main__":
    main()
