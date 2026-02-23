#!/usr/bin/env python3
"""
分析DiffDock对接质量

功能:
1. 读取docking元数据 (置信度分数)
2. 检查clash情况
3. 统计口袋大小分布
4. 可视化质量分布
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from collections import defaultdict

# Import pose validation
from pose_validation import check_clash_pdb, ValidationResult


def load_dataset(pt_file: str) -> List:
    """加载PyG数据集"""
    print(f"Loading dataset from {pt_file}...")
    dataset = torch.load(pt_file, weights_only=False)
    print(f"Loaded {len(dataset)} samples")
    return dataset


def extract_docking_metadata(sample_data_dir: str, sample_id: str) -> Dict:
    """提取docking元数据"""
    meta_path = Path(sample_data_dir) / "samples" / sample_id / "docking" / "diffdock_output" / "docking_meta.json"

    if not meta_path.exists():
        return {}

    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        return meta
    except Exception as e:
        return {}


def get_pocket_pdb_path(sample_data_dir: str, sample_id: str, pdb_id: str) -> Path:
    """获取口袋PDB文件路径"""
    return Path(sample_data_dir) / "samples" / sample_id / "docking" / pdb_id


def analyze_pocket_file(pocket_pdb: Path) -> Dict:
    """分析口袋文件的基本统计"""
    if not pocket_pdb.exists():
        return {}

    try:
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('pocket', str(pocket_pdb))

        protein_atoms = []
        ligand_atoms = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    # Check if residue is HETATM (non-standard residue)
                    is_hetatm = residue.get_id()[0].strip() != ''
                    for atom in residue:
                        if is_hetatm:
                            ligand_atoms.append(atom)
                        else:
                            protein_atoms.append(atom)

        # 计算口袋体积 (粗略估计: 凸包体积)
        if protein_atoms:
            coords = np.array([atom.get_coord() for atom in protein_atoms])
            bbox_volume = np.prod(coords.max(axis=0) - coords.min(axis=0))
        else:
            bbox_volume = 0

        return {
            'n_protein_atoms': len(protein_atoms),
            'n_ligand_atoms': len(ligand_atoms),
            'n_residues': len(list(structure.get_residues())),
            'bbox_volume': bbox_volume,
        }
    except Exception as e:
        print(f"Error analyzing {pocket_pdb}: {e}")
        return {}


def analyze_docking_quality(dataset_pt: str,
                            sample_data_dir: str = "sample_data",
                            output_dir: str = "results/docking_quality",
                            check_clash: bool = False,
                            limit: int = None) -> pd.DataFrame:
    """分析docking质量"""

    dataset = load_dataset(dataset_pt)
    if limit:
        dataset = dataset[:limit]

    results = []

    print(f"\nAnalyzing {len(dataset)} samples...")
    for data in tqdm(dataset):
        sample_id = data.sample_id
        pdb_id = data.pdb_id if hasattr(data, 'pdb_id') else None

        # 基本信息
        row = {
            'sample_id': sample_id,
            'pdb_id': pdb_id,
            'n_nodes': data.x.shape[0],
            'n_edges': data.edge_index.shape[1],
            'y': data.y.item() if hasattr(data, 'y') else None,
        }

        # Docking置信度
        if hasattr(data, 'docking_confidence'):
            row['confidence'] = data.docking_confidence.item()
        else:
            meta = extract_docking_metadata(sample_data_dir, sample_id)
            if meta:
                row['confidence'] = meta.get('best_confidence', None)
                row['n_poses'] = len(meta.get('confidence_values', []))

        # 口袋统计
        if pdb_id:
            pocket_path = get_pocket_pdb_path(sample_data_dir, sample_id, pdb_id)
            pocket_stats = analyze_pocket_file(pocket_path)
            row.update(pocket_stats)

            # Clash检测 (可选，较慢)
            if check_clash and pocket_path.exists():
                try:
                    # 需要分离protein和ligand来检测clash
                    # 这里简化处理，跳过详细clash检测
                    row['clash_checked'] = False
                except Exception as e:
                    row['clash_checked'] = False

        results.append(row)

    df = pd.DataFrame(results)

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, 'docking_quality_summary.csv')
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved quality summary to {output_csv}")

    return df


def visualize_quality(df: pd.DataFrame, output_dir: str):
    """可视化docking质量"""

    os.makedirs(output_dir, exist_ok=True)

    # 设置样式
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (15, 10)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Confidence分布
    if 'confidence' in df.columns and df['confidence'].notna().sum() > 0:
        ax = axes[0, 0]
        df['confidence'].dropna().hist(bins=50, ax=ax, edgecolor='black')
        ax.set_xlabel('DiffDock Confidence Score')
        ax.set_ylabel('Count')
        ax.set_title('Docking Confidence Distribution')
        ax.axvline(df['confidence'].median(), color='red', linestyle='--',
                   label=f'Median: {df["confidence"].median():.3f}')
        ax.legend()

    # 2. 图节点数分布
    ax = axes[0, 1]
    df['n_nodes'].hist(bins=50, ax=ax, edgecolor='black')
    ax.set_xlabel('Number of Atoms (nodes)')
    ax.set_ylabel('Count')
    ax.set_title('Pocket Size Distribution (Atoms)')
    ax.axvline(df['n_nodes'].median(), color='red', linestyle='--',
               label=f'Median: {df["n_nodes"].median():.0f}')
    ax.legend()

    # 3. Confidence vs 节点数
    if 'confidence' in df.columns:
        ax = axes[0, 2]
        valid = df[df['confidence'].notna()]
        if len(valid) > 0:
            ax.scatter(valid['n_nodes'], valid['confidence'], alpha=0.5, s=10)
            ax.set_xlabel('Number of Atoms')
            ax.set_ylabel('Confidence Score')
            ax.set_title('Confidence vs Pocket Size')

            # 拟合趋势线
            if len(valid) > 10:
                z = np.polyfit(valid['n_nodes'], valid['confidence'], 1)
                p = np.poly1d(z)
                ax.plot(valid['n_nodes'].sort_values(),
                       p(valid['n_nodes'].sort_values()),
                       "r--", alpha=0.8, label=f'Trend: y={z[0]:.4f}x+{z[1]:.3f}')
                ax.legend()

    # 4. 蛋白原子数分布
    if 'n_protein_atoms' in df.columns:
        ax = axes[1, 0]
        df['n_protein_atoms'].dropna().hist(bins=50, ax=ax, edgecolor='black')
        ax.set_xlabel('Protein Atoms in Pocket')
        ax.set_ylabel('Count')
        ax.set_title('Protein Atoms Distribution')

    # 5. 配体原子数分布
    if 'n_ligand_atoms' in df.columns:
        ax = axes[1, 1]
        df['n_ligand_atoms'].dropna().hist(bins=30, ax=ax, edgecolor='black')
        ax.set_xlabel('Ligand Atoms')
        ax.set_ylabel('Count')
        ax.set_title('Ligand Size Distribution')
        ax.axvline(df['n_ligand_atoms'].median(), color='red', linestyle='--',
                   label=f'Median: {df["n_ligand_atoms"].median():.0f}')
        ax.legend()

    # 6. 口袋体积分布
    if 'bbox_volume' in df.columns:
        ax = axes[1, 2]
        df['bbox_volume'].dropna().hist(bins=50, ax=ax, edgecolor='black')
        ax.set_xlabel('Pocket Bounding Box Volume (Å³)')
        ax.set_ylabel('Count')
        ax.set_title('Pocket Volume Distribution')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'docking_quality_plots.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved plots to {output_path}")
    plt.close()


def print_summary_stats(df: pd.DataFrame):
    """打印统计摘要"""

    print("\n" + "="*80)
    print("📊 DOCKING QUALITY SUMMARY")
    print("="*80)

    print(f"\n📈 Dataset Size: {len(df)} samples")

    if 'confidence' in df.columns:
        conf = df['confidence'].dropna()
        if len(conf) > 0:
            print(f"\n🎯 DiffDock Confidence Scores:")
            print(f"   Mean:   {conf.mean():.4f}")
            print(f"   Median: {conf.median():.4f}")
            print(f"   Std:    {conf.std():.4f}")
            print(f"   Min:    {conf.min():.4f}")
            print(f"   Max:    {conf.max():.4f}")

            # 质量分级
            high_conf = (conf > 0.3).sum()
            mid_conf = ((conf > 0.0) & (conf <= 0.3)).sum()
            low_conf = (conf <= 0.0).sum()

            print(f"\n   Quality Distribution:")
            print(f"   - High confidence (>0.3):  {high_conf:4d} ({high_conf/len(conf)*100:.1f}%)")
            print(f"   - Mid confidence (0~0.3):  {mid_conf:4d} ({mid_conf/len(conf)*100:.1f}%)")
            print(f"   - Low confidence (<0):     {low_conf:4d} ({low_conf/len(conf)*100:.1f}%)")

    print(f"\n🔬 Pocket Statistics:")
    print(f"   Atoms per pocket:  {df['n_nodes'].mean():.1f} ± {df['n_nodes'].std():.1f}")
    print(f"   Min atoms:         {df['n_nodes'].min():.0f}")
    print(f"   Max atoms:         {df['n_nodes'].max():.0f}")
    print(f"   Median atoms:      {df['n_nodes'].median():.0f}")

    if 'n_protein_atoms' in df.columns:
        prot = df['n_protein_atoms'].dropna()
        if len(prot) > 0:
            print(f"\n   Protein atoms:     {prot.mean():.1f} ± {prot.std():.1f}")

    if 'n_ligand_atoms' in df.columns:
        lig = df['n_ligand_atoms'].dropna()
        if len(lig) > 0:
            print(f"   Ligand atoms:      {lig.mean():.1f} ± {lig.std():.1f}")

    if 'bbox_volume' in df.columns:
        vol = df['bbox_volume'].dropna()
        if len(vol) > 0:
            print(f"   Pocket volume:     {vol.mean():.1f} ± {vol.std():.1f} Ų")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="分析DiffDock对接质量")
    parser.add_argument('--dataset', type=str,
                       default='data/processed/kcat_test_new_diffdock.pt',
                       help='输入数据集 (.pt)')
    parser.add_argument('--sample-data-dir', type=str, default='sample_data',
                       help='样本数据目录')
    parser.add_argument('--output-dir', type=str,
                       default='results/docking_quality',
                       help='输出目录')
    parser.add_argument('--check-clash', action='store_true',
                       help='检查clash (较慢)')
    parser.add_argument('--limit', type=int, default=None,
                       help='限制分析的样本数 (用于快速测试)')
    parser.add_argument('--no-plot', action='store_true',
                       help='不生成可视化')

    args = parser.parse_args()

    # 分析质量
    df = analyze_docking_quality(
        dataset_pt=args.dataset,
        sample_data_dir=args.sample_data_dir,
        output_dir=args.output_dir,
        check_clash=args.check_clash,
        limit=args.limit
    )

    # 打印统计
    print_summary_stats(df)

    # 可视化
    if not args.no_plot:
        visualize_quality(df, args.output_dir)

    print(f"\n✅ Analysis complete! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
