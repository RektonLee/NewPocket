#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析对接流程中的文件生成情况
"""

import os
import sys
from Bio.PDB import PDBParser
import pandas as pd

def analyze_docking_files(base_dir, sample_id):
    """分析指定样本的对接文件"""
    sample_dir = os.path.join(base_dir, "samples", sample_id)
    
    if not os.path.exists(sample_dir):
        print(f"❌ 样本目录不存在: {sample_dir}")
        return
    
    print(f"🔍 分析样本: {sample_id}")
    print(f"📁 目录: {sample_dir}")
    
    # 列出所有文件
    files = os.listdir(sample_dir)
    print(f"📄 文件列表: {files}")
    
    # 分析每个文件
    for file in files:
        file_path = os.path.join(sample_dir, file)
        if file.endswith('.pdb'):
            analyze_pdb_file(file_path)

def analyze_pdb_file(pdb_file):
    """分析PDB文件内容"""
    if not os.path.exists(pdb_file):
        print(f"❌ 文件不存在: {pdb_file}")
        return
    
    print(f"\n📊 分析文件: {os.path.basename(pdb_file)}")
    
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('structure', pdb_file)
        
        atom_count = 0
        residue_types = {}
        chain_info = {}
        
        for model in structure:
            for chain in model:
                chain_id = chain.get_id()
                chain_info[chain_id] = {
                    'residues': 0,
                    'atoms': 0,
                    'residue_types': set()
                }
                
                for residue in chain:
                    residue_name = residue.get_resname()
                    residue_types[residue_name] = residue_types.get(residue_name, 0) + 1
                    chain_info[chain_id]['residues'] += 1
                    chain_info[chain_id]['residue_types'].add(residue_name)
                    
                    for atom in residue:
                        atom_count += 1
                        chain_info[chain_id]['atoms'] += 1
        
        print(f"   总原子数: {atom_count}")
        print(f"   链信息:")
        for chain_id, info in chain_info.items():
            print(f"     链 {chain_id}: {info['residues']} 残基, {info['atoms']} 原子")
            print(f"       残基类型: {', '.join(sorted(info['residue_types']))}")
        
        # 判断文件类型
        if any(res in residue_types for res in ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 
                                               'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 
                                               'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']):
            print("   🧬 包含蛋白质残基")
        
        # 检查是否有配体（非标准氨基酸残基）
        ligand_residues = [res for res in residue_types.keys() 
                          if res not in ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 
                                       'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 
                                       'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HOH', 'WAT']]
        
        if ligand_residues:
            print(f"   💊 包含配体残基: {ligand_residues}")
        else:
            print("   ⚠️  未检测到配体残基")
            
    except Exception as e:
        print(f"❌ 分析失败: {e}")

def explain_vina_output():
    """解释AutoDock Vina的输出"""
    print("\n" + "="*60)
    print("🔬 AutoDock Vina 输出解释")
    print("="*60)
    
    print("""
AutoDock Vina 的输出文件说明：

1. 📥 输入文件：
   - receptor.pdbqt: 蛋白质受体（刚性）
   - ligand.pdbqt: 配体分子（柔性）

2. 🔄 对接过程：
   - Vina 在指定的盒子内搜索配体的最佳结合位姿
   - 蛋白质保持刚性，配体可以旋转和移动
   - 计算结合亲和力分数

3. 📤 输出文件：
   - vina_out.pdbqt: 包含多个对接位姿的配体结构
     * 每个位姿都有对应的结合分数
     * 只包含配体原子，不包含蛋白质
     * 配体已经移动到最佳结合位置

4. 🔧 后续处理：
   - vina_split: 将多个位姿分离成单独文件
   - pose0.pdb: 最佳位姿的配体结构（PDB格式）
   - 这个配体结构用于后续的口袋提取

5. 🎯 关键点：
   - Vina输出的是对接后的配体结构
   - 配体已经移动到蛋白质的结合位点
   - 蛋白质结构保持不变
   - 最终需要将对接后的配体与蛋白质结合位点区域结合
""")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python analyze_docking_flow.py <base_dir> [sample_id]")
        print("  python analyze_docking_flow.py --explain-vina")
        print("\n示例:")
        print("  python analyze_docking_flow.py kcat_full_after kcat_017576")
        print("  python analyze_docking_flow.py --explain-vina")
        return
    
    if sys.argv[1] == "--explain-vina":
        explain_vina_output()
        return
    
    base_dir = sys.argv[1]
    sample_id = sys.argv[2] if len(sys.argv) > 2 else None
    
    if sample_id:
        analyze_docking_files(base_dir, sample_id)
    else:
        # 分析所有样本
        samples_dir = os.path.join(base_dir, "samples")
        if os.path.exists(samples_dir):
            samples = os.listdir(samples_dir)
            print(f"🔍 找到 {len(samples)} 个样本")
            for sample in samples[:5]:  # 只分析前5个样本
                analyze_docking_files(base_dir, sample)
                print("\n" + "-"*50)
        else:
            print(f"❌ 样本目录不存在: {samples_dir}")

if __name__ == "__main__":
    main()

