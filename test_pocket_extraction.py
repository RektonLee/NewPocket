#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试口袋提取功能，验证配体原子是否被正确包含
"""

import os
import sys
from Bio.PDB import PDBParser
import numpy as np

def analyze_pocket_file(pocket_file):
    """分析口袋文件，统计蛋白质和配体原子数量"""
    if not os.path.exists(pocket_file):
        print(f"❌ 文件不存在: {pocket_file}")
        return
    
    parser = PDBParser(QUIET=True)
    
    try:
        structure = parser.get_structure('pocket', pocket_file)
        
        protein_atoms = 0
        ligand_atoms = 0
        protein_residues = set()
        ligand_residues = set()
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    residue_name = residue.get_resname()
                    residue_id = f"{chain.get_id()}_{residue.get_id()[1]}"
                    
                    for atom in residue:
                        if residue_name in ['HOH', 'WAT']:  # 跳过水分子
                            continue
                            
                        if atom.get_name().startswith('H'):  # 跳过氢原子
                            continue
                            
                        if residue_name in ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 
                                          'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 
                                          'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']:
                            # 标准氨基酸
                            protein_atoms += 1
                            protein_residues.add(residue_id)
                        else:
                            # 配体或其他分子
                            ligand_atoms += 1
                            ligand_residues.add(residue_id)
        
        print(f"📊 口袋文件分析: {pocket_file}")
        print(f"   蛋白质原子数: {protein_atoms}")
        print(f"   配体原子数: {ligand_atoms}")
        print(f"   蛋白质残基数: {len(protein_residues)}")
        print(f"   配体残基数: {len(ligand_residues)}")
        print(f"   总原子数: {protein_atoms + ligand_atoms}")
        
        if ligand_atoms == 0:
            print("⚠️  警告: 口袋文件中没有配体原子！")
        else:
            print("✅ 口袋文件包含配体原子")
            
        return {
            'protein_atoms': protein_atoms,
            'ligand_atoms': ligand_atoms,
            'protein_residues': len(protein_residues),
            'ligand_residues': len(ligand_residues)
        }
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return None

def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("使用方法: python test_pocket_extraction.py <pocket_file.pdb>")
        print("示例: python test_pocket_extraction.py kcat_017576_1560_10A.pdb")
        return
    
    pocket_file = sys.argv[1]
    analyze_pocket_file(pocket_file)

if __name__ == "__main__":
    main()

