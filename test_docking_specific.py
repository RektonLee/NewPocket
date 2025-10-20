#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试特定蛋白质和底物的对接过程
蛋白质: kcat_017576_protein.pdb
底物: CC(C)=CCC/C(C)=C/CC/C(C)=C/COP(=O)(O)OP(=O)(O)O
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
import hashlib

# 导入对接相关函数
from docking import run_preprocess, extract_pocket_pymol, convert_pdbqt_to_pdb
from graph_builder_rbf import parse_pocket

def test_docking_specific():
    """测试特定蛋白质和底物的对接"""
    
    # 测试参数
    uniprot_id = "kcat_017576"
    smiles = "CC(C)=CCC/C(C)=C/CC/C(C)=C/COP(=O)(O)OP(=O)(O)O"
    protein_pdb = "kcat_017576_protein.pdb"
    
    print("🧪 对接测试开始")
    print("="*60)
    print(f"蛋白质: {protein_pdb}")
    print(f"底物SMILES: {smiles}")
    print(f"UniProt ID: {uniprot_id}")
    
    # 检查蛋白质文件是否存在
    if not os.path.exists(protein_pdb):
        print(f"❌ 蛋白质文件不存在: {protein_pdb}")
        print("请确保文件在当前目录中")
        return False
    
    # 创建临时输出目录
    test_output_dir = "test_docking_output"
    os.makedirs(test_output_dir, exist_ok=True)
    
    # 计算底物哈希
    pocket_hash = int(hashlib.sha256(smiles.encode()).hexdigest(), 16) & 0xffff
    base_name = f"{uniprot_id}_{pocket_hash}"
    output_pocket_path = os.path.join(test_output_dir, f"{base_name}_10A.pdb")
    
    print(f"\n📁 输出目录: {test_output_dir}")
    print(f"📄 口袋文件: {output_pocket_path}")
    print(f"🔢 底物哈希: {pocket_hash}")
    
    try:
        # 执行对接流程
        print(f"\n🔄 开始对接流程...")
        result = run_preprocess(uniprot_id, smiles, protein_pdb, output_pocket_path, 0)
        
        if result:
            print(f"✅ 对接成功完成!")
            
            # 分析生成的口袋文件
            print(f"\n📊 分析口袋文件...")
            analyze_pocket_file(output_pocket_path)
            
            # 测试图构建
            print(f"\n🔗 测试图构建...")
            test_graph_construction(output_pocket_path)
            
            return True
        else:
            print(f"❌ 对接失败")
            return False
            
    except Exception as e:
        print(f"❌ 对接过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_pocket_file(pocket_file):
    """分析口袋文件内容"""
    if not os.path.exists(pocket_file):
        print(f"❌ 口袋文件不存在: {pocket_file}")
        return
    
    print(f"📄 分析文件: {pocket_file}")
    
    try:
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('pocket', pocket_file)
        
        protein_atoms = 0
        ligand_atoms = 0
        protein_residues = set()
        ligand_residues = set()
        residue_types = {}
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    residue_name = residue.get_resname()
                    residue_id = f"{chain.get_id()}_{residue.get_id()[1]}"
                    residue_types[residue_name] = residue_types.get(residue_name, 0) + 1
                    
                    for atom in residue:
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
        
        print(f"   总原子数: {protein_atoms + ligand_atoms}")
        print(f"   蛋白质原子数: {protein_atoms}")
        print(f"   配体原子数: {ligand_atoms}")
        print(f"   蛋白质残基数: {len(protein_residues)}")
        print(f"   配体残基数: {len(ligand_residues)}")
        
        # 显示残基类型
        print(f"   残基类型分布:")
        for res_type, count in sorted(residue_types.items()):
            print(f"     {res_type}: {count}")
        
        if ligand_atoms == 0:
            print("   ⚠️  警告: 口袋文件中没有配体原子！")
        else:
            print("   ✅ 口袋文件包含配体原子")
            
        # 检查文件大小
        file_size = os.path.getsize(pocket_file)
        print(f"   文件大小: {file_size} bytes")
        
        return {
            'protein_atoms': protein_atoms,
            'ligand_atoms': ligand_atoms,
            'protein_residues': len(protein_residues),
            'ligand_residues': len(ligand_residues),
            'file_size': file_size
        }
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return None

def test_graph_construction(pocket_file):
    """测试图构建"""
    try:
        print(f"🔗 测试图构建...")
        
        # 解析口袋原子
        atoms = parse_pocket(pocket_file)
        print(f"   解析到 {len(atoms)} 个原子")
        
        if len(atoms) == 0:
            print("   ❌ 没有解析到原子，图构建失败")
            return False
        
        # 构建图
        from graph_builder_rbf import build_graph
        graph_data = build_graph(atoms, temperature=303.15)
        
        print(f"   图节点数: {graph_data.x.shape[0]}")
        print(f"   图边数: {graph_data.edge_index.shape[1]}")
        print(f"   节点特征维度: {graph_data.x.shape[1]}")
        print(f"   边特征维度: {graph_data.edge_attr.shape[1]}")
        
        # 检查是否有配体原子
        ligand_atoms = 0
        for i, atom in enumerate(atoms):
            if atom.get('is_ligand', False):
                ligand_atoms += 1
        
        print(f"   配体原子数: {ligand_atoms}")
        
        if ligand_atoms > 0:
            print("   ✅ 图构建成功，包含配体原子")
        else:
            print("   ⚠️  图构建成功，但没有配体原子")
        
        return True
        
    except Exception as e:
        print(f"❌ 图构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_test_files():
    """清理测试文件"""
    print(f"\n🧹 清理测试文件...")
    
    # 清理临时目录
    temp_dirs = [d for d in os.listdir('.') if d.startswith('tmp/')]
    for temp_dir in temp_dirs:
        try:
            shutil.rmtree(temp_dir)
            print(f"   删除临时目录: {temp_dir}")
        except Exception as e:
            print(f"   清理失败: {e}")
    
    # 清理测试输出目录
    test_output_dir = "test_docking_output"
    if os.path.exists(test_output_dir):
        try:
            shutil.rmtree(test_output_dir)
            print(f"   删除测试输出目录: {test_output_dir}")
        except Exception as e:
            print(f"   清理失败: {e}")

def main():
    """主函数"""
    print("🧪 特定蛋白质-底物对接测试")
    print("="*60)
    
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "--cleanup":
        cleanup_test_files()
        return
    
    # 运行测试
    success = test_docking_specific()
    
    if success:
        print(f"\n🎉 测试成功完成!")
        print(f"💡 运行 'python {sys.argv[0]} --cleanup' 清理测试文件")
    else:
        print(f"\n❌ 测试失败")
    
    return success

if __name__ == "__main__":
    main()

