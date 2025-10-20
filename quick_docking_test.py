#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速对接测试脚本
"""

import os
import sys
import hashlib

def quick_test():
    """快速测试对接功能"""
    
    # 测试参数
    uniprot_id = "kcat_017576"
    smiles = "CC(C)=CCC/C(C)=C/CC/C(C)=C/COP(=O)(O)OP(=O)(O)O"
    protein_pdb = "kcat_017576_protein.pdb"
    
    print("🚀 快速对接测试")
    print("="*50)
    print(f"蛋白质: {protein_pdb}")
    print(f"底物: {smiles}")
    
    # 检查文件
    if not os.path.exists(protein_pdb):
        print(f"❌ 蛋白质文件不存在: {protein_pdb}")
        print("请确保文件在当前目录中")
        return False
    
    # 计算哈希
    pocket_hash = int(hashlib.sha256(smiles.encode()).hexdigest(), 16) & 0xffff
    print(f"底物哈希: {pocket_hash}")
    
    # 创建输出目录
    output_dir = "quick_test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 输出文件路径
    output_pocket = os.path.join(output_dir, f"{uniprot_id}_{pocket_hash}_10A.pdb")
    print(f"输出文件: {output_pocket}")
    
    try:
        # 导入并运行对接
        from docking import run_preprocess
        
        print(f"\n🔄 开始对接...")
        result = run_preprocess(uniprot_id, smiles, protein_pdb, output_pocket, 0)
        
        if result:
            print(f"✅ 对接成功!")
            
            # 快速分析结果
            if os.path.exists(output_pocket):
                file_size = os.path.getsize(output_pocket)
                print(f"📄 口袋文件大小: {file_size} bytes")
                
                # 统计原子数
                with open(output_pocket, 'r') as f:
                    lines = f.readlines()
                
                atom_count = len([line for line in lines if line.startswith(('ATOM', 'HETATM'))])
                ligand_count = len([line for line in lines if line.startswith('HETATM')])
                
                print(f"🔢 总原子数: {atom_count}")
                print(f"💊 配体原子数: {ligand_count}")
                
                if ligand_count > 0:
                    print("✅ 成功包含配体原子!")
                else:
                    print("❌ 没有配体原子")
                
                return True
            else:
                print("❌ 口袋文件未生成")
                return False
        else:
            print("❌ 对接失败")
            return False
            
    except Exception as e:
        print(f"❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    success = quick_test()
    
    if success:
        print(f"\n🎉 快速测试成功!")
    else:
        print(f"\n❌ 快速测试失败")

if __name__ == "__main__":
    main()

