#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分步测试对接流程的各个步骤
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

# 导入必要的函数
from docking import (
    clean_altlocs, smiles_to_3d, convert_pdbqt_to_pdb,
    extract_pocket_pymol, find_binding_site_center,
    extract_box_from_pdb, move_ligand_to_box_center
)

def test_step_by_step():
    """分步测试对接流程"""
    
    # 测试参数
    protein_pdb = "kcat_017576_protein.pdb"
    smiles = "CC(C)=CCC/C(C)=C/CC/C(C)=C/COP(=O)(O)OP(=O)(O)O"
    uniprot_id = "kcat_017576"
    
    print("🧪 分步对接测试")
    print("="*60)
    print(f"蛋白质: {protein_pdb}")
    print(f"底物SMILES: {smiles}")
    
    # 检查输入文件
    if not os.path.exists(protein_pdb):
        print(f"❌ 蛋白质文件不存在: {protein_pdb}")
        return False
    
    # 创建测试目录
    test_dir = "test_docking_steps"
    os.makedirs(test_dir, exist_ok=True)
    os.chdir(test_dir)
    
    try:
        # 步骤1: 清理蛋白质文件
        print(f"\n1️⃣ 步骤1: 清理蛋白质文件")
        clean_protein = "protein_clean.pdb"
        clean_altlocs(f"../{protein_pdb}", clean_protein)
        print(f"   ✅ 生成: {clean_protein}")
        
        # 步骤2: 生成配体3D结构
        print(f"\n2️⃣ 步骤2: 生成配体3D结构")
        ligand_pdb = "ligand.pdb"
        smiles_to_3d(smiles, ligand_pdb)
        print(f"   ✅ 生成: {ligand_pdb}")
        
        # 步骤3: 准备配体PDBQT
        print(f"\n3️⃣ 步骤3: 准备配体PDBQT")
        ligand_pdbqt = "ligand.pdbqt"
        prep_bin = "/home/lizihao/ADFRsuite_x86_64Linux_1.0/bin"
        subprocess.run([
            f"{prep_bin}/prepare_ligand", 
            "-l", ligand_pdb, 
            "-A", "hydrogens", 
            "-o", ligand_pdbqt
        ], check=True)
        print(f"   ✅ 生成: {ligand_pdbqt}")
        
        # 步骤4: 准备受体PDBQT
        print(f"\n4️⃣ 步骤4: 准备受体PDBQT")
        receptor_pdbqt = "receptor.pdbqt"
        subprocess.run([
            f"{prep_bin}/prepare_receptor", 
            "-r", clean_protein, 
            "-A", "hydrogens", 
            "-U", "lps_waters_nonstdres", 
            "-o", receptor_pdbqt
        ], check=True)
        print(f"   ✅ 生成: {receptor_pdbqt}")
        
        # 步骤5: AutoSite口袋检测
        print(f"\n5️⃣ 步骤5: AutoSite口袋检测")
        os.makedirs("autosite_out", exist_ok=True)
        try:
            subprocess.run([
                f"{prep_bin}/autosite", 
                "-r", receptor_pdbqt, 
                "-o", "autosite_out"
            ], check=True)
            
            pocket_file = "autosite_out/receptor_cl_001.pdb"
            if os.path.exists(pocket_file):
                print(f"   ✅ AutoSite成功，生成: {pocket_file}")
                center, size = extract_box_from_pdb(pocket_file)
                print(f"   📍 口袋中心: {center}")
                print(f"   📏 口袋大小: {size}")
            else:
                print(f"   ⚠️ AutoSite未找到口袋，使用fallback")
                center = [0.0, 0.0, 0.0]
                size = [20.0, 20.0, 20.0]
        except subprocess.CalledProcessError:
            print(f"   ⚠️ AutoSite失败，使用fallback")
            center = [0.0, 0.0, 0.0]
            size = [20.0, 20.0, 20.0]
        
        # 步骤6: 移动配体到盒子中心
        print(f"\n6️⃣ 步骤6: 移动配体到盒子中心")
        ligand_moved = "ligand_moved.pdbqt"
        move_ligand_to_box_center(ligand_pdbqt, center, ligand_moved)
        print(f"   ✅ 生成: {ligand_moved}")
        
        # 步骤7: Vina对接
        print(f"\n7️⃣ 步骤7: Vina对接")
        try:
            from vina import Vina
            v = Vina()
            v.set_receptor(rigid_pdbqt_filename=receptor_pdbqt)
            v.set_ligand_from_file(ligand_moved)
            v.compute_vina_maps(center=center, box_size=size)
            v.dock(exhaustiveness=16, n_poses=5)
            v.write_poses("vina_out.pdbqt", n_poses=5, overwrite=True)
            print(f"   ✅ Vina对接完成，生成: vina_out.pdbqt")
        except Exception as e:
            print(f"   ❌ Vina对接失败: {e}")
            return False
        
        # 步骤8: 分离最佳位姿
        print(f"\n8️⃣ 步骤8: 分离最佳位姿")
        vina_bin = "/home/lizihao/autodock_vina_1_1_2_linux_x86/bin"
        subprocess.run([
            f"{vina_bin}/vina_split", 
            "--input", "vina_out.pdbqt", 
            "--ligand", "vina_out_ligand_"
        ], check=True)
        
        # 转换为PDB格式
        pose_pdb = "pose0.pdb"
        with open(pose_pdb, "w") as w:
            for ln in open("vina_out_ligand_1.pdbqt"):
                if ln.startswith(("ATOM", "HETATM")):
                    w.write(ln[:66] + "\n")
        print(f"   ✅ 生成最佳位姿: {pose_pdb}")
        
        # 步骤9: 转换受体PDBQT为PDB
        print(f"\n9️⃣ 步骤9: 转换受体PDBQT为PDB")
        receptor_pdb = "receptor.pdb"
        convert_pdbqt_to_pdb(receptor_pdbqt, receptor_pdb)
        print(f"   ✅ 生成: {receptor_pdb}")
        
        # 步骤10: 口袋提取
        print(f"\n🔟 步骤10: 口袋提取")
        output_pocket = f"{uniprot_id}_test_10A.pdb"
        extract_pocket_pymol(receptor_pdb, pose_pdb, output_pocket, cutoff=5.0)
        print(f"   ✅ 生成口袋文件: {output_pocket}")
        
        # 分析结果
        print(f"\n📊 分析结果:")
        analyze_result_files()
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 返回上级目录
        os.chdir("..")

def analyze_result_files():
    """分析结果文件"""
    test_dir = "test_docking_steps"
    
    # 分析口袋文件
    pocket_file = os.path.join(test_dir, "kcat_017576_test_10A.pdb")
    if os.path.exists(pocket_file):
        print(f"📄 口袋文件: {pocket_file}")
        
        # 统计原子数
        with open(pocket_file, 'r') as f:
            lines = f.readlines()
        
        atom_lines = [line for line in lines if line.startswith(('ATOM', 'HETATM'))]
        hetatm_lines = [line for line in lines if line.startswith('HETATM')]
        
        print(f"   总原子数: {len(atom_lines)}")
        print(f"   配体原子数: {len(hetatm_lines)}")
        
        if len(hetatm_lines) > 0:
            print("   ✅ 口袋文件包含配体原子")
        else:
            print("   ❌ 口袋文件不包含配体原子")
    
    # 显示文件列表
    print(f"\n📁 生成的文件:")
    for file in os.listdir(test_dir):
        file_path = os.path.join(test_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            print(f"   {file}: {size} bytes")

def cleanup():
    """清理测试文件"""
    test_dir = "test_docking_steps"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        print(f"🧹 清理测试目录: {test_dir}")

def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == "--cleanup":
        cleanup()
        return
    
    success = test_step_by_step()
    
    if success:
        print(f"\n🎉 分步测试成功完成!")
        print(f"💡 运行 'python {sys.argv[0]} --cleanup' 清理测试文件")
    else:
        print(f"\n❌ 分步测试失败")

if __name__ == "__main__":
    main()

