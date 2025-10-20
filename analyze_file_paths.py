#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析pred_range_fixed.py运行时的文件路径
"""

import os
import hashlib

def analyze_pred_range_paths():
    """分析pred_range_fixed.py运行时的文件路径"""
    
    print("🔍 pred_range_fixed.py 文件路径分析")
    print("="*60)
    
    # 模拟参数
    sample_id = "kcat_017576"
    uniprot_id = "P12345"  # 示例
    smiles = "CCO"  # 示例
    sample_data_dir = "kcat_full_after"
    output_dir = "results/test_output"
    
    print(f"📋 输入参数:")
    print(f"   sample_id: {sample_id}")
    print(f"   uniprot_id: {uniprot_id}")
    print(f"   smiles: {smiles}")
    print(f"   sample_data_dir: {sample_data_dir}")
    print(f"   output_dir: {output_dir}")
    
    print(f"\n📁 文件路径分析:")
    
    # 1. 蛋白质结构路径
    print(f"\n1️⃣ 蛋白质结构路径:")
    if True:  # use_sample_manager = True
        pocket_dir = os.path.join(sample_data_dir, "samples", sample_id)
        print(f"   pocket_dir: {pocket_dir}")
        
        # 假设通过SampleManager获取蛋白质路径
        protein_path = os.path.join(sample_data_dir, "samples", sample_id, f"{sample_id}.pdb")
        print(f"   protein_path: {protein_path}")
    else:
        pocket_dir = output_dir
        print(f"   pocket_dir: {pocket_dir}")
    
    # 2. 口袋文件路径
    print(f"\n2️⃣ 口袋文件路径:")
    pocket_hash = int(hashlib.sha256(smiles.encode()).hexdigest(), 16) & 0xffff
    base_name = f"{uniprot_id or sample_id}_{pocket_hash}"
    pocket_pdb = os.path.join(pocket_dir, f"{base_name}_10A.pdb")
    print(f"   pocket_hash: {pocket_hash}")
    print(f"   base_name: {base_name}")
    print(f"   pocket_pdb: {pocket_pdb}")
    
    # 3. 对接过程中的临时文件路径
    print(f"\n3️⃣ 对接过程临时文件路径:")
    tmp_dir = f"tmp/{base_name}"
    print(f"   tmp_dir: {tmp_dir}")
    
    # 在tmp_dir中的文件
    tmp_files = {
        "prot_clean.pdb": "清理后的蛋白质结构",
        "ligand.pdb": "从SMILES生成的配体结构",
        "ligand.pdbqt": "配体的PDBQT格式",
        "receptor.pdbqt": "蛋白质受体的PDBQT格式",
        "autosite_out/": "AutoSite输出目录",
        "receptor_cl_001.pdb": "AutoSite识别的口袋",
        "ligand_moved.pdbqt": "移动到盒子中心的配体",
        "vina_out.pdbqt": "Vina对接输出（多个位姿）",
        "vina_out_ligand_1.pdbqt": "最佳位姿的配体",
        "pose0.pdb": "最佳位姿的配体（PDB格式）"
    }
    
    for filename, description in tmp_files.items():
        full_path = os.path.join(tmp_dir, filename)
        print(f"   {filename}: {full_path}")
        print(f"     用途: {description}")
    
    # 4. 最终输出路径
    print(f"\n4️⃣ 最终输出路径:")
    print(f"   口袋文件: {pocket_pdb}")
    print(f"   预测结果: {output_dir}/predictions.csv")
    print(f"   成功预测: {output_dir}/successful_predictions.csv")
    print(f"   失败预测: {output_dir}/failed_predictions.csv")
    print(f"   统计信息: {output_dir}/stats.csv")
    
    # 5. 文件检查逻辑
    print(f"\n5️⃣ 文件检查逻辑:")
    print(f"   检查口袋文件是否存在: {pocket_pdb}")
    print(f"   如果存在且大小>0: 跳过对接，直接使用")
    print(f"   如果不存在: 调用run_preprocess进行对接")
    
    # 6. 实际运行示例
    print(f"\n6️⃣ 实际运行示例:")
    print(f"   python pred_range_fixed.py \\")
    print(f"     --input your_data.csv \\")
    print(f"     --model your_model.pt \\")
    print(f"     --output {output_dir} \\")
    print(f"     --start 0 --end 100 \\")
    print(f"     --use-sample-manager \\")
    print(f"     --sample-data-dir {sample_data_dir}")
    
    print(f"\n📊 文件流程总结:")
    print(f"   输入: 蛋白质PDB + SMILES字符串")
    print(f"   处理: 对接 → 口袋提取 → 图构建 → 预测")
    print(f"   输出: 口袋PDB + 预测结果CSV")

def show_file_relationships():
    """显示文件之间的关系"""
    print(f"\n🔗 文件关系图:")
    print(f"""
    CSV输入数据
         ↓
    蛋白质PDB (通过SampleManager获取)
         ↓
    ┌─────────────────────────────────────┐
    │           对接过程 (tmp目录)          │
    │  ┌─────────────────────────────────┐ │
    │  │ 1. clean_altlocs()             │ │
    │  │    prot.pdb → prot_clean.pdb   │ │
    │  └─────────────────────────────────┘ │
    │  ┌─────────────────────────────────┐ │
    │  │ 2. smiles_to_3d()              │ │
    │  │    SMILES → ligand.pdb         │ │
    │  └─────────────────────────────────┘ │
    │  ┌─────────────────────────────────┐ │
    │  │ 3. prepare_ligand/receptor     │ │
    │  │    ligand.pdb → ligand.pdbqt   │ │
    │  │    prot_clean.pdb → receptor.pdbqt │ │
    │  └─────────────────────────────────┘ │
    │  ┌─────────────────────────────────┐ │
    │  │ 4. AutoSite                    │ │
    │  │    receptor.pdbqt → receptor_cl_001.pdb │ │
    │  └─────────────────────────────────┘ │
    │  ┌─────────────────────────────────┐ │
    │  │ 5. Vina Docking                │ │
    │  │    ligand.pdbqt + receptor.pdbqt │ │
    │  │    → vina_out.pdbqt            │ │
    │  └─────────────────────────────────┘ │
    │  ┌─────────────────────────────────┐ │
    │  │ 6. vina_split                  │ │
    │  │    vina_out.pdbqt → pose0.pdb  │ │
    │  └─────────────────────────────────┘ │
    └─────────────────────────────────────┘
         ↓
    extract_pocket_pymol()
    receptor.pdbqt + pose0.pdb → {base_name}_10A.pdb
         ↓
    图构建和预测
         ↓
    预测结果CSV
    """)

if __name__ == "__main__":
    analyze_pred_range_paths()
    show_file_relationships()
