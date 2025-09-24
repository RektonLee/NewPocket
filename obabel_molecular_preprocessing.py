#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用OpenBabel进行分子预处理的脚本
替代ADFRsuite的prepare_ligand和prepare_receptor功能
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import subprocess
import tempfile
import shutil
from datetime import datetime
from tqdm import tqdm
from Bio.PDB import PDBParser, NeighborSearch, Select, PDBIO
import re

def setup_logging():
    """设置日志"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = 'logs/obabel_preprocessing_{}.log'.format(timestamp)
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def clean_pdb_file(input_pdb, output_pdb):
    """清理PDB文件，移除水分子、金属离子等"""
    logger = logging.getLogger()
    logger.info("清理PDB文件: {} -> {}".format(input_pdb, output_pdb))
    
    # 需要移除的残基类型
    remove_residues = {
        'HOH', 'WAT', 'TIP', 'TIP3', 'TIP4',  # 水分子
        'MG', 'MN', 'FE', 'ZN', 'CA', 'CU', 'CO', 'NI', 'NA', 'K', 'CL',  # 金属离子
        'SO4', 'PO4', 'NO3', 'CO3',  # 无机离子
    }
    
    with open(input_pdb, 'r') as fin, open(output_pdb, 'w') as fout:
        for line in fin:
            if line.startswith(('ATOM', 'HETATM')):
                resname = line[17:20].strip()
                atomname = line[12:16].strip()
                
                # 跳过需要移除的残基
                if resname in remove_residues:
                    continue
                    
                # 跳过金属原子
                if atomname in {'K', 'SE', 'ZN', 'MG', 'MN', 'FE', 'CA', 'CU', 'CO', 'NI', 'NA', 'CL'}:
                    continue
                    
                # 清理替代位置标记
                if line[16] != ' ' and line[16] != 'A':
                    line = line[:16] + ' ' + line[17:]
                
                fout.write(line)
            else:
                fout.write(line)
    
    logger.info("✅ PDB文件清理完成: {}".format(output_pdb))

def prepare_ligand_with_obabel(ligand_pdb, output_pdbqt, timeout=60):
    """使用OpenBabel准备配体文件"""
    logger = logging.getLogger()
    logger.info("使用OpenBabel准备配体: {} -> {}".format(ligand_pdb, output_pdbqt))
    
    try:
        # 首先尝试简化的转换（无能量最小化）
        cmd_simple = [
            'obabel', 
            ligand_pdb, 
            '-O', output_pdbqt,
            '--partialcharge', 'gasteiger',  # 计算Gasteiger电荷
            '--addpolarh'  # 添加极性氢
        ]
        
        result = subprocess.run(cmd_simple, capture_output=True, text=True, 
                              check=True, timeout=timeout)
        logger.info("✅ 配体准备成功（简化模式）: {}".format(output_pdbqt))
        return True
        
    except subprocess.TimeoutExpired:
        logger.error("❌ 配体准备超时: {}秒".format(timeout))
        
        # 尝试更简单的转换
        try:
            cmd_basic = ['obabel', ligand_pdb, '-O', output_pdbqt]
            result = subprocess.run(cmd_basic, capture_output=True, text=True, 
                                  check=True, timeout=30)
            logger.info("✅ 配体准备成功（基础模式）: {}".format(output_pdbqt))
            return True
        except Exception as e2:
            logger.error("❌ 基础模式也失败: {}".format(e2))
            return False
        
    except subprocess.CalledProcessError as e:
        logger.error("❌ 配体准备失败: {}".format(e))
        logger.error("错误输出: {}".format(e.stderr))
        
        # 尝试基础转换
        try:
            cmd_basic = ['obabel', ligand_pdb, '-O', output_pdbqt]
            result = subprocess.run(cmd_basic, capture_output=True, text=True, 
                                  check=True, timeout=30)
            logger.info("✅ 配体准备成功（基础模式）: {}".format(output_pdbqt))
            return True
        except Exception as e2:
            logger.error("❌ 基础模式也失败: {}".format(e2))
            return False
            
    except Exception as e:
        logger.error("❌ 配体准备异常: {}".format(e))
        return False

def prepare_receptor_with_obabel(protein_pdb, output_pdbqt, timeout=120):
    """使用OpenBabel准备受体文件"""
    logger = logging.getLogger()
    logger.info("使用OpenBabel准备受体: {} -> {}".format(protein_pdb, output_pdbqt))
    
    try:
        # 使用OpenBabel转换蛋白质PDB到PDBQT格式
        cmd = [
            'obabel',
            protein_pdb,
            '-O', output_pdbqt,
            '--partialcharge', 'gasteiger',  # 计算Gasteiger电荷
            '--addpolarh'  # 添加极性氢
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              check=True, timeout=timeout)
        logger.info("✅ 受体准备成功: {}".format(output_pdbqt))
        return True
        
    except subprocess.TimeoutExpired:
        logger.error("❌ 受体准备超时: {}秒".format(timeout))
        
        # 尝试更简单的转换
        try:
            cmd_basic = ['obabel', protein_pdb, '-O', output_pdbqt]
            result = subprocess.run(cmd_basic, capture_output=True, text=True, 
                                  check=True, timeout=60)
            logger.info("✅ 受体准备成功（基础模式）: {}".format(output_pdbqt))
            return True
        except Exception as e2:
            logger.error("❌ 基础模式也失败: {}".format(e2))
            return False
        
    except subprocess.CalledProcessError as e:
        logger.error("❌ 受体准备失败: {}".format(e))
        logger.error("错误输出: {}".format(e.stderr))
        
        # 尝试基础转换
        try:
            cmd_basic = ['obabel', protein_pdb, '-O', output_pdbqt]
            result = subprocess.run(cmd_basic, capture_output=True, text=True, 
                                  check=True, timeout=60)
            logger.info("✅ 受体准备成功（基础模式）: {}".format(output_pdbqt))
            return True
        except Exception as e2:
            logger.error("❌ 基础模式也失败: {}".format(e2))
            return False
            
    except Exception as e:
        logger.error("❌ 受体准备异常: {}".format(e))
        return False

def find_binding_site_center(protein_pdb, ligand_pdb=None, radius=10.0):
    """识别结合位点中心"""
    logger = logging.getLogger()
    logger.info("识别结合位点中心: {}".format(protein_pdb))
    
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', protein_pdb)
        
        # 如果有配体文件，基于配体位置计算中心
        if ligand_pdb and os.path.exists(ligand_pdb):
            ligand_structure = parser.get_structure('ligand', ligand_pdb)
            ligand_atoms = []
            for model in ligand_structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            ligand_atoms.append(atom.get_coord())
            
            if ligand_atoms:
                center = np.mean(ligand_atoms, axis=0)
                logger.info("✅ 基于配体计算结合位点中心: {}".format(center))
                return center
        
        # 否则基于蛋白质结构计算中心
        protein_atoms = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        protein_atoms.append(atom.get_coord())
        
        if protein_atoms:
            center = np.mean(protein_atoms, axis=0)
            logger.info("✅ 基于蛋白质计算结合位点中心: {}".format(center))
            return center
        
        # 默认中心
        default_center = [0.0, 0.0, 0.0]
        logger.warning("⚠️ 使用默认结合位点中心: {}".format(default_center))
        return default_center
        
    except Exception as e:
        logger.error("❌ 结合位点识别失败: {}".format(e))
        return [0.0, 0.0, 0.0]

def extract_pocket_region(protein_pdb, center, radius=10.0, output_pdb=None):
    """提取口袋区域"""
    logger = logging.getLogger()
    logger.info("提取口袋区域，中心: {}, 半径: {}".format(center, radius))
    
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', protein_pdb)
        
        class PocketSelect(Select):
            def accept_atom(self, atom):
                coord = atom.get_coord()
                distance = np.linalg.norm(coord - np.array(center))
                return distance <= radius
        
        if output_pdb is None:
            output_pdb = protein_pdb.replace('.pdb', '_pocket.pdb')
        
        io = PDBIO()
        io.set_structure(structure)
        io.save(output_pdb, PocketSelect())
        
        logger.info("✅ 口袋区域已保存: {}".format(output_pdb))
        return output_pdb
        
    except Exception as e:
        logger.error("❌ 口袋提取失败: {}".format(e))
        return None

class OpenBabelMolecularPreprocessor:
    """OpenBabel分子预处理器"""
    
    def __init__(self, output_dir='output/obabel_preprocessing'):
        self.logger = logging.getLogger()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def preprocess_molecule_pair(self, protein_pdb, ligand_pdb, sample_id):
        """预处理蛋白质-配体对"""
        try:
            self.logger.info("开始预处理样本 {}...".format(sample_id))
            
            # 创建工作目录
            work_dir = os.path.join(self.output_dir, "sample_{}".format(sample_id))
            os.makedirs(work_dir, exist_ok=True)
            
            # 1. 清理蛋白质结构
            clean_prot_path = os.path.join(work_dir, "protein_clean.pdb")
            clean_pdb_file(protein_pdb, clean_prot_path)
            
            # 2. 准备配体
            lig_pdbqt = os.path.join(work_dir, "ligand.pdbqt")
            if not prepare_ligand_with_obabel(ligand_pdb, lig_pdbqt):
                raise ValueError("配体准备失败")
            
            # 3. 准备受体
            receptor_pdbqt = os.path.join(work_dir, "receptor.pdbqt")
            if not prepare_receptor_with_obabel(clean_prot_path, receptor_pdbqt):
                raise ValueError("受体准备失败")
            
            # 4. 识别结合位点
            center = find_binding_site_center(clean_prot_path, ligand_pdb)
            
            # 5. 提取口袋区域
            pocket_pdb = os.path.join(work_dir, "pocket.pdb")
            extract_pocket_region(clean_prot_path, center, output_pdb=pocket_pdb)
            
            self.logger.info("✅ 样本 {} 预处理完成".format(sample_id))
            
            return {
                'sample_id': sample_id,
                'protein_clean': clean_prot_path,
                'ligand_pdbqt': lig_pdbqt,
                'receptor_pdbqt': receptor_pdbqt,
                'pocket_pdb': pocket_pdb,
                'binding_center': center,
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error("❌ 样本 {} 预处理失败: {}".format(sample_id, e))
            return {
                'sample_id': sample_id,
                'status': 'failed',
                'error': str(e)
            }

def process_test_structures_with_obabel(input_file, output_dir='output/obabel_preprocessing'):
    """批量处理测试结构"""
    logger = setup_logging()
    logger.info("开始使用OpenBabel批量预处理: {}".format(input_file))
    
    # 读取数据
    df = pd.read_csv(input_file)
    logger.info("共 {} 个样本需要处理".format(len(df)))
    
    # 初始化预处理器
    preprocessor = OpenBabelMolecularPreprocessor(output_dir)
    
    # 处理结果
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="OpenBabel预处理"):
        sample_id = row['sample_id']
        protein_pdb = row['protein_pdb']
        molecule_pdb = row['molecule_pdb']
        
        # 检查数据类型和空值
        if pd.isna(protein_pdb) or pd.isna(molecule_pdb):
            logger.warning("⚠️ 样本 {} 的结构文件路径为空，跳过".format(sample_id))
            continue
            
        if not isinstance(protein_pdb, str) or not isinstance(molecule_pdb, str):
            logger.warning("⚠️ 样本 {} 的结构文件路径格式错误，跳过".format(sample_id))
            continue
        
        # 检查文件是否存在
        if not os.path.exists(protein_pdb):
            logger.warning("⚠️ 蛋白质文件不存在: {}".format(protein_pdb))
            continue
            
        if not os.path.exists(molecule_pdb):
            logger.warning("⚠️ 分子文件不存在: {}".format(molecule_pdb))
            continue
        
        # 预处理
        result = preprocessor.preprocess_molecule_pair(protein_pdb, molecule_pdb, sample_id)
        results.append(result)
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_file = os.path.join(output_dir, 'obabel_preprocessing_results.csv')
    results_df.to_csv(results_file, index=False)
    
    logger.info("✅ OpenBabel预处理完成，结果保存到: {}".format(results_file))
    
    # 统计成功/失败数量
    success_count = (results_df['status'] == 'success').sum()
    logger.info("预处理成功: {}/{}".format(success_count, len(df)))
    
    return results_df

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='使用OpenBabel进行分子预处理')
    parser.add_argument('--input', default='output/structures/structure_results.csv',
                       help='输入结构结果文件')
    parser.add_argument('--output-dir', default='output/obabel_preprocessing',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 处理结构
    results_df = process_test_structures_with_obabel(args.input, args.output_dir)
    
    print("\n预处理完成！")
    print("成功: {}".format((results_df['status'] == 'success').sum()))
    print("失败: {}".format((results_df['status'] == 'failed').sum()))
    print("结果保存在: {}".format(args.output_dir))

if __name__ == "__main__":
    main()
