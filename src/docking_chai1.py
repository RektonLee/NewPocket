#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chai-1 对接模块
使用 Chai-1 co-folding 模型进行蛋白质-配体复合物结构预测

Chai-1 优势：
- 不需要单独的对接步骤（直接预测复合物结构）
- 支持 apo 结构输入
- PoseBusters benchmark ~80% 成功率
- 支持约束条件（pocket residues, contacts）

安装：pip install chai_lab
硬件要求：推荐 A100/H100 GPU（至少 40GB 显存）
"""

import os
import sys
import tempfile
import logging
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from dataclasses import dataclass
import numpy as np

# RDKit for SMILES handling
from rdkit import Chem
from rdkit.Chem import AllChem

# Bio.PDB for structure processing
from Bio.PDB import PDBParser, PDBIO, Select

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Chai1Config:
    """Chai-1 配置"""
    num_trunk_recycles: int = 3      # Trunk recycle 次数
    num_diffn_timesteps: int = 200   # 扩散时间步数
    seed: int = 42                   # 随机种子
    use_esm_embeddings: bool = True  # 是否使用 ESM embeddings
    device: str = "cuda"             # 设备
    output_format: str = "pdb"       # 输出格式


# ============================================================================
# Chai-1 Availability Check
# ============================================================================

def check_chai1_available() -> Tuple[bool, str]:
    """
    检查 Chai-1 是否可用
    
    Returns:
        (available, message): 是否可用和消息
    """
    try:
        import chai_lab
        from chai_lab.chai1 import run_inference
        return True, f"Chai-1 available (version: {getattr(chai_lab, '__version__', 'unknown')})"
    except ImportError as e:
        return False, f"Chai-1 not installed: {e}. Install with: pip install chai_lab"
    except Exception as e:
        return False, f"Chai-1 check failed: {e}"


def check_gpu_memory() -> Tuple[bool, str]:
    """
    检查 GPU 显存是否足够
    
    Returns:
        (sufficient, message): 是否足够和消息
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        
        # 获取 GPU 显存
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9  # GB
        
        # Chai-1 推荐至少 40GB 显存
        if total_memory < 40:
            return False, f"GPU memory {total_memory:.1f}GB < 40GB (recommended)"
        
        return True, f"GPU memory {total_memory:.1f}GB"
    except Exception as e:
        return False, f"GPU check failed: {e}"


# ============================================================================
# Input Preparation
# ============================================================================

def prepare_fasta_input(protein_sequence: str,
                        ligand_smiles: str,
                        protein_name: str = "enzyme",
                        ligand_name: str = "substrate") -> str:
    """
    准备 Chai-1 输入的 FASTA 格式
    
    Chai-1 FASTA 格式:
    >protein|name=xxx
    SEQUENCE...
    >ligand|name=xxx
    SMILES
    
    Args:
        protein_sequence: 蛋白质序列
        ligand_smiles: 配体 SMILES
        protein_name: 蛋白质名称
        ligand_name: 配体名称
    
    Returns:
        FASTA 格式字符串
    """
    fasta_content = f""">protein|name={protein_name}
{protein_sequence}
>ligand|name={ligand_name}
{ligand_smiles}
"""
    return fasta_content


def prepare_fasta_with_constraints(protein_sequence: str,
                                    ligand_smiles: str,
                                    pocket_residues: Optional[List[int]] = None,
                                    protein_name: str = "enzyme",
                                    ligand_name: str = "substrate") -> str:
    """
    准备带约束的 Chai-1 输入
    
    Args:
        protein_sequence: 蛋白质序列
        ligand_smiles: 配体 SMILES
        pocket_residues: 口袋残基索引列表（1-based）
        protein_name: 蛋白质名称
        ligand_name: 配体名称
    
    Returns:
        FASTA 格式字符串
    """
    # 基础 FASTA
    fasta_content = prepare_fasta_input(protein_sequence, ligand_smiles, 
                                        protein_name, ligand_name)
    
    # 添加约束（如果提供）
    # Chai-1 约束格式需要根据实际 API 调整
    # 这里是一个示例格式
    if pocket_residues:
        constraint_line = f"# pocket_residues: {','.join(map(str, pocket_residues))}\n"
        fasta_content = constraint_line + fasta_content
    
    return fasta_content


# ============================================================================
# Chai-1 Inference
# ============================================================================

def run_chai1_inference(fasta_content: str,
                        output_dir: str,
                        config: Optional[Chai1Config] = None) -> Optional[str]:
    """
    运行 Chai-1 推理
    
    Args:
        fasta_content: FASTA 格式输入
        output_dir: 输出目录
        config: Chai-1 配置
    
    Returns:
        output_pdb: 输出 PDB 文件路径，失败返回 None
    """
    if config is None:
        config = Chai1Config()
    
    # 检查可用性
    available, msg = check_chai1_available()
    if not available:
        logger.error(msg)
        return None
    
    try:
        from chai_lab.chai1 import run_inference
        
        # 创建临时 FASTA 文件
        os.makedirs(output_dir, exist_ok=True)
        fasta_file = os.path.join(output_dir, "input.fasta")
        with open(fasta_file, 'w') as f:
            f.write(fasta_content)
        
        logger.info(f"Running Chai-1 inference...")
        logger.info(f"  Input: {fasta_file}")
        logger.info(f"  Output: {output_dir}")
        logger.info(f"  Config: recycles={config.num_trunk_recycles}, "
                   f"diffn_steps={config.num_diffn_timesteps}")
        
        # 运行推理
        run_inference(
            fasta_file=Path(fasta_file),
            output_dir=Path(output_dir),
            num_trunk_recycles=config.num_trunk_recycles,
            num_diffn_timesteps=config.num_diffn_timesteps,
            seed=config.seed,
            use_esm_embeddings=config.use_esm_embeddings,
            device=config.device
        )
        
        # 查找输出文件
        output_pdb = find_chai1_output(output_dir)
        if output_pdb:
            logger.info(f"✅ Chai-1 inference completed: {output_pdb}")
        else:
            logger.error("❌ Chai-1 did not generate output file")
        
        return output_pdb
        
    except Exception as e:
        logger.error(f"❌ Chai-1 inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def find_chai1_output(output_dir: str) -> Optional[str]:
    """
    查找 Chai-1 输出文件
    
    Args:
        output_dir: 输出目录
    
    Returns:
        PDB 文件路径，未找到返回 None
    """
    # Chai-1 输出格式可能包含 pred_model_X.pdb 或类似文件
    possible_patterns = [
        "pred_model_0.pdb",
        "pred.pdb",
        "output.pdb",
        "complex.pdb",
    ]
    
    for pattern in possible_patterns:
        pdb_path = os.path.join(output_dir, pattern)
        if os.path.exists(pdb_path):
            return pdb_path
    
    # 搜索任何 .pdb 文件
    for f in os.listdir(output_dir):
        if f.endswith('.pdb'):
            return os.path.join(output_dir, f)
    
    return None


# ============================================================================
# Pocket Extraction from Chai-1 Output
# ============================================================================

def extract_pocket_from_complex(complex_pdb: str,
                                 output_pocket_pdb: str,
                                 ligand_chain: str = "L",
                                 cutoff: float = 5.0) -> bool:
    """
    从 Chai-1 复合物结构中提取口袋区域
    
    Args:
        complex_pdb: 复合物 PDB 文件路径
        output_pocket_pdb: 输出口袋 PDB 文件路径
        ligand_chain: 配体链 ID
        cutoff: 口袋提取距离阈值（Å）
    
    Returns:
        bool: 是否成功
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('complex', complex_pdb)
        
        # 找到配体原子
        ligand_atoms = []
        protein_residues = []
        
        for model in structure:
            for chain in model:
                chain_id = chain.get_id()
                for residue in chain:
                    if chain_id == ligand_chain or residue.get_resname() in ['UNL', 'LIG', 'MOL']:
                        # 配体
                        for atom in residue:
                            ligand_atoms.append(atom.get_coord())
                    else:
                        # 蛋白质
                        protein_residues.append(residue)
        
        if not ligand_atoms:
            logger.warning("No ligand atoms found in complex")
            return False
        
        ligand_coords = np.array(ligand_atoms)
        
        # 选择口袋残基
        class PocketSelect(Select):
            def accept_residue(self, residue):
                for atom in residue:
                    coord = atom.get_coord()
                    distances = np.linalg.norm(ligand_coords - coord, axis=1)
                    if np.min(distances) <= cutoff:
                        return True
                return False
        
        # 保存口袋
        os.makedirs(os.path.dirname(output_pocket_pdb), exist_ok=True)
        io = PDBIO()
        io.set_structure(structure)
        io.save(output_pocket_pdb, PocketSelect())
        
        logger.info(f"✅ Pocket extracted: {output_pocket_pdb}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pocket extraction failed: {e}")
        return False


# ============================================================================
# High-level API
# ============================================================================

def dock_with_chai1(protein_sequence: str,
                    ligand_smiles: str,
                    output_dir: str,
                    pocket_residues: Optional[List[int]] = None,
                    config: Optional[Chai1Config] = None,
                    extract_pocket: bool = True,
                    pocket_cutoff: float = 5.0) -> Dict:
    """
    使用 Chai-1 进行对接（高级 API）
    
    Args:
        protein_sequence: 蛋白质序列
        ligand_smiles: 配体 SMILES
        output_dir: 输出目录
        pocket_residues: 可选的口袋残基约束
        config: Chai-1 配置
        extract_pocket: 是否提取口袋
        pocket_cutoff: 口袋提取距离阈值
    
    Returns:
        dict: {
            'success': bool,
            'complex_pdb': str or None,
            'pocket_pdb': str or None,
            'error': str or None
        }
    """
    result = {
        'success': False,
        'complex_pdb': None,
        'pocket_pdb': None,
        'error': None
    }
    
    try:
        # 准备输入
        if pocket_residues:
            fasta_content = prepare_fasta_with_constraints(
                protein_sequence, ligand_smiles, pocket_residues
            )
        else:
            fasta_content = prepare_fasta_input(protein_sequence, ligand_smiles)
        
        # 运行推理
        complex_pdb = run_chai1_inference(fasta_content, output_dir, config)
        
        if complex_pdb is None:
            result['error'] = "Chai-1 inference failed"
            return result
        
        result['complex_pdb'] = complex_pdb
        
        # 提取口袋
        if extract_pocket:
            pocket_pdb = os.path.join(output_dir, "pocket.pdb")
            if extract_pocket_from_complex(complex_pdb, pocket_pdb, cutoff=pocket_cutoff):
                result['pocket_pdb'] = pocket_pdb
        
        result['success'] = True
        return result
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"❌ Chai-1 docking failed: {e}")
        return result


# ============================================================================
# Fallback: Mock Implementation for Testing
# ============================================================================

def dock_with_chai1_mock(protein_sequence: str,
                         ligand_smiles: str,
                         output_dir: str,
                         **kwargs) -> Dict:
    """
    Mock 实现，用于测试（当 Chai-1 不可用时）
    
    警告：这不是真正的对接，只是生成一个假的复合物结构用于测试
    """
    logger.warning("⚠️ Using mock Chai-1 implementation (not real docking!)")
    
    result = {
        'success': False,
        'complex_pdb': None,
        'pocket_pdb': None,
        'error': None,
        'is_mock': True
    }
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成一个简单的假 PDB
        complex_pdb = os.path.join(output_dir, "mock_complex.pdb")
        
        with open(complex_pdb, 'w') as f:
            f.write("HEADER    MOCK CHAI-1 OUTPUT\n")
            f.write("REMARK    This is a mock structure for testing only!\n")
            f.write("REMARK    Not a real Chai-1 prediction.\n")
            
            # 生成蛋白质骨架
            for i, aa in enumerate(protein_sequence[:100]):  # 只取前 100 个残基
                x = i * 3.8
                y = 0.0
                z = 0.0
                f.write(f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n")
            
            # 生成配体
            mol = Chem.MolFromSmiles(ligand_smiles)
            if mol:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, randomSeed=42)
                conf = mol.GetConformer()
                for j in range(mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(j)
                    atom = mol.GetAtomWithIdx(j)
                    f.write(f"HETATM{1000+j:5d} {atom.GetSymbol():4s} LIG L   1    "
                           f"{pos.x+50:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00 20.00          "
                           f" {atom.GetSymbol():2s}\n")
            
            f.write("END\n")
        
        result['complex_pdb'] = complex_pdb
        result['success'] = True
        logger.info(f"✅ Mock complex generated: {complex_pdb}")
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chai-1 docking module")
    parser.add_argument("--sequence", type=str, help="Protein sequence")
    parser.add_argument("--smiles", type=str, help="Ligand SMILES")
    parser.add_argument("--output", type=str, default="output/chai1", help="Output directory")
    parser.add_argument("--check", action="store_true", help="Check Chai-1 availability")
    parser.add_argument("--mock", action="store_true", help="Use mock implementation")
    
    args = parser.parse_args()
    
    if args.check:
        available, msg = check_chai1_available()
        print(f"Chai-1 available: {available}")
        print(f"Message: {msg}")
        
        gpu_ok, gpu_msg = check_gpu_memory()
        print(f"GPU sufficient: {gpu_ok}")
        print(f"GPU info: {gpu_msg}")
        
    elif args.sequence and args.smiles:
        if args.mock:
            result = dock_with_chai1_mock(args.sequence, args.smiles, args.output)
        else:
            result = dock_with_chai1(args.sequence, args.smiles, args.output)
        
        print(f"Result: {result}")
    else:
        parser.print_help()

