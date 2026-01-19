#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pose Validation Module
对接结果的物理合理性验证：clash check, stereochemistry, bond geometry

用于验证 DiffDock/Vina/Chai-1 等工具输出的 pose 是否物理上合理
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors
from Bio.PDB import PDBParser, NeighborSearch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ValidationResult:
    """Pose 验证结果"""
    is_valid: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def __str__(self):
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        result = [f"Validation Result: {status}"]
        if self.issues:
            result.append(f"  Issues ({len(self.issues)}):")
            for issue in self.issues:
                result.append(f"    - {issue}")
        if self.warnings:
            result.append(f"  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                result.append(f"    ⚠️ {warning}")
        if self.metrics:
            result.append(f"  Metrics:")
            for key, value in self.metrics.items():
                result.append(f"    {key}: {value:.4f}")
        return "\n".join(result)


@dataclass
class ClashCheckResult:
    """Clash 检测结果"""
    has_clash: bool
    clash_count: int
    clash_pairs: List[Tuple[int, int, float]] = field(default_factory=list)  # (atom_i, atom_j, distance)
    min_distance: float = float('inf')


# ============================================================================
# Clash Detection
# ============================================================================

# VDW 半径（Å）- 用于 clash 检测
VDW_RADII = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80,
    'P': 1.80, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98,
    'Fe': 1.80, 'Zn': 1.39, 'Mg': 1.73, 'Ca': 1.97, 'Na': 2.27,
    'K': 2.75, 'Cu': 1.40, 'Mn': 1.80, 'Co': 1.80, 'Ni': 1.63,
}

# 默认 VDW 半径
DEFAULT_VDW_RADIUS = 1.70

# Clash 阈值：如果两个非键合原子之间的距离 < sum(vdw) * threshold，则为 clash
CLASH_THRESHOLD = 0.6  # 60% of VDW sum


def get_vdw_radius(element: str) -> float:
    """获取原子的 VDW 半径"""
    element = element.strip().upper()
    if len(element) > 1:
        element = element[0] + element[1:].lower()
    return VDW_RADII.get(element, DEFAULT_VDW_RADIUS)


def check_clash_pdb(protein_pdb: str, ligand_pdb: str, 
                    clash_threshold: float = CLASH_THRESHOLD,
                    ignore_hydrogens: bool = True) -> ClashCheckResult:
    """
    检测蛋白质和配体之间的 clash
    
    Args:
        protein_pdb: 蛋白质 PDB 文件路径
        ligand_pdb: 配体 PDB 文件路径
        clash_threshold: Clash 阈值（VDW 距离的比例）
        ignore_hydrogens: 是否忽略氢原子
    
    Returns:
        ClashCheckResult: Clash 检测结果
    """
    parser = PDBParser(QUIET=True)
    
    # 加载结构
    protein_structure = parser.get_structure('protein', protein_pdb)
    ligand_structure = parser.get_structure('ligand', ligand_pdb)
    
    # 收集原子
    protein_atoms = []
    for model in protein_structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if ignore_hydrogens and atom.element.strip() == 'H':
                        continue
                    protein_atoms.append(atom)
    
    ligand_atoms = []
    for model in ligand_structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if ignore_hydrogens and atom.element.strip() == 'H':
                        continue
                    ligand_atoms.append(atom)
    
    # 检测 clash
    clash_pairs = []
    min_distance = float('inf')
    
    for lig_atom in ligand_atoms:
        lig_coord = lig_atom.get_coord()
        lig_radius = get_vdw_radius(lig_atom.element)
        
        for prot_atom in protein_atoms:
            prot_coord = prot_atom.get_coord()
            prot_radius = get_vdw_radius(prot_atom.element)
            
            distance = np.linalg.norm(lig_coord - prot_coord)
            vdw_sum = lig_radius + prot_radius
            threshold_distance = vdw_sum * clash_threshold
            
            min_distance = min(min_distance, distance)
            
            if distance < threshold_distance:
                clash_pairs.append((
                    lig_atom.get_serial_number(),
                    prot_atom.get_serial_number(),
                    distance
                ))
    
    return ClashCheckResult(
        has_clash=len(clash_pairs) > 0,
        clash_count=len(clash_pairs),
        clash_pairs=clash_pairs,
        min_distance=min_distance
    )


def check_internal_clash_sdf(sdf_file: str, 
                              clash_threshold: float = 0.5,
                              bond_factor: float = 1.3) -> ClashCheckResult:
    """
    检测配体分子内部的 clash（排除键合原子）
    
    Args:
        sdf_file: 配体 SDF 文件路径
        clash_threshold: Clash 阈值
        bond_factor: 键合原子距离因子（用于排除键合原子对）
    
    Returns:
        ClashCheckResult: Clash 检测结果
    """
    mol = Chem.SDMolSupplier(sdf_file)[0]
    if mol is None:
        return ClashCheckResult(has_clash=False, clash_count=0, min_distance=0.0)
    
    # 获取键合原子对
    bonded_pairs = set()
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonded_pairs.add((min(i, j), max(i, j)))
    
    # 获取坐标
    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    
    clash_pairs = []
    min_distance = float('inf')
    
    for i in range(num_atoms):
        atom_i = mol.GetAtomWithIdx(i)
        if atom_i.GetSymbol() == 'H':
            continue
        coord_i = np.array(conf.GetAtomPosition(i))
        radius_i = get_vdw_radius(atom_i.GetSymbol())
        
        for j in range(i + 1, num_atoms):
            atom_j = mol.GetAtomWithIdx(j)
            if atom_j.GetSymbol() == 'H':
                continue
            
            # 跳过键合原子
            if (i, j) in bonded_pairs:
                continue
            
            coord_j = np.array(conf.GetAtomPosition(j))
            radius_j = get_vdw_radius(atom_j.GetSymbol())
            
            distance = np.linalg.norm(coord_i - coord_j)
            vdw_sum = radius_i + radius_j
            threshold_distance = vdw_sum * clash_threshold
            
            min_distance = min(min_distance, distance)
            
            if distance < threshold_distance:
                clash_pairs.append((i, j, distance))
    
    return ClashCheckResult(
        has_clash=len(clash_pairs) > 0,
        clash_count=len(clash_pairs),
        clash_pairs=clash_pairs,
        min_distance=min_distance
    )


# ============================================================================
# Stereochemistry Validation
# ============================================================================

def check_stereochemistry(original_smiles: str, pose_sdf: str) -> Tuple[bool, List[str]]:
    """
    验证 pose 的立体化学是否与原始 SMILES 一致
    
    Args:
        original_smiles: 原始配体 SMILES
        pose_sdf: Docked pose 的 SDF 文件
    
    Returns:
        (is_valid, issues): 是否有效和问题列表
    """
    issues = []
    
    # 解析原始 SMILES
    original_mol = Chem.MolFromSmiles(original_smiles)
    if original_mol is None:
        issues.append(f"Cannot parse original SMILES: {original_smiles}")
        return False, issues
    
    # 解析 pose
    pose_mol = Chem.SDMolSupplier(pose_sdf)[0]
    if pose_mol is None:
        issues.append(f"Cannot parse pose SDF: {pose_sdf}")
        return False, issues
    
    # 获取手性中心
    original_chiral = Chem.FindMolChiralCenters(original_mol, includeUnassigned=True)
    pose_chiral = Chem.FindMolChiralCenters(pose_mol, includeUnassigned=True)
    
    # 比较手性中心数量
    if len(original_chiral) != len(pose_chiral):
        issues.append(
            f"Chiral center count mismatch: original={len(original_chiral)}, pose={len(pose_chiral)}"
        )
    
    # 比较手性配置
    original_chiral_dict = {idx: config for idx, config in original_chiral}
    pose_chiral_dict = {idx: config for idx, config in pose_chiral}
    
    for idx, orig_config in original_chiral_dict.items():
        if idx in pose_chiral_dict:
            pose_config = pose_chiral_dict[idx]
            if orig_config != pose_config and orig_config != '?' and pose_config != '?':
                issues.append(
                    f"Chiral center {idx} configuration changed: {orig_config} -> {pose_config}"
                )
    
    # 检查 E/Z 双键
    # TODO: 添加 E/Z 双键检查
    
    is_valid = len(issues) == 0
    return is_valid, issues


# ============================================================================
# Bond Geometry Validation
# ============================================================================

# 典型键长范围（Å）
BOND_LENGTH_RANGES = {
    ('C', 'C', 1): (1.45, 1.60),   # C-C 单键
    ('C', 'C', 2): (1.30, 1.40),   # C=C 双键
    ('C', 'C', 3): (1.15, 1.25),   # C≡C 三键
    ('C', 'N', 1): (1.40, 1.55),   # C-N 单键
    ('C', 'N', 2): (1.25, 1.35),   # C=N 双键
    ('C', 'O', 1): (1.35, 1.50),   # C-O 单键
    ('C', 'O', 2): (1.18, 1.28),   # C=O 双键
    ('C', 'S', 1): (1.75, 1.90),   # C-S 单键
    ('N', 'N', 1): (1.35, 1.50),   # N-N 单键
    ('N', 'O', 1): (1.35, 1.50),   # N-O 单键
    ('O', 'H', 1): (0.90, 1.05),   # O-H
    ('N', 'H', 1): (0.95, 1.10),   # N-H
    ('C', 'H', 1): (1.00, 1.15),   # C-H
}

# 典型键角范围（度）
BOND_ANGLE_RANGES = {
    'sp3': (100, 115),   # 四面体，理想 109.5°
    'sp2': (115, 125),   # 平面三角，理想 120°
    'sp': (170, 180),    # 线性，理想 180°
}


def check_bond_geometry(sdf_file: str, 
                        tolerance: float = 0.15) -> Tuple[bool, List[str], Dict]:
    """
    检查配体的键长和键角是否在合理范围内
    
    Args:
        sdf_file: 配体 SDF 文件路径
        tolerance: 容差（比例）
    
    Returns:
        (is_valid, issues, metrics): 是否有效、问题列表、度量指标
    """
    issues = []
    metrics = {}
    
    mol = Chem.SDMolSupplier(sdf_file)[0]
    if mol is None:
        issues.append(f"Cannot parse SDF: {sdf_file}")
        return False, issues, metrics
    
    conf = mol.GetConformer()
    
    # 检查键长
    bond_length_issues = 0
    total_bonds = 0
    
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        atom_i = mol.GetAtomWithIdx(i)
        atom_j = mol.GetAtomWithIdx(j)
        
        coord_i = np.array(conf.GetAtomPosition(i))
        coord_j = np.array(conf.GetAtomPosition(j))
        distance = np.linalg.norm(coord_i - coord_j)
        
        # 查找预期范围
        elem_i = atom_i.GetSymbol()
        elem_j = atom_j.GetSymbol()
        bond_order = int(bond.GetBondTypeAsDouble())
        
        key = tuple(sorted([elem_i, elem_j])) + (bond_order,)
        
        if key in BOND_LENGTH_RANGES:
            min_len, max_len = BOND_LENGTH_RANGES[key]
            # 应用容差
            min_len *= (1 - tolerance)
            max_len *= (1 + tolerance)
            
            total_bonds += 1
            
            if distance < min_len or distance > max_len:
                bond_length_issues += 1
                issues.append(
                    f"Bond {elem_i}{i}-{elem_j}{j} length {distance:.3f}Å "
                    f"outside range [{min_len:.2f}, {max_len:.2f}]"
                )
    
    metrics['total_bonds'] = total_bonds
    metrics['bond_length_issues'] = bond_length_issues
    metrics['bond_length_issue_rate'] = bond_length_issues / max(total_bonds, 1)
    
    # 键长问题比例超过 10% 则认为无效
    is_valid = metrics['bond_length_issue_rate'] < 0.1
    
    return is_valid, issues, metrics


# ============================================================================
# Main Validation Function
# ============================================================================

def validate_pose(protein_pdb: str,
                  ligand_sdf: str,
                  original_smiles: Optional[str] = None,
                  check_clash: bool = True,
                  check_stereo: bool = True,
                  check_geometry: bool = True,
                  clash_threshold: float = CLASH_THRESHOLD) -> ValidationResult:
    """
    综合验证对接 pose 的物理合理性
    
    Args:
        protein_pdb: 蛋白质 PDB 文件路径
        ligand_sdf: 配体 SDF 文件路径（docked pose）
        original_smiles: 原始配体 SMILES（用于手性验证）
        check_clash: 是否检查 clash
        check_stereo: 是否检查立体化学
        check_geometry: 是否检查键几何
        clash_threshold: Clash 阈值
    
    Returns:
        ValidationResult: 验证结果
    """
    result = ValidationResult(is_valid=True)
    
    # 1. Clash 检测
    if check_clash:
        try:
            # 转换 SDF 为 PDB 用于 clash 检测
            mol = Chem.SDMolSupplier(ligand_sdf)[0]
            if mol is not None:
                ligand_pdb = ligand_sdf.replace('.sdf', '_temp.pdb')
                Chem.MolToPDBFile(mol, ligand_pdb)
                
                clash_result = check_clash_pdb(protein_pdb, ligand_pdb, clash_threshold)
                result.metrics['min_distance'] = clash_result.min_distance
                result.metrics['clash_count'] = clash_result.clash_count
                
                if clash_result.has_clash:
                    result.is_valid = False
                    result.issues.append(
                        f"Protein-ligand clash detected: {clash_result.clash_count} clashes, "
                        f"min distance {clash_result.min_distance:.2f}Å"
                    )
                
                # 清理临时文件
                if os.path.exists(ligand_pdb):
                    os.remove(ligand_pdb)
                    
            # 内部 clash
            internal_clash = check_internal_clash_sdf(ligand_sdf)
            result.metrics['internal_clash_count'] = internal_clash.clash_count
            
            if internal_clash.has_clash:
                result.warnings.append(
                    f"Internal ligand clash: {internal_clash.clash_count} clashes"
                )
                
        except Exception as e:
            result.warnings.append(f"Clash check failed: {e}")
    
    # 2. 立体化学验证
    if check_stereo and original_smiles:
        try:
            stereo_valid, stereo_issues = check_stereochemistry(original_smiles, ligand_sdf)
            if not stereo_valid:
                result.is_valid = False
                result.issues.extend(stereo_issues)
        except Exception as e:
            result.warnings.append(f"Stereochemistry check failed: {e}")
    
    # 3. 键几何验证
    if check_geometry:
        try:
            geom_valid, geom_issues, geom_metrics = check_bond_geometry(ligand_sdf)
            result.metrics.update(geom_metrics)
            
            if not geom_valid:
                result.is_valid = False
                # 只添加最多 5 个问题
                result.issues.extend(geom_issues[:5])
                if len(geom_issues) > 5:
                    result.issues.append(f"... and {len(geom_issues) - 5} more bond issues")
        except Exception as e:
            result.warnings.append(f"Bond geometry check failed: {e}")
    
    return result


def validate_pose_quick(ligand_sdf: str,
                        max_clash_count: int = 5,
                        min_distance_threshold: float = 1.0) -> bool:
    """
    快速验证 pose（只检查内部 clash）
    
    Args:
        ligand_sdf: 配体 SDF 文件路径
        max_clash_count: 最大允许的 clash 数量
        min_distance_threshold: 最小距离阈值
    
    Returns:
        bool: 是否通过验证
    """
    try:
        clash_result = check_internal_clash_sdf(ligand_sdf)
        
        if clash_result.clash_count > max_clash_count:
            return False
        if clash_result.min_distance < min_distance_threshold:
            return False
        
        return True
    except:
        return False


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate docking pose")
    parser.add_argument("--protein", type=str, required=True, help="Protein PDB file")
    parser.add_argument("--ligand", type=str, required=True, help="Ligand SDF file")
    parser.add_argument("--smiles", type=str, default=None, help="Original SMILES for stereochemistry check")
    parser.add_argument("--no-clash", action="store_true", help="Skip clash check")
    parser.add_argument("--no-stereo", action="store_true", help="Skip stereochemistry check")
    parser.add_argument("--no-geometry", action="store_true", help="Skip geometry check")
    
    args = parser.parse_args()
    
    result = validate_pose(
        protein_pdb=args.protein,
        ligand_sdf=args.ligand,
        original_smiles=args.smiles,
        check_clash=not args.no_clash,
        check_stereo=not args.no_stereo,
        check_geometry=not args.no_geometry
    )
    
    print(result)

