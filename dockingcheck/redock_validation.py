#!/usr/bin/env python3
"""
Re-docking 验证脚本：使用 Vina 重新对接，计算 RMSD

使用方法：
    python src/redock_validation.py --pdb 1ABC --output results/validation/

功能：
1. 从 PDB 下载 Holo 状态的复合物
2. 提取配体真实坐标（参考）
3. 移除配体，只留蛋白
4. 使用 Vina 重新对接
5. 计算对接结果与真实位置的 RMSD
"""

import os
import argparse
import subprocess
import numpy as np
from pathlib import Path
from typing import Tuple, List
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio import PDB
import logging
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReDockingValidator:
    """Re-docking 验证工具"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_pdb(self, pdb_id: str) -> str:
        """从 RCSB PDB 下载结构"""
        pdb_id = pdb_id.lower()
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        output_path = self.output_dir / f"{pdb_id}_original.pdb"

        logger.info(f"Downloading {pdb_id} from RCSB...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_path, 'w') as f:
                f.write(response.text)
            logger.info(f"Saved to {output_path}")
            return str(output_path)
        else:
            raise ValueError(f"Failed to download {pdb_id}: {response.status_code}")

    def extract_ligand(self, pdb_file: str, ligand_resname: str = None) -> Tuple[str, str]:
        """
        从 PDB 中提取配体和蛋白

        Returns:
            (protein_pdb, ligand_sdf)
        """
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('complex', pdb_file)

        # 分离蛋白和配体
        protein_atoms = []
        ligand_atoms = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    # 跳过水分子
                    if residue.get_resname() == 'HOH':
                        continue

                    # 判断是否为配体（HETATM 且不是标准氨基酸）
                    is_hetero = residue.id[0] != ' '
                    is_standard_aa = residue.get_resname() in PDB.Polypeptide.standard_aa_names

                    if is_hetero and not is_standard_aa:
                        # 如果指定了配体名称，只提取该配体
                        if ligand_resname and residue.get_resname() != ligand_resname:
                            continue
                        ligand_atoms.extend(residue.get_atoms())
                    else:
                        protein_atoms.extend(residue.get_atoms())

        if not ligand_atoms:
            raise ValueError("No ligand found in PDB file")

        # 保存蛋白
        protein_pdb = self.output_dir / f"{Path(pdb_file).stem}_protein.pdb"
        io = PDB.PDBIO()
        io.set_structure(structure)

        class ProteinSelect(PDB.Select):
            def accept_residue(self, residue):
                is_hetero = residue.id[0] != ' '
                is_standard_aa = residue.get_resname() in PDB.Polypeptide.standard_aa_names
                return not (is_hetero and not is_standard_aa) and residue.get_resname() != 'HOH'

        io.save(str(protein_pdb), ProteinSelect())
        logger.info(f"Saved protein to {protein_pdb}")

        # 保存配体坐标（作为参考）
        ligand_coords = np.array([atom.coord for atom in ligand_atoms])
        ligand_pdb = self.output_dir / f"{Path(pdb_file).stem}_ligand_ref.pdb"

        with open(ligand_pdb, 'w') as f:
            for i, atom in enumerate(ligand_atoms):
                f.write(f"HETATM{i+1:5d}  {atom.name:4s} LIG A   1    "
                       f"{atom.coord[0]:8.3f}{atom.coord[1]:8.3f}{atom.coord[2]:8.3f}"
                       f"  1.00  0.00           {atom.element:>2s}\n")

        logger.info(f"Saved reference ligand to {ligand_pdb}")

        return str(protein_pdb), str(ligand_pdb), ligand_coords

    def run_vina_docking(self, protein_pdb: str, ligand_sdf: str, center: np.ndarray,
                        box_size: Tuple[float, float, float] = (20, 20, 20)) -> str:
        """
        使用 Vina 进行对接

        Args:
            protein_pdb: 蛋白 PDB 文件
            ligand_sdf: 配体 SDF 文件
            center: 对接中心坐标 (x, y, z)
            box_size: 对接盒子大小

        Returns:
            对接结果 PDBQT 文件路径
        """
        # 准备输入文件
        protein_pdbqt = protein_pdb.replace('.pdb', '.pdbqt')
        ligand_pdbqt = ligand_sdf.replace('.pdb', '.pdbqt')
        output_pdbqt = self.output_dir / "docked_ligand.pdbqt"

        # 转换为 PDBQT 格式（需要 AutoDockTools）
        logger.info("Converting to PDBQT format...")
        subprocess.run([
            'prepare_receptor4.py',
            '-r', protein_pdb,
            '-o', protein_pdbqt
        ], check=True)

        subprocess.run([
            'prepare_ligand4.py',
            '-l', ligand_sdf,
            '-o', ligand_pdbqt
        ], check=True)

        # 运行 Vina
        logger.info("Running Vina docking...")
        config_file = self.output_dir / "vina_config.txt"
        with open(config_file, 'w') as f:
            f.write(f"receptor = {protein_pdbqt}\n")
            f.write(f"ligand = {ligand_pdbqt}\n")
            f.write(f"center_x = {center[0]:.3f}\n")
            f.write(f"center_y = {center[1]:.3f}\n")
            f.write(f"center_z = {center[2]:.3f}\n")
            f.write(f"size_x = {box_size[0]}\n")
            f.write(f"size_y = {box_size[1]}\n")
            f.write(f"size_z = {box_size[2]}\n")
            f.write(f"out = {output_pdbqt}\n")
            f.write("exhaustiveness = 8\n")

        subprocess.run(['vina', '--config', str(config_file)], check=True)

        logger.info(f"Docking complete: {output_pdbqt}")
        return str(output_pdbqt)

    def calculate_rmsd(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        """
        计算两个构象的 RMSD（不考虑原子对应关系）

        使用最小距离匹配法
        """
        from scipy.spatial.distance import cdist

        # 中心化
        coords1 = coords1 - coords1.mean(axis=0)
        coords2 = coords2 - coords2.mean(axis=0)

        # 计算每个原子到最近原子的距离
        distances = cdist(coords1, coords2)
        min_distances = np.min(distances, axis=1)

        # RMSD
        rmsd = np.sqrt(np.mean(min_distances ** 2))
        return rmsd

    def parse_pdbqt_coords(self, pdbqt_file: str) -> np.ndarray:
        """从 PDBQT 文件中提取坐标"""
        coords = []
        with open(pdbqt_file) as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
        return np.array(coords)

    def validate(self, pdb_id: str, ligand_resname: str = None) -> dict:
        """
        完整的 Re-docking 验证流程

        Returns:
            验证结果字典
        """
        # 1. 下载 PDB
        pdb_file = self.download_pdb(pdb_id)

        # 2. 提取配体和蛋白
        protein_pdb, ligand_pdb, ref_coords = self.extract_ligand(pdb_file, ligand_resname)

        # 3. 计算对接中心（配体质心）
        center = ref_coords.mean(axis=0)

        # 4. 运行对接
        docked_pdbqt = self.run_vina_docking(protein_pdb, ligand_pdb, center)

        # 5. 提取对接结果坐标
        docked_coords = self.parse_pdbqt_coords(docked_pdbqt)

        # 6. 计算 RMSD
        rmsd = self.calculate_rmsd(ref_coords, docked_coords)

        logger.info(f"RMSD: {rmsd:.3f} Å")

        return {
            'pdb_id': pdb_id,
            'rmsd': rmsd,
            'success': rmsd < 2.0,  # 通常 RMSD < 2Å 认为对接成功
            'protein_pdb': protein_pdb,
            'docked_pdbqt': docked_pdbqt
        }


def main():
    parser = argparse.ArgumentParser(description="Re-docking validation")
    parser.add_argument('--pdb', type=str, required=True, help='PDB ID (e.g., 1ABC)')
    parser.add_argument('--ligand', type=str, help='Ligand residue name (optional)')
    parser.add_argument('--output', type=str, default='results/validation/', help='Output directory')
    args = parser.parse_args()

    validator = ReDockingValidator(args.output)
    result = validator.validate(args.pdb, args.ligand)

    print("\n=== Re-docking Validation Result ===")
    print(f"PDB ID: {result['pdb_id']}")
    print(f"RMSD: {result['rmsd']:.3f} Å")
    print(f"Success: {'✓' if result['success'] else '✗'} (threshold: 2.0 Å)")


if __name__ == '__main__':
    main()
