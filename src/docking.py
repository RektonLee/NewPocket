#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子对接模块 (Molecular Docking Module)

支持的对接工具：
- DiffDock: 基于扩散模型的分子对接（当前默认）
- Chai-1: Co-folding 模型（见 docking_chai1.py）
- AutoDock Vina: 传统对接（已移除，可作为参考）

修复记录 (2026-01-19):
- 修复硬编码路径问题
- 添加 pose 验证模块集成
- 改进错误处理
- 支持可配置的 GPU ID
"""
import __main__
__main__.pymol_argv = ['pymol', '-c']
import os
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser, Select, PDBIO
import hashlib
import json
import shutil
import logging
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DockingConfig:
    """对接配置"""
    diffdock_path: Optional[str] = None  # DiffDock 路径，None 则自动检测
    diffdock_python: Optional[str] = None  # DiffDock Python 解释器
    gpu_id: int = 0  # 默认 GPU ID（从环境变量 CUDA_VISIBLE_DEVICES 获取，或使用此默认值）
    inference_steps: int = 20  # DiffDock 推理步数
    samples_per_complex: int = 5  # 每个复合物生成的 pose 数量
    pocket_cutoff: float = 5.0  # 口袋提取距离阈值（Å）
    validate_pose: bool = True  # 是否验证 pose
    failed_log_path: Optional[str] = None  # 失败日志路径，None 则自动生成


def get_default_config() -> DockingConfig:
    """获取默认配置（从环境变量读取）"""
    config = DockingConfig()
    
    # GPU ID: 优先从环境变量读取
    if 'DOCKING_GPU_ID' in os.environ:
        config.gpu_id = int(os.environ['DOCKING_GPU_ID'])
    elif 'CUDA_VISIBLE_DEVICES' in os.environ:
        # 使用 CUDA_VISIBLE_DEVICES 中的第一个 GPU
        devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        if devices and devices[0].strip().isdigit():
            config.gpu_id = int(devices[0].strip())
    
    return config


# 全局配置（可在运行时修改）
_config = get_default_config()


def set_config(config: DockingConfig):
    """设置全局配置"""
    global _config
    _config = config


def get_config() -> DockingConfig:
    """获取当前配置"""
    return _config


# ============================================================================
# Path Detection
# ============================================================================

def _get_project_root() -> str:
    """获取项目根目录"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)  # src 的父目录


def _find_diffdock_path() -> str:
    """自动查找 DiffDock 路径"""
    # 1. 配置优先
    if _config.diffdock_path and os.path.exists(_config.diffdock_path):
        return _config.diffdock_path
    
    # 2. 环境变量
    env_path = os.environ.get("DIFFDOCK_PATH")
    if env_path and os.path.exists(env_path):
        return os.path.abspath(env_path)
    
    # 3. 项目根目录下的 DiffDock
    project_root = _get_project_root()
    default_path = os.path.join(project_root, "DiffDock")
    if os.path.exists(default_path):
        return default_path
    
    raise FileNotFoundError(
        "DiffDock not found. Please:\n"
        "  1. Set DIFFDOCK_PATH environment variable, or\n"
        "  2. Place DiffDock in project root directory, or\n"
        "  3. Set config.diffdock_path"
    )


def _find_diffdock_python() -> str:
    """自动查找 DiffDock Python 解释器"""
    # 1. 配置优先
    if _config.diffdock_python and os.path.exists(_config.diffdock_python):
        return _config.diffdock_python
    
    # 2. 环境变量
    env_python = os.environ.get("DIFFDOCK_PYTHON")
    if env_python and os.path.exists(env_python):
        return env_python
    
    # 3. 自动查找 conda 环境
    conda_base = os.environ.get("CONDA_PREFIX", "")
    if conda_base:
        conda_envs_dir = os.path.join(os.path.dirname(conda_base), "envs")
        diffdock_python = os.path.join(conda_envs_dir, "diffdock", "bin", "python")
        if os.path.exists(diffdock_python):
            return diffdock_python
    
    # 4. 常见路径
    home_dir = os.path.expanduser("~")
    common_paths = [
        os.path.join(home_dir, "miniforge3", "envs", "diffdock", "bin", "python"),
        os.path.join(home_dir, "anaconda3", "envs", "diffdock", "bin", "python"),
        os.path.join(home_dir, "miniconda3", "envs", "diffdock", "bin", "python"),
        os.path.join(home_dir, "conda", "envs", "diffdock", "bin", "python"),
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    # 5. 回退到当前 Python
    logger.warning("DiffDock Python not found, using current Python (may not work)")
    return sys.executable


def _get_failed_log_path() -> str:
    """获取失败日志路径"""
    if _config.failed_log_path:
        return _config.failed_log_path
    
    # 默认路径：项目根目录/logs/docking_failed.txt
    project_root = _get_project_root()
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, "docking_failed.txt")


# 延迟初始化路径（避免导入时就报错）
DIFFDOCK_PATH = None
DIFFDOCK_PYTHON = None


def _ensure_paths_initialized():
    """确保路径已初始化"""
    global DIFFDOCK_PATH, DIFFDOCK_PYTHON
    if DIFFDOCK_PATH is None:
        DIFFDOCK_PATH = _find_diffdock_path()
    if DIFFDOCK_PYTHON is None:
        DIFFDOCK_PYTHON = _find_diffdock_python()


def clean_altlocs(infile, outfile):
    metals = {"MG", "MN", "FE", "ZN", "CA", "CU", "CO", "NI", "NA", "K", "CL"}  # 常见金属离子
    with open(infile) as fin, open(outfile, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")):
                resname = line[17:20].strip()
                atomname = line[12:16].strip()
                if atomname in {"K", "SE", "ZN"} or resname in metals:
                    continue  # 删除金属离子、杂原子
                altloc = line[16]
                if altloc in (" ", "A"):
                    fout.write(line[:16] + " " + line[17:])
            else:
                fout.write(line)


def smiles_to_3d(smiles, outfile):
    """从SMILES生成3D结构（PDB格式）"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无法解析 SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
    AllChem.UFFOptimizeMolecule(mol)
    Chem.MolToPDBFile(mol, outfile)

def smiles_to_sdf(smiles, outfile):
    """从SMILES生成3D结构（SDF格式，用于DiffDock）"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无法解析 SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xf00d
        params.useSmallRingTorsions = True
        params.useBasicKnowledge = True
        AllChem.EmbedMolecule(mol, params)
        AllChem.UFFOptimizeMolecule(mol)
    except Exception:
        # Fallback to standard embedding if ETKDG fails
        AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
        AllChem.UFFOptimizeMolecule(mol)
    writer = Chem.SDWriter(outfile)
    writer.write(mol)
    writer.close()

def sdf_to_pdb(sdf_file, pdb_file):
    """将SDF文件转换为PDB格式"""
    try:
        suppl = Chem.SDMolSupplier(sdf_file)
        mol = suppl[0]  # 取第一个分子
        if mol is None:
            raise ValueError(f"无法从SDF文件读取分子: {sdf_file}")
        Chem.MolToPDBFile(mol, pdb_file)
        print(f"✅ SDF转PDB成功: {sdf_file} -> {pdb_file}")
        return True
    except Exception as e:
        print(f"❌ SDF转PDB失败: {e}")
        return False

# 移除PDBQT相关函数，DiffDock不需要PDBQT格式

def find_binding_site_center(protein_pdb, ligand_pdb=None):
    """识别结合位点中心"""
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
                print(f"✅ 基于配体计算结合位点中心: {center}")
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
            print(f"✅ 基于蛋白质计算结合位点中心: {center}")
            return center
        
        # 默认中心
        default_center = [0.0, 0.0, 0.0]
        print(f"⚠️ 使用默认结合位点中心: {default_center}")
        return default_center
        
    except Exception as e:
        print(f"❌ 结合位点识别失败: {e}")
        return [0.0, 0.0, 0.0]

def extract_pocket_region(protein_pdb, center, radius=10.0, output_pdb=None):
    """提取口袋区域"""
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
        
        print(f"✅ 口袋区域已保存: {output_pdb}")
        return output_pdb
        
    except Exception as e:
        print(f"❌ 口袋提取失败: {e}")
        return None


# 移除move_ligand_to_box_center函数，DiffDock会自动处理配体位置


def extract_box_from_pdb(pocket_pdb, margin=2.0):
    coords = []
    with open(pocket_pdb) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except:
                    continue
    if len(coords) == 0:
        raise ValueError(f"❌ 未提取到任何坐标点: {pocket_pdb}")
    coords = np.array(coords)
    min_coord = coords.min(axis=0)
    max_coord = coords.max(axis=0)
    center = ((max_coord + min_coord) / 2).tolist()
    size = (max_coord - min_coord + margin).tolist()
    return center, size


def create_fallback_pocket(protein_pdb_path, output_pocket_path):
    """
    当AutoSite失败时，创建基于蛋白质几何中心的简单口袋
    """
    import numpy as np
    
    # 读取蛋白质坐标
    coords = []
    with open(protein_pdb_path, 'r') as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except:
                    continue
    
    if len(coords) == 0:
        raise ValueError("❌ 未找到蛋白质原子坐标")
    
    coords = np.array(coords)
    center = coords.mean(axis=0).tolist()
    size = [20.0, 20.0, 20.0]  # 固定大小的盒子
    
    print(f"🔧 Fallback口袋中心: {center}")
    print(f"🔧 Fallback口袋大小: {size}")
    
    # 创建一个简单的口袋PDB文件（包含蛋白质中心附近的原子）
    radius = 10.0  # 10Å半径
    with open(output_pocket_path, 'w') as out_f:
        with open(protein_pdb_path, 'r') as in_f:
            for line in in_f:
                if line.startswith(("ATOM", "HETATM")):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        
                        # 计算到中心的距离
                        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
                        if dist <= radius:
                            out_f.write(line)
                    except:
                        continue
    
    print(f"✅ Fallback口袋文件创建: {output_pocket_path}")
    return center, size


# 移除convert_pdbqt_to_pdb函数，DiffDock输出SDF格式，使用sdf_to_pdb转换

def extract_pocket_pymol(protein_path, ligand_path, output_path, cutoff=5.0):
    """使用Bio.PDB提取口袋区域（包含配体原子）"""
    try:
        from Bio.PDB import PDBParser, PDBIO, Select
        import numpy as np
        
        print(f"🔍 加载蛋白质: {protein_path}")
        print(f"🔍 加载配体: {ligand_path}")
        
        parser = PDBParser(QUIET=True)
        
        # 读取蛋白质和配体结构
        protein_structure = parser.get_structure('protein', protein_path)
        ligand_structure = parser.get_structure('ligand', ligand_path)
        
        # 获取配体原子坐标
        ligand_atoms = []
        for model in ligand_structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        ligand_atoms.append(atom.get_coord())
        
        if not ligand_atoms:
            raise ValueError("配体文件中没有找到原子")
        
        ligand_coords = np.array(ligand_atoms)
        print(f"🔍 配体包含 {len(ligand_atoms)} 个原子")
        
        class PocketSelect(Select):
            def accept_residue(self, residue):
                # 检查残基中是否有原子在cutoff范围内
                for atom in residue:
                    atom_coord = atom.get_coord()
                    distances = np.linalg.norm(ligand_coords - atom_coord, axis=1)
                    if np.min(distances) <= cutoff:
                        return True
                return False
        
        # 使用绝对路径，避免相对路径问题
        abs_output_path = os.path.abspath(output_path)
        print(f"🔍 保存口袋到: {abs_output_path}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)
        
        # 保存口袋区域（只包含蛋白质残基）
        io = PDBIO()
        io.set_structure(protein_structure)
        io.save(abs_output_path, PocketSelect())
        
        # 手动添加配体原子到口袋文件中
        print(f"🔍 添加配体原子到口袋文件...")
        with open(abs_output_path, 'a') as f:
            # 为配体原子添加HETATM记录
            atom_count = 0
            for model in ligand_structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            atom_count += 1
                            coord = atom.get_coord()
                            # 格式化配体原子为PDB格式
                            hetatm_line = f"HETATM{atom_count:5d} {atom.get_name():4s} {residue.get_resname():3s} L{residue.get_id()[1]:4d}    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00 20.00           {atom.element:2s}\n"
                            f.write(hetatm_line)
        
        print(f"🔍 添加了 {atom_count} 个配体原子")
        
        # 强制刷新文件系统缓存
        import time
        time.sleep(0.1)  # 短暂等待确保文件写入完成
        
        # 检查文件是否真的保存了
        if os.path.exists(abs_output_path):
            file_size = os.path.getsize(abs_output_path)
            if file_size > 0:
                print(f"✅ Bio.PDB extracted pocket with ligand saved to {abs_output_path} (size: {file_size} bytes)")
            else:
                raise ValueError(f"口袋文件为空: {abs_output_path}")
        else:
            print(f"❌ 口袋文件未保存: {abs_output_path}")
            raise FileNotFoundError(f"口袋文件未保存: {abs_output_path}")
        
    except Exception as e:
        print(f"❌ 口袋提取失败: {e}")
        # 输出更详细的调试信息
        print(f"   蛋白质文件: {protein_path} (存在: {os.path.exists(protein_path)})")
        print(f"   配体文件: {ligand_path} (存在: {os.path.exists(ligand_path)})")
        print(f"   输出路径: {abs_output_path}")
        print(f"   输出目录: {os.path.dirname(abs_output_path)} (存在: {os.path.exists(os.path.dirname(abs_output_path))})")
        raise


def run_diffdock(protein_pdb: str, 
                 ligand_sdf: str, 
                 output_dir: str, 
                 inference_steps: Optional[int] = None, 
                 samples_per_complex: Optional[int] = None, 
                 gpu_id: Optional[int] = None,
                 validate: Optional[bool] = None,
                 original_smiles: Optional[str] = None) -> Optional[str]:
    """
    使用 DiffDock 进行分子对接
    
    Args:
        protein_pdb: 蛋白质 PDB 文件路径
        ligand_sdf: 配体 SDF 文件路径
        output_dir: 输出目录
        inference_steps: 推理步数（默认从配置读取）
        samples_per_complex: 每个复合物生成的 pose 数量
        gpu_id: GPU 设备 ID（默认从配置读取）
        validate: 是否验证 pose（默认从配置读取）
        original_smiles: 原始 SMILES（用于立体化学验证）
    
    Returns:
        output_sdf: 最佳 pose 的 SDF 文件路径，失败返回 None
    """
    # 确保路径已初始化
    _ensure_paths_initialized()
    
    # 使用配置默认值
    if inference_steps is None:
        inference_steps = _config.inference_steps
    if samples_per_complex is None:
        samples_per_complex = _config.samples_per_complex
    if gpu_id is None:
        gpu_id = _config.gpu_id
    if validate is None:
        validate = _config.validate_pose
    
    try:
        # 设置 GPU
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logger.info(f"🔧 Using GPU {gpu_id}")
        
        # DiffDock 路径（已在 _ensure_paths_initialized 中验证）
        diffdock_path = os.path.abspath(DIFFDOCK_PATH)
        python_exec = DIFFDOCK_PYTHON
        
        logger.info(f"🔬 DiffDock path: {diffdock_path}")
        logger.info(f"🔬 Python: {python_exec}")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 方法1: 使用CSV文件方式（DiffDock推荐方式）
        # 创建临时CSV文件
        complex_name = os.path.splitext(os.path.basename(protein_pdb))[0]
        csv_file = os.path.join(output_dir, "diffdock_input.csv")
        with open(csv_file, 'w') as f:
            f.write("complex_name,protein_path,ligand_description,protein_sequence\n")
            f.write(f"{complex_name},{os.path.abspath(protein_pdb)},{os.path.abspath(ligand_sdf)},\n")
        
        # 查找 DiffDock inference 脚本
        possible_scripts = [
            os.path.join(diffdock_path, "inference.py"),
            os.path.join(diffdock_path, "scripts", "inference.py"),
            os.path.join(diffdock_path, "diffdock", "inference.py"),
        ]
        
        cmd = None
        for script_path in possible_scripts:
            if os.path.exists(script_path):
                # 使用inference.py脚本
                cmd = [
                    python_exec, script_path,
                    "--protein_ligand_csv", os.path.abspath(csv_file),
                    "--out_dir", os.path.abspath(output_dir),
                    "--inference_steps", str(inference_steps),
                    "--samples_per_complex", str(samples_per_complex)
                ]
                break
        
        # 如果找不到脚本，尝试使用命令行模块方式
        if cmd is None:
            # 尝试使用命令行模块
            cmd = [
                python_exec, "-m", "diffdock.dock",
                "--protein_ligand_csv", os.path.abspath(csv_file),
                "--out_dir", os.path.abspath(output_dir),
                "--inference_steps", str(inference_steps),
                "--samples_per_complex", str(samples_per_complex)
            ]
        
        logger.info(f"🔬 Running DiffDock...")
        logger.info(f"   Protein: {protein_pdb}")
        logger.info(f"   Ligand: {ligand_sdf}")
        logger.info(f"   Output: {output_dir}")
        logger.info(f"   Command: {' '.join(cmd)}")
        logger.info(f"   ⚠️ First run may take minutes to download models...")
        
        # 运行 DiffDock
        process = subprocess.Popen(
            cmd,
            cwd=diffdock_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 实时输出
        logger.info(f"   DiffDock output:")
        logger.info(f"   {'='*60}")
        stdout_lines = []
        try:
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    logger.info(f"   {line}")
                    stdout_lines.append(line)
            process.wait()
        except KeyboardInterrupt:
            process.kill()
            raise
        
        if process.returncode != 0:
            error_msg = '\n'.join(stdout_lines[-50:])
            raise subprocess.CalledProcessError(process.returncode, cmd, output=error_msg)
        
        logger.info(f"   {'='*60}")
        logger.info(f"✅ DiffDock completed")
        
        # 查找最佳 pose（DiffDock 输出通常在 out_dir/complex_name）
        search_dir = os.path.join(output_dir, complex_name)
        if os.path.isdir(search_dir):
            best_pose = _find_best_pose(search_dir)
        else:
            best_pose = _find_best_pose(output_dir)
        
        if best_pose is None:
            raise FileNotFoundError(
                f"❌ DiffDock did not generate output files\n"
                f"   Output dir: {output_dir}\n"
                f"   Contents: {os.listdir(output_dir) if os.path.exists(output_dir) else 'not exists'}\n"
                f"   Searched: {search_dir if os.path.isdir(search_dir) else output_dir}"
            )
        
        # 可选：验证 pose
        if validate:
            try:
                from pose_validation import validate_pose_quick
                if not validate_pose_quick(best_pose):
                    logger.warning(f"⚠️ Pose validation failed for {best_pose}")
                    # 不直接失败，只是警告
            except ImportError:
                logger.warning("pose_validation module not available, skipping validation")
        
        logger.info(f"✅ Best pose: {best_pose}")
        return best_pose
        
    except subprocess.TimeoutExpired:
        logger.error(f"❌ DiffDock timeout")
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ DiffDock failed: {e}")
        if e.stdout:
            logger.error(f"   stdout: {e.stdout[:1000]}")
        return None
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        return None
    except Exception as e:
        logger.error(f"❌ DiffDock docking failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _find_best_pose(output_dir: str) -> Optional[str]:
    """
    从 DiffDock 输出目录中查找最佳 pose
    
    查找优先级：
    1. confidence 最高的文件（如果存在 confidence 输出）
    2. rank1_*.sdf
    3. 任何 .sdf 文件
    """
    if not os.path.exists(output_dir):
        return None
    
    import re
    sdf_files = []
    rank1_path = None
    best_conf = None
    best_conf_path = None
    for f in os.listdir(output_dir):
        if f.endswith('.sdf'):
            file_path = os.path.join(output_dir, f)
            m = re.search(r"confidence(-?\d+(?:\.\d+)?)", f)
            if m:
                try:
                    conf = float(m.group(1))
                    if best_conf is None or conf > best_conf:
                        best_conf = conf
                        best_conf_path = file_path
                except ValueError:
                    pass
            if 'rank1' in f.lower() or 'rank_1' in f.lower():
                rank1_path = file_path
            sdf_files.append((f, file_path))
    
    if best_conf_path:
        return best_conf_path
    if rank1_path:
        return rank1_path

    # 按文件名排序（通常 confidence 较高的排前面）
    sdf_files.sort(key=lambda x: x[0])
    
    if sdf_files:
        return sdf_files[0][1]
    
    return None


def run_preprocess(uniprot_id: str, 
                   smiles: str, 
                   prot_pdb_path: str, 
                   output_pocket_path: str, 
                   index: int,
                   gpu_id: Optional[int] = None,
                   validate_pose: bool = True,
                   persist_dir: Optional[str] = None,
                   keep_tmp_dir: bool = False) -> bool:
    """
    使用 DiffDock 进行分子对接和口袋提取
    
    Args:
        uniprot_id: UniProt ID
        smiles: 底物 SMILES 字符串
        prot_pdb_path: 蛋白质 PDB 文件路径
        output_pocket_path: 输出口袋文件路径
        index: 样本索引
        gpu_id: GPU ID（默认使用配置）
        validate_pose: 是否验证 pose
    
    Returns:
        bool: 处理是否成功
    """
    # 使用临时目录避免污染工作目录
    base_name = f"{uniprot_id}_{int(hashlib.sha256(smiles.encode()).hexdigest(), 16) & 0xffff}"
    tmp_dir = tempfile.mkdtemp(prefix=f"docking_{base_name}_")
    orig_dir = os.getcwd()
    
    # 确保路径是绝对路径
    prot_pdb_path = os.path.abspath(prot_pdb_path)
    output_pocket_path = os.path.abspath(output_pocket_path)
    if persist_dir:
        persist_dir = os.path.abspath(persist_dir)
    
    os.chdir(tmp_dir)

    try:
        # 验证 SMILES
        if not _validate_smiles(smiles):
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # 清理蛋白质 PDB
        clean_pdb_path = os.path.join(tmp_dir, "protein_clean.pdb")
        clean_altlocs(prot_pdb_path, clean_pdb_path)
        
        # 生成配体 SDF
        ligand_sdf = os.path.join(tmp_dir, "ligand.sdf")
        smiles_to_sdf(smiles, ligand_sdf)
        logger.info(f"✅ Ligand SDF: {ligand_sdf}")
        
        # 运行 DiffDock
        diffdock_output_dir = os.path.join(tmp_dir, "diffdock_output")
        os.makedirs(diffdock_output_dir, exist_ok=True)
        
        docked_ligand_sdf = run_diffdock(
            protein_pdb=clean_pdb_path,
            ligand_sdf=ligand_sdf,
            output_dir=diffdock_output_dir,
            gpu_id=gpu_id,
            validate=validate_pose,
            original_smiles=smiles
        )
        
        if docked_ligand_sdf is None:
            raise RuntimeError("DiffDock failed to generate pose")
        
        # 如果需要保留中间结果，复制 DiffDock 输出
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            try:
                import re
                complex_dir = os.path.join(
                    diffdock_output_dir,
                    os.path.splitext(os.path.basename(clean_pdb_path))[0]
                )
                copy_src = complex_dir if os.path.isdir(complex_dir) else diffdock_output_dir
                sdf_files = []
                confidence_map = {}
                for fname in os.listdir(copy_src):
                    if fname.endswith(".sdf"):
                        sdf_files.append(fname)
                        m = re.search(r"confidence(-?\d+(?:\.\d+)?)", fname)
                        if m:
                            try:
                                confidence_map[fname] = float(m.group(1))
                            except ValueError:
                                pass
                        shutil.copyfile(os.path.join(copy_src, fname), os.path.join(persist_dir, fname))
                sdf_files.sort()
                confidences = [confidence_map[f] for f in sdf_files if f in confidence_map]
                best_confidence = confidence_map.get(os.path.basename(docked_ligand_sdf))
                meta = {
                    "best_pose": os.path.basename(docked_ligand_sdf),
                    "sdf_files": sdf_files,
                    "confidence_values": confidences,
                    "confidence_map": confidence_map,
                    "best_confidence": best_confidence,
                }
                with open(os.path.join(persist_dir, "docking_meta.json"), "w") as f:
                    json.dump(meta, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to persist DiffDock outputs: {e}")

        # SDF -> PDB
        pose_pdb = os.path.join(tmp_dir, "pose.pdb")
        if not sdf_to_pdb(docked_ligand_sdf, pose_pdb):
            raise RuntimeError("Failed to convert SDF to PDB")
        
        # 提取口袋
        os.makedirs(os.path.dirname(output_pocket_path), exist_ok=True)
        extract_pocket_pymol(clean_pdb_path, pose_pdb, output_pocket_path, 
                            cutoff=_config.pocket_cutoff)
        
        # 验证输出
        if os.path.exists(output_pocket_path):
            file_size = os.path.getsize(output_pocket_path)
            logger.info(f"✅ {base_name} completed -> {output_pocket_path} ({file_size} bytes)")
            return True
        else:
            raise FileNotFoundError(f"Pocket file not saved: {output_pocket_path}")
            
    except Exception as e:
        logger.error(f"❌ Failed {uniprot_id}: {e}")
        import traceback
        traceback.print_exc()
        
        # 记录失败
        failed_log = _get_failed_log_path()
        with open(failed_log, "a") as f:
            f.write(f"{uniprot_id},{index},{str(e)[:100]}\n")
        return False
        
    finally:
        os.chdir(orig_dir)
        # 保留或清理临时目录
        if keep_tmp_dir:
            if persist_dir:
                try:
                    keep_root = os.path.join(persist_dir, "tmp_keep")
                    os.makedirs(keep_root, exist_ok=True)
                    dest_dir = os.path.join(keep_root, os.path.basename(tmp_dir))
                    if not os.path.exists(dest_dir):
                        shutil.move(tmp_dir, dest_dir)
                        logger.info(f"📦 Kept temp dir: {dest_dir}")
                except Exception as keep_e:
                    logger.warning(f"⚠️ Failed to keep temp dir {tmp_dir}: {keep_e}")
        else:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception as cleanup_e:
                logger.warning(f"⚠️ Failed to cleanup {tmp_dir}: {cleanup_e}")


def _validate_smiles(smiles: str) -> bool:
    """
    验证 SMILES 是否有效
    
    Returns:
        bool: 是否有效
    """
    # 已知有问题的 SMILES
    problematic_smiles = {
        "C(=O)=O",  # CO2，RDKit 无法处理
        "",
        ".",
    }
    
    if smiles in problematic_smiles:
        logger.warning(f"⚠️ Problematic SMILES skipped: {smiles}")
        return False
    
    # 尝试解析
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        return True
    except:
        return False



# ============================================================================
# CLI
# ============================================================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Molecular Docking Module")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # dock 子命令
    dock_parser = subparsers.add_parser('dock', help='Run docking')
    dock_parser.add_argument('--protein', type=str, required=True, help='Protein PDB file')
    dock_parser.add_argument('--ligand', type=str, required=True, help='Ligand SDF file')
    dock_parser.add_argument('--output', type=str, default='output', help='Output directory')
    dock_parser.add_argument('--gpu', type=int, default=None, help='GPU ID')
    dock_parser.add_argument('--no-validate', action='store_true', help='Skip pose validation')
    
    # batch 子命令
    batch_parser = subparsers.add_parser('batch', help='Batch docking from CSV')
    batch_parser.add_argument('--csv', type=str, required=True, help='Input CSV file')
    batch_parser.add_argument('--pdb-dir', type=str, required=True, help='Directory containing PDB files')
    batch_parser.add_argument('--output-dir', type=str, default='pockets', help='Output directory')
    batch_parser.add_argument('--gpu', type=int, default=None, help='GPU ID')
    
    # check 子命令
    check_parser = subparsers.add_parser('check', help='Check DiffDock installation')
    
    args = parser.parse_args()
    
    if args.command == 'check':
        print("Checking DiffDock installation...")
        try:
            _ensure_paths_initialized()
            print(f"✅ DiffDock path: {DIFFDOCK_PATH}")
            print(f"✅ Python: {DIFFDOCK_PYTHON}")
        except Exception as e:
            print(f"❌ Error: {e}")
            
    elif args.command == 'dock':
        if args.gpu is not None:
            _config.gpu_id = args.gpu
        _config.validate_pose = not args.no_validate
        
        result = run_diffdock(
            protein_pdb=args.protein,
            ligand_sdf=args.ligand,
            output_dir=args.output
        )
        if result:
            print(f"✅ Docking completed: {result}")
        else:
            print("❌ Docking failed")
            
    elif args.command == 'batch':
        if args.gpu is not None:
            _config.gpu_id = args.gpu
            
        df = pd.read_csv(args.csv)
        os.makedirs(args.output_dir, exist_ok=True)
        
        success_count = 0
        for idx, row in enumerate(df.itertuples()):
            uniprot = getattr(row, 'uniprot', None)
            smiles = getattr(row, 'substrate_smiles', None)
            
            if not uniprot or not smiles:
                logger.warning(f"Skipping row {idx}: missing uniprot or smiles")
                continue
            
            smiles = smiles.split(';')[0]  # 取第一个 SMILES
            
            pdb_path = os.path.join(args.pdb_dir, f"{uniprot}.pdb")
            if not os.path.exists(pdb_path):
                logger.warning(f"⚠️ Missing PDB: {pdb_path}")
                continue
            
            # 生成输出路径
            pocket_hash = int(hashlib.sha256(smiles.encode()).hexdigest(), 16) & 0xffff
            output_path = os.path.join(args.output_dir, f"{uniprot}_{pocket_hash}_pocket.pdb")
            
            if run_preprocess(uniprot, smiles, pdb_path, output_path, idx):
                success_count += 1
            
            print(f"Progress: {idx+1}/{len(df)}")
        
        print(f"✅ Completed: {success_count}/{len(df)} successful")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
