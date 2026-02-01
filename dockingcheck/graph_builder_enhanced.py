"""
增强版图构建模块：几何截断 + 同残基原子聚合

改进点：
1. 保持原有的几何截断（如 10Å）
2. 对于边界原子，把它所在的整个残基的其他原子也包含进来
3. 避免切断氨基酸的化学完整性

使用方法：
    from graph_builder_enhanced import parse_pocket_enhanced, build_graph_enhanced
"""

import numpy as np
import torch
from torch_geometric.data import Data
from Bio.PDB import PDBParser
from sklearn.preprocessing import OneHotEncoder
import logging
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==== 复用原有的编码器和特征表 ====
element_list = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H']
residue_list = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
                'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','LIG']

element_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
element_encoder.fit(np.array([[e] for e in element_list]))
residue_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
residue_encoder.fit(np.array([[r] for r in residue_list]))

atomic_property_table = {
    'H':  {'Z': 1, 'mass': 1.008,  'electronegativity': 2.20, 'radius': 0.31},
    'C':  {'Z': 6, 'mass': 12.011, 'electronegativity': 2.55, 'radius': 0.76},
    'N':  {'Z': 7, 'mass': 14.007, 'electronegativity': 3.04, 'radius': 0.71},
    'O':  {'Z': 8, 'mass': 15.999, 'electronegativity': 3.44, 'radius': 0.66},
    'S':  {'Z':16, 'mass': 32.06,  'electronegativity': 2.58, 'radius': 1.05},
    'P':  {'Z':15, 'mass': 30.974, 'electronegativity': 2.19, 'radius': 1.07},
    'F':  {'Z': 9, 'mass': 18.998, 'electronegativity': 3.98, 'radius': 0.57},
    'Cl': {'Z':17, 'mass': 35.45,  'electronegativity': 3.16, 'radius': 1.02},
    'Br': {'Z':35, 'mass': 79.904, 'electronegativity': 2.96, 'radius': 1.20},
    'I':  {'Z':53, 'mass': 126.90, 'electronegativity': 2.66, 'radius': 1.39},
}


def get_elec_feature(max_atomic_number=20):
    """电子构型特征（从原始代码复用）"""
    return torch.tensor([
        [0]*16,
        [0,1]+[0]*14,
        [2,0]+[0]*14,
        [2,0,0,1]+[0]*12,
        [2,0,2,0]+[0]*12,
        [2,1,2,0]+[0]*12,
        [2,2,2,0]+[0]*12,
        [2,3,2,0]+[0]*12,
        [2,4,2,0]+[0]*12,
        [2,5,2,0]+[0]*12,
        [2,6,2,0]+[0]*12,
    ], dtype=torch.float)

atomic_electronic_features = get_elec_feature()


def parse_pocket_enhanced(pdb_path: str, ligand_coords: np.ndarray = None,
                         cutoff: float = 10.0, expand_residues: bool = True) -> List[Dict]:
    """
    增强版 Pocket 解析：几何截断 + 同残基原子聚合

    Args:
        pdb_path: PDB 文件路径
        ligand_coords: 配体坐标 [N_lig, 3]，用于计算截断中心
        cutoff: 几何截断半径（Å）
        expand_residues: 是否扩展到完整残基

    Returns:
        原子字典列表
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pocket', pdb_path)

    # Step 1: 提取所有原子信息
    all_atoms = []
    for atom in structure.get_atoms():
        if atom.element == 'H':  # 跳过氢原子
            continue

        residue = atom.get_parent()
        res_name = residue.get_resname()
        chain_id = residue.get_full_id()[2]
        res_id = residue.get_id()[1]  # 残基编号

        is_ligand = 1 if (res_name == 'UNL' or chain_id == ' ') else 0

        all_atoms.append({
            'coord': atom.coord,
            'element': atom.element,
            'residue': res_name if res_name in residue_list else 'LIG',
            'is_ligand': is_ligand,
            'residue_id': (chain_id, res_id),  # 唯一标识残基
            'atom_name': atom.name
        })

    if len(all_atoms) == 0:
        logger.warning(f"No atoms found in {pdb_path}")
        return []

    # Step 2: 几何截断
    if ligand_coords is None:
        # 如果没有提供配体坐标，使用所有配体原子的质心
        ligand_atoms = [a for a in all_atoms if a['is_ligand'] == 1]
        if len(ligand_atoms) > 0:
            ligand_coords = np.array([a['coord'] for a in ligand_atoms])
        else:
            # 如果没有配体，使用所有原子的质心
            logger.warning("No ligand found, using center of all atoms")
            ligand_coords = np.array([a['coord'] for a in all_atoms])

    # 计算截断中心（配体质心）
    center = ligand_coords.mean(axis=0)

    # 找出在截断半径内的原子
    coords = np.array([a['coord'] for a in all_atoms])
    distances = np.linalg.norm(coords - center, axis=1)
    within_cutoff = distances <= cutoff

    # Step 3: 扩展到完整残基
    if expand_residues:
        # 获取所有在截断范围内的残基ID
        pocket_residues = set()
        for i, atom in enumerate(all_atoms):
            if within_cutoff[i]:
                pocket_residues.add(atom['residue_id'])

        logger.info(f"Found {len(pocket_residues)} residues within {cutoff}Å cutoff")

        # 扩展：包含这些残基的所有原子
        selected_atoms = []
        for atom in all_atoms:
            if atom['residue_id'] in pocket_residues:
                selected_atoms.append(atom)

        logger.info(f"Expanded from {within_cutoff.sum()} atoms to {len(selected_atoms)} atoms")

    else:
        # 不扩展，直接使用几何截断的原子
        selected_atoms = [all_atoms[i] for i in range(len(all_atoms)) if within_cutoff[i]]

    return selected_atoms


def gaussian_rbf(edge_dist: torch.Tensor, num_centers: int = 16,
                D_min: float = 0.0, D_max: float = 8.0, gamma: float = 20.0) -> torch.Tensor:
    """Gaussian RBF 边特征（从原始代码复用）"""
    centers = torch.linspace(D_min, D_max, num_centers, device=edge_dist.device)
    diff = edge_dist - centers.view(1, -1)
    return torch.exp(-gamma * diff**2)


def build_edges_manual(pos: torch.Tensor, cutoff: float) -> torch.Tensor:
    """手动构建边（从原始代码复用）"""
    n_atoms = pos.shape[0]
    edges = []

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = torch.norm(pos[i] - pos[j])
            if dist <= cutoff:
                edges.append([i, j])
                edges.append([j, i])

    if len(edges) == 0:
        edges = [[i, i] for i in range(n_atoms)]

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_graph_enhanced(atoms: List[Dict], temperature: float, dist_cutoff: float = 4.0) -> Data:
    """
    增强版图构建函数

    Args:
        atoms: 原子字典列表（来自 parse_pocket_enhanced）
        temperature: 温度（K）
        dist_cutoff: 边构建的距离截断

    Returns:
        PyTorch Geometric Data 对象
    """
    if len(atoms) == 0:
        raise ValueError("Cannot build graph from empty atom list")

    # 提取特征
    coords = np.array([a['coord'] for a in atoms])
    elements = np.array([[a['element']] for a in atoms])
    residues = np.array([[a['residue']] for a in atoms])
    is_lig = np.array([[a['is_ligand']] for a in atoms])

    # One-hot 编码
    el_feat = element_encoder.transform(elements)
    res_feat = residue_encoder.transform(residues)

    # 最近邻距离
    dmat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    np.fill_diagonal(dmat, np.inf)
    min_dists = np.min(dmat, axis=1, keepdims=True)

    # 电子构型特征
    nums = np.array([
        {'C':6,'N':7,'O':8,'S':16,'P':15,'F':9,'Cl':17,'Br':35,'I':53,'H':1}.get(e[0], 0)
        for e in elements
    ])
    elec = atomic_electronic_features[np.clip(nums, 0, 10)]

    # 原子属性
    props = np.array([
        [
            atomic_property_table.get(e[0], {'mass':0,'electronegativity':0,'radius':0})['mass'],
            atomic_property_table.get(e[0], {'mass':0,'electronegativity':0,'radius':0})['electronegativity'],
            atomic_property_table.get(e[0], {'mass':0,'electronegativity':0,'radius':0})['radius'],
        ]
        for e in elements
    ])

    # 拼接节点特征
    x = np.hstack([el_feat, res_feat, is_lig, min_dists, elec, props])
    if np.isnan(x).any():
        raise ValueError("Node features contain NaN")

    pos = torch.tensor(coords, dtype=torch.float)

    # 构建边
    try:
        from torch_cluster import radius_graph
        edge_index = radius_graph(pos, r=dist_cutoff)
    except ImportError:
        logger.warning("torch_cluster not available, using manual edge construction")
        edge_index = build_edges_manual(pos, dist_cutoff)

    # 边特征（Gaussian RBF）
    row, col = edge_index
    dists = torch.norm(pos[row] - pos[col], dim=1, keepdim=True)
    edge_attr = gaussian_rbf(dists)

    if torch.isnan(edge_attr).any():
        raise ValueError("Edge features contain NaN")

    # 构建 Data 对象
    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    # 标准化温度
    temp_scaled = (temperature - 303.15) / 10.0
    data.temperature = torch.tensor([temp_scaled], dtype=torch.float)

    return data


# ==== 使用示例 ====
if __name__ == '__main__':
    # 测试增强版图构建
    pdb_path = 'data/processed/pockets/example_10A.pdb'

    # 方法1：自动检测配体
    atoms = parse_pocket_enhanced(pdb_path, cutoff=10.0, expand_residues=True)
    print(f"Extracted {len(atoms)} atoms with residue expansion")

    # 方法2：手动指定配体坐标
    # ligand_coords = np.array([[x1, y1, z1], [x2, y2, z2], ...])
    # atoms = parse_pocket_enhanced(pdb_path, ligand_coords=ligand_coords, cutoff=10.0)

    if len(atoms) > 0:
        data = build_graph_enhanced(atoms, temperature=298.15)
        print(f"Built graph: {data.num_nodes} nodes, {data.num_edges} edges")
