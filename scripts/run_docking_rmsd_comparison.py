#!/usr/bin/env python3
"""
DiffDock vs AutoDock Vina RMSD Comparison on Astex Diverse Set

用 astex_diverse_set 中的晶体结构（蛋白质+配体）分别用 Vina 和 DiffDock 对接，
计算 RMSD vs 晶体构象，生成比较图。

Usage:
    python scripts/run_docking_rmsd_comparison.py \
        --dataset data/posebench/data/astex_diverse_set \
        --output results/docking_rmsd_comparison \
        --vina_bin /home/lizihao/autodock_vina_1_1_2_linux_x86/bin/vina \
        --adfr_bin /home/lizihao/ADFRsuite_x86_64Linux_1.0/bin \
        --n_samples 96
"""

import os
import sys
import argparse
import subprocess
import shutil
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import tempfile
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  RMSD 计算（不需要 DiffDock env）
# ─────────────────────────────────────────────

def parse_coords_from_pdbqt(pdbqt_file: str) -> tuple:
    """从 Vina 输出 PDBQT 中解析第一个 MODEL 的重原子坐标和元素。"""
    coords, elements = [], []
    in_model = False
    with open(pdbqt_file) as f:
        for line in f:
            if line.startswith('MODEL'):
                in_model = True
                continue
            if line.startswith('ENDMDL'):
                break
            if line.startswith(('ATOM', 'HETATM')):
                elem = line[76:78].strip() if len(line) > 76 else line[12:16].strip().rstrip('0123456789')
                if not elem:
                    elem = line[12:14].strip()
                if elem in ('H', 'HD', 'HS'):
                    continue  # 跳过氢
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    elem_clean = ''.join(c for c in elem if c.isalpha())[:2].capitalize()
                    coords.append([x, y, z])
                    elements.append(elem_clean)
                except ValueError:
                    continue
    return np.array(coords) if coords else np.empty((0, 3)), elements


def compute_rmsd_coords(pred_coords: np.ndarray, ref_coords: np.ndarray,
                         pred_elems: list = None, ref_elems: list = None) -> float:
    """用匈牙利算法对原子排序，计算最佳 RMSD（支持对称分子）。"""
    from scipy.optimize import linear_sum_assignment
    n_pred = len(pred_coords)
    n_ref  = len(ref_coords)
    if n_pred == 0 or n_ref == 0:
        return np.nan

    # 如果原子数不匹配，取两者中少的那个
    n = min(n_pred, n_ref)
    if n_pred != n_ref:
        logger.debug(f"Atom count mismatch: pred={n_pred}, ref={n_ref}, using {n}")

    # 距离矩阵
    diff = pred_coords[:n, None, :] - ref_coords[None, :n, :]  # [n, n, 3]
    dist_mat = np.sqrt((diff ** 2).sum(axis=-1))              # [n, n]
    row_ind, col_ind = linear_sum_assignment(dist_mat)
    rmsd = np.sqrt(((pred_coords[row_ind] - ref_coords[col_ind]) ** 2).sum(axis=-1).mean())
    return float(rmsd)


def compute_rmsd_sdf_vs_pdb(pred_file: str, ref_sdf: str) -> float:
    """计算预测配体 pose 与晶体 pose 的 RMSD（重原子，坐标匹配法）。"""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        # 解析预测坐标
        if pred_file.endswith('.pdbqt'):
            pred_coords, pred_elems = parse_coords_from_pdbqt(pred_file)
        elif pred_file.endswith('.sdf'):
            mol = Chem.SDMolSupplier(pred_file, sanitize=True)[0]
            if mol is None:
                return np.nan
            mol = Chem.RemoveHs(mol)
            conf = mol.GetConformer()
            pred_coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
            pred_elems = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
        else:
            return np.nan

        # 解析参考坐标（晶体 SDF）
        mol_ref = Chem.SDMolSupplier(ref_sdf, sanitize=True)[0]
        if mol_ref is None:
            return np.nan
        mol_ref = Chem.RemoveHs(mol_ref)
        conf_ref = mol_ref.GetConformer()
        ref_coords = np.array([conf_ref.GetAtomPosition(i) for i in range(mol_ref.GetNumAtoms())])

        if len(pred_coords) == 0:
            return np.nan

        return compute_rmsd_coords(pred_coords, ref_coords)

    except Exception as e:
        logger.debug(f"RMSD calc error: {e}")
        return np.nan


def pdbqt_to_rdkit(pdbqt_file: str):
    """将 pdbqt 转为 RDKit mol（保留向后兼容）。"""
    try:
        from rdkit import Chem
        lines = []
        with open(pdbqt_file) as f:
            for line in f:
                if line.startswith('ENDMDL'):
                    break
                if line.startswith(('HETATM', 'ATOM')):
                    elem = line[76:78].strip() if len(line) > 76 else ''
                    if elem in ('H', 'HD', 'HS'):
                        continue
                    lines.append(line[:66] + '\n')
        pdb_block = ''.join(lines) + 'END\n'
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=True)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            except Exception:
                pass
        return mol
    except Exception as e:
        logger.debug(f"pdbqt_to_rdkit error: {e}")
        return None


# ─────────────────────────────────────────────
#  AutoDock Vina 对接流程
# ─────────────────────────────────────────────

def get_ligand_center(sdf_file: str):
    """从 SDF 获取配体质心作为对接 box 中心。"""
    from rdkit import Chem
    import numpy as np
    mol = Chem.SDMolSupplier(sdf_file, sanitize=True)[0]
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    pos = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    return pos.mean(axis=0)


def sdf_to_pdbqt_ligand(sdf_file: str, out_pdbqt: str, adfr_bin: str = None) -> bool:
    """将 SDF 转换为 PDBQT（使用 meeko + RDKit）。"""
    try:
        import warnings
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from meeko import MoleculePreparation
        mol = Chem.SDMolSupplier(sdf_file, sanitize=True)[0]
        if mol is None:
            return False
        mol_h = AllChem.AddHs(mol, addCoords=True)
        if mol_h.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol_h)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            preparator = MoleculePreparation()
            preparator.prepare(mol_h)
            pdbqt_str = preparator.write_pdbqt_string()
        with open(out_pdbqt, 'w') as f:
            f.write(pdbqt_str)
        return len(pdbqt_str) > 100
    except Exception as e:
        logger.debug(f"sdf_to_pdbqt error: {e}")
        return False


def pdb_to_pdbqt_receptor(pdb_file: str, out_pdbqt: str, adfr_bin: str) -> bool:
    """用 ADFR prepare_receptor 将 PDB 转 PDBQT。"""
    prepare_receptor = os.path.join(adfr_bin, 'prepare_receptor')
    cmd = [prepare_receptor, '-r', pdb_file, '-o', out_pdbqt,
           '-A', 'checkhydrogens', '-U', 'nphs_lps_waters_deleteAltB']
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        return os.path.exists(out_pdbqt)
    except Exception as e:
        logger.debug(f"prepare_receptor error: {e}")
        return False


def run_vina(receptor_pdbqt: str, ligand_pdbqt: str, center: np.ndarray,
             out_pdbqt: str, vina_bin: str, box_size: float = 22.0,
             exhaustiveness: int = 8, num_modes: int = 9) -> bool:
    """运行 AutoDock Vina 对接。"""
    cmd = [
        vina_bin,
        '--receptor', receptor_pdbqt,
        '--ligand', ligand_pdbqt,
        '--out', out_pdbqt,
        '--center_x', f'{center[0]:.3f}',
        '--center_y', f'{center[1]:.3f}',
        '--center_z', f'{center[2]:.3f}',
        '--size_x', str(box_size),
        '--size_y', str(box_size),
        '--size_z', str(box_size),
        '--exhaustiveness', str(exhaustiveness),
        '--num_modes', str(num_modes),
        '--seed', '42',
        '--cpu', '2',
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        return os.path.exists(out_pdbqt)
    except Exception as e:
        logger.debug(f"Vina error: {e}")
        return False


def dock_vina_one_sample(sample_dir: Path, out_dir: Path, vina_bin: str, adfr_bin: str) -> dict:
    """对单个样本运行 Vina，返回 RMSD 结果。
    使用 _ligand_start_conf.sdf 作为输入（随机起始构象），_ligand.sdf 作为 RMSD 参考（晶体位置）。
    """
    name = sample_dir.name
    protein_pdb = sample_dir / f'{name}_protein.pdb'
    ref_sdf     = sample_dir / f'{name}_ligand.sdf'    # 晶体 SDF（用于 RMSD 计算）
    start_sdf   = sample_dir / f'{name}_ligand_start_conf.sdf'  # 起始构象（用于 Vina 输入）

    # 如果没有 start_conf，用晶体 SDF（但 RMSD 可能偏低）
    if not start_sdf.exists():
        start_sdf = ref_sdf

    if not protein_pdb.exists() or not ref_sdf.exists():
        ref_sdf_alt = sample_dir / f'{name}_ligands.sdf'
        if ref_sdf_alt.exists():
            ref_sdf = ref_sdf_alt
            if not start_sdf.exists():
                start_sdf = ref_sdf
        else:
            return {'sample': name, 'vina_rmsd': np.nan, 'error': 'missing_files'}

    sample_out = out_dir / name
    sample_out.mkdir(parents=True, exist_ok=True)

    receptor_pdbqt = sample_out / 'receptor.pdbqt'
    ligand_pdbqt   = sample_out / 'ligand.pdbqt'
    vina_out_pdbqt = sample_out / 'vina_out.pdbqt'

    # 获取 box 中心（从晶体位置，代表真实口袋中心）
    center = get_ligand_center(str(ref_sdf))
    if center is None:
        return {'sample': name, 'vina_rmsd': np.nan, 'error': 'center_failed'}

    # 准备受体
    if not receptor_pdbqt.exists():
        ok = pdb_to_pdbqt_receptor(str(protein_pdb), str(receptor_pdbqt), adfr_bin)
        if not ok:
            return {'sample': name, 'vina_rmsd': np.nan, 'error': 'receptor_prep_failed'}

    # 准备配体（用 start_conf 作为 Vina 输入）
    if not ligand_pdbqt.exists():
        ok = sdf_to_pdbqt_ligand(str(start_sdf), str(ligand_pdbqt), adfr_bin)
        if not ok:
            return {'sample': name, 'vina_rmsd': np.nan, 'error': 'ligand_prep_failed'}

    # 运行 Vina
    ok = run_vina(str(receptor_pdbqt), str(ligand_pdbqt), center,
                  str(vina_out_pdbqt), vina_bin)
    if not ok:
        return {'sample': name, 'vina_rmsd': np.nan, 'error': 'vina_failed'}

    # 计算 RMSD（top-1 pose vs 晶体）
    rmsd = compute_rmsd_sdf_vs_pdb(str(vina_out_pdbqt), str(ref_sdf))
    return {'sample': name, 'vina_rmsd': rmsd, 'error': None}


# ─────────────────────────────────────────────
#  DiffDock 对接流程
# ─────────────────────────────────────────────

def dock_diffdock_one_sample(sample_dir: Path, out_dir: Path,
                              diffdock_dir: str, diffdock_python: str) -> dict:
    """对单个样本运行 DiffDock，返回 RMSD 结果。"""
    name = sample_dir.name
    protein_pdb = sample_dir / f'{name}_protein.pdb'
    ligand_sdf  = sample_dir / f'{name}_ligand.sdf'
    if not ligand_sdf.exists():
        ligand_sdf = sample_dir / f'{name}_ligands.sdf'

    if not protein_pdb.exists() or not ligand_sdf.exists():
        return {'sample': name, 'diffdock_rmsd': np.nan, 'error': 'missing_files'}

    sample_out = out_dir / name
    sample_out.mkdir(parents=True, exist_ok=True)
    dd_out = sample_out / 'diffdock_out'

    inference_script = os.path.join(diffdock_dir, 'inference.py')
    cmd = [
        diffdock_python, inference_script,
        '--protein_path', str(protein_pdb),
        '--ligand', str(ligand_sdf),
        '--out_dir', str(dd_out),
        '--inference_steps', '20',
        '--samples_per_complex', '5',
        '--batch_size', '8',
        '--actual_steps', '19',
        '--no_final_step_noise',
    ]
    env = os.environ.copy()
    env['PYTHONPATH'] = diffdock_dir + ':' + env.get('PYTHONPATH', '')

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300,
                                cwd=diffdock_dir, env=env)
    except subprocess.TimeoutExpired:
        return {'sample': name, 'diffdock_rmsd': np.nan, 'error': 'timeout'}
    except Exception as e:
        return {'sample': name, 'diffdock_rmsd': np.nan, 'error': str(e)}

    # 找输出的 top-1 pose
    pose_files = sorted(dd_out.glob('rank1*.sdf')) if dd_out.exists() else []
    if not pose_files:
        pose_files = sorted(dd_out.glob('**/*.sdf')) if dd_out.exists() else []

    if not pose_files:
        return {'sample': name, 'diffdock_rmsd': np.nan, 'error': 'no_output_pose'}

    best_pose = pose_files[0]
    rmsd = compute_rmsd_sdf_vs_pdb(str(best_pose), str(ligand_sdf))
    return {'sample': name, 'diffdock_rmsd': rmsd, 'error': None}


# ─────────────────────────────────────────────
#  可视化
# ─────────────────────────────────────────────

def plot_rmsd_comparison(df: pd.DataFrame, out_dir: str):
    """生成 DiffDock vs Vina RMSD 对比图（CDF + bar + scatter）。"""
    os.makedirs(out_dir, exist_ok=True)

    has_diffdock = 'diffdock_rmsd' in df.columns and df['diffdock_rmsd'].notna().sum() > 5
    has_vina     = 'vina_rmsd' in df.columns and df['vina_rmsd'].notna().sum() > 5

    if not has_vina and not has_diffdock:
        logger.warning("No valid RMSD data to plot")
        return

    # 颜色设定
    COLOR_VINA = '#E07B54'
    COLOR_DD   = '#4C90C0'

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    # ─── 1. CDF ───
    ax = axes[0]
    rmsd_range = np.linspace(0, 10, 200)
    if has_vina:
        vals = df['vina_rmsd'].dropna().values
        cdf = [(vals < t).mean() for t in rmsd_range]
        ax.plot(rmsd_range, cdf, color=COLOR_VINA, lw=2, label='AutoDock Vina')
    if has_diffdock:
        vals = df['diffdock_rmsd'].dropna().values
        cdf = [(vals < t).mean() for t in rmsd_range]
        ax.plot(rmsd_range, cdf, color=COLOR_DD, lw=2.5, label='DiffDock (ours)')
    ax.axvline(2.0, color='gray', ls='--', alpha=0.6, label='2 Å threshold')
    ax.set_xlabel('RMSD (Å)', fontsize=11)
    ax.set_ylabel('Fraction of success', fontsize=11)
    ax.set_title('CDF of Top-1 RMSD', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 10); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # ─── 2. Success rates at various thresholds ───
    ax = axes[1]
    x = np.arange(len(thresholds))
    w = 0.35
    methods = []
    if has_vina:
        vina_rates = [(df['vina_rmsd'].dropna() < t).mean() * 100 for t in thresholds]
        ax.bar(x - w/2, vina_rates, w, color=COLOR_VINA, alpha=0.85, label='AutoDock Vina')
        methods.append(('Vina', vina_rates))
    if has_diffdock:
        dd_rates = [(df['diffdock_rmsd'].dropna() < t).mean() * 100 for t in thresholds]
        ax.bar(x + w/2, dd_rates, w, color=COLOR_DD, alpha=0.85, label='DiffDock (ours)')
        methods.append(('DiffDock', dd_rates))
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t}Å' for t in thresholds], rotation=30, fontsize=9)
    ax.set_ylabel('Success rate (%)', fontsize=11)
    ax.set_title('Success Rate by RMSD Threshold', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    # ─── 3. 箱线图 / 分布图 ───
    ax = axes[2]
    plot_data = []
    if has_vina:
        v = df['vina_rmsd'].dropna().clip(upper=15)
        for x_val in v:
            plot_data.append({'Method': 'AutoDock Vina', 'RMSD (Å)': x_val})
    if has_diffdock:
        v = df['diffdock_rmsd'].dropna().clip(upper=15)
        for x_val in v:
            plot_data.append({'Method': 'DiffDock (ours)', 'RMSD (Å)': x_val})
    plot_df = pd.DataFrame(plot_data)
    palette = {'AutoDock Vina': COLOR_VINA, 'DiffDock (ours)': COLOR_DD}
    sns.violinplot(data=plot_df, x='Method', y='RMSD (Å)', palette=palette,
                   inner='box', ax=ax, cut=0)
    ax.axhline(2.0, color='gray', ls='--', alpha=0.6, label='2 Å')
    ax.set_title('RMSD Distribution', fontsize=12)
    ax.set_ylim(0, 12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Molecular Docking Accuracy on Astex Diverse Set (n=96)', fontsize=13, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'docking_rmsd_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Figure saved: {out_path}")

    # 数字摘要
    print("\n" + "="*50)
    print("RMSD Comparison Summary (Astex Diverse Set)")
    print("="*50)
    for thresh in [1.0, 1.5, 2.0, 2.5]:
        line = f"  <{thresh:.1f}Å: "
        if has_vina:
            r = (df['vina_rmsd'].dropna() < thresh).mean() * 100
            line += f"Vina={r:.1f}%  "
        if has_diffdock:
            r = (df['diffdock_rmsd'].dropna() < thresh).mean() * 100
            line += f"DiffDock={r:.1f}%"
        print(line)
    if has_vina:
        print(f"\n  Vina median RMSD: {df['vina_rmsd'].median():.2f} Å")
    if has_diffdock:
        print(f"  DiffDock median RMSD: {df['diffdock_rmsd'].median():.2f} Å")
    print("="*50)

    return out_path


def print_summary_table(df: pd.DataFrame):
    """打印 LaTeX 格式的结果表。"""
    has_vina     = 'vina_rmsd' in df.columns and df['vina_rmsd'].notna().sum() > 5
    has_diffdock = 'diffdock_rmsd' in df.columns and df['diffdock_rmsd'].notna().sum() > 5
    methods = []
    if has_vina:     methods.append(('AutoDock Vina', 'vina_rmsd'))
    if has_diffdock: methods.append(('DiffDock (ours)', 'diffdock_rmsd'))

    print("\n--- LaTeX Table ---")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Molecular docking accuracy on Astex Diverse Set.}")
    print(r"\label{tab:docking_rmsd}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Method & Median RMSD (Å) & \%<1.5Å & \%<2.0Å & \%<2.5Å \\")
    print(r"\midrule")
    for name, col in methods:
        vals = df[col].dropna()
        med  = vals.median()
        p15  = (vals < 1.5).mean() * 100
        p20  = (vals < 2.0).mean() * 100
        p25  = (vals < 2.5).mean() * 100
        bold = r"\textbf{" if name.startswith('DiffDock') else ""
        bend = r"}" if bold else ""
        print(f"{bold}{name}{bend} & {bold}{med:.2f}{bend} & {bold}{p15:.1f}{bend} & {bold}{p20:.1f}{bend} & {bold}{p25:.1f}{bend} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ─────────────────────────────────────────────
#  主流程
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='data/posebench/data/astex_diverse_set')
    parser.add_argument('--output', type=str,
                        default='results/docking_rmsd_comparison')
    parser.add_argument('--vina_bin', type=str,
                        default='/home/lizihao/autodock_vina_1_1_2_linux_x86/bin/vina')
    parser.add_argument('--adfr_bin', type=str,
                        default='/home/lizihao/ADFRsuite_x86_64Linux_1.0/bin')
    parser.add_argument('--diffdock_dir', type=str,
                        default='DiffDock')
    parser.add_argument('--diffdock_python', type=str, default=None,
                        help='diffdock env python path')
    parser.add_argument('--n_samples', type=int, default=96,
                        help='max samples to process')
    parser.add_argument('--skip_vina', action='store_true')
    parser.add_argument('--skip_diffdock', action='store_true')
    parser.add_argument('--n_workers', type=int, default=4)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    out_dir     = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])[:args.n_samples]
    logger.info(f"Found {len(samples)} samples in {dataset_dir}")

    results_path = out_dir / 'rmsd_results.csv'

    # 加载已有结果（支持断点续跑）
    if results_path.exists():
        df_existing = pd.read_csv(results_path)
        done_samples = set(df_existing['sample'].tolist())
        logger.info(f"Resuming: {len(done_samples)} already done")
    else:
        df_existing = pd.DataFrame()
        done_samples = set()

    new_rows = []

    # ─── Vina ───
    if not args.skip_vina:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        vina_out = out_dir / 'vina_poses'
        vina_out.mkdir(exist_ok=True)
        todo_samples = [s for s in samples if s.name not in done_samples]
        logger.info(f"Running AutoDock Vina on {len(todo_samples)} samples (workers={args.n_workers})...")
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futures = {
                pool.submit(dock_vina_one_sample, s, vina_out, args.vina_bin, args.adfr_bin): s
                for s in todo_samples
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc='Vina'):
                try:
                    row = fut.result()
                except Exception as e:
                    row = {'sample': futures[fut].name, 'vina_rmsd': np.nan, 'error': str(e)}
                new_rows.append(row)
                # 写中间结果（防止意外中断）
                if len(new_rows) % 10 == 0:
                    df_mid = pd.concat([df_existing, pd.DataFrame(new_rows)], ignore_index=True) if not df_existing.empty else pd.DataFrame(new_rows)
                    df_mid.to_csv(results_path, index=False)

    # ─── DiffDock ───
    if not args.skip_diffdock and args.diffdock_python:
        dd_out = out_dir / 'diffdock_poses'
        dd_out.mkdir(exist_ok=True)
        logger.info(f"Running DiffDock on {len(samples)} samples...")
        for sample_dir in tqdm(samples, desc='DiffDock'):
            name = sample_dir.name
            row = dock_diffdock_one_sample(sample_dir, dd_out,
                                           args.diffdock_dir, args.diffdock_python)
            # 合并到 Vina 结果
            for r in new_rows:
                if r.get('sample') == name:
                    r['diffdock_rmsd'] = row.get('diffdock_rmsd', np.nan)
                    break
            else:
                new_rows.append(row)

    # 合并结果
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_all = pd.concat([df_existing, df_new], ignore_index=True) if not df_existing.empty else df_new
        df_all.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
    else:
        df_all = df_existing

    if df_all.empty:
        logger.warning("No results to plot")
        return

    # 生成图
    plot_rmsd_comparison(df_all, str(out_dir))
    print_summary_table(df_all)
    logger.info(f"All done. Results in {out_dir}")


if __name__ == '__main__':
    main()
