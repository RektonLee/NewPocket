#!/usr/bin/env python3
"""
Quick DiffDock test on PoseBench samples
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
from rdkit import Chem
import time
import tempfile
import shutil

def parse_sdf_coords(sdf_file: str) -> np.ndarray:
    """Extract coordinates from SDF file"""
    mol = Chem.MolFromMolFile(str(sdf_file), removeHs=False)
    if mol is None:
        # Try without removing hydrogens
        mol = Chem.MolFromMolFile(str(sdf_file))
    if mol is None:
        return np.array([])
    conf = mol.GetConformer()
    return np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])

def compute_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute RMSD"""
    if len(coords1) == 0 or len(coords2) == 0:
        return float('inf')
    c1 = coords1 - coords1.mean(axis=0)
    c2 = coords2 - coords2.mean(axis=0)
    dist = cdist(c1, c2)
    min_d = dist.min(axis=1)
    return np.sqrt((min_d ** 2).mean())

def run_diffdock(protein_pdb: str, ligand_sdf: str, output_dir: Path, diffdock_dir: str) -> tuple:
    """Run DiffDock"""
    start = time.time()

    try:
        # Use absolute paths
        protein_pdb = str(Path(protein_pdb).absolute())
        ligand_sdf = str(Path(ligand_sdf).absolute())
        output_dir = Path(output_dir).absolute()

        # Create input CSV
        csv_file = output_dir / "input.csv"
        with open(csv_file, 'w') as f:
            f.write("complex_name,protein_path,ligand_description,protein_sequence\n")
            f.write(f"test,{protein_pdb},{ligand_sdf},\n")

        out_path = output_dir / "diffdock_out"

        # Run DiffDock
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'

        cmd = [
            sys.executable, "inference.py",
            "--protein_ligand_csv", str(csv_file),
            "--out_dir", str(out_path),
            "--inference_steps", "20",
            "--samples_per_complex", "1",
            "--batch_size", "1",
            "--no_final_step_noise"
        ]

        result = subprocess.run(
            cmd,
            cwd=diffdock_dir,
            capture_output=True,
            text=True,
            timeout=300,
            env=env
        )

        # Find output
        if out_path.exists():
            pose_files = list(out_path.glob("**/rank1*.sdf"))
            if not pose_files:
                pose_files = list(out_path.glob("**/*.sdf"))
            if pose_files:
                return str(pose_files[0]), time.time() - start

        print(f"    DiffDock stderr: {result.stderr[:500] if result.stderr else 'None'}")
        return None, time.time() - start

    except subprocess.TimeoutExpired:
        print("    DiffDock timeout")
        return None, time.time() - start
    except Exception as e:
        print(f"    DiffDock error: {e}")
        return None, time.time() - start

def test_posebench_diffdock(complex_id: str, posebench_dir: Path, output_dir: Path, diffdock_dir: str) -> dict:
    """Test DiffDock on a single PoseBench sample"""
    complex_dir = posebench_dir / complex_id
    protein = complex_dir / f"{complex_id}_protein.pdb"
    ligand = complex_dir / f"{complex_id}_ligand.sdf"

    if not protein.exists() or not ligand.exists():
        return {"complex_id": complex_id, "error": "Missing files"}

    ref_coords = parse_sdf_coords(str(ligand))
    if len(ref_coords) == 0:
        return {"complex_id": complex_id, "error": "Could not parse ligand"}

    result = {"complex_id": complex_id, "ref_atoms": len(ref_coords)}

    sample_dir = output_dir / complex_id
    sample_dir.mkdir(exist_ok=True)

    print(f"  Running DiffDock...")
    dd_pose, dd_time = run_diffdock(str(protein), str(ligand), sample_dir, diffdock_dir)

    if dd_pose and Path(dd_pose).exists():
        dd_coords = parse_sdf_coords(dd_pose)
        if len(dd_coords) > 0:
            dd_rmsd = compute_rmsd(ref_coords, dd_coords)
            result["diffdock_rmsd"] = dd_rmsd
            result["diffdock_success"] = dd_rmsd < 2.0
            result["diffdock_time"] = dd_time
            print(f"    DiffDock RMSD: {dd_rmsd:.2f} Å ({'✓' if dd_rmsd < 2.0 else '✗'})")
        else:
            result["diffdock_rmsd"] = None
            result["diffdock_success"] = False
            print(f"    DiffDock: Failed to parse output")
    else:
        result["diffdock_rmsd"] = None
        result["diffdock_success"] = False
        print(f"    DiffDock: No output")

    return result

def main():
    posebench_dir = Path("data/posebench/data/astex_diverse_set")
    output_dir = Path("results/diffdock_rmsd_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    diffdock_dir = str(Path("DiffDock").absolute())

    # Test on same 5 complexes as Vina
    complexes = ["1G9V_RQ3", "1GKC_NFH", "1GM8_SOX", "1GPK_HUP", "1HNN_SKF"]

    print("=" * 60)
    print("PoseBench RMSD Test (DiffDock)")
    print("=" * 60)
    print(f"\nTesting {len(complexes)} complexes\n")

    results = []
    for i, cid in enumerate(complexes):
        print(f"[{i+1}/{len(complexes)}] {cid}")
        result = test_posebench_diffdock(cid, posebench_dir, output_dir, diffdock_dir)
        results.append(result)

    # Summary
    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("DIFFDOCK RESULTS")
    print("=" * 60)

    valid = df[df['diffdock_rmsd'].notna()]
    if len(valid) > 0:
        print(f"\n  Valid docking: {len(valid)}/{len(df)}")
        print(f"  Mean RMSD: {valid['diffdock_rmsd'].mean():.2f} ± {valid['diffdock_rmsd'].std():.2f} Å")
        print(f"  Success rate: {100 * valid['diffdock_success'].mean():.1f}%")

        print(f"\nPer-complex RMSD:")
        for _, row in df.iterrows():
            rmsd = row.get('diffdock_rmsd', None)
            if rmsd is not None:
                print(f"  {row['complex_id']}: {rmsd:.2f} Å ({'✓' if rmsd < 2.0 else '✗'})")

    csv_path = output_dir / "diffdock_rmsd_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

if __name__ == "__main__":
    main()
