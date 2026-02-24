#!/usr/bin/env python3
"""
Quick docking RMSD test on PoseBench samples
Test a few samples with Vina first (faster), then DiffDock
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
from rdkit import Chem
import time
import subprocess

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent))

def parse_sdf_coords(sdf_file: str) -> np.ndarray:
    """Extract coordinates from SDF file"""
    mol = Chem.MolFromMolFile(str(sdf_file))
    if mol is None:
        return np.array([])
    conf = mol.GetConformer()
    return np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])

def parse_pdbqt_coords(pdbqt_file: str) -> np.ndarray:
    """Extract coordinates from PDBQT"""
    coords = []
    with open(pdbqt_file) as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except:
                    pass
            if line.startswith('ENDMDL'):
                break
    return np.array(coords) if coords else np.array([])

def compute_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute RMSD using minimum distance matching"""
    if len(coords1) == 0 or len(coords2) == 0:
        return float('inf')

    # Center both
    c1 = coords1 - coords1.mean(axis=0)
    c2 = coords2 - coords2.mean(axis=0)

    # Min distance matching
    dist = cdist(c1, c2)
    min_d = dist.min(axis=1)
    return np.sqrt((min_d ** 2).mean())

def convert_pdb_to_pdbqt(pdb_file: str, output_file: str) -> bool:
    """Convert PDB to PDBQT using obabel"""
    try:
        cmd = ['obabel', pdb_file, '-O', output_file, '-xr']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return Path(output_file).exists()
    except Exception as e:
        print(f"    Conversion error: {e}")
        return False

def convert_sdf_to_pdbqt(sdf_file: str, output_file: str) -> bool:
    """Convert SDF to PDBQT using obabel"""
    try:
        cmd = ['obabel', sdf_file, '-O', output_file]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return Path(output_file).exists()
    except Exception as e:
        print(f"    Conversion error: {e}")
        return False

def run_vina_dock(protein_pdbqt: str, ligand_pdbqt: str, center: list, output_file: str) -> tuple:
    """Run Vina docking with PDBQT files"""
    try:
        from vina import Vina
    except ImportError:
        print("Vina not available")
        return None, 0.0

    start = time.time()

    try:
        # Setup Vina
        v = Vina(sf_name='vina')
        v.set_receptor(protein_pdbqt)
        v.set_ligand_from_file(ligand_pdbqt)

        # Set box
        v.compute_vina_maps(center=center, box_size=[20, 20, 20])

        # Dock
        v.dock(exhaustiveness=8, n_poses=1)

        # Save
        v.write_poses(output_file, n_poses=1, overwrite=True)

        return output_file, time.time() - start

    except Exception as e:
        print(f"    Vina error: {e}")
        return None, time.time() - start

def test_posebench_sample(complex_id: str, posebench_dir: Path, output_dir: Path) -> dict:
    """Test a single PoseBench sample"""
    complex_dir = posebench_dir / complex_id
    protein = complex_dir / f"{complex_id}_protein.pdb"
    ligand = complex_dir / f"{complex_id}_ligand.sdf"

    if not protein.exists() or not ligand.exists():
        return {"error": "Missing files", "complex_id": complex_id}

    # Reference coords
    ref_coords = parse_sdf_coords(str(ligand))
    if len(ref_coords) == 0:
        return {"error": "Could not parse ligand", "complex_id": complex_id}

    result = {
        "complex_id": complex_id,
        "ref_atoms": len(ref_coords),
    }

    # Create output dir
    sample_dir = output_dir / complex_id
    sample_dir.mkdir(exist_ok=True)

    # Convert to PDBQT
    print(f"  Converting to PDBQT...")
    protein_pdbqt = sample_dir / "receptor.pdbqt"
    ligand_pdbqt = sample_dir / "ligand.pdbqt"

    if not convert_pdb_to_pdbqt(str(protein), str(protein_pdbqt)):
        result["vina_rmsd"] = None
        result["vina_success"] = False
        result["error"] = "Failed to convert protein"
        return result

    if not convert_sdf_to_pdbqt(str(ligand), str(ligand_pdbqt)):
        result["vina_rmsd"] = None
        result["vina_success"] = False
        result["error"] = "Failed to convert ligand"
        return result

    # Get center from reference ligand
    center = ref_coords.mean(axis=0).tolist()

    # Vina docking
    print(f"  Running Vina...")
    output_pdbqt = str(sample_dir / "vina_pose.pdbqt")
    vina_pose, vina_time = run_vina_dock(str(protein_pdbqt), str(ligand_pdbqt), center, output_pdbqt)

    if vina_pose and Path(vina_pose).exists():
        vina_coords = parse_pdbqt_coords(vina_pose)
        if len(vina_coords) > 0:
            vina_rmsd = compute_rmsd(ref_coords, vina_coords)
            result["vina_rmsd"] = vina_rmsd
            result["vina_success"] = vina_rmsd < 2.0
            result["vina_time"] = vina_time
            print(f"    Vina RMSD: {vina_rmsd:.2f} Å ({'✓' if vina_rmsd < 2.0 else '✗'})")
        else:
            result["vina_rmsd"] = None
            result["vina_success"] = False
            result["vina_time"] = vina_time
            print(f"    Vina: Failed to parse output")
    else:
        result["vina_rmsd"] = None
        result["vina_success"] = False
        result["vina_time"] = vina_time
        print(f"    Vina: Failed")

    return result

def main():
    posebench_dir = Path("data/posebench/data/astex_diverse_set")
    output_dir = Path("results/docking_rmsd_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get first 5 complexes
    complexes = sorted([d.name for d in posebench_dir.iterdir() if d.is_dir()])[:5]

    print("=" * 60)
    print("PoseBench RMSD Test (Vina)")
    print("=" * 60)
    print(f"\nTesting {len(complexes)} complexes\n")

    results = []
    for i, cid in enumerate(complexes):
        print(f"[{i+1}/{len(complexes)}] {cid}")
        result = test_posebench_sample(cid, posebench_dir, output_dir)
        results.append(result)

    # Summary
    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    valid = df[df['vina_rmsd'].notna()]
    if len(valid) > 0:
        print(f"\nVina Results:")
        print(f"  Valid docking: {len(valid)}/{len(df)}")
        print(f"  Mean RMSD: {valid['vina_rmsd'].mean():.2f} ± {valid['vina_rmsd'].std():.2f} Å")
        print(f"  Success rate: {100 * valid['vina_success'].mean():.1f}%")

        # Per-complex results
        print(f"\nPer-complex RMSD:")
        for _, row in df.iterrows():
            rmsd = row.get('vina_rmsd', None)
            if rmsd is not None:
                print(f"  {row['complex_id']}: {rmsd:.2f} Å ({'✓' if rmsd < 2.0 else '✗'})")

    # Save
    csv_path = output_dir / "vina_rmsd_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    print("\n✅ Test complete!")

if __name__ == "__main__":
    main()
