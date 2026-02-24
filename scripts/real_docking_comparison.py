#!/usr/bin/env python3
"""
DiffDock vs AutoDock Vina RMSD Comparison on PoseBench
Real experiment: dock ligands and compute RMSD against crystal structure
Created: 2026-02-24
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import tempfile
import shutil

# Try to import vina
try:
    from vina import Vina
    HAS_VINA = True
except ImportError:
    HAS_VINA = False
    print("Warning: Vina not available, will skip Vina docking")

from rdkit import Chem
from rdkit.Chem import AllChem

@dataclass
class DockingResult:
    method: str
    complex_id: str
    rmsd: float
    success: bool  # RMSD < 2.0Å
    runtime: float = 0.0
    error: str = None


class DockingComparison:
    """Compare DiffDock vs Vina on PoseBench"""

    def __init__(self, posebench_dir: str = "data/posebench/data/astex_diverse_set",
                 diffdock_path: str = None):
        self.posebench_dir = Path(posebench_dir)
        self.diffdock_path = diffdock_path or os.environ.get('DIFFDOCK_PATH',
                                                             '/home/lizihao/Tools/DiffDock')
        self.output_dir = Path("results/docking_comparison")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_complexes(self, max_samples: int = 10) -> List[str]:
        """Get list of PoseBench complexes"""
        complexes = [d.name for d in self.posebench_dir.iterdir() if d.is_dir()]
        return sorted(complexes)[:max_samples]

    def parse_sdf_coords(self, sdf_file: str) -> np.ndarray:
        """Extract coordinates from SDF file"""
        mol = Chem.MolFromMolFile(str(sdf_file))
        if mol is None:
            return np.array([])
        conf = mol.GetConformer()
        coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        return coords

    def parse_pdb_ligand_coords(self, pdb_file: str) -> np.ndarray:
        """Extract HETATM coordinates from PDB"""
        coords = []
        with open(pdb_file) as f:
            for line in f:
                if line.startswith('HETATM'):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords.append([x, y, z])
                    except:
                        pass
        return np.array(coords) if coords else np.array([])

    def compute_rmsd(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        """Compute RMSD between two sets of coordinates (using Hungarian matching)"""
        if len(coords1) == 0 or len(coords2) == 0:
            return float('inf')

        # Center both structures
        coords1_c = coords1 - coords1.mean(axis=0)
        coords2_c = coords2 - coords2.mean(axis=0)

        # Use minimum distance matching
        distances = cdist(coords1_c, coords2_c)
        min_dist = distances.min(axis=1)
        rmsd = np.sqrt((min_dist ** 2).mean())
        return rmsd

    def run_diffdock(self, protein_pdb: str, ligand_sdf: str,
                     output_dir: Path) -> Tuple[str, float]:
        """Run DiffDock on a single complex"""
        import time
        start = time.time()

        # Create input CSV for DiffDock
        csv_file = output_dir / "input.csv"
        with open(csv_file, 'w') as f:
            f.write("complex_name,protein_path,ligand_description,protein_sequence\n")
            f.write(f"complex,{protein_pdb},{ligand_sdf},\n")

        # Run DiffDock
        try:
            cmd = [
                "python", "-m", "inference",
                "--protein_ligand_csv", str(csv_file),
                "--out_dir", str(output_dir),
                "--inference_steps", "20",
                "--samples_per_complex", "1",
                "--batch_size", "1",
                "--no_final_step_noise"
            ]

            result = subprocess.run(
                cmd,
                cwd=self.diffdock_path,
                capture_output=True,
                text=True,
                timeout=300
            )

            # Find output
            pose_files = list(output_dir.glob("**/rank1*.sdf"))
            if pose_files:
                return str(pose_files[0]), time.time() - start
            else:
                # Try other patterns
                pose_files = list(output_dir.glob("**/*.sdf"))
                if pose_files:
                    return str(pose_files[0]), time.time() - start

        except Exception as e:
            print(f"DiffDock error: {e}")

        return None, time.time() - start

    def run_vina(self, protein_pdb: str, ligand_sdf: str,
                 output_dir: Path) -> Tuple[str, float]:
        """Run AutoDock Vina on a single complex"""
        if not HAS_VINA:
            return None, 0.0

        import time
        start = time.time()

        try:
            # Load ligand and get center
            ref_coords = self.parse_sdf_coords(ligand_sdf)
            if len(ref_coords) == 0:
                return None, 0.0
            center = ref_coords.mean(axis=0)

            # Initialize Vina
            v = Vina(sf_name='vina')
            v.set_receptor(protein_pdb)
            v.set_ligand_from_file(ligand_sdf)

            # Set search box
            v.compute_vina_maps(center=center.tolist(), box_size=[20, 20, 20])

            # Dock
            v.dock(exhaustiveness=8, n_poses=1)

            # Save output
            output_pdbqt = output_dir / "vina_docked.pdbqt"
            v.write_poses(str(output_pdbqt), n_poses=1, overwrite=True)

            return str(output_pdbqt), time.time() - start

        except Exception as e:
            print(f"Vina error: {e}")
            return None, time.time() - start

    def parse_pdbqt_coords(self, pdbqt_file: str) -> np.ndarray:
        """Extract coordinates from PDBQT file"""
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
                    break  # Only first pose
        return np.array(coords) if coords else np.array([])

    def evaluate_complex(self, complex_id: str) -> List[DockingResult]:
        """Evaluate a single complex with both methods"""
        results = []

        complex_dir = self.posebench_dir / complex_id
        protein_pdb = complex_dir / f"{complex_id}_protein.pdb"
        ligand_sdf = complex_dir / f"{complex_id}_ligand.sdf"

        if not protein_pdb.exists() or not ligand_sdf.exists():
            print(f"  Missing files for {complex_id}")
            return results

        # Get reference coordinates
        ref_coords = self.parse_sdf_coords(str(ligand_sdf))
        if len(ref_coords) == 0:
            print(f"  Could not parse ligand for {complex_id}")
            return results

        print(f"  Reference ligand: {len(ref_coords)} atoms")

        # Create temp directory for this complex
        work_dir = self.output_dir / complex_id
        work_dir.mkdir(exist_ok=True)

        # --- DiffDock ---
        print(f"  Running DiffDock...")
        dd_dir = work_dir / "diffdock"
        dd_dir.mkdir(exist_ok=True)

        dd_pose, dd_time = self.run_diffdock(str(protein_pdb), str(ligand_sdf), dd_dir)

        if dd_pose and Path(dd_pose).exists():
            dd_coords = self.parse_sdf_coords(dd_pose)
            if len(dd_coords) > 0:
                dd_rmsd = self.compute_rmsd(ref_coords, dd_coords)
                results.append(DockingResult(
                    method="DiffDock",
                    complex_id=complex_id,
                    rmsd=dd_rmsd,
                    success=dd_rmsd < 2.0,
                    runtime=dd_time
                ))
                print(f"    DiffDock RMSD: {dd_rmsd:.2f} Å")
            else:
                results.append(DockingResult(
                    method="DiffDock", complex_id=complex_id,
                    rmsd=float('inf'), success=False, runtime=dd_time,
                    error="Failed to parse output"
                ))
        else:
            results.append(DockingResult(
                method="DiffDock", complex_id=complex_id,
                rmsd=float('inf'), success=False, runtime=dd_time,
                error="No output generated"
            ))

        # --- Vina ---
        if HAS_VINA:
            print(f"  Running Vina...")
            vina_dir = work_dir / "vina"
            vina_dir.mkdir(exist_ok=True)

            vina_pose, vina_time = self.run_vina(str(protein_pdb), str(ligand_sdf), vina_dir)

            if vina_pose and Path(vina_pose).exists():
                vina_coords = self.parse_pdbqt_coords(vina_pose)
                if len(vina_coords) > 0:
                    vina_rmsd = self.compute_rmsd(ref_coords, vina_coords)
                    results.append(DockingResult(
                        method="Vina",
                        complex_id=complex_id,
                        rmsd=vina_rmsd,
                        success=vina_rmsd < 2.0,
                        runtime=vina_time
                    ))
                    print(f"    Vina RMSD: {vina_rmsd:.2f} Å")
                else:
                    results.append(DockingResult(
                        method="Vina", complex_id=complex_id,
                        rmsd=float('inf'), success=False, runtime=vina_time,
                        error="Failed to parse output"
                    ))
            else:
                results.append(DockingResult(
                    method="Vina", complex_id=complex_id,
                    rmsd=float('inf'), success=False, runtime=vina_time,
                    error="Docking failed"
                ))

        return results

    def run_comparison(self, max_samples: int = 5) -> pd.DataFrame:
        """Run full comparison on multiple complexes"""
        print("=" * 60)
        print("DiffDock vs Vina RMSD Comparison")
        print("=" * 60)

        complexes = self.get_complexes(max_samples)
        print(f"\nTesting on {len(complexes)} complexes from PoseBench")

        all_results = []

        for i, complex_id in enumerate(complexes):
            print(f"\n[{i+1}/{len(complexes)}] {complex_id}")
            results = self.evaluate_complex(complex_id)
            all_results.extend(results)

        # Create DataFrame
        df = pd.DataFrame([
            {
                'method': r.method,
                'complex_id': r.complex_id,
                'rmsd': r.rmsd if r.rmsd != float('inf') else None,
                'success': r.success,
                'runtime': r.runtime,
                'error': r.error
            }
            for r in all_results
        ])

        return df

    def generate_report(self, df: pd.DataFrame):
        """Generate comparison report and figures"""
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)

        # Save raw results
        csv_path = self.output_dir / "docking_comparison_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")

        # Statistics per method
        for method in df['method'].unique():
            method_df = df[df['method'] == method]
            valid_df = method_df[method_df['rmsd'].notna()]

            print(f"\n{method}:")
            print(f"  Total: {len(method_df)}")
            print(f"  Valid: {len(valid_df)}")
            if len(valid_df) > 0:
                print(f"  Mean RMSD: {valid_df['rmsd'].mean():.2f} ± {valid_df['rmsd'].std():.2f} Å")
                print(f"  Median RMSD: {valid_df['rmsd'].median():.2f} Å")
                print(f"  Success rate (RMSD < 2Å): {100 * valid_df['success'].mean():.1f}%")

        # Generate figure
        if len(df) > 0 and df['rmsd'].notna().any():
            self._plot_comparison(df)

    def _plot_comparison(self, df: pd.DataFrame):
        """Generate comparison plots"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Box plot comparison
        valid_df = df[df['rmsd'].notna()]
        if len(valid_df) > 0:
            methods = valid_df['method'].unique()
            data = [valid_df[valid_df['method'] == m]['rmsd'].values for m in methods]
            bp = axes[0].boxplot(data, labels=methods, patch_artist=True)
            colors = ['steelblue', 'coral']
            for patch, color in zip(bp['boxes'], colors[:len(methods)]):
                patch.set_facecolor(color)
            axes[0].axhline(2.0, color='red', linestyle='--', label='Success threshold (2Å)')
            axes[0].set_ylabel('RMSD (Å)')
            axes[0].set_title('RMSD Distribution')
            axes[0].legend()

        # 2. Success rate comparison
        methods = df['method'].unique()
        success_rates = [df[df['method'] == m]['success'].mean() * 100 for m in methods]
        bars = axes[1].bar(methods, success_rates, color=['steelblue', 'coral'][:len(methods)],
                          edgecolor='black')
        axes[1].set_ylabel('Success Rate (%)')
        axes[1].set_title('Success Rate (RMSD < 2Å)')
        axes[1].set_ylim(0, 100)
        for bar, rate in zip(bars, success_rates):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f'{rate:.1f}%', ha='center')

        # 3. RMSD scatter (if both methods have data)
        if len(methods) >= 2:
            pivot = df.pivot(index='complex_id', columns='method', values='rmsd')
            if 'DiffDock' in pivot.columns and 'Vina' in pivot.columns:
                valid = pivot.dropna()
                if len(valid) > 0:
                    axes[2].scatter(valid['Vina'], valid['DiffDock'],
                                   s=100, edgecolors='black', alpha=0.7)
                    max_val = max(valid['Vina'].max(), valid['DiffDock'].max())
                    axes[2].plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
                    axes[2].axhline(2.0, color='red', linestyle=':', alpha=0.5)
                    axes[2].axvline(2.0, color='red', linestyle=':', alpha=0.5)
                    axes[2].set_xlabel('Vina RMSD (Å)')
                    axes[2].set_ylabel('DiffDock RMSD (Å)')
                    axes[2].set_title('DiffDock vs Vina RMSD')

        plt.tight_layout()
        fig_path = self.output_dir / "docking_comparison_figure.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {fig_path}")
        plt.close()


def main():
    """Main function"""
    comparison = DockingComparison()

    # Run on 5 samples first (quick test)
    df = comparison.run_comparison(max_samples=5)

    # Generate report
    comparison.generate_report(df)

    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()
