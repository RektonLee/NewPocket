#!/usr/bin/env python3
"""
Quick docking validation: DiffDock quality assessment
Simplified approach: validate existing DiffDock poses instead of full Vina comparison
Created: 2026-02-24
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import PDB
from scipy.spatial.distance import cdist

@dataclass
class DockingResult:
    sample_id: str
    pdb_path: str
    confidence: float = None
    ligand_atoms: int = 0
    pocket_atoms: int = 0
    valid: bool = True
    error: str = None

class DockingValidator:
    """Validate DiffDock docking results quality"""

    def __init__(self, sample_dir: str = "sample_data/samples"):
        self.sample_dir = Path(sample_dir)
        self.parser = PDB.PDBParser(QUIET=True)

    def find_docked_samples(self) -> List[Path]:
        """Find all samples with docking results"""
        samples = []
        for sample_folder in sorted(self.sample_dir.glob("kcat_test_*")):
            docking_dir = sample_folder / "docking"
            if docking_dir.exists():
                pdb_files = list(docking_dir.glob("*.pdb"))
                if pdb_files:
                    samples.append(sample_folder)
        return samples

    def extract_ligand_coords(self, pdb_path: Path) -> np.ndarray:
        """Extract ligand (HETATM) coordinates by parsing file directly"""
        coords = []
        with open(pdb_path) as f:
            for line in f:
                if line.startswith('HETATM'):
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        coords.append([x, y, z])
                    except:
                        pass
        return np.array(coords) if coords else np.array([])

    def extract_protein_coords(self, pdb_path: Path) -> np.ndarray:
        """Extract protein (ATOM) coordinates by parsing file directly"""
        coords = []
        with open(pdb_path) as f:
            for line in f:
                if line.startswith('ATOM  '):
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        coords.append([x, y, z])
                    except:
                        pass
        return np.array(coords) if coords else np.array([])

    def compute_binding_metrics(self, pdb_path: Path) -> Dict:
        """Compute docking quality metrics"""
        try:
            lig_coords = self.extract_ligand_coords(pdb_path)
            prot_coords = self.extract_protein_coords(pdb_path)

            if len(lig_coords) == 0:
                return {"error": "No ligand atoms found"}

            if len(prot_coords) == 0:
                return {"error": "No protein atoms found"}

            # Compute distances
            distances = cdist(lig_coords, prot_coords)
            min_distances = distances.min(axis=1)

            # Metrics
            metrics = {
                "ligand_atoms": len(lig_coords),
                "pocket_atoms": len(prot_coords),
                "min_distance": float(min_distances.min()),
                "mean_distance": float(min_distances.mean()),
                "max_distance": float(min_distances.max()),
                "contacts_3A": int((min_distances < 3.0).sum()),
                "contacts_4A": int((min_distances < 4.0).sum()),
                "contacts_5A": int((min_distances < 5.0).sum()),
                "ligand_centroid": lig_coords.mean(axis=0).tolist(),
                "binding_score": float(np.exp(-min_distances.mean())),  # Simple score
            }

            return metrics

        except Exception as e:
            return {"error": str(e)}

    def validate_sample(self, sample_folder: Path) -> DockingResult:
        """Validate a single docked sample"""
        sample_id = sample_folder.name
        docking_dir = sample_folder / "docking"

        # Find PDB file
        pdb_files = list(docking_dir.glob("*.pdb"))
        if not pdb_files:
            return DockingResult(sample_id, None, valid=False, error="No PDB found")

        pdb_path = pdb_files[0]  # Use first (typically best pose)

        # Load docking metadata if exists
        meta_path = docking_dir / "docking_meta.json"
        confidence = None
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                    confidence = meta.get('confidence_score')
            except:
                pass

        # Compute metrics
        metrics = self.compute_binding_metrics(pdb_path)

        if "error" in metrics:
            return DockingResult(sample_id, str(pdb_path), confidence=confidence,
                               valid=False, error=metrics["error"])

        return DockingResult(
            sample_id=sample_id,
            pdb_path=str(pdb_path),
            confidence=confidence,
            ligand_atoms=metrics["ligand_atoms"],
            pocket_atoms=metrics["pocket_atoms"],
            valid=True
        )

    def validate_all(self) -> pd.DataFrame:
        """Validate all docked samples"""
        samples = self.find_docked_samples()
        results = []

        print(f"Found {len(samples)} samples with docking results")

        for i, sample_folder in enumerate(samples):
            if (i + 1) % 50 == 0:
                print(f"Processed {i+1}/{len(samples)}...")

            result = self.validate_sample(sample_folder)
            results.append({
                "sample_id": result.sample_id,
                "pdb_path": result.pdb_path,
                "confidence": result.confidence,
                "ligand_atoms": result.ligand_atoms,
                "pocket_atoms": result.pocket_atoms,
                "valid": result.valid,
                "error": result.error
            })

        df = pd.DataFrame(results)
        return df


def generate_validation_report(df: pd.DataFrame, output_dir: str = "results/docking_validation"):
    """Generate validation report with figures"""
    os.makedirs(output_dir, exist_ok=True)

    # Summary statistics
    print("\n" + "="*60)
    print("DIFFDOCK VALIDATION SUMMARY")
    print("="*60)
    print(f"Total samples: {len(df)}")
    print(f"Valid docking: {df['valid'].sum()} ({100*df['valid'].mean():.1f}%)")
    print(f"Failed: {(~df['valid']).sum()}")

    valid_df = df[df['valid']]
    if len(valid_df) > 0:
        print(f"\nLigand atoms: {valid_df['ligand_atoms'].mean():.1f} ± {valid_df['ligand_atoms'].std():.1f}")
        print(f"Pocket atoms: {valid_df['pocket_atoms'].mean():.1f} ± {valid_df['pocket_atoms'].std():.1f}")

        if valid_df['confidence'].notna().any():
            print(f"Confidence: {valid_df['confidence'].mean():.3f} ± {valid_df['confidence'].std():.3f}")

    # Save CSV
    csv_path = f"{output_dir}/docking_validation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Generate figures
    if len(valid_df) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Ligand atom distribution
        axes[0, 0].hist(valid_df['ligand_atoms'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Number of Ligand Atoms')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Ligand Size Distribution')
        axes[0, 0].axvline(valid_df['ligand_atoms'].mean(), color='red', linestyle='--',
                          label=f'Mean: {valid_df["ligand_atoms"].mean():.1f}')
        axes[0, 0].legend()

        # 2. Pocket atom distribution
        axes[0, 1].hist(valid_df['pocket_atoms'], bins=30, edgecolor='black', alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Number of Pocket Atoms')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Pocket Size Distribution')
        axes[0, 1].axvline(valid_df['pocket_atoms'].mean(), color='red', linestyle='--',
                          label=f'Mean: {valid_df["pocket_atoms"].mean():.1f}')
        axes[0, 1].legend()

        # 3. Success rate pie chart
        success_counts = df['valid'].value_counts()
        labels = ['Valid' if idx else 'Failed' for idx in success_counts.index]
        axes[1, 0].pie(success_counts, labels=labels, autopct='%1.1f%%',
                       colors=['#2ecc71', '#e74c3c'], startangle=90)
        axes[1, 0].set_title(f'Docking Success Rate (n={len(df)})')

        # 4. Confidence distribution (if available)
        if valid_df['confidence'].notna().any():
            conf_data = valid_df.dropna(subset=['confidence'])
            axes[1, 1].hist(conf_data['confidence'], bins=30, edgecolor='black', alpha=0.7, color='purple')
            axes[1, 1].set_xlabel('Confidence Score')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('DiffDock Confidence Distribution')
            axes[1, 1].axvline(conf_data['confidence'].mean(), color='red', linestyle='--',
                              label=f'Mean: {conf_data["confidence"].mean():.3f}')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'No confidence scores available',
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Confidence Scores')

        plt.tight_layout()
        fig_path = f"{output_dir}/docking_validation_summary.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {fig_path}")
        plt.close()

    print("="*60 + "\n")

    return df


def main():
    """Main execution"""
    print("DiffDock Docking Validation")
    print("=" * 60)

    validator = DockingValidator()

    # Validate all samples
    df = validator.validate_all()

    # Generate report
    generate_validation_report(df)

    print("\n✅ Validation complete!")


if __name__ == "__main__":
    main()
