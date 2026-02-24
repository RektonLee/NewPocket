#!/usr/bin/env python3
"""
Parallel DiffDock docking for training set (kcat_full_1213.csv)
"""
import pandas as pd
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os

DIFFDOCK_DIR = Path("/home/lizihao/Work/enzyme_prediction/PGNN_clean/DiffDock")
SAMPLES_DIR = Path("/home/lizihao/Work/enzyme_prediction/PGNN/sample_data/samples")
OUTPUT_BASE = Path("sample_data/samples")

def find_protein_pdb(sample_id: str) -> Path:
    """Find protein PDB file for sample"""
    sample_dir = SAMPLES_DIR / sample_id
    if not sample_dir.exists():
        return None

    # Look for .pdb files
    pdb_files = list(sample_dir.glob("*.pdb"))
    if pdb_files:
        return pdb_files[0]
    return None

def run_diffdock_single(sample_id: str, smiles: str, protein_pdb: Path) -> dict:
    """Run DiffDock for a single sample"""
    try:
        # Create output directory
        output_dir = OUTPUT_BASE / sample_id / "docking_diffdock"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if already done
        existing_pdbs = list(output_dir.glob("*.pdb"))
        if existing_pdbs:
            return {
                "sample_id": sample_id,
                "status": "skipped",
                "output": str(existing_pdbs[0])
            }

        # Create CSV for DiffDock
        csv_file = output_dir / "input.csv"
        with open(csv_file, 'w') as f:
            f.write("complex_name,protein_path,ligand_description,protein_sequence\n")
            f.write(f"{sample_id},{protein_pdb.absolute()},{smiles},\n")

        out_path = output_dir / "diffdock_out"

        # Run DiffDock
        cmd = [
            sys.executable, "inference.py",
            "--protein_ligand_csv", str(csv_file.absolute()),
            "--out_dir", str(out_path.absolute()),
            "--inference_steps", "20",
            "--samples_per_complex", "1",
            "--batch_size", "1",
            "--no_final_step_noise"
        ]

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ''  # Use CPU for parallel

        result = subprocess.run(
            cmd,
            cwd=str(DIFFDOCK_DIR),
            capture_output=True,
            text=True,
            timeout=180,
            env=env
        )

        # Find output SDF
        if out_path.exists():
            sdf_files = list(out_path.glob("**/*.sdf"))
            if sdf_files:
                # Convert SDF to PDB
                output_pdb = output_dir / f"{sample_id}_diffdock.pdb"
                # Simple extraction - just copy for now
                return {
                    "sample_id": sample_id,
                    "status": "success",
                    "output": str(sdf_files[0])
                }

        return {
            "sample_id": sample_id,
            "status": "failed",
            "error": "No output found"
        }

    except Exception as e:
        return {
            "sample_id": sample_id,
            "status": "error",
            "error": str(e)
        }

def main():
    # Load CSV
    csv_path = Path("data/processed/kcat_full_1213.csv")
    df = pd.read_csv(csv_path)
    print(f"📊 Loaded {len(df)} samples")

    # Filter samples with existing PDB
    tasks = []
    for _, row in df.iterrows():
        sample_id = row['sample_id']
        smiles = row['substrate_smiles']

        protein_pdb = find_protein_pdb(sample_id)
        if protein_pdb:
            tasks.append((sample_id, smiles, protein_pdb))

    print(f"✅ Found {len(tasks)} samples with protein structures")

    if len(tasks) == 0:
        print("❌ No samples to process!")
        return

    # Ask for confirmation
    print(f"\n🚀 Will run DiffDock on {len(tasks)} samples")
    print(f"   Estimated time: ~{len(tasks) * 30 / 3600:.1f} hours with 4 workers")
    response = input("Continue? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled")
        return

    # Run parallel DiffDock
    n_workers = 4
    results = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(run_diffdock_single, sample_id, smiles, pdb): sample_id
            for sample_id, smiles, pdb in tasks
        }

        with tqdm(total=len(tasks), desc="DiffDock Progress") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)

                # Update progress
                success = sum(1 for r in results if r['status'] == 'success')
                skipped = sum(1 for r in results if r['status'] == 'skipped')
                failed = sum(1 for r in results if r['status'] in ['failed', 'error'])
                pbar.set_postfix({
                    'success': success,
                    'skipped': skipped,
                    'failed': failed
                })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/diffdock_batch_results.csv", index=False)

    print(f"\n📈 Summary:")
    print(f"   Success: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"   Skipped: {sum(1 for r in results if r['status'] == 'skipped')}")
    print(f"   Failed: {sum(1 for r in results if r['status'] in ['failed', 'error'])}")

if __name__ == '__main__':
    main()
