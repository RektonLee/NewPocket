#!/usr/bin/env python3
"""
Extract protein from complex PDB and batch DiffDock with env2
Usage: python batch_diffdock_clean.py --max_samples N --n_workers 4
"""
import pandas as pd
import subprocess
from pathlib import Path
from tqdm import tqdm
import shutil
import argparse

PYTHON_ENV2 = "/home/lizihao/miniforge3/envs/env2/bin/python"
DIFFDOCK_DIR = Path("/home/lizihao/Work/enzyme_prediction/PGNN_clean/DiffDock")
SAMPLES_DIR = Path("/home/lizihao/Work/enzyme_prediction/PGNN/sample_data/samples")
OUTPUT_DIR = Path("sample_data/samples")
PROTEIN_CACHE = Path("temp/protein_only")

def extract_protein_from_pdb(complex_pdb: Path, output_pdb: Path) -> bool:
    """Extract protein atoms (remove HETATM) from complex PDB"""
    try:
        output_pdb.parent.mkdir(parents=True, exist_ok=True)

        with open(complex_pdb, 'r') as fin, open(output_pdb, 'w') as fout:
            for line in fin:
                # Keep ATOM lines (protein), skip HETATM (ligand)
                if line.startswith('ATOM'):
                    fout.write(line)
                elif line.startswith('END'):
                    fout.write(line)
                    break

        return output_pdb.exists() and output_pdb.stat().st_size > 0
    except Exception as e:
        print(f"Error extracting protein: {e}")
        return False

def run_diffdock_sample(sample_id: str, protein_pdb: Path, smiles: str) -> dict:
    """Run DiffDock for one sample"""
    try:
        # Output directory
        output_dir = OUTPUT_DIR / sample_id / "docking"

        # Check if already done
        if output_dir.exists():
            existing = list(output_dir.glob("*.sdf")) or list(output_dir.glob("*.pdb"))
            if existing:
                return {"sample_id": sample_id, "status": "exists"}

        output_dir.mkdir(parents=True, exist_ok=True)

        # Create temp input CSV
        temp_dir = Path(f"temp/diffdock_work/{sample_id}")
        temp_dir.mkdir(parents=True, exist_ok=True)

        csv_file = temp_dir / "input.csv"
        with open(csv_file, 'w') as f:
            f.write("complex_name,protein_path,ligand_description,protein_sequence\n")
            f.write(f"{sample_id},{protein_pdb.absolute()},{smiles},\n")

        out_dir = temp_dir / "output"

        # Run DiffDock
        cmd = [
            PYTHON_ENV2, "inference.py",
            "--protein_ligand_csv", str(csv_file.absolute()),
            "--out_dir", str(out_dir.absolute()),
            "--inference_steps", "20",
            "--samples_per_complex", "1",
            "--batch_size", "1",
            "--no_final_step_noise"
        ]

        import os
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

        result = subprocess.run(
            cmd,
            cwd=str(DIFFDOCK_DIR),
            capture_output=True,
            text=True,
            timeout=180,
            env=env
        )

        # Copy output
        if out_dir.exists():
            sdf_files = list(out_dir.glob(f"**/{sample_id}/**/*.sdf"))
            if sdf_files:
                dest_file = output_dir / f"{sample_id}_diffdock.sdf"
                shutil.copy(sdf_files[0], dest_file)
                shutil.rmtree(temp_dir, ignore_errors=True)
                return {"sample_id": sample_id, "status": "success"}

        # Save stderr for debugging
        if result.returncode != 0:
            error_file = temp_dir / "error.log"
            with open(error_file, 'w') as f:
                f.write(result.stderr)

        return {"sample_id": sample_id, "status": "failed"}

    except subprocess.TimeoutExpired:
        return {"sample_id": sample_id, "status": "timeout"}
    except Exception as e:
        return {"sample_id": sample_id, "status": "error", "error": str(e)[:200]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples to process')
    parser.add_argument('--n_workers', type=int, default=1, help='Sequential workers (not parallel)')
    parser.add_argument('--start_from', type=int, default=0, help='Start from index')
    args = parser.parse_args()

    # Load CSV
    df = pd.read_csv("data/processed/kcat_full_1213.csv")

    if args.max_samples:
        df = df.iloc[args.start_from:args.start_from + args.max_samples]
    else:
        df = df.iloc[args.start_from:]

    print(f"📊 Processing {len(df)} samples (from index {args.start_from})")

    # Prepare tasks
    tasks = []
    protein_cache_created = 0

    for idx, row in df.iterrows():
        sample_id = row['sample_id']
        smiles = row['substrate_smiles']

        # Skip if already done
        output_dir = OUTPUT_DIR / sample_id / "docking"
        if output_dir.exists() and (list(output_dir.glob("*.sdf")) or list(output_dir.glob("*.pdb"))):
            continue

        # Find complex PDB
        sample_dir = SAMPLES_DIR / sample_id
        if not sample_dir.exists():
            continue

        complex_pdbs = list(sample_dir.glob("*.pdb"))
        if not complex_pdbs:
            continue

        # Extract protein
        protein_only_pdb = PROTEIN_CACHE / f"{sample_id}_protein.pdb"
        if not protein_only_pdb.exists():
            if extract_protein_from_pdb(complex_pdbs[0], protein_only_pdb):
                protein_cache_created += 1
            else:
                continue

        tasks.append((sample_id, protein_only_pdb, smiles))

    print(f"✅ {len(tasks)} samples to process")
    print(f"   Created {protein_cache_created} protein-only PDBs")

    if len(tasks) == 0:
        print("Nothing to do!")
        return

    # Process sequentially (GPU bottleneck)
    results = []

    for sample_id, protein_pdb, smiles in tqdm(tasks, desc="DiffDock"):
        result = run_diffdock_sample(sample_id, protein_pdb, smiles)
        results.append(result)

        # Print progress
        if len(results) % 10 == 0:
            success = sum(1 for r in results if r['status'] == 'success')
            failed = sum(1 for r in results if r['status'] in ['failed', 'error', 'timeout'])
            print(f"  Progress: {len(results)}/{len(tasks)} | Success: {success} | Failed: {failed}")

    # Save results
    results_df = pd.DataFrame(results)
    output_file = f"results/diffdock_batch_results_from_{args.start_from}.csv"
    Path("results").mkdir(exist_ok=True)
    results_df.to_csv(output_file, index=False)

    print(f"\n📈 Summary:")
    print(f"   Success: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"   Failed: {sum(1 for r in results if r['status'] in ['failed', 'error', 'timeout'])}")
    print(f"   Exists: {sum(1 for r in results if r['status'] == 'exists')}")
    print(f"\n💾 Results saved to: {output_file}")

if __name__ == '__main__':
    main()
