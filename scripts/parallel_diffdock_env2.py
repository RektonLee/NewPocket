#!/usr/bin/env python3
"""
Parallel DiffDock with env2 for full training set
Run multiple GPU workers in parallel
"""
import pandas as pd
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

PYTHON_ENV2 = "/home/lizihao/miniforge3/envs/env2/bin/python"
DIFFDOCK_DIR = Path("/home/lizihao/Work/enzyme_prediction/PGNN_clean/DiffDock")
SAMPLES_DIR = Path("/home/lizihao/Work/enzyme_prediction/PGNN/sample_data/samples")

def run_single_sample(sample_id: str, protein_pdb: str, smiles: str, gpu_id: int) -> dict:
    """Run DiffDock for a single sample"""
    try:
        # Check if already done
        dest_dir = Path("sample_data/samples") / sample_id / "docking"
        if dest_dir.exists():
            existing = list(dest_dir.glob("*.sdf"))
            if existing:
                return {"sample_id": sample_id, "status": "skipped"}

        # Create temp directory
        temp_dir = Path(f"temp/diffdock_single/{sample_id}")
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Create CSV
        csv_file = temp_dir / "input.csv"
        with open(csv_file, 'w') as f:
            f.write("complex_name,protein_path,ligand_description,protein_sequence\n")
            f.write(f"{sample_id},{protein_pdb},{smiles},\n")

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

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        result = subprocess.run(
            cmd,
            cwd=str(DIFFDOCK_DIR),
            capture_output=True,
            text=True,
            timeout=120,  # 2 min per sample
            env=env
        )

        # Find output
        if out_dir.exists():
            sdf_files = list(out_dir.glob("**/*.sdf"))
            if sdf_files:
                # Copy to destination
                dest_dir.mkdir(parents=True, exist_ok=True)
                import shutil
                dest_file = dest_dir / f"{sample_id}_diffdock.sdf"
                shutil.copy(sdf_files[0], dest_file)

                # Cleanup temp
                shutil.rmtree(temp_dir, ignore_errors=True)

                return {"sample_id": sample_id, "status": "success"}

        return {"sample_id": sample_id, "status": "failed", "error": "No output"}

    except subprocess.TimeoutExpired:
        return {"sample_id": sample_id, "status": "timeout"}
    except Exception as e:
        return {"sample_id": sample_id, "status": "error", "error": str(e)[:100]}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--gpu_ids', type=str, default='0,1')
    parser.add_argument('--n_workers', type=int, default=4)
    args = parser.parse_args()

    # Load CSV
    df = pd.read_csv("data/processed/kcat_full_1213.csv")
    if args.max_samples:
        df = df.head(args.max_samples)

    print(f"📊 Processing {len(df)} samples")

    # Prepare tasks
    tasks = []
    for _, row in df.iterrows():
        sample_id = row['sample_id']
        smiles = row['substrate_smiles']

        # Skip if done
        dest_dir = Path("sample_data/samples") / sample_id / "docking"
        if dest_dir.exists() and list(dest_dir.glob("*.sdf")):
            continue

        # Find protein
        sample_dir = SAMPLES_DIR / sample_id
        if not sample_dir.exists():
            continue

        pdbs = list(sample_dir.glob("*.pdb"))
        if not pdbs:
            continue

        tasks.append((sample_id, str(pdbs[0].absolute()), smiles))

    print(f"✅ {len(tasks)} samples to process")

    if len(tasks) == 0:
        print("Nothing to do!")
        return

    # Parse GPUs
    gpu_ids = [int(x) for x in args.gpu_ids.split(',')]

    print(f"🚀 Using {args.n_workers} workers on GPUs: {gpu_ids}")
    print(f"   Estimated time: ~{len(tasks) * 30 / (args.n_workers * 60):.0f} minutes")

    # Run parallel
    results = []

    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {}
        for i, (sample_id, pdb, smiles) in enumerate(tasks):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            future = executor.submit(run_single_sample, sample_id, pdb, smiles, gpu_id)
            futures[future] = sample_id

        with tqdm(total=len(tasks), desc="DiffDock") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                success = sum(1 for r in results if r['status'] == 'success')
                skipped = sum(1 for r in results if r['status'] == 'skipped')
                failed = sum(1 for r in results if r['status'] in ['failed', 'error', 'timeout'])

                pbar.update(1)
                pbar.set_postfix({
                    'success': success,
                    'skip': skipped,
                    'fail': failed
                })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/diffdock_parallel_results.csv", index=False)

    print(f"\n📈 Final:")
    print(f"   Success: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"   Skipped: {sum(1 for r in results if r['status'] == 'skipped')}")
    print(f"   Failed: {sum(1 for r in results if r['status'] in ['failed', 'error', 'timeout'])}")

if __name__ == '__main__':
    main()
