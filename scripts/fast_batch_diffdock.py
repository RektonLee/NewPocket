#!/usr/bin/env python3
"""
Fast parallel DiffDock with GPU support
Process kcat_full_1213.csv in batches
"""
import pandas as pd
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
import os
import time

DIFFDOCK_DIR = Path("/home/lizihao/Work/enzyme_prediction/PGNN_clean/DiffDock")
SAMPLES_DIR = Path("/home/lizihao/Work/enzyme_prediction/PGNN/sample_data/samples")

def create_batch_csv(samples: list, batch_dir: Path) -> Path:
    """Create CSV for a batch of samples"""
    csv_file = batch_dir / "batch_input.csv"

    with open(csv_file, 'w') as f:
        f.write("complex_name,protein_path,ligand_description,protein_sequence\n")
        for sample_id, protein_pdb, smiles in samples:
            f.write(f"{sample_id},{protein_pdb},{smiles},\n")

    return csv_file

def run_diffdock_batch(batch_samples: list, batch_id: int, gpu_id: int) -> dict:
    """Run DiffDock on a batch"""
    batch_dir = Path(f"temp/diffdock_batch_{batch_id}")
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Create batch CSV
    csv_file = create_batch_csv(batch_samples, batch_dir)
    out_dir = batch_dir / "output"

    # Run DiffDock
    cmd = [
        sys.executable, "inference.py",
        "--protein_ligand_csv", str(csv_file.absolute()),
        "--out_dir", str(out_dir.absolute()),
        "--inference_steps", "20",
        "--samples_per_complex", "1",
        "--batch_size", str(min(10, len(batch_samples))),
        "--no_final_step_noise"
    ]

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(DIFFDOCK_DIR),
            capture_output=True,
            text=True,
            timeout=600,  # 10 min per batch
            env=env
        )

        # Copy results back to sample directories
        success_count = 0
        for sample_id, _, _ in batch_samples:
            sample_output = out_dir / sample_id
            if not sample_output.exists():
                continue

            sdf_files = list(sample_output.glob("**/*rank1*.sdf"))
            if not sdf_files:
                sdf_files = list(sample_output.glob("**/*.sdf"))

            if sdf_files:
                # Copy to sample directory
                dest_dir = Path("sample_data/samples") / sample_id / "docking"
                dest_dir.mkdir(parents=True, exist_ok=True)

                import shutil
                dest_file = dest_dir / f"{sample_id}_diffdock.sdf"
                shutil.copy(sdf_files[0], dest_file)
                success_count += 1

        return {
            "batch_id": batch_id,
            "success": success_count,
            "total": len(batch_samples),
            "time": time.time() - start
        }

    except Exception as e:
        return {
            "batch_id": batch_id,
            "success": 0,
            "total": len(batch_samples),
            "error": str(e),
            "time": time.time() - start
        }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples to process (for testing)')
    parser.add_argument('--batch_size', type=int, default=50, help='Samples per batch')
    parser.add_argument('--gpu_ids', type=str, default='0,1', help='GPU IDs to use (comma-separated)')
    args = parser.parse_args()

    # Load CSV
    csv_path = Path("data/processed/kcat_full_1213.csv")
    df = pd.read_csv(csv_path)

    if args.max_samples:
        df = df.head(args.max_samples)

    print(f"📊 Processing {len(df)} samples")

    # Prepare tasks
    tasks = []
    skipped = 0

    for _, row in df.iterrows():
        sample_id = row['sample_id']
        smiles = row['substrate_smiles']

        # Check if already done
        dest_dir = Path("sample_data/samples") / sample_id / "docking"
        if dest_dir.exists() and list(dest_dir.glob("*.sdf")):
            skipped += 1
            continue

        # Find protein PDB
        sample_dir = SAMPLES_DIR / sample_id
        if not sample_dir.exists():
            continue

        pdb_files = list(sample_dir.glob("*.pdb"))
        if not pdb_files:
            continue

        tasks.append((sample_id, str(pdb_files[0].absolute()), smiles))

    print(f"✅ Found {len(tasks)} samples to process ({skipped} already done)")

    if len(tasks) == 0:
        print("Nothing to do!")
        return

    # Create batches
    batch_size = args.batch_size
    batches = [tasks[i:i+batch_size] for i in range(0, len(tasks), batch_size)]
    gpu_ids = [int(x) for x in args.gpu_ids.split(',')]

    print(f"🚀 Running {len(batches)} batches on GPUs: {gpu_ids}")
    print(f"   Batch size: {batch_size}")
    print(f"   Estimated time: ~{len(batches) * 5 / len(gpu_ids):.0f} minutes")

    # Process batches
    results = []
    pbar = tqdm(total=len(batches), desc="Batches")

    for i, batch in enumerate(batches):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        result = run_diffdock_batch(batch, i, gpu_id)
        results.append(result)

        pbar.update(1)
        pbar.set_postfix({
            'success': sum(r['success'] for r in results),
            'total': sum(r['total'] for r in results)
        })

    pbar.close()

    # Summary
    total_success = sum(r['success'] for r in results)
    total_processed = sum(r['total'] for r in results)

    print(f"\n📈 Summary:")
    print(f"   Success: {total_success}/{total_processed} ({100*total_success/total_processed:.1f}%)")
    print(f"   Already done: {skipped}")
    print(f"   Total time: {sum(r['time'] for r in results)/60:.1f} min")

if __name__ == '__main__':
    main()
