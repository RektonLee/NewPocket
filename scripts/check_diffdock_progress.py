#!/usr/bin/env python3
"""
Check DiffDock batch processing progress
"""
import pandas as pd
from pathlib import Path
from datetime import datetime

def main():
    print("=" * 60)
    print("DiffDock Batch Processing Progress")
    print("=" * 60)
    print()

    # Check result files
    result_files = list(Path("results").glob("diffdock_batch_results_from_*.csv"))

    total_success = 0
    total_failed = 0
    total_exists = 0

    for result_file in sorted(result_files):
        df = pd.read_csv(result_file)
        success = (df['status'] == 'success').sum()
        failed = (df['status'].isin(['failed', 'error', 'timeout'])).sum()
        exists = (df['status'] == 'exists').sum()

        total_success += success
        total_failed += failed
        total_exists += exists

        start_idx = int(result_file.stem.split('_')[-1])
        print(f"Batch from {start_idx:5d}: Success={success:4d} Failed={failed:4d} Exists={exists:4d}")

    # Check output directories
    output_base = Path("sample_data/samples")
    if output_base.exists():
        docked_samples = []
        for sample_dir in output_base.iterdir():
            docking_dir = sample_dir / "docking"
            if docking_dir.exists():
                files = list(docking_dir.glob("*.sdf")) + list(docking_dir.glob("*.pdb"))
                if files:
                    docked_samples.append(sample_dir.name)

        print()
        print(f"Total docked samples on disk: {len(docked_samples)}")

    print()
    print("Summary:")
    print(f"  Total Success: {total_success}")
    print(f"  Total Failed:  {total_failed}")
    print(f"  Already Done:  {total_exists}")
    print(f"  Total Target:  9062")

    if total_success + total_exists > 0:
        progress = (total_success + total_exists) / 9062 * 100
        print(f"  Progress:      {progress:.1f}%")

    # Check if processes are running
    import subprocess
    try:
        result = subprocess.run(
            ["pgrep", "-f", "batch_diffdock_clean.py"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            print(f"\n  Running workers: {len(pids)} PIDs: {', '.join(pids)}")
        else:
            print(f"\n  Running workers: 0 (all finished or not started)")
    except:
        pass

    # Check log files
    log_dir = Path("logs")
    if log_dir.exists():
        print("\nRecent log activity:")
        for log_file in sorted(log_dir.glob("diffdock_gpu*.log")):
            if log_file.stat().st_size > 0:
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                size_kb = log_file.stat().st_size / 1024
                print(f"  {log_file.name:25s} Last modified: {mtime:%Y-%m-%d %H:%M:%S} Size: {size_kb:6.1f} KB")

if __name__ == '__main__':
    main()
