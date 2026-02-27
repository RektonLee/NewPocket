#!/usr/bin/env python3
"""
Check combined progress of 8-GPU DiffDock batch processing
"""
import subprocess
from pathlib import Path
from datetime import datetime

def count_sdf_files():
    """Count total docked samples"""
    try:
        result = subprocess.run(
            ["find", "sample_data/samples", "-name", "*_diffdock.sdf", "-type", "f"],
            capture_output=True,
            text=True,
            check=True
        )
        files = [f for f in result.stdout.strip().split('\n') if f]
        return len(files)
    except:
        return 0

def check_worker_status():
    """Check if workers are running"""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            check=True
        )
        workers = [line for line in result.stdout.split('\n')
                  if 'batch_diffdock_clean.py' in line and 'grep' not in line]
        return len(workers), workers
    except:
        return 0, []

def get_log_stats():
    """Get latest stats from each GPU log"""
    stats = {}
    log_dir = Path("logs")

    for gpu in range(8):
        log_file = log_dir / f"diffdock_gpu{gpu}_8gpu_run.log"
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()

                # Find latest progress line
                for line in reversed(lines):
                    if 'Progress:' in line:
                        stats[f"GPU{gpu}"] = line.strip()
                        break
                    elif 'Summary:' in line:
                        stats[f"GPU{gpu}"] = "COMPLETED"
                        break
            except:
                stats[f"GPU{gpu}"] = "ERROR reading log"
        else:
            stats[f"GPU{gpu}"] = "Log not found"

    return stats

def main():
    print("="*80)
    print("🔍 8-GPU DiffDock Progress Check")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Overall progress
    total_docked = count_sdf_files()
    target = 4072
    remaining = target - total_docked
    progress = total_docked / target * 100

    print(f"📊 Overall Progress:")
    print(f"   Docked: {total_docked}/{target} ({progress:.1f}%)")
    print(f"   Remaining: {remaining}")

    if total_docked >= 3868:
        print(f"   ✅ Reached 95% threshold! ({3868} samples)")
    else:
        print(f"   🎯 Need {3868 - total_docked} more for 95% threshold")

    # Worker status
    num_workers, workers = check_worker_status()
    print(f"\n🔧 Active Workers: {num_workers}/8")

    if num_workers > 0:
        print("   Running processes:")
        for i, worker in enumerate(workers[:8], 1):
            # Extract GPU info from process
            parts = worker.split()
            if len(parts) > 10:
                gpu_info = "GPU?"
                for part in parts:
                    if 'CUDA_VISIBLE_DEVICES' in part:
                        gpu_info = part
                print(f"   {i}. PID {parts[1]} - {gpu_info}")
    else:
        print("   ⚠️  No workers running!")

    # Per-GPU stats
    log_stats = get_log_stats()
    if log_stats:
        print(f"\n📋 Per-GPU Status:")
        for gpu, status in sorted(log_stats.items()):
            status_icon = "✅" if status == "COMPLETED" else "⏳" if "Progress:" in status else "❌"
            print(f"   {status_icon} {gpu}: {status}")

    # Time estimate
    if num_workers > 0 and remaining > 0:
        # Assume ~1 min per sample, distributed across active workers
        estimated_mins = remaining / num_workers
        estimated_hours = estimated_mins / 60
        print(f"\n⏱️  Time Estimate:")
        print(f"   ~{estimated_mins:.0f} minutes ({estimated_hours:.1f} hours)")
        print(f"   With {num_workers} workers @ ~1 min/sample")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
