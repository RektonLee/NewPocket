#!/usr/bin/env python3
"""
智能分配8个GPU的任务 - 基于实际剩余样本列表
读取remaining_samples.txt,均匀分配给8个GPU
"""
import json
from pathlib import Path

def load_remaining_samples():
    """Load list of remaining samples"""
    remaining_file = Path("results/remaining_samples.txt")
    if not remaining_file.exists():
        print("❌ remaining_samples.txt not found!")
        print("   Run: python scripts/reconcile_docking_status.py")
        return []

    with open(remaining_file, 'r') as f:
        samples = [line.strip() for line in f if line.strip()]

    return samples

def load_csv_mapping():
    """Load CSV to get index mapping for sample_ids"""
    import pandas as pd

    df = pd.read_csv("data/processed/kcat_full_1213.csv")
    # Create mapping: sample_id -> index
    mapping = {row['sample_id']: idx for idx, row in df.iterrows()}
    return mapping

def distribute_samples(samples, n_gpus=8):
    """Distribute samples evenly across GPUs"""
    samples_per_gpu = len(samples) // n_gpus
    remainder = len(samples) % n_gpus

    distributions = []
    start = 0

    for gpu in range(n_gpus):
        # Give extra sample to first 'remainder' GPUs
        count = samples_per_gpu + (1 if gpu < remainder else 0)
        end = start + count

        distributions.append({
            'gpu': gpu,
            'samples': samples[start:end],
            'count': count,
            'first_sample': samples[start] if start < len(samples) else None,
            'last_sample': samples[end-1] if end > 0 and end <= len(samples) else None
        })

        start = end

    return distributions

def main():
    print("="*80)
    print("🧮 计算8-GPU任务分配")
    print("="*80)

    # Load remaining samples
    remaining = load_remaining_samples()
    if not remaining:
        return

    print(f"\n📊 剩余样本: {len(remaining)}")

    # Distribute
    distributions = distribute_samples(remaining, n_gpus=8)

    print(f"\n📋 分配方案:")
    print("-"*80)

    total_assigned = 0
    for dist in distributions:
        print(f"GPU {dist['gpu']}: {dist['count']:4d} 样本")
        if dist['first_sample']:
            print(f"         首样本: {dist['first_sample']}")
            print(f"         末样本: {dist['last_sample']}")
        total_assigned += dist['count']

    print("-"*80)
    print(f"总计: {total_assigned}/{len(remaining)} 样本")

    # Save distribution
    output_file = Path("results/8gpu_distribution.json")
    with open(output_file, 'w') as f:
        json.dump(distributions, f, indent=2)

    print(f"\n💾 分配方案保存到: {output_file}")

    # Generate launch commands
    print(f"\n🚀 生成启动命令:")
    print("-"*80)

    for dist in distributions:
        if dist['count'] == 0:
            continue

        # Save sample list for this GPU
        sample_list_file = Path(f"results/gpu{dist['gpu']}_samples.txt")
        with open(sample_list_file, 'w') as f:
            for sample_id in dist['samples']:
                f.write(f"{sample_id}\n")

        print(f"\n# GPU {dist['gpu']} ({dist['count']} samples)")
        print(f"CUDA_VISIBLE_DEVICES={dist['gpu']} nohup python scripts/batch_diffdock_clean.py \\")
        print(f"  --sample_list results/gpu{dist['gpu']}_samples.txt \\")
        print(f"  > logs/diffdock_gpu{dist['gpu']}_8gpu_run.log 2>&1 &")

    print("\n" + "="*80)
    print("💡 提示: 脚本需要支持 --sample_list 参数")
    print("   如果batch_diffdock_clean.py不支持,需要先修改脚本添加此参数")
    print("="*80)

if __name__ == "__main__":
    main()
