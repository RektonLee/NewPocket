#!/usr/bin/env python3
"""
Automated pipeline: Monitor DiffDock → Build Dataset → Train Model
Run this to automate the entire workflow while you sleep
"""
import subprocess
import time
from pathlib import Path
from datetime import datetime
import pandas as pd

def check_diffdock_progress():
    """Check how many samples are docked"""
    output_base = Path('sample_data/samples')
    if not output_base.exists():
        return 0, 0

    df = pd.read_csv('data/processed/kcat_full_1213.csv')
    train_ids = set(df['sample_id'].iloc[:4072])  # Only samples with PDB

    docked = set()
    for sample_dir in output_base.iterdir():
        if not sample_dir.is_dir():
            continue
        if sample_dir.name not in train_ids:
            continue

        docking_dir = sample_dir / 'docking'
        if docking_dir.exists():
            sdfs = list(docking_dir.glob('*.sdf')) + list(docking_dir.glob('**/*.sdf'))
            if sdfs:
                docked.add(sample_dir.name)

    return len(docked), len(train_ids)

def wait_for_diffdock_completion(check_interval=300):
    """Wait for DiffDock to complete, checking every 5 minutes"""
    print(f"[{datetime.now():%H:%M:%S}] Waiting for DiffDock batch processing...")
    print(f"Checking every {check_interval}s")

    last_count = 0
    stalled_checks = 0

    while True:
        docked, total = check_diffdock_progress()
        progress = docked / total * 100 if total > 0 else 0

        print(f"[{datetime.now():%H:%M:%S}] Progress: {docked}/{total} ({progress:.1f}%)")

        # Check if making progress
        if docked == last_count:
            stalled_checks += 1
            if stalled_checks >= 6:  # 30 minutes without progress
                print("⚠️  No progress for 30 minutes, checking if workers are alive...")
                result = subprocess.run(['pgrep', '-f', 'batch_diffdock_clean.py'],
                                      capture_output=True, text=True)
                if not result.stdout.strip():
                    print("❌ No workers running! Stopping.")
                    break
        else:
            stalled_checks = 0
            last_count = docked

        # Check if complete
        if docked >= total * 0.95:  # 95% completion is enough
            print(f"✅ DiffDock ~95% complete ({docked}/{total})")
            break

        time.sleep(check_interval)

    return docked, total

def build_graph_dataset():
    """Build PyG dataset from docked structures"""
    print(f"\n[{datetime.now():%H:%M:%S}] Building graph dataset...")

    # First, create a CSV with only successfully docked samples
    df = pd.read_csv('data/processed/kcat_full_1213.csv')
    output_base = Path('sample_data/samples')

    docked_samples = []
    for idx, row in df.iterrows():
        sample_id = row['sample_id']
        sample_dir = output_base / sample_id

        if not sample_dir.exists():
            continue

        docking_dir = sample_dir / 'docking'
        if docking_dir.exists():
            sdfs = list(docking_dir.glob('*.sdf')) + list(docking_dir.glob('**/*.sdf'))
            if sdfs:
                docked_samples.append(row)

    if not docked_samples:
        print("❌ No docked samples found!")
        return None

    docked_df = pd.DataFrame(docked_samples)
    csv_path = 'data/processed/kcat_train_diffdock.csv'
    docked_df.to_csv(csv_path, index=False)
    print(f"   Created CSV with {len(docked_df)} docked samples")

    # Build graph dataset
    output_pt = 'data/processed/kcat_train_diffdock.pt'

    cmd = [
        'python', 'src/build_graph_dataset.py',
        '--input', csv_path,
        '--output', output_pt,
        '--pocket_dir', 'sample_data/samples'
    ]

    print(f"   Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if Path(output_pt).exists():
        print(f"✅ Dataset built: {output_pt}")
        return output_pt
    else:
        print(f"❌ Dataset build failed")
        print(result.stderr[-500:] if result.stderr else "")
        return None

def train_model(dataset_path):
    """Train model on the new dataset"""
    print(f"\n[{datetime.now():%H:%M:%S}] Training model...")

    cmd = [
        'python', 'src/train.py',
        '--dataset', dataset_path,
        '--save_dir', 'outputs/diffdock_trained_auto',
        '--hidden_dim', '128',
        '--num_layers', '3',
        '--heads', '4',
        '--max_epochs', '300',
        '--lr', '1e-3',
        '--pooling_type', 'mean',
        '--split_ratios', '0.8,0.1,0.1',
        '--scheduler', 'plateau',
        '--weight_decay', '1e-4'
    ]

    print(f"   Running: {' '.join(cmd)}")

    log_file = f'logs/train_auto_{datetime.now():%Y%m%d_%H%M%S}.log'
    with open(log_file, 'w') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

    print(f"   Training log: {log_file}")

    # Check if model was saved
    model_dir = Path('outputs/diffdock_trained_auto')
    if model_dir.exists():
        models = list(model_dir.glob('**/best_model.pt'))
        if models:
            print(f"✅ Model trained: {models[0]}")
            return models[0]

    print(f"❌ Training failed or no model saved")
    return None

def evaluate_model(model_path):
    """Evaluate on test set"""
    print(f"\n[{datetime.now():%H:%M:%S}] Evaluating on test set...")

    cmd = [
        'python', 'src/test.py',
        '--test_dataset', 'data/processed/kcat_test_new_diffdock.pt',
        '--model', str(model_path),
        '--save_dir', 'results/diffdock_trained_auto_test',
        '--hidden_dim', '128',
        '--num_layers', '3',
        '--heads', '4',
        '--pooling_type', 'mean'
    ]

    print(f"   Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout[-500:] if result.stdout else "")

    return Path('results/diffdock_trained_auto_test').exists()

def main():
    print("="*60)
    print("Automated DiffDock → Train → Evaluate Pipeline")
    print("="*60)
    print()

    # Step 1: Wait for DiffDock
    docked, total = wait_for_diffdock_completion(check_interval=300)

    if docked < total * 0.5:
        print(f"❌ Too few samples docked ({docked}/{total}), aborting")
        return

    # Step 2: Build dataset
    dataset_path = build_graph_dataset()
    if not dataset_path:
        print("❌ Dataset building failed, aborting")
        return

    # Step 3: Train model
    model_path = train_model(dataset_path)
    if not model_path:
        print("❌ Training failed, aborting")
        return

    # Step 4: Evaluate
    success = evaluate_model(model_path)

    print()
    print("="*60)
    if success:
        print("✅ Pipeline completed successfully!")
        print(f"   Model: {model_path}")
        print(f"   Results: results/diffdock_trained_auto_test/")
    else:
        print("⚠️  Pipeline completed with errors")
    print("="*60)

if __name__ == '__main__':
    main()
