#!/usr/bin/env python3
"""
Reconcile docking status between disk files and CSV records.
Provides comprehensive analysis of what's been docked and what remains.
"""

import pandas as pd
from pathlib import Path
import json
from datetime import datetime

# Configuration
CSV_PATH = Path("data/processed/kcat_full_1213.csv")
SAMPLES_DIR = Path("sample_data/samples")
RESULTS_CSV_GPU0 = Path("results/diffdock_batch_results_from_0.csv")
RESULTS_CSV_GPU1 = Path("results/diffdock_batch_results_from_2036.csv")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

def find_docked_samples():
    """Find all samples with SDF files on disk."""
    print("🔍 Scanning disk for docked samples...")
    docked = []

    if not SAMPLES_DIR.exists():
        print(f"❌ Samples directory not found: {SAMPLES_DIR}")
        return []

    for sdf_file in SAMPLES_DIR.rglob("*_diffdock.sdf"):
        # Extract sample_id from filename
        sample_id = sdf_file.stem.replace("_diffdock", "")
        docked.append({
            'sample_id': sample_id,
            'sdf_path': str(sdf_file),
            'size_kb': sdf_file.stat().st_size / 1024
        })

    print(f"✅ Found {len(docked)} docked samples on disk")
    return docked

def load_csv_records():
    """Load CSV records from batch processing."""
    records = {}

    for csv_path in [RESULTS_CSV_GPU0, RESULTS_CSV_GPU1]:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            gpu_name = "GPU0" if "from_0" in str(csv_path) else "GPU1"

            for _, row in df.iterrows():
                sample_id = row['sample_id']
                status = row['status']

                if sample_id not in records:
                    records[sample_id] = {
                        'status': status,
                        'gpu': gpu_name
                    }

            print(f"📊 Loaded {len(df)} records from {csv_path.name}")

    return records

def analyze_discrepancy(docked_on_disk, csv_records, target_csv):
    """Analyze discrepancy between disk and CSV."""
    print("\n" + "="*80)
    print("📊 DISCREPANCY ANALYSIS")
    print("="*80)

    # Load target samples
    if not target_csv.exists():
        print(f"❌ Target CSV not found: {target_csv}")
        return None

    df_target = pd.read_csv(target_csv)
    first_4072 = df_target.head(4072)

    # Convert to sets for comparison
    disk_ids = {s['sample_id'] for s in docked_on_disk}
    csv_success_ids = {sid for sid, info in csv_records.items() if info['status'] == 'success'}
    target_ids = set(first_4072['sample_id'])

    # Find discrepancies
    on_disk_not_csv = disk_ids - csv_success_ids
    in_csv_not_disk = csv_success_ids - disk_ids

    # Categorize disk samples
    disk_in_target = disk_ids & target_ids
    disk_beyond_target = disk_ids - target_ids

    print(f"\n📁 Disk State:")
    print(f"  Total SDF files: {len(disk_ids)}")
    print(f"  In target range (0-4071): {len(disk_in_target)}")
    print(f"  Beyond target range (4072+): {len(disk_beyond_target)}")

    print(f"\n📋 CSV Records:")
    print(f"  Total success records: {len(csv_success_ids)}")
    print(f"  Total failed records: {sum(1 for info in csv_records.values() if info['status'] == 'failed')}")

    print(f"\n⚠️  Discrepancies:")
    print(f"  On disk but NOT in CSV: {len(on_disk_not_csv)}")
    print(f"  In CSV but NOT on disk: {len(in_csv_not_disk)}")

    if len(in_csv_not_disk) > 0:
        print(f"  ⚠️  WARNING: {len(in_csv_not_disk)} samples marked success but no SDF file!")

    # Calculate actual target coverage
    actual_coverage = len(disk_in_target) / len(target_ids) * 100
    csv_coverage = len(csv_success_ids & target_ids) / len(target_ids) * 100

    print(f"\n📈 Coverage of Target 4072 Samples:")
    print(f"  Actual (on disk): {len(disk_in_target)}/{len(target_ids)} ({actual_coverage:.1f}%)")
    print(f"  CSV records: {len(csv_success_ids & target_ids)}/{len(target_ids)} ({csv_coverage:.1f}%)")

    # Identify source of extra samples
    print(f"\n🔍 Source of Extra {len(on_disk_not_csv)} Samples:")
    if len(on_disk_not_csv) > 0:
        sample_list = sorted(list(on_disk_not_csv))[:20]
        print(f"  First 20: {sample_list}")

        # Check if they're in target range
        extra_in_target = on_disk_not_csv & target_ids
        extra_beyond_target = on_disk_not_csv - target_ids
        print(f"  In target range: {len(extra_in_target)}")
        print(f"  Beyond target: {len(extra_beyond_target)}")
        print(f"  → These likely came from previous runs before GPU0/GPU1 batch")

    return {
        'disk_total': len(disk_ids),
        'disk_in_target': len(disk_in_target),
        'disk_beyond_target': len(disk_beyond_target),
        'csv_success': len(csv_success_ids),
        'discrepancy_on_disk_not_csv': len(on_disk_not_csv),
        'discrepancy_in_csv_not_disk': len(in_csv_not_disk),
        'actual_coverage_pct': actual_coverage,
        'csv_coverage_pct': csv_coverage,
        'target_total': len(target_ids)
    }

def find_remaining_gaps(docked_on_disk, target_csv):
    """Find which target samples still need docking."""
    print("\n" + "="*80)
    print("🎯 REMAINING GAPS ANALYSIS")
    print("="*80)

    df_target = pd.read_csv(target_csv)
    first_4072 = df_target.head(4072)

    disk_ids = {s['sample_id'] for s in docked_on_disk}
    target_ids = set(first_4072['sample_id'])

    remaining = target_ids - disk_ids

    print(f"\n📊 Gap Analysis:")
    print(f"  Target samples: {len(target_ids)}")
    print(f"  Docked: {len(disk_ids & target_ids)}")
    print(f"  Remaining: {len(remaining)}")

    if len(remaining) > 0:
        # Get indices of remaining samples
        remaining_df = first_4072[first_4072['sample_id'].isin(remaining)]
        indices = remaining_df.index.tolist()

        print(f"\n  Index range: {min(indices)} to {max(indices)}")
        print(f"  First 20: {sorted(indices)[:20]}")
        print(f"  Last 20: {sorted(indices)[-20:]}")

        # Check if continuous or scattered
        if max(indices) - min(indices) + 1 == len(indices):
            print(f"  Pattern: Continuous block")
        else:
            print(f"  Pattern: Scattered throughout range")

    return remaining

def generate_recommendations(analysis, remaining_count):
    """Generate recommendations based on analysis."""
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)

    actual_coverage = analysis['actual_coverage_pct']
    docked_count = analysis['disk_in_target']

    print(f"\nCurrent Status:")
    print(f"  ✅ {docked_count} samples docked ({actual_coverage:.1f}% coverage)")
    print(f"  ⏳ {remaining_count} samples remaining")
    print(f"  🎯 Need 3868 samples for 95% threshold → {3868 - docked_count} more needed")

    print(f"\nOptions:")

    print(f"\n1️⃣  Use Existing Samples ({actual_coverage:.1f}% coverage)")
    print(f"   • Pros: Can start training immediately")
    print(f"   • Cons: May reduce model performance")
    if actual_coverage < 50:
        print(f"   • ⚠️  Coverage is low - may significantly impact results")
    else:
        print(f"   • ✅ Coverage >50% - may be acceptable for initial experiments")

    print(f"\n2️⃣  Continue Processing Remaining {remaining_count} Samples")
    print(f"   • Time estimate: ~{remaining_count * 60 / 3600:.1f} hours on GPU0 (1 min/sample)")
    print(f"   • Will reach 100% coverage")
    print(f"   • Recommended if aiming for publication-quality results")

    print(f"\n3️⃣  Lower Threshold to {actual_coverage:.0f}%")
    print(f"   • Modify auto_pipeline.py: REQUIRED_THRESHOLD = {actual_coverage/100:.2f}")
    print(f"   • Start training with current samples")
    print(f"   • Can always rerun with more samples later")

    if analysis['discrepancy_on_disk_not_csv'] > 0:
        print(f"\n4️⃣  Update CSV Records")
        print(f"   • {analysis['discrepancy_on_disk_not_csv']} samples on disk not recorded in CSV")
        print(f"   • Create merged CSV with all docked samples")
        print(f"   • This won't change actual coverage but improves record accuracy")

def main():
    print("="*80)
    print("🔍 DIFFDOCK DOCKING STATUS RECONCILIATION")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Step 1: Find all docked samples
    docked_on_disk = find_docked_samples()

    # Step 2: Load CSV records
    csv_records = load_csv_records()

    # Step 3: Analyze discrepancy
    analysis = analyze_discrepancy(docked_on_disk, csv_records, CSV_PATH)

    # Step 4: Find remaining gaps
    remaining = find_remaining_gaps(docked_on_disk, CSV_PATH)

    # Step 5: Generate recommendations
    generate_recommendations(analysis, len(remaining))

    # Save report
    report = {
        'timestamp': datetime.now().isoformat(),
        'analysis': analysis,
        'docked_samples': len(docked_on_disk),
        'csv_records': len(csv_records),
        'remaining_count': len(remaining),
        'remaining_samples': sorted(list(remaining))[:100]  # First 100 for brevity
    }

    report_path = OUTPUT_DIR / "docking_status_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n💾 Report saved to: {report_path}")

    # Also save remaining samples list
    remaining_path = OUTPUT_DIR / "remaining_samples.txt"
    with open(remaining_path, 'w') as f:
        for sample_id in sorted(remaining):
            f.write(f"{sample_id}\n")

    print(f"💾 Remaining samples list saved to: {remaining_path}")

    print("\n" + "="*80)
    print("✅ RECONCILIATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
