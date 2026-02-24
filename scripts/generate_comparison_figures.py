#!/usr/bin/env python3
"""
Generate comprehensive comparison figures for DiffDock vs Vina
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def generate_comparison_figure():
    """Generate comparison figure from experimental results"""

    # Load results
    vina_df = pd.read_csv("results/docking_rmsd_test/vina_rmsd_results.csv")
    diffdock_df = pd.read_csv("results/diffdock_rmsd_test/diffdock_rmsd_results.csv")

    # Merge
    merged = vina_df.merge(diffdock_df, on='complex_id', suffixes=('_vina', '_diffdock'))

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Bar plot of RMSD comparison
    complexes = merged['complex_id'].tolist()
    x = np.arange(len(complexes))
    width = 0.35

    vina_rmsd = merged['vina_rmsd'].values
    dd_rmsd = merged['diffdock_rmsd'].values

    axes[0, 0].bar(x - width/2, vina_rmsd, width, label='AutoDock Vina', color='steelblue', edgecolor='black')
    axes[0, 0].bar(x + width/2, dd_rmsd, width, label='DiffDock', color='coral', edgecolor='black')
    axes[0, 0].axhline(2.0, color='red', linestyle='--', linewidth=2, label='Success threshold (2.0Å)')
    axes[0, 0].set_ylabel('RMSD (Å)', fontsize=12)
    axes[0, 0].set_xlabel('Complex ID', fontsize=12)
    axes[0, 0].set_title('RMSD Comparison by Complex', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([c.split('_')[0] for c in complexes], rotation=45, ha='right')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(axis='y', alpha=0.3)

    # 2. Success rate comparison
    vina_success = (vina_rmsd < 2.0).sum() / len(vina_rmsd) * 100
    dd_success = (dd_rmsd < 2.0).sum() / len(dd_rmsd) * 100

    methods = ['AutoDock Vina', 'DiffDock']
    success_rates = [vina_success, dd_success]
    colors = ['steelblue', 'coral']

    bars = axes[0, 1].bar(methods, success_rates, color=colors, edgecolor='black', width=0.6)
    axes[0, 1].set_ylabel('Success Rate (%)', fontsize=12)
    axes[0, 1].set_title('Docking Success Rate (RMSD < 2.0Å)', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylim(0, 110)
    for bar, rate in zip(bars, success_rates):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                       f'{rate:.0f}%', ha='center', fontsize=14, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # 3. Mean RMSD comparison
    vina_mean = np.mean(vina_rmsd)
    vina_std = np.std(vina_rmsd)
    dd_mean = np.mean(dd_rmsd)
    dd_std = np.std(dd_rmsd)

    bars = axes[1, 0].bar(methods, [vina_mean, dd_mean], yerr=[vina_std, dd_std],
                          color=colors, edgecolor='black', width=0.6, capsize=10)
    axes[1, 0].axhline(2.0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_ylabel('Mean RMSD (Å)', fontsize=12)
    axes[1, 0].set_title('Mean RMSD Comparison', fontsize=14, fontweight='bold')
    for bar, mean, std in zip(bars, [vina_mean, dd_mean], [vina_std, dd_std]):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.1,
                       f'{mean:.2f}±{std:.2f}Å', ha='center', fontsize=11)
    axes[1, 0].grid(axis='y', alpha=0.3)

    # 4. Scatter plot
    axes[1, 1].scatter(vina_rmsd, dd_rmsd, s=150, c='purple', edgecolors='black', alpha=0.7)
    max_val = max(max(vina_rmsd), max(dd_rmsd)) + 0.5
    axes[1, 1].plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x')
    axes[1, 1].axhline(2.0, color='red', linestyle=':', alpha=0.7)
    axes[1, 1].axvline(2.0, color='red', linestyle=':', alpha=0.7)
    axes[1, 1].fill_between([0, 2], [0, 0], [2, 2], alpha=0.1, color='green', label='Both succeed')
    axes[1, 1].set_xlabel('Vina RMSD (Å)', fontsize=12)
    axes[1, 1].set_ylabel('DiffDock RMSD (Å)', fontsize=12)
    axes[1, 1].set_title('DiffDock vs Vina RMSD Scatter', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlim(0, max_val)
    axes[1, 1].set_ylim(0, max_val)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(alpha=0.3)

    # Add complex labels to scatter
    for i, cid in enumerate(complexes):
        axes[1, 1].annotate(cid.split('_')[0], (vina_rmsd[i], dd_rmsd[i]),
                           textcoords="offset points", xytext=(5, 5), fontsize=9)

    plt.tight_layout()

    # Save
    output_dir = Path("results/docking_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "diffdock_vs_vina_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {fig_path}")
    plt.close()

    # Print summary
    print("\n" + "=" * 60)
    print("DOCKING METHOD COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\nPoseBench Astex Diverse Set (n={len(complexes)})")
    print("\n                   Vina        DiffDock")
    print(f"Success Rate:     {vina_success:.0f}%         {dd_success:.0f}%")
    print(f"Mean RMSD:        {vina_mean:.2f}Å       {dd_mean:.2f}Å")
    print(f"Std RMSD:         {vina_std:.2f}Å       {dd_std:.2f}Å")

    # Per-complex table
    print("\nPer-complex RMSD (Å):")
    print("-" * 45)
    print(f"{'Complex':<12} {'Vina':<10} {'DiffDock':<10} {'Winner':<10}")
    print("-" * 45)
    for i, row in merged.iterrows():
        cid = row['complex_id'].split('_')[0]
        v = row['vina_rmsd']
        d = row['diffdock_rmsd']
        winner = 'Vina' if v < d else 'DiffDock' if d < v else 'Tie'
        print(f"{cid:<12} {v:<10.2f} {d:<10.2f} {winner:<10}")
    print("-" * 45)

    vina_wins = sum(1 for _, r in merged.iterrows() if r['vina_rmsd'] < r['diffdock_rmsd'])
    dd_wins = len(merged) - vina_wins
    print(f"\nVina wins: {vina_wins}/{len(merged)}, DiffDock wins: {dd_wins}/{len(merged)}")

    return merged

if __name__ == "__main__":
    generate_comparison_figure()
