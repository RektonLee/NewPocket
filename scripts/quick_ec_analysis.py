#!/usr/bin/env python3
"""
Quick EC-wise performance analysis for JCIM paper.

Analyzes model performance by EC class to show generalization.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
from collections import defaultdict
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from GNN_model import PocketGNNKcatOnly


def analyze_ec_performance(model_path, test_dataset_path, output_dir, device='cuda'):
    """Analyze performance by EC class."""

    # Load model
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    hidden_dim = state_dict['node_encoder.weight'].shape[0]
    num_layers = sum(1 for k in state_dict.keys() if k.startswith('att_layers.') and '.lin.weight' in k)
    heads = state_dict['att_layers.0.att_src'].shape[1] if 'att_layers.0.att_src' in state_dict else 4
    use_mlp_layernorm = len(state_dict['mlp.0.weight'].shape) == 1

    model = PocketGNNKcatOnly(
        node_input_dim=52, edge_input_dim=24,
        hidden_dim=hidden_dim, num_layers=num_layers, heads=heads,
        pooling_type='mean', dropout=0.1, use_seq_embedding=False,
        use_mlp_layernorm=use_mlp_layernorm
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load dataset
    print("Loading dataset...")
    dataset = torch.load(test_dataset_path)

    # Collect predictions by EC class
    ec_data = defaultdict(lambda: {'true': [], 'pred': []})

    with torch.no_grad():
        for data in dataset:
            ec = getattr(data, 'ec', None)
            if ec is None or ec == 'unknown':
                continue

            # Get EC level 1 (first digit)
            ec_str = str(ec)
            ec_level1 = ec_str.split('.')[0] if '.' in ec_str else ec_str

            data_device = data.to(device)
            pred = model(data_device)

            ec_data[ec_level1]['true'].append(data.y.item())
            ec_data[ec_level1]['pred'].append(pred.item())

    # Compute metrics per EC
    results = []
    for ec_class, data in sorted(ec_data.items()):
        true = np.array(data['true'])
        pred = np.array(data['pred'])

        if len(true) < 5:  # Skip if too few samples
            continue

        r, _ = pearsonr(true, pred)
        r2 = 1 - np.sum((true - pred)**2) / np.sum((true - true.mean())**2)
        mae = np.mean(np.abs(true - pred))
        rmse = np.sqrt(np.mean((true - pred)**2))

        results.append({
            'EC': f'EC {ec_class}',
            'N': len(true),
            'Pearson_r': r,
            'R2': r2,
            'MAE': mae,
            'RMSE': rmse
        })

    df = pd.DataFrame(results)

    # Save CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'ec_wise_performance.csv'
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"✅ Saved EC-wise performance: {csv_path}")

    # Print summary
    print("\n" + "="*70)
    print("EC-WISE PERFORMANCE ANALYSIS")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: Pearson r by EC
    ax = axes[0, 0]
    colors = plt.cm.Set2(range(len(df)))
    ax.barh(df['EC'], df['Pearson_r'], color=colors, alpha=0.7)
    ax.axvline(df['Pearson_r'].mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {df["Pearson_r"].mean():.3f}')
    ax.set_xlabel('Pearson r', fontsize=11)
    ax.set_ylabel('EC Class', fontsize=11)
    ax.set_title('A. Correlation by EC Class', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    # Panel B: Sample size
    ax = axes[0, 1]
    ax.barh(df['EC'], df['N'], color=colors, alpha=0.7)
    ax.set_xlabel('Number of Samples', fontsize=11)
    ax.set_ylabel('EC Class', fontsize=11)
    ax.set_title('B. Sample Distribution', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Panel C: MAE by EC
    ax = axes[1, 0]
    ax.barh(df['EC'], df['MAE'], color=colors, alpha=0.7)
    ax.axvline(df['MAE'].mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {df["MAE"].mean():.3f}')
    ax.set_xlabel('MAE (log₁₀ units)', fontsize=11)
    ax.set_ylabel('EC Class', fontsize=11)
    ax.set_title('C. Mean Absolute Error', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    # Panel D: R² by EC
    ax = axes[1, 1]
    ax.barh(df['EC'], df['R2'], color=colors, alpha=0.7)
    ax.axvline(df['R2'].mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {df["R2"].mean():.3f}')
    ax.set_xlabel('R²', fontsize=11)
    ax.set_ylabel('EC Class', fontsize=11)
    ax.set_title('D. R² Score', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / 'figS_ec_wise_performance.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved figure: {fig_path}")

    return df


if __name__ == '__main__':
    df = analyze_ec_performance(
        model_path='outputs/kcat_hom40_train_20251213_221248/best_model.pt',
        test_dataset_path='data/processed/kcat_merged_hom40_test.pt',
        output_dir=Path('results/ec_analysis'),
        device='cuda'
    )
