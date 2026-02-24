#!/usr/bin/env python3
"""
Analyze model performance by EC class for JCIM supplementary materials

Usage:
    python scripts/analyze_ec_performance.py \
        --predictions results/diffdock_trained_auto_test/test_predictions.csv \
        --dataset data/processed/kcat_test_new_diffdock.pt \
        --output_dir paperwriting/figures
"""
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from collections import defaultdict

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10


def load_predictions(pred_csv):
    """Load predictions CSV"""
    df = pd.read_csv(pred_csv)
    return df


def load_dataset(dataset_pt):
    """Load test dataset to extract EC numbers"""
    data_list = torch.load(dataset_pt)

    # Extract sample_id and EC
    sample_ec_map = {}
    for data in data_list:
        if hasattr(data, 'sample_id') and hasattr(data, 'ec'):
            sample_ec_map[data.sample_id] = data.ec

    return sample_ec_map


def get_ec_class(ec_number):
    """
    Extract EC class from EC number

    Examples:
        3.2.1.1 → 3 (Hydrolases)
        2.7.1.1 → 2 (Transferases)
    """
    if pd.isna(ec_number):
        return 'Unknown'

    try:
        ec_str = str(ec_number)
        ec_class = ec_str.split('.')[0]
        return int(ec_class)
    except:
        return 'Unknown'


def get_ec_class_name(ec_class):
    """Get enzyme class name"""
    EC_NAMES = {
        1: 'Oxidoreductases',
        2: 'Transferases',
        3: 'Hydrolases',
        4: 'Lyases',
        5: 'Isomerases',
        6: 'Ligases',
    }
    return EC_NAMES.get(ec_class, 'Unknown')


def compute_metrics(true_vals, pred_vals):
    """
    Compute performance metrics

    Returns:
        dict with pearson_r, r2, mae, rmse, n_samples
    """
    true_vals = np.array(true_vals)
    pred_vals = np.array(pred_vals)

    # Remove NaNs
    mask = ~(np.isnan(true_vals) | np.isnan(pred_vals))
    true_vals = true_vals[mask]
    pred_vals = pred_vals[mask]

    if len(true_vals) < 5:  # Need at least 5 samples
        return None

    # Pearson correlation
    r, p_value = stats.pearsonr(true_vals, pred_vals)

    # R²
    ss_res = np.sum((true_vals - pred_vals)**2)
    ss_tot = np.sum((true_vals - np.mean(true_vals))**2)
    r2 = 1 - (ss_res / ss_tot)

    # MAE and RMSE
    mae = np.mean(np.abs(true_vals - pred_vals))
    rmse = np.sqrt(np.mean((true_vals - pred_vals)**2))

    return {
        'pearson_r': r,
        'p_value': p_value,
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'n_samples': len(true_vals)
    }


def analyze_by_ec_class(df, sample_ec_map):
    """
    Analyze performance by EC class

    Returns:
        DataFrame with per-EC-class metrics
    """
    # Add EC class to predictions
    df['ec'] = df['sample_id'].map(sample_ec_map)
    df['ec_class'] = df['ec'].apply(get_ec_class)

    # Filter valid EC classes
    df_valid = df[df['ec_class'] != 'Unknown'].copy()

    # Compute metrics per EC class
    ec_results = []

    for ec_class in sorted(df_valid['ec_class'].unique()):
        subset = df_valid[df_valid['ec_class'] == ec_class]

        metrics = compute_metrics(
            subset['true'].values,
            subset['predicted'].values
        )

        if metrics is not None:
            ec_results.append({
                'EC_Class': ec_class,
                'EC_Name': get_ec_class_name(ec_class),
                'N_Samples': metrics['n_samples'],
                'Pearson_r': metrics['pearson_r'],
                'R2': metrics['r2'],
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse'],
                'P_value': metrics['p_value']
            })

    return pd.DataFrame(ec_results)


def plot_ec_performance(ec_df, output_dir):
    """
    Plot per-EC-class performance

    Figure S2: Performance by EC Class
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Sort by EC class
    ec_df = ec_df.sort_values('EC_Class')

    # X-axis labels
    x_labels = [f"EC {row['EC_Class']}\n{row['EC_Name']}"
                for _, row in ec_df.iterrows()]
    x_pos = np.arange(len(x_labels))

    # (A) Pearson correlation
    ax = axes[0]
    bars = ax.bar(x_pos, ec_df['Pearson_r'],
                  color='#93c5fd', edgecolor='black', linewidth=1)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, ec_df['Pearson_r'])):
        ax.text(bar.get_x() + bar.get_width()/2, val,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=8)

    # Add sample counts as secondary text
    for i, (bar, n) in enumerate(zip(bars, ec_df['N_Samples'])):
        ax.text(bar.get_x() + bar.get_width()/2, 0.05,
                f'n={n}',
                ha='center', va='bottom', fontsize=7, color='black')

    ax.set_ylabel('Pearson Correlation (r)', fontweight='bold')
    ax.set_title('(A) Correlation by EC Class', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (B) R²
    ax = axes[1]
    bars = ax.bar(x_pos, ec_df['R2'],
                  color='#fca5a5', edgecolor='black', linewidth=1)

    for bar, val in zip(bars, ec_df['R2']):
        ax.text(bar.get_x() + bar.get_width()/2, val,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('$R^2$', fontweight='bold')
    ax.set_title('(B) $R^2$ by EC Class', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, max(ec_df['R2']) * 1.2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (C) MAE
    ax = axes[2]
    bars = ax.bar(x_pos, ec_df['MAE'],
                  color='#a5f3fc', edgecolor='black', linewidth=1)

    for bar, val in zip(bars, ec_df['MAE']):
        ax.text(bar.get_x() + bar.get_width()/2, val,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('MAE (log$_{10}$ units)', fontweight='bold')
    ax.set_title('(C) Mean Absolute Error by EC Class', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, max(ec_df['MAE']) * 1.3)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'figS2_ec_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figS2_ec_performance.pdf', bbox_inches='tight')
    plt.close()

    print("✅ Figure S2: EC class performance saved")


def plot_ec_scatter(df, sample_ec_map, output_dir):
    """
    Plot scatter plots for each EC class

    Figure S3: Per-EC-Class Predictions
    """
    # Add EC class
    df['ec'] = df['sample_id'].map(sample_ec_map)
    df['ec_class'] = df['ec'].apply(get_ec_class)
    df_valid = df[df['ec_class'] != 'Unknown'].copy()

    ec_classes = sorted(df_valid['ec_class'].unique())
    n_classes = len(ec_classes)

    # Create subplot grid
    n_cols = 3
    n_rows = (n_classes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    axes = axes.flatten() if n_classes > 1 else [axes]

    for idx, ec_class in enumerate(ec_classes):
        ax = axes[idx]
        subset = df_valid[df_valid['ec_class'] == ec_class]

        # Scatter plot
        ax.scatter(subset['true'], subset['predicted'],
                  alpha=0.6, s=30, edgecolors='black', linewidth=0.5)

        # Perfect prediction line
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, 'r--', alpha=0.75, linewidth=2, label='Perfect')

        # Compute metrics
        metrics = compute_metrics(subset['true'].values, subset['predicted'].values)

        # Add text box
        if metrics is not None:
            textstr = f"$r$ = {metrics['pearson_r']:.3f}\n$R^2$ = {metrics['r2']:.3f}\nMAE = {metrics['mae']:.2f}\nn = {metrics['n_samples']}"
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', bbox=props)

        ax.set_xlabel('True log$_{10}$($k_{cat}$)', fontweight='bold')
        ax.set_ylabel('Predicted log$_{10}$($k_{cat}$)', fontweight='bold')
        ax.set_title(f'EC {ec_class}: {get_ec_class_name(ec_class)}',
                    fontweight='bold', fontsize=10)
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(loc='lower right')

    # Hide unused subplots
    for idx in range(n_classes, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'figS3_ec_scatter.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figS3_ec_scatter.pdf', bbox_inches='tight')
    plt.close()

    print("✅ Figure S3: EC class scatter plots saved")


def main():
    parser = argparse.ArgumentParser(description='Analyze model performance by EC class')
    parser.add_argument('--predictions', type=str,
                       default='results/diffdock_trained_auto_test/test_predictions.csv',
                       help='Path to predictions CSV')
    parser.add_argument('--dataset', type=str,
                       default='data/processed/kcat_test_new_diffdock.pt',
                       help='Path to test dataset .pt file (for EC numbers)')
    parser.add_argument('--output_dir', type=str,
                       default='paperwriting/figures',
                       help='Output directory for figures')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("EC Class Performance Analysis for JCIM")
    print("="*60)
    print()

    # Load data
    print("📊 Loading predictions and dataset...")

    if not Path(args.predictions).exists():
        print(f"⚠️  Predictions file not found: {args.predictions}")
        print("This analysis will run once the model training completes.")
        print("Expected location: results/diffdock_trained_auto_test/test_predictions.csv")
        return

    df = load_predictions(args.predictions)
    print(f"   Loaded {len(df)} predictions")

    sample_ec_map = load_dataset(args.dataset)
    print(f"   Loaded EC numbers for {len(sample_ec_map)} samples")
    print()

    # Analyze by EC class
    print("🔍 Analyzing performance by EC class...")
    ec_df = analyze_by_ec_class(df, sample_ec_map)
    print(ec_df.to_string(index=False))
    print()

    # Save results table
    ec_df.to_csv(output_dir / 'ec_class_performance.csv', index=False)
    print(f"💾 Saved results to: {output_dir / 'ec_class_performance.csv'}")
    print()

    # Generate figures
    print("📊 Generating figures...")
    plot_ec_performance(ec_df, output_dir)
    plot_ec_scatter(df, sample_ec_map, output_dir)
    print()

    print("="*60)
    print("✅ EC Class Analysis Complete!")
    print(f"📁 Output directory: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
