#!/usr/bin/env python3
"""
Analyze training curves and generate supplementary figures for JCIM

Usage:
    python scripts/analyze_training_curves.py \
        --exp_dir experiments/kcat_diffdock_20260225_*/run_* \
        --output_dir paperwriting/figures
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import glob

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10


def find_latest_experiment(pattern='experiments/kcat_diffdock_*/run_*'):
    """
    Find the latest experiment directory

    Returns:
        Path object of latest experiment
    """
    exp_dirs = sorted(glob.glob(pattern))

    if not exp_dirs:
        return None

    # Sort by modification time
    exp_dirs_with_time = [(Path(d), Path(d).stat().st_mtime) for d in exp_dirs]
    exp_dirs_with_time.sort(key=lambda x: x[1], reverse=True)

    return exp_dirs_with_time[0][0]


def load_training_log(exp_dir):
    """
    Load training metrics from experiment directory

    Expected format:
        - epoch_metrics.csv: epoch,train_loss,val_loss,val_pearson,val_r2,...
        OR
        - training_log.txt: text file with metrics
    """
    exp_dir = Path(exp_dir)

    # Try CSV format first
    csv_path = exp_dir / 'epoch_metrics.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df

    # Try JSON format
    json_path = exp_dir / 'training_metrics.json'
    if json_path.exists():
        with open(json_path) as f:
            metrics = json.load(f)

        # Convert to DataFrame
        df = pd.DataFrame(metrics)
        return df

    # Try loading from WandB logs (if saved locally)
    wandb_path = exp_dir / 'wandb_metrics.csv'
    if wandb_path.exists():
        df = pd.read_csv(wandb_path)
        return df

    return None


def plot_loss_curves(df, output_dir):
    """
    Plot training and validation loss curves

    Figure S4: Training Curves
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = df['epoch'].values

    # Plot losses
    ax.plot(epochs, df['train_loss'], label='Training Loss',
           color='#3b82f6', linewidth=2)
    ax.plot(epochs, df['val_loss'], label='Validation Loss',
           color='#ef4444', linewidth=2)

    # Mark best epoch
    best_epoch = df['val_loss'].idxmin()
    best_val_loss = df['val_loss'].iloc[best_epoch]

    ax.scatter([epochs[best_epoch]], [best_val_loss],
              color='red', s=100, zorder=5, marker='*',
              label=f'Best (Epoch {epochs[best_epoch]})')

    # Add vertical line at best epoch
    ax.axvline(x=epochs[best_epoch], color='red', linestyle='--',
              alpha=0.5, linewidth=1)

    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss (MSE)', fontweight='bold')
    ax.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'figS4_loss_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figS4_loss_curves.pdf', bbox_inches='tight')
    plt.close()

    print("✅ Figure S4: Training curves saved")


def plot_metrics_curves(df, output_dir):
    """
    Plot Pearson r and R² curves

    Figure S5: Validation Metrics
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    epochs = df['epoch'].values

    # (A) Pearson correlation
    ax = axes[0]
    ax.plot(epochs, df['val_pearson'], label='Validation Pearson r',
           color='#3b82f6', linewidth=2)

    # Mark best
    best_epoch = df['val_pearson'].idxmax()
    best_r = df['val_pearson'].iloc[best_epoch]
    ax.scatter([epochs[best_epoch]], [best_r],
              color='red', s=100, zorder=5, marker='*',
              label=f'Best (Epoch {epochs[best_epoch]})')
    ax.axvline(x=epochs[best_epoch], color='red', linestyle='--',
              alpha=0.5, linewidth=1)

    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Pearson Correlation (r)', fontweight='bold')
    ax.set_title('(A) Validation Pearson Correlation', fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (B) R²
    ax = axes[1]
    ax.plot(epochs, df['val_r2'], label='Validation $R^2$',
           color='#ef4444', linewidth=2)

    # Mark best
    best_epoch = df['val_r2'].idxmax()
    best_r2 = df['val_r2'].iloc[best_epoch]
    ax.scatter([epochs[best_epoch]], [best_r2],
              color='red', s=100, zorder=5, marker='*',
              label=f'Best (Epoch {epochs[best_epoch]})')
    ax.axvline(x=epochs[best_epoch], color='red', linestyle='--',
              alpha=0.5, linewidth=1)

    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('$R^2$', fontweight='bold')
    ax.set_title('(B) Validation $R^2$', fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'figS5_metrics_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figS5_metrics_curves.pdf', bbox_inches='tight')
    plt.close()

    print("✅ Figure S5: Validation metrics curves saved")


def plot_learning_rate_schedule(df, output_dir):
    """
    Plot learning rate schedule

    Figure S6: Learning Rate Schedule
    """
    if 'learning_rate' not in df.columns:
        print("⚠️  Learning rate not logged, skipping LR schedule plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = df['epoch'].values
    lr = df['learning_rate'].values

    ax.plot(epochs, lr, color='#3b82f6', linewidth=2)

    # Mark LR reductions
    lr_changes = np.where(np.diff(lr) < 0)[0]
    for idx in lr_changes:
        ax.axvline(x=epochs[idx+1], color='red', linestyle='--',
                  alpha=0.5, linewidth=1)
        ax.text(epochs[idx+1], lr[idx+1], f'  {lr[idx+1]:.2e}',
               verticalalignment='bottom', fontsize=8, color='red')

    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Learning Rate', fontweight='bold')
    ax.set_title('Learning Rate Schedule (ReduceLROnPlateau)',
                fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'figS6_lr_schedule.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figS6_lr_schedule.pdf', bbox_inches='tight')
    plt.close()

    print("✅ Figure S6: Learning rate schedule saved")


def plot_overfitting_analysis(df, output_dir):
    """
    Plot train vs validation loss to analyze overfitting

    Figure S7: Overfitting Analysis
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = df['epoch'].values
    train_loss = df['train_loss'].values
    val_loss = df['val_loss'].values

    # Compute train-val gap
    gap = val_loss - train_loss

    # Plot on dual y-axis
    ax.plot(epochs, train_loss, label='Training Loss',
           color='#3b82f6', linewidth=2)
    ax.plot(epochs, val_loss, label='Validation Loss',
           color='#ef4444', linewidth=2)

    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss (MSE)', fontweight='bold')
    ax.set_title('Overfitting Analysis (Train vs Validation Loss)',
                fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add secondary axis for gap
    ax2 = ax.twinx()
    ax2.plot(epochs, gap, label='Val-Train Gap',
            color='#f59e0b', linewidth=2, linestyle='--', alpha=0.7)
    ax2.set_ylabel('Validation - Training Loss', fontweight='bold', color='#f59e0b')
    ax2.tick_params(axis='y', labelcolor='#f59e0b')
    ax2.spines['top'].set_visible(False)
    ax2.legend(loc='upper left', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_dir / 'figS7_overfitting.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figS7_overfitting.pdf', bbox_inches='tight')
    plt.close()

    print("✅ Figure S7: Overfitting analysis saved")


def print_training_summary(df, exp_dir):
    """
    Print training summary statistics
    """
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)

    print(f"\n📁 Experiment: {exp_dir.name}")
    print(f"📊 Total Epochs: {len(df)}")

    # Best epoch
    best_epoch = df['val_loss'].idxmin()
    print(f"\n🏆 Best Epoch: {df['epoch'].iloc[best_epoch]}")
    print(f"   Val Loss: {df['val_loss'].iloc[best_epoch]:.4f}")

    if 'val_pearson' in df.columns:
        print(f"   Val Pearson r: {df['val_pearson'].iloc[best_epoch]:.4f}")
    if 'val_r2' in df.columns:
        print(f"   Val R²: {df['val_r2'].iloc[best_epoch]:.4f}")
    if 'val_mae' in df.columns:
        print(f"   Val MAE: {df['val_mae'].iloc[best_epoch]:.4f}")

    # Final epoch
    print(f"\n📌 Final Epoch: {df['epoch'].iloc[-1]}")
    print(f"   Val Loss: {df['val_loss'].iloc[-1]:.4f}")

    if 'val_pearson' in df.columns:
        print(f"   Val Pearson r: {df['val_pearson'].iloc[-1]:.4f}")

    # Convergence analysis
    val_loss = df['val_loss'].values
    if len(val_loss) > 20:
        recent_std = np.std(val_loss[-20:])
        print(f"\n📈 Recent Stability (last 20 epochs):")
        print(f"   Std Dev: {recent_std:.6f}")

        if recent_std < 0.001:
            print("   ✅ Model converged (very stable)")
        elif recent_std < 0.01:
            print("   ✅ Model mostly converged")
        else:
            print("   ⚠️  Model still learning (high variance)")

    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze training curves for JCIM')
    parser.add_argument('--exp_dir', type=str, default=None,
                       help='Path to experiment directory (auto-detect if not specified)')
    parser.add_argument('--output_dir', type=str,
                       default='paperwriting/figures',
                       help='Output directory for figures')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Training Curve Analysis for JCIM")
    print("="*60)
    print()

    # Find experiment directory
    if args.exp_dir is None:
        print("🔍 Auto-detecting latest experiment...")
        exp_dir = find_latest_experiment()

        if exp_dir is None:
            print("⚠️  No experiment directory found.")
            print("Expected location: experiments/kcat_diffdock_*/run_*")
            print("This analysis will run once the model training completes.")
            return

        print(f"   Found: {exp_dir}")
    else:
        exp_dir = Path(args.exp_dir)

    if not exp_dir.exists():
        print(f"❌ Experiment directory not found: {exp_dir}")
        return

    print()

    # Load training log
    print("📊 Loading training metrics...")
    df = load_training_log(exp_dir)

    if df is None:
        print("❌ Could not find training metrics in experiment directory")
        print("Expected files:")
        print("  - epoch_metrics.csv")
        print("  - training_metrics.json")
        print("  - wandb_metrics.csv")
        return

    print(f"   Loaded {len(df)} epochs of training data")
    print()

    # Print summary
    print_training_summary(df, exp_dir)

    # Generate figures
    print("📊 Generating figures...")
    print()

    plot_loss_curves(df, output_dir)
    plot_metrics_curves(df, output_dir)
    plot_learning_rate_schedule(df, output_dir)
    plot_overfitting_analysis(df, output_dir)

    print()
    print("="*60)
    print("✅ Training Curve Analysis Complete!")
    print(f"📁 Output directory: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
