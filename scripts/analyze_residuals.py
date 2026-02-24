#!/usr/bin/env python3
"""
Residual analysis for model error characterization

Generates comprehensive error analysis including:
- Residual distribution plots
- Error vs true value analysis
- Error vs predicted value analysis
- Outlier identification

Usage:
    python scripts/analyze_residuals.py \
        --predictions results/diffdock_trained_auto_test/test_predictions.csv \
        --output_dir paperwriting/figures
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10


def load_predictions(pred_csv):
    """Load predictions CSV"""
    df = pd.read_csv(pred_csv)

    # Compute residuals
    df['residual'] = df['predicted'] - df['true']
    df['abs_residual'] = np.abs(df['residual'])

    return df


def plot_residual_distribution(df, output_dir):
    """
    Plot residual distribution with histogram and KDE

    Figure S8: Residual Distribution
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    residuals = df['residual'].values

    # (A) Histogram with normal fit
    ax = axes[0]

    # Histogram
    n, bins, patches = ax.hist(residuals, bins=50, density=True,
                               color='#93c5fd', edgecolor='black',
                               linewidth=0.5, alpha=0.7, label='Residuals')

    # Fit normal distribution
    mu, sigma = stats.norm.fit(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
           label=f'Normal fit\nμ={mu:.3f}, σ={sigma:.3f}')

    # Add vertical line at zero
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2,
              alpha=0.5, label='Zero error')

    ax.set_xlabel('Residual (Predicted - True)', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('(A) Residual Distribution', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (B) Q-Q plot
    ax = axes[1]

    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('(B) Q-Q Plot', fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'figS8_residual_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figS8_residual_distribution.pdf', bbox_inches='tight')
    plt.close()

    print("✅ Figure S8: Residual distribution saved")


def plot_error_vs_magnitude(df, output_dir):
    """
    Plot error vs true value to check for heteroscedasticity

    Figure S9: Error vs Magnitude
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (A) Residual vs True Value
    ax = axes[0]

    ax.scatter(df['true'], df['residual'], alpha=0.5, s=20,
              edgecolors='black', linewidth=0.3)

    # Add zero line
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)

    # Add ±1 log unit lines
    ax.axhline(y=1, color='orange', linestyle=':', linewidth=1.5, alpha=0.5,
              label='±1 log unit')
    ax.axhline(y=-1, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)

    # Add ±2 log units
    ax.axhline(y=2, color='red', linestyle=':', linewidth=1, alpha=0.3,
              label='±2 log units')
    ax.axhline(y=-2, color='red', linestyle=':', linewidth=1, alpha=0.3)

    ax.set_xlabel('True log$_{10}$($k_{cat}$)', fontweight='bold')
    ax.set_ylabel('Residual (Predicted - True)', fontweight='bold')
    ax.set_title('(A) Residual vs True Value', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (B) Absolute Error vs True Value
    ax = axes[1]

    # Hexbin for density visualization
    hb = ax.hexbin(df['true'], df['abs_residual'],
                   gridsize=30, cmap='Blues', mincnt=1, edgecolors='black',
                   linewidths=0.2)

    # Add median line (running median)
    sorted_idx = np.argsort(df['true'].values)
    sorted_true = df['true'].values[sorted_idx]
    sorted_abs_res = df['abs_residual'].values[sorted_idx]

    # Compute running median with window size 50
    window = 50
    running_median = []
    running_x = []

    for i in range(len(sorted_true) - window):
        running_median.append(np.median(sorted_abs_res[i:i+window]))
        running_x.append(np.mean(sorted_true[i:i+window]))

    ax.plot(running_x, running_median, 'r-', linewidth=2,
           label='Running median', zorder=10)

    ax.set_xlabel('True log$_{10}$($k_{cat}$)', fontweight='bold')
    ax.set_ylabel('Absolute Error', fontweight='bold')
    ax.set_title('(B) Absolute Error vs True Value', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add colorbar
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label('Count', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'figS9_error_vs_magnitude.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figS9_error_vs_magnitude.pdf', bbox_inches='tight')
    plt.close()

    print("✅ Figure S9: Error vs magnitude saved")


def identify_outliers(df, threshold=2.0):
    """
    Identify outliers with |residual| > threshold

    Returns:
        DataFrame of outliers
    """
    outliers = df[df['abs_residual'] > threshold].copy()
    outliers = outliers.sort_values('abs_residual', ascending=False)

    return outliers


def plot_error_distribution_by_range(df, output_dir):
    """
    Plot error distribution for different kcat ranges

    Figure S10: Error by kcat Range
    """
    # Define kcat ranges
    bins = [-3, -1, 0, 1, 2, 3, 5]
    labels = [
        '< 0.1 s⁻¹',
        '0.1-1 s⁻¹',
        '1-10 s⁻¹',
        '10-100 s⁻¹',
        '100-1k s⁻¹',
        '> 1k s⁻¹'
    ]

    df['kcat_range'] = pd.cut(df['true'], bins=bins, labels=labels)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (label, group) in enumerate(df.groupby('kcat_range', observed=True)):
        ax = axes[idx]

        # Histogram
        residuals = group['residual'].values

        ax.hist(residuals, bins=30, density=True,
               color='#93c5fd', edgecolor='black', linewidth=0.5, alpha=0.7)

        # Fit normal
        mu, sigma = stats.norm.fit(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2)

        # Add zero line
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # Statistics text
        textstr = f'n = {len(residuals)}\nμ = {mu:.3f}\nσ = {sigma:.3f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', bbox=props)

        ax.set_xlabel('Residual', fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        ax.set_title(f'{label}', fontweight='bold', fontsize=10)
        ax.grid(alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'figS10_error_by_range.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figS10_error_by_range.pdf', bbox_inches='tight')
    plt.close()

    print("✅ Figure S10: Error by kcat range saved")


def print_residual_summary(df):
    """
    Print comprehensive residual analysis summary
    """
    print("\n" + "="*60)
    print("Residual Analysis Summary")
    print("="*60)

    residuals = df['residual'].values
    abs_residuals = df['abs_residual'].values

    # Basic statistics
    print("\n📊 Residual Statistics:")
    print(f"   Mean: {np.mean(residuals):.4f}")
    print(f"   Std Dev: {np.std(residuals):.4f}")
    print(f"   Median: {np.median(residuals):.4f}")
    print(f"   MAD: {np.median(abs_residuals):.4f}")

    # Percentiles
    print("\n📈 Absolute Error Percentiles:")
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(abs_residuals, p)
        print(f"   {p}th: {val:.3f} log units ({10**val:.1f}x)")

    # Normality test
    print("\n🔍 Normality Tests:")
    _, p_shapiro = stats.shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)
    print(f"   Shapiro-Wilk p-value: {p_shapiro:.4f}")

    if p_shapiro > 0.05:
        print("   ✅ Residuals are approximately normal")
    else:
        print("   ⚠️  Residuals deviate from normality")

    # Heteroscedasticity test (Breusch-Pagan)
    # Correlate squared residuals with predictions
    r_hetero, p_hetero = stats.pearsonr(df['predicted'].values, abs_residuals)
    print(f"\n🔍 Heteroscedasticity:")
    print(f"   Correlation(|residual|, prediction): {r_hetero:.4f}")
    print(f"   p-value: {p_hetero:.4f}")

    if p_hetero > 0.05:
        print("   ✅ Homoscedastic (constant variance)")
    else:
        print("   ⚠️  Heteroscedastic (variance depends on magnitude)")

    # Outliers
    outlier_threshold = 2.0
    outliers = df[df['abs_residual'] > outlier_threshold]
    print(f"\n⚠️  Outliers (|error| > {outlier_threshold} log units):")
    print(f"   Count: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")

    if len(outliers) > 0:
        print(f"   Worst case: {outliers['abs_residual'].max():.3f} log units")

    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Residual analysis for JCIM')
    parser.add_argument('--predictions', type=str,
                       default='results/diffdock_trained_auto_test/test_predictions.csv',
                       help='Path to predictions CSV')
    parser.add_argument('--output_dir', type=str,
                       default='paperwriting/figures',
                       help='Output directory for figures')
    parser.add_argument('--outlier_threshold', type=float, default=2.0,
                       help='Threshold for outlier identification (log units)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Residual Analysis for JCIM")
    print("="*60)
    print()

    # Load predictions
    print("📊 Loading predictions...")

    if not Path(args.predictions).exists():
        print(f"⚠️  Predictions file not found: {args.predictions}")
        print("This analysis will run once the model training completes.")
        print("Expected location: results/diffdock_trained_auto_test/test_predictions.csv")
        return

    df = load_predictions(args.predictions)
    print(f"   Loaded {len(df)} predictions")
    print()

    # Print summary
    print_residual_summary(df)

    # Generate figures
    print("📊 Generating figures...")
    print()

    plot_residual_distribution(df, output_dir)
    plot_error_vs_magnitude(df, output_dir)
    plot_error_distribution_by_range(df, output_dir)

    # Identify and save outliers
    outliers = identify_outliers(df, threshold=args.outlier_threshold)
    if len(outliers) > 0:
        outlier_path = output_dir / 'outliers.csv'
        outliers[['sample_id', 'true', 'predicted', 'residual', 'abs_residual']].to_csv(
            outlier_path, index=False
        )
        print(f"💾 Saved {len(outliers)} outliers to: {outlier_path}")

    print()
    print("="*60)
    print("✅ Residual Analysis Complete!")
    print(f"📁 Output directory: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
