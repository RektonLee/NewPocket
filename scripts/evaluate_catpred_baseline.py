#!/usr/bin/env python3
"""Evaluate CatPred baseline performance and compare with PocketGNN."""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load CatPred predictions
catpred_df = pd.read_csv('results/baseline_comparison/catpred_output.csv')
print(f"CatPred predictions: {len(catpred_df)} samples")
print(f"Columns: {catpred_df.columns.tolist()}")

# Load ground truth from test results
test_results_df = pd.read_csv('results/baseline_comparison/kcat_test_results.csv')
print(f"\nGround truth: {len(test_results_df)} samples")

# Merge predictions with ground truth
merged_df = catpred_df.merge(
    test_results_df[['sample_id', 'experimental value[log10]']],
    on='sample_id',
    how='inner'
)
print(f"\nMerged dataset: {len(merged_df)} samples")

# Extract predictions and ground truth
y_true = merged_df['experimental value[log10]'].values
y_pred = merged_df['log10kcat_max'].values

# Remove any NaN values
valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
y_true = y_true[valid_mask]
y_pred = y_pred[valid_mask]
print(f"Valid samples (after removing NaN): {len(y_true)}")

# Calculate metrics
r2 = r2_score(y_true, y_pred)
pearson_r, pearson_p = pearsonr(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print("\n" + "="*60)
print("CatPred Baseline Performance")
print("="*60)
print(f"R² Score:        {r2:.4f}")
print(f"Pearson r:       {pearson_r:.4f} (p={pearson_p:.2e})")
print(f"MAE:             {mae:.4f}")
print(f"RMSE:            {rmse:.4f}")
print("="*60)

# Load PocketGNN results for comparison
pocketgnn_results = {
    'Model': 'PocketGNN (kcat_esm_norm_fix)',
    'R²': 0.4432,
    'Pearson': 0.6728,
    'MAE': 0.6890,
    'RMSE': 0.9348
}

# Load CataPro results
catapro_results = {
    'Model': 'CataPro',
    'R²': 0.4444,
    'Pearson': 0.6827,
    'MAE': 0.6838,
    'RMSE': 0.9332
}

print("\n" + "="*60)
print("Comparison with Other Methods")
print("="*60)
print(f"{'Method':<30} {'R²':>10} {'Pearson':>10} {'MAE':>10} {'RMSE':>10}")
print("-"*60)
print(f"{'CatPred (This Run)':<30} {r2:>10.4f} {pearson_r:>10.4f} {mae:>10.4f} {rmse:>10.4f}")
print(f"{catapro_results['Model']:<30} {catapro_results['R²']:>10.4f} {catapro_results['Pearson']:>10.4f} {catapro_results['MAE']:>10.4f} {catapro_results['RMSE']:>10.4f}")
print(f"{pocketgnn_results['Model']:<30} {pocketgnn_results['R²']:>10.4f} {pocketgnn_results['Pearson']:>10.4f} {pocketgnn_results['MAE']:>10.4f} {pocketgnn_results['RMSE']:>10.4f}")
print("="*60)

# Save detailed results
results_dict = {
    'method': 'CatPred',
    'n_samples': len(y_true),
    'r2': r2,
    'pearson_r': pearson_r,
    'pearson_p': pearson_p,
    'mae': mae,
    'rmse': rmse,
    'y_true_mean': np.mean(y_true),
    'y_true_std': np.std(y_true),
    'y_pred_mean': np.mean(y_pred),
    'y_pred_std': np.std(y_pred),
}

# Save to JSON
import json
with open('results/baseline_comparison/catpred_metrics.json', 'w') as f:
    json.dump(results_dict, f, indent=2)
print(f"\nMetrics saved to: results/baseline_comparison/catpred_metrics.json")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot
ax = axes[0]
ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
        'r--', lw=2, label='Perfect prediction')
ax.set_xlabel('Experimental log10(kcat)', fontsize=12)
ax.set_ylabel('CatPred Predicted log10(kcat)', fontsize=12)
ax.set_title(f'CatPred: True vs Predicted\nR²={r2:.4f}, Pearson r={pearson_r:.4f}', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)

# Residual plot
ax = axes[1]
residuals = y_pred - y_true
ax.scatter(y_true, residuals, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('Experimental log10(kcat)', fontsize=12)
ax.set_ylabel('Residual (Predicted - True)', fontsize=12)
ax.set_title(f'Residual Plot\nMAE={mae:.4f}, RMSE={rmse:.4f}', fontsize=13)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/baseline_comparison/catpred_prediction_analysis.png', dpi=300, bbox_inches='tight')
print(f"Visualization saved to: results/baseline_comparison/catpred_prediction_analysis.png")

# Create comparison bar chart
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

methods = ['CatPred', 'CataPro', 'PocketGNN']
metrics_data = {
    'R²': [r2, catapro_results['R²'], pocketgnn_results['R²']],
    'Pearson r': [pearson_r, catapro_results['Pearson'], pocketgnn_results['Pearson']],
    'MAE': [mae, catapro_results['MAE'], pocketgnn_results['MAE']],
    'RMSE': [rmse, catapro_results['RMSE'], pocketgnn_results['RMSE']]
}

for idx, (metric_name, values) in enumerate(metrics_data.items()):
    ax = axes[idx // 2, idx % 2]
    bars = ax.bar(methods, values, color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.8, edgecolor='black')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} Comparison', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Highlight best performance
    best_idx = np.argmax(values) if metric_name in ['R²', 'Pearson r'] else np.argmin(values)
    bars[best_idx].set_edgecolor('green')
    bars[best_idx].set_linewidth(3)

plt.suptitle('CatPred vs PocketGNN vs CataPro\nBaseline Comparison on kcat Test Set',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('results/baseline_comparison/catpred_comparison_barplot.png', dpi=300, bbox_inches='tight')
print(f"Comparison plot saved to: results/baseline_comparison/catpred_comparison_barplot.png")

print("\n✅ CatPred baseline evaluation completed!")
