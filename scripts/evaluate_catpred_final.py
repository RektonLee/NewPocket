#!/usr/bin/env python3
"""Extract ground truth from test dataset and evaluate CatPred."""

import sys
sys.path.insert(0, 'src')

import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json

# Load test dataset
print("Loading test dataset...")
test_data = torch.load('data/processed/kcat_test_new.pt', weights_only=False)
print(f"Test dataset: {len(test_data)} samples")

# Extract ground truth
ground_truth = []
for data in test_data:
    sample_id = data.sample_id
    y_true = float(data.y.item())
    ground_truth.append({'sample_id': sample_id, 'experimental_value_log10': y_true})

gt_df = pd.DataFrame(ground_truth)
print(f"Ground truth extracted: {len(gt_df)} samples")

# Load CatPred predictions
catpred_df = pd.read_csv('results/baseline_comparison/catpred_output.csv')
print(f"CatPred predictions: {len(catpred_df)} samples")

# Merge
merged_df = catpred_df.merge(gt_df, on='sample_id', how='inner')
print(f"Merged dataset: {len(merged_df)} samples")

# Extract predictions and ground truth
y_true = merged_df['experimental_value_log10'].values
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

# Determine ranking
methods = ['CatPred', 'CataPro', 'PocketGNN']
r2_values = [r2, catapro_results['R²'], pocketgnn_results['R²']]
pearson_values = [pearson_r, catapro_results['Pearson'], pocketgnn_results['Pearson']]

print("\n" + "="*60)
print("Performance Ranking (by R²)")
print("="*60)
sorted_idx = np.argsort(r2_values)[::-1]
for rank, idx in enumerate(sorted_idx, 1):
    print(f"{rank}. {methods[idx]:<30} R²={r2_values[idx]:.4f}")

print("\n" + "="*60)
print("Performance Ranking (by Pearson r)")
print("="*60)
sorted_idx = np.argsort(pearson_values)[::-1]
for rank, idx in enumerate(sorted_idx, 1):
    print(f"{rank}. {methods[idx]:<30} Pearson={pearson_values[idx]:.4f}")
print("="*60)

# Save detailed results
results_dict = {
    'method': 'CatPred',
    'n_samples': int(len(y_true)),
    'r2': float(r2),
    'pearson_r': float(pearson_r),
    'pearson_p': float(pearson_p),
    'mae': float(mae),
    'rmse': float(rmse),
    'y_true_mean': float(np.mean(y_true)),
    'y_true_std': float(np.std(y_true)),
    'y_pred_mean': float(np.mean(y_pred)),
    'y_pred_std': float(np.std(y_pred)),
}

# Save to JSON
with open('results/baseline_comparison/catpred_metrics.json', 'w') as f:
    json.dump(results_dict, f, indent=2)
print(f"\nMetrics saved to: results/baseline_comparison/catpred_metrics.json")

# Save comparison results
comparison_dict = {
    'CatPred': {
        'R²': float(r2),
        'Pearson': float(pearson_r),
        'MAE': float(mae),
        'RMSE': float(rmse),
        'n_samples': int(len(y_true))
    },
    'CataPro': catapro_results,
    'PocketGNN': pocketgnn_results
}

with open('results/baseline_comparison/baseline_comparison.json', 'w') as f:
    json.dump(comparison_dict, f, indent=2)
print(f"Comparison results saved to: results/baseline_comparison/baseline_comparison.json")

# Save predictions with ground truth
pred_df = pd.DataFrame({
    'sample_id': merged_df['sample_id'],
    'experimental_value_log10': y_true,
    'catpred_prediction_log10': y_pred,
    'residual': y_pred - y_true,
    'absolute_error': np.abs(y_pred - y_true)
})
pred_df.to_csv('results/baseline_comparison/catpred_predictions_with_truth.csv', index=False)
print(f"Predictions saved to: results/baseline_comparison/catpred_predictions_with_truth.csv")

print("\n✅ CatPred baseline evaluation completed!")
