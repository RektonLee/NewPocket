#!/usr/bin/env python3
"""
Quick test script for kcat_test_new.csv
Tests the best model and generates comprehensive results
Created: 2026-02-24
"""

import sys
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Add src to path
sys.path.insert(0, 'src')

from GNN_model import PocketGNNKcatOnly
from torch_geometric.loader import DataLoader

def load_model(checkpoint_path, device='cuda'):
    """Load trained model"""
    print(f"Loading model from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model config from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Initialize model with default parameters
    model = PocketGNNKcatOnly(
        node_input_dim=52,
        edge_input_dim=24,
        hidden_dim=128,
        num_layers=3,
        heads=4,
        pooling_type='mean',
        use_seq_embedding=False,
        dropout=0.1
    )

    # Load state dict
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    return model

def load_test_data(dataset_path):
    """Load test dataset"""
    print(f"Loading test data from: {dataset_path}")
    data_list = torch.load(dataset_path)
    print(f"✓ Loaded {len(data_list)} samples")
    return data_list

def run_inference(model, data_list, device='cuda', batch_size=32):
    """Run inference on test data"""
    print(f"Running inference on {len(data_list)} samples...")

    # Create dataloader
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Forward pass
            output = model(batch)

            # Store results
            all_preds.append(output.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())

    # Concatenate
    predictions = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    print(f"✓ Inference complete")
    return predictions, labels

def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pearson_r, pearson_p = pearsonr(y_true, y_pred)

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Pearson_r': pearson_r,
        'Pearson_p': pearson_p
    }

    return metrics

def generate_report(predictions, labels, output_dir='results/test_kcat_new'):
    """Generate comprehensive test report"""
    os.makedirs(output_dir, exist_ok=True)

    # Compute metrics
    metrics = compute_metrics(labels, predictions)

    # Print results
    print("\n" + "="*60)
    print("TEST RESULTS ON kcat_test_new.csv")
    print("="*60)
    print(f"Number of samples: {len(labels)}")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"R²: {metrics['R2']:.4f}")
    print(f"Pearson r: {metrics['Pearson_r']:.4f} (p={metrics['Pearson_p']:.2e})")
    print("="*60 + "\n")

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = f"{output_dir}/test_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✓ Metrics saved to: {metrics_path}")

    # Save predictions
    results_df = pd.DataFrame({
        'true_log10_kcat': labels,
        'pred_log10_kcat': predictions,
        'error': predictions - labels,
        'abs_error': np.abs(predictions - labels)
    })
    results_path = f"{output_dir}/predictions.csv"
    results_df.to_csv(results_path, index=False)
    print(f"✓ Predictions saved to: {results_path}")

    # Generate figures
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Scatter plot
    axes[0, 0].scatter(labels, predictions, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
    axes[0, 0].plot([labels.min(), labels.max()], [labels.min(), labels.max()],
                    'r--', lw=2, label='Perfect prediction')
    axes[0, 0].set_xlabel('True log₁₀(kcat)', fontsize=12)
    axes[0, 0].set_ylabel('Predicted log₁₀(kcat)', fontsize=12)
    axes[0, 0].set_title(f'Test Set Performance\n(R²={metrics["R2"]:.3f}, r={metrics["Pearson_r"]:.3f})',
                         fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Error distribution
    errors = predictions - labels
    axes[0, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    axes[0, 1].axvline(errors.mean(), color='green', linestyle='--',
                      label=f'Mean: {errors.mean():.3f}')
    axes[0, 1].set_xlabel('Prediction Error (log₁₀ units)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title(f'Error Distribution\n(MAE={metrics["MAE"]:.3f}, RMSE={metrics["RMSE"]:.3f})',
                         fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Residual plot
    axes[1, 0].scatter(predictions, errors, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
    axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Predicted log₁₀(kcat)', fontsize=12)
    axes[1, 0].set_ylabel('Residual Error', fontsize=12)
    axes[1, 0].set_title('Residual Plot', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Absolute error distribution
    abs_errors = np.abs(errors)
    axes[1, 1].hist(abs_errors, bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[1, 1].axvline(abs_errors.mean(), color='red', linestyle='--',
                      label=f'Mean: {abs_errors.mean():.3f}')
    axes[1, 1].axvline(np.median(abs_errors), color='blue', linestyle='--',
                      label=f'Median: {np.median(abs_errors):.3f}')
    axes[1, 1].set_xlabel('Absolute Error (log₁₀ units)', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Absolute Error Distribution', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = f"{output_dir}/test_results_comprehensive.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to: {fig_path}")
    plt.close()

    return metrics

def main():
    """Main execution"""
    print("="*60)
    print("Testing Best Model on kcat_test_new.csv")
    print("="*60 + "\n")

    # Paths
    model_path = "outputs/kcat_hom40_train_20251213_221248/best_model.pt"
    test_data_path = "data/processed/kcat_test_new_diffdock_full.pt"

    # Check if files exist
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("Trying alternative model...")
        model_path = "outputs/kcat_quantile/best_model.pt"

    if not Path(test_data_path).exists():
        print(f"❌ Test data not found: {test_data_path}")
        print("Please ensure kcat_test_new is properly processed.")
        return

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Load model and data
    model = load_model(model_path, device)
    data_list = load_test_data(test_data_path)

    # Run inference
    predictions, labels = run_inference(model, data_list, device)

    # Generate report
    metrics = generate_report(predictions, labels)

    print("\n✅ Testing complete!")
    print(f"Results saved to: results/test_kcat_new/")

if __name__ == "__main__":
    main()
