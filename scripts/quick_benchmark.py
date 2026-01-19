#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Benchmark: Evaluate PocketGNN and compare with SOTA

This script evaluates existing PocketGNN models and generates comparison reports.

Usage:
    python scripts/quick_benchmark.py
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from GNN_model import PocketGNNKcatOnly

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Literature-reported results for comparison
SOTA_RESULTS = {
    'DLKcat (Nat. Catal. 2022)': {
        'split': 'Random 80/20',
        'R2': 0.64,
        'Pearson_r': 0.80,
        'note': 'Sequence + SMILES'
    },
    'UniKP (Nat. Commun. 2023)': {
        'split': 'Random',
        'R2': 0.56,
        'Pearson_r': 0.75,
        'note': 'ProtTrans embeddings'
    },
    'CatPred (Nat. Commun. 2025)': {
        'split': 'OOD (low similarity)',
        'R2': 0.27,
        'Pearson_r': 0.52,
        'note': 'ESM-2 + Seq-Attn + D-MPNN'
    },
    'CataPro (Nat. Commun. 2025)': {
        'split': '10-fold CV (0.4 similarity)',
        'Pearson_r': 0.497,
        'Spearman_rho': 0.502,
        'note': 'ProtT5 + MolT5 + MACCS'
    }
}


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_model(model_path: str, node_dim: int = 52, edge_dim: int = 24, 
               hidden_dim: int = 128, device: str = 'cuda'):
    """Load trained PocketGNN model"""
    model = PocketGNNKcatOnly(
        node_input_dim=node_dim,
        edge_input_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=3,
        heads=4,
        dropout=0.1,
        pooling_type='mean',
        use_seq_embedding=False
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def evaluate_model(model, dataloader, device='cuda'):
    """Evaluate model on dataloader"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            out = model(batch)
            
            # Handle different output formats
            if isinstance(out, tuple):
                out = out[0]
            
            preds = out.squeeze().cpu().numpy()
            labels = batch.y.squeeze().cpu().numpy()
            
            if preds.ndim == 0:
                preds = np.array([preds])
            if labels.ndim == 0:
                labels = np.array([labels])
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    return np.array(all_preds), np.array(all_labels)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute evaluation metrics"""
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {}
    
    metrics = {
        'n_samples': len(y_true),
        'R2': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'Pearson_r': pearsonr(y_true, y_pred)[0],
        'Spearman_rho': spearmanr(y_true, y_pred)[0],
    }
    
    # p_1mag: percentage within 1 order of magnitude
    within_1mag = np.abs(y_true - y_pred) <= 1.0
    metrics['p_1mag'] = np.mean(within_1mag)
    
    return metrics


def plot_scatter(y_true, y_pred, metrics, output_path, title='PocketGNN'):
    """Create scatter plot"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, c='steelblue')
    
    # Diagonal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x')
    
    # +/- 1 order of magnitude lines
    ax.plot([min_val, max_val], [min_val+1, max_val+1], 'g--', lw=1, alpha=0.5)
    ax.plot([min_val, max_val], [min_val-1, max_val-1], 'g--', lw=1, alpha=0.5)
    
    ax.set_xlabel('True log10(kcat)', fontsize=12)
    ax.set_ylabel('Predicted log10(kcat)', fontsize=12)
    ax.set_title(f'{title}\nPearson r = {metrics["Pearson_r"]:.3f}, R² = {metrics["R2"]:.3f}', fontsize=14)
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.info(f"Saved scatter plot to {output_path}")


def plot_comparison_bar(pocketgnn_metrics, output_path):
    """Create comparison bar chart"""
    metrics_to_plot = ['R2', 'Pearson_r', 'Spearman_rho']
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 5))
    
    for ax, metric in zip(axes, metrics_to_plot):
        methods = ['PocketGNN\n(This work)']
        values = [pocketgnn_metrics.get(metric, 0)]
        colors = ['steelblue']
        
        for name, sota in SOTA_RESULTS.items():
            if metric in sota:
                methods.append(name.replace(' ', '\n'))
                values.append(sota[metric])
                colors.append('lightcoral')
        
        x = np.arange(len(methods))
        bars = ax.bar(x, values, color=colors, alpha=0.8)
        
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric} Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.info(f"Saved comparison bar chart to {output_path}")


def generate_report(metrics, output_path):
    """Generate markdown report"""
    report = []
    report.append("# PocketGNN Benchmark Report\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    report.append("## PocketGNN Results\n\n")
    report.append("| Metric | Value |\n")
    report.append("|--------|-------|\n")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            report.append(f"| {key} | {value:.4f} |\n")
        else:
            report.append(f"| {key} | {value} |\n")
    
    report.append("\n## SOTA Comparison\n\n")
    report.append("| Method | Split | R² | Pearson r | Note |\n")
    report.append("|--------|-------|-----|-----------|------|\n")
    
    # Add PocketGNN
    report.append(f"| **PocketGNN (This work)** | Random 80/20 | "
                  f"{metrics.get('R2', '-'):.3f} | {metrics.get('Pearson_r', '-'):.3f} | "
                  f"Pocket Graph + ESM-2 |\n")
    
    # Add SOTA
    for name, sota in SOTA_RESULTS.items():
        r2 = f"{sota['R2']:.3f}" if 'R2' in sota else '-'
        pearson = f"{sota['Pearson_r']:.3f}" if 'Pearson_r' in sota else '-'
        report.append(f"| {name} | {sota['split']} | {r2} | {pearson} | {sota['note']} |\n")
    
    report.append("\n## Analysis\n\n")
    
    # Compare with each method
    if 'Pearson_r' in metrics:
        our_r = metrics['Pearson_r']
        report.append("### Comparison Summary\n\n")
        
        for name, sota in SOTA_RESULTS.items():
            if 'Pearson_r' in sota:
                sota_r = sota['Pearson_r']
                diff = our_r - sota_r
                if diff > 0:
                    report.append(f"- vs {name}: **+{diff:.3f}** (better)\n")
                else:
                    report.append(f"- vs {name}: {diff:.3f}\n")
    
    with open(output_path, 'w') as f:
        f.writelines(report)
    
    logger.info(f"Saved report to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Quick benchmark for PocketGNN")
    parser.add_argument('--model', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--test_data', type=str, default=None, help='Test dataset path')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    project_root = get_project_root()
    
    # Find model checkpoint
    if args.model:
        model_path = args.model
    else:
        # Use the latest model
        possible_models = [
            'outputs/kcat_after_new/best_model.pt',
            'experiments/kcat_quantile/run_04/best_model.pt',
        ]
        model_path = None
        for p in possible_models:
            full_path = os.path.join(project_root, p)
            if os.path.exists(full_path):
                model_path = full_path
                break
        
        if model_path is None:
            logger.error("No model checkpoint found")
            return
    
    logger.info(f"Using model: {model_path}")
    
    # Find test data
    if args.test_data:
        test_data_path = args.test_data
    else:
        possible_data = [
            'data/processed/kcat_merged_hom40_test.pt',
            'data/processed/kcat_test_new.pt',
            'data/processed/kcat_full_1213.pt',
        ]
        test_data_path = None
        for p in possible_data:
            full_path = os.path.join(project_root, p)
            if os.path.exists(full_path):
                test_data_path = full_path
                break
        
        if test_data_path is None:
            logger.error("No test data found")
            return
    
    logger.info(f"Using test data: {test_data_path}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir:
        output_dir = os.path.join(project_root, args.output_dir)
    else:
        output_dir = os.path.join(project_root, 'results', f'benchmark_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading test data...")
    test_data = torch.load(test_data_path, weights_only=False)
    logger.info(f"Loaded {len(test_data)} samples")
    
    # Create dataloader
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    
    # Load model
    logger.info("Loading model...")
    try:
        model = load_model(model_path, device=device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Try with different hidden_dim
        try:
            model = load_model(model_path, hidden_dim=256, device=device)
        except Exception as e2:
            logger.error(f"Still failed: {e2}")
            return
    
    # Evaluate
    logger.info("Evaluating model...")
    y_pred, y_true = evaluate_model(model, test_loader, device)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)
    
    # Print results
    logger.info(f"""
    =====================================
    PocketGNN Results
    =====================================
    Samples: {metrics['n_samples']}
    R²: {metrics['R2']:.4f}
    Pearson r: {metrics['Pearson_r']:.4f}
    Spearman ρ: {metrics['Spearman_rho']:.4f}
    MAE: {metrics['MAE']:.4f}
    RMSE: {metrics['RMSE']:.4f}
    p_1mag: {metrics['p_1mag']:.4f}
    =====================================
    """)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'pocketgnn_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create plots
    plot_scatter(y_true, y_pred, metrics, 
                 os.path.join(output_dir, 'pocketgnn_scatter.png'))
    plot_comparison_bar(metrics, 
                       os.path.join(output_dir, 'sota_comparison.png'))
    
    # Generate report
    generate_report(metrics, os.path.join(output_dir, 'benchmark_report.md'))
    
    # Save predictions
    pred_df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'error': y_pred - y_true,
        'abs_error': np.abs(y_pred - y_true)
    })
    pred_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    
    logger.info(f"All results saved to: {output_dir}")


if __name__ == '__main__':
    main()

