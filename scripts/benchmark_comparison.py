#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark Comparison: PocketGNN vs CatPred vs CataPro

This script compares the performance of different methods on kcat prediction.

Usage:
    python scripts/benchmark_comparison.py --results_dir results/benchmark
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Literature-reported results for reference
LITERATURE_RESULTS = {
    'CatPred': {
        'source': 'Nature Communications 2025',
        'task': 'kcat',
        'split': 'OOD (low similarity)',
        'metrics': {
            'R2': 0.27,  # Approximate from paper figures
            'Pearson_r': 0.52,
            'p_1mag': 0.65,
        }
    },
    'CataPro': {
        'source': 'Nature Communications 2025',
        'task': 'kcat',
        'split': '10-fold CV (0.4 similarity)',
        'metrics': {
            'Pearson_r': 0.497,
            'Spearman_rho': 0.502,
        }
    },
    'DLKcat': {
        'source': 'Nature Catalysis 2022',
        'task': 'kcat',
        'split': 'Random',
        'metrics': {
            'R2': 0.64,
            'Pearson_r': 0.80,
        }
    },
    'UniKP': {
        'source': 'Nature Communications 2023',
        'task': 'kcat',
        'split': 'Random',
        'metrics': {
            'R2': 0.56,
            'Pearson_r': 0.75,
        }
    }
}


def load_results(results_dir: str) -> dict:
    """
    Load all results from directory
    
    Returns:
        dict: {method_name: {fold: metrics_dict}}
    """
    results = {}
    
    for method_dir in os.listdir(results_dir):
        method_path = os.path.join(results_dir, method_dir)
        if not os.path.isdir(method_path):
            continue
        
        method_name = method_dir.split('_')[0]  # e.g., 'pocketgnn_fold0' -> 'pocketgnn'
        
        if method_name not in results:
            results[method_name] = {}
        
        # Load metrics
        for file in os.listdir(method_path):
            if file.endswith('.json') and 'metrics' in file:
                metrics_path = os.path.join(method_path, file)
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                # Extract fold number
                fold = -1
                if 'fold' in method_dir:
                    try:
                        fold = int(method_dir.split('fold')[1].split('_')[0])
                    except:
                        pass
                
                results[method_name][fold] = metrics
    
    return results


def aggregate_results(results: dict) -> pd.DataFrame:
    """
    Aggregate results across folds
    
    Returns:
        DataFrame with mean ± std for each metric
    """
    records = []
    
    for method, fold_results in results.items():
        all_metrics = {}
        
        for fold, metrics in fold_results.items():
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
        
        record = {'Method': method}
        for key, values in all_metrics.items():
            if len(values) > 0:
                record[f'{key}_mean'] = np.mean(values)
                record[f'{key}_std'] = np.std(values)
        
        records.append(record)
    
    return pd.DataFrame(records)


def create_comparison_table(results: dict, output_path: str):
    """
    Create comparison table
    """
    # Aggregate experimental results
    exp_df = aggregate_results(results)
    
    # Add literature results
    lit_records = []
    for method, data in LITERATURE_RESULTS.items():
        record = {
            'Method': f"{method} (Literature)",
            'Source': data['source'],
            'Split': data['split'],
        }
        for key, value in data['metrics'].items():
            record[f'{key}_mean'] = value
            record[f'{key}_std'] = 0.0
        lit_records.append(record)
    
    lit_df = pd.DataFrame(lit_records)
    
    # Combine
    combined_df = pd.concat([exp_df, lit_df], ignore_index=True)
    
    # Save
    combined_df.to_csv(output_path, index=False)
    logger.info(f"Saved comparison table to {output_path}")
    
    return combined_df


def plot_metric_comparison(results: dict, output_dir: str):
    """
    Create bar charts comparing metrics
    """
    metrics_to_plot = ['R2', 'Pearson_r', 'Spearman_rho', 'MAE', 'p_1mag']
    
    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = []
        values = []
        errors = []
        colors = []
        
        # Experimental results
        for method, fold_results in results.items():
            method_values = []
            for fold, metrics in fold_results.items():
                if metric in metrics:
                    method_values.append(metrics[metric])
            
            if method_values:
                methods.append(method)
                values.append(np.mean(method_values))
                errors.append(np.std(method_values))
                colors.append('steelblue')
        
        # Literature results
        for method, data in LITERATURE_RESULTS.items():
            if metric in data['metrics']:
                methods.append(f"{method}\n(Literature)")
                values.append(data['metrics'][metric])
                errors.append(0)
                colors.append('lightcoral')
        
        if not methods:
            continue
        
        x = np.arange(len(methods))
        bars = ax.bar(x, values, yerr=errors, capsize=5, color=colors, alpha=0.8)
        
        ax.set_xlabel('Method')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_{metric}.png'), dpi=150)
        plt.close()
        
        logger.info(f"Saved {metric} comparison plot")


def plot_scatter_comparison(predictions: dict, output_dir: str):
    """
    Create scatter plots comparing predictions
    
    predictions: {method: {'y_true': array, 'y_pred': array}}
    """
    n_methods = len(predictions)
    if n_methods == 0:
        return
    
    fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
    if n_methods == 1:
        axes = [axes]
    
    for ax, (method, data) in zip(axes, predictions.items()):
        y_true = data['y_true']
        y_pred = data['y_pred']
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=10)
        
        # Diagonal line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        # Compute metrics
        from scipy.stats import pearsonr
        r, _ = pearsonr(y_true, y_pred)
        
        ax.set_xlabel('True log10(kcat)')
        ax.set_ylabel('Predicted log10(kcat)')
        ax.set_title(f'{method}\nPearson r = {r:.3f}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_comparison.png'), dpi=150)
    plt.close()
    
    logger.info("Saved scatter comparison plot")


def generate_report(results: dict, output_path: str):
    """
    Generate markdown report
    """
    report = []
    report.append("# Benchmark Comparison Report\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    report.append("## Summary\n\n")
    report.append("| Method | R² | Pearson r | Spearman ρ | MAE | p_1mag |\n")
    report.append("|--------|-----|-----------|------------|-----|--------|\n")
    
    # Experimental results
    for method, fold_results in results.items():
        metrics_agg = {}
        for fold, metrics in fold_results.items():
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key not in metrics_agg:
                        metrics_agg[key] = []
                    metrics_agg[key].append(value)
        
        r2 = f"{np.mean(metrics_agg.get('R2', [0])):.3f}" if 'R2' in metrics_agg else '-'
        pearson = f"{np.mean(metrics_agg.get('Pearson_r', [0])):.3f}" if 'Pearson_r' in metrics_agg else '-'
        spearman = f"{np.mean(metrics_agg.get('Spearman_rho', [0])):.3f}" if 'Spearman_rho' in metrics_agg else '-'
        mae = f"{np.mean(metrics_agg.get('MAE', [0])):.3f}" if 'MAE' in metrics_agg else '-'
        p1mag = f"{np.mean(metrics_agg.get('p_1mag', [0])):.3f}" if 'p_1mag' in metrics_agg else '-'
        
        report.append(f"| {method} | {r2} | {pearson} | {spearman} | {mae} | {p1mag} |\n")
    
    # Literature results
    report.append("\n## Literature Reference\n\n")
    for method, data in LITERATURE_RESULTS.items():
        report.append(f"### {method}\n")
        report.append(f"- Source: {data['source']}\n")
        report.append(f"- Split: {data['split']}\n")
        report.append(f"- Metrics: {data['metrics']}\n\n")
    
    with open(output_path, 'w') as f:
        f.writelines(report)
    
    logger.info(f"Saved report to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark comparison")
    parser.add_argument('--results_dir', type=str, default='results/benchmark',
                        help='Directory containing results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots and reports')
    
    args = parser.parse_args()
    
    project_root = get_project_root()
    results_dir = os.path.join(project_root, args.results_dir)
    
    if args.output_dir:
        output_dir = os.path.join(project_root, args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(project_root, 'results', f'benchmark_comparison_{timestamp}')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    if os.path.exists(results_dir):
        results = load_results(results_dir)
        logger.info(f"Loaded results for {len(results)} methods")
    else:
        logger.warning(f"Results directory not found: {results_dir}")
        results = {}
    
    # Create comparison table
    table_path = os.path.join(output_dir, 'comparison_table.csv')
    create_comparison_table(results, table_path)
    
    # Create plots
    plot_metric_comparison(results, output_dir)
    
    # Generate report
    report_path = os.path.join(output_dir, 'benchmark_report.md')
    generate_report(results, report_path)
    
    logger.info(f"""
    =====================================
    Benchmark Comparison Complete
    =====================================
    Output: {output_dir}
    - comparison_table.csv
    - comparison_*.png
    - benchmark_report.md
    =====================================
    """)


if __name__ == '__main__':
    main()





