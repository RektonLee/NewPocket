#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run CataPro Baseline for kcat prediction

This script runs CataPro on the test set and collects metrics for comparison.

Usage:
    python scripts/run_catapro_baseline.py --fold 0
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import subprocess
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def prepare_catapro_input(df: pd.DataFrame, output_path: str):
    """
    Prepare input file in CataPro format
    
    CataPro format:
    Enzyme_id, type, sequence, smiles
    """
    catapro_df = pd.DataFrame({
        'Enzyme_id': df.index if df.index.name else df['UniProtID'],
        'type': df['EnzymeType'],
        'sequence': df['Sequence'],
        'smiles': df['Smiles']
    })
    
    catapro_df.to_csv(output_path, index=False)
    logger.info(f"Saved CataPro input to {output_path}")
    return output_path


def run_catapro_inference(input_path: str, output_path: str, 
                          catapro_dir: str, device: str = 'cuda:0') -> bool:
    """
    Run CataPro inference
    
    Args:
        input_path: Input CSV file path
        output_path: Output prediction CSV path
        catapro_dir: CataPro repository directory
        device: CUDA device
    
    Returns:
        bool: Success or not
    """
    inference_dir = os.path.join(catapro_dir, 'inference')
    predict_script = os.path.join(inference_dir, 'predict.py')
    models_dir = os.path.join(catapro_dir, 'models')
    
    if not os.path.exists(predict_script):
        logger.error(f"CataPro predict.py not found at {predict_script}")
        return False
    
    cmd = [
        'python', predict_script,
        '-inp_fpath', input_path,
        '-model_dpath', models_dir,
        '-batch_size', '64',
        '-device', device,
        '-out_fpath', output_path
    ]
    
    logger.info(f"Running CataPro: {' '.join(cmd)}")
    
    try:
        # Need to activate catapro conda environment
        env = os.environ.copy()
        result = subprocess.run(
            cmd,
            cwd=inference_dir,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            logger.error(f"CataPro failed: {result.stderr}")
            return False
        
        logger.info(f"CataPro completed: {output_path}")
        return True
        
    except subprocess.TimeoutExpired:
        logger.error("CataPro timeout")
        return False
    except Exception as e:
        logger.error(f"CataPro error: {e}")
        return False


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute evaluation metrics
    
    Args:
        y_true: True log10(kcat) values
        y_pred: Predicted log10(kcat) values
    
    Returns:
        dict: Metrics
    """
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    
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


def main():
    parser = argparse.ArgumentParser(description="Run CataPro baseline")
    parser.add_argument('--fold', type=int, default=0, help='Test fold (0-9)')
    parser.add_argument('--device', type=str, default='cuda:0', help='CUDA device')
    parser.add_argument('--output_dir', type=str, default='results/catapro_baseline',
                        help='Output directory')
    
    args = parser.parse_args()
    
    project_root = get_project_root()
    catapro_dir = os.path.join(project_root, 'benchmark_tools', 'CataPro')
    data_path = os.path.join(catapro_dir, 'datasets', 'kcat-data_0.4simi-10fold.csv')
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(project_root, args.output_dir, f'fold{args.fold}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, index_col=0)
    
    # Split by fold
    test_df = df[df['fold'] == args.fold]
    train_df = df[df['fold'] != args.fold]
    
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Prepare input for CataPro
    test_input_path = os.path.join(output_dir, 'catapro_test_input.csv')
    prepare_catapro_input(test_df, test_input_path)
    
    # Run CataPro inference
    prediction_path = os.path.join(output_dir, 'catapro_predictions.csv')
    success = run_catapro_inference(
        test_input_path, 
        prediction_path, 
        catapro_dir, 
        args.device
    )
    
    if success and os.path.exists(prediction_path):
        # Load predictions and compute metrics
        pred_df = pd.read_csv(prediction_path)
        
        # Get true values
        y_true = np.log10(test_df['kcat(s^-1)'].values)
        
        # Get predicted values (assuming CataPro outputs log10 or need to convert)
        if 'kcat_pred' in pred_df.columns:
            y_pred = pred_df['kcat_pred'].values
            # Check if need to take log10
            if np.mean(y_pred) > 10:  # Likely not log scale
                y_pred = np.log10(y_pred)
        else:
            logger.error("Cannot find prediction column in CataPro output")
            return
        
        # Compute metrics
        metrics = compute_metrics(y_true, y_pred)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"""
        =====================================
        CataPro Results (Fold {args.fold})
        =====================================
        Samples: {metrics['n_samples']}
        R²: {metrics['R2']:.4f}
        Pearson r: {metrics['Pearson_r']:.4f}
        Spearman ρ: {metrics['Spearman_rho']:.4f}
        MAE: {metrics['MAE']:.4f}
        RMSE: {metrics['RMSE']:.4f}
        p_1mag: {metrics['p_1mag']:.4f}
        =====================================
        Results saved to: {output_dir}
        """)
    else:
        logger.error("CataPro inference failed or output not found")
        
        # Use literature-reported values as fallback
        logger.info("""
        =====================================
        CataPro Literature Results (Reference)
        =====================================
        From Nature Communications 2025 paper:
        - kcat (10-fold CV, 0.4 similarity):
          - PCC (Pearson): ~0.497
          - SCC (Spearman): ~0.502
        =====================================
        """)
        
        # Save reference metrics
        reference_metrics = {
            'source': 'CataPro Nature Communications 2025',
            'note': 'Literature-reported values for 10-fold CV at 0.4 similarity',
            'Pearson_r': 0.497,
            'Spearman_rho': 0.502,
        }
        
        metrics_path = os.path.join(output_dir, 'reference_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(reference_metrics, f, indent=2)


if __name__ == '__main__':
    main()

