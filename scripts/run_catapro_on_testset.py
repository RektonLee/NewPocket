#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run CataPro on converted test set

This script directly runs CataPro on our converted test set.
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


def run_catapro_inference(input_path: str, output_path: str, 
                          catapro_dir: str, device: str = 'cuda:0') -> bool:
    """
    Run CataPro inference
    
    Args:
        input_path: Input CSV file path (CataPro format)
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
    logger.info(f"Working directory: {inference_dir}")
    
    try:
        env = os.environ.copy()
        result = subprocess.run(
            cmd,
            cwd=inference_dir,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        if result.returncode != 0:
            logger.error(f"CataPro failed with return code {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")
            logger.error(f"STDOUT: {result.stdout}")
            return False
        
        logger.info(f"CataPro completed: {output_path}")
        logger.info(f"STDOUT: {result.stdout[-500:]}")  # Last 500 chars
        return True
        
    except subprocess.TimeoutExpired:
        logger.error("CataPro timeout (2 hours)")
        return False
    except Exception as e:
        logger.error(f"CataPro error: {e}")
        import traceback
        traceback.print_exc()
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
    parser = argparse.ArgumentParser(description="Run CataPro on converted test set")
    parser.add_argument('--test_csv', type=str, required=True,
                        help='Test CSV file (CataPro format)')
    parser.add_argument('--device', type=str, default='cuda:0', help='CUDA device')
    parser.add_argument('--output_dir', type=str, default='results/catapro_real',
                        help='Output directory')
    parser.add_argument('--true_kcat_col', type=str, default=None,
                        help='Column name for true kcat values (if available)')
    
    args = parser.parse_args()
    
    project_root = get_project_root()
    catapro_dir = os.path.join(project_root, 'benchmark_tools', 'CataPro')
    test_csv = os.path.join(project_root, args.test_csv) if not os.path.isabs(args.test_csv) else args.test_csv
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(project_root, args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("CataPro Inference on Test Set")
    logger.info("="*60)
    logger.info(f"Test CSV: {test_csv}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Device: {args.device}")
    
    # Check input file
    if not os.path.exists(test_csv):
        logger.error(f"Test CSV not found: {test_csv}")
        return
    
    # Load test data to check format
    logger.info(f"\nLoading test data...")
    test_df = pd.read_csv(test_csv)
    logger.info(f"Loaded {len(test_df)} samples")
    logger.info(f"Columns: {test_df.columns.tolist()}")
    
    # Verify required columns
    required_cols = ['Enzyme_id', 'type', 'sequence', 'smiles']
    missing_cols = [col for col in required_cols if col not in test_df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return
    
    # Prepare input for CataPro (save to output dir)
    catapro_input = os.path.join(output_dir, 'catapro_input.csv')
    test_df.to_csv(catapro_input, index=False)
    logger.info(f"Saved CataPro input to {catapro_input}")
    
    # Run CataPro inference
    prediction_path = os.path.join(output_dir, 'catapro_predictions.csv')
    logger.info(f"\nRunning CataPro inference...")
    success = run_catapro_inference(
        catapro_input, 
        prediction_path, 
        catapro_dir, 
        args.device
    )
    
    if success and os.path.exists(prediction_path):
        logger.info(f"\n✅ CataPro inference completed!")
        logger.info(f"Predictions saved to: {prediction_path}")
        
        # Load predictions
        pred_df = pd.read_csv(prediction_path)
        logger.info(f"Prediction columns: {pred_df.columns.tolist()}")
        
        # Try to find prediction column
        pred_col = None
        for col in ['kcat_pred', 'prediction', 'kcat', 'log10kcat', 'Prediction']:
            if col in pred_df.columns:
                pred_col = col
                break
        
        if pred_col is None:
            logger.warning("Could not find prediction column. Available columns:")
            logger.warning(pred_df.columns.tolist())
            logger.warning("Please check the output file manually.")
        else:
            logger.info(f"Using prediction column: {pred_col}")
            y_pred = pred_df[pred_col].values
            
            # Check if need to take log10
            if np.mean(y_pred) > 10:  # Likely not log scale
                logger.info("Converting predictions to log10 scale")
                y_pred = np.log10(y_pred)
            
            # Get true values if available
            if args.true_kcat_col and args.true_kcat_col in test_df.columns:
                y_true = np.log10(test_df[args.true_kcat_col].values)
                
                # Compute metrics
                metrics = compute_metrics(y_true, y_pred)
                
                # Save metrics
                metrics_path = os.path.join(output_dir, 'metrics.json')
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                logger.info(f"""
                =====================================
                CataPro Results
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
                logger.info("No true kcat column provided. Predictions saved without metrics.")
    else:
        logger.error("❌ CataPro inference failed or output not found")
        logger.error(f"Check logs in: {output_dir}")


if __name__ == '__main__':
    main()





