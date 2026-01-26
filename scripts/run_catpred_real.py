#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run CatPred on converted test set

This script actually runs CatPred inference on the test set.
"""

import os
import sys
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


def check_catpred_environment():
    """Check if CatPred environment is set up"""
    catpred_dir = os.path.join(get_project_root(), 'benchmark_tools', 'CatPred')
    
    # Check if capsule_data exists
    capsule_data = os.path.join(catpred_dir, 'capsule_data')
    if not os.path.exists(capsule_data):
        logger.warning(f"Capsule data not found at {capsule_data}")
        logger.info("Extracting capsule_data_update.tar.gz...")
        tar_file = os.path.join(catpred_dir, 'capsule_data_update.tar.gz')
        if os.path.exists(tar_file):
            import tarfile
            with tarfile.open(tar_file, 'r:gz') as tar:
                tar.extractall(catpred_dir)
            logger.info("Extraction complete")
        else:
            logger.error(f"Tar file not found: {tar_file}")
            return False
    
    # Check if catpred package is installed
    try:
        import catpred
        logger.info("✅ CatPred package is available")
        return True
    except ImportError:
        logger.warning("CatPred package not installed. Need to run: pip install -e .")
        return False


def prepare_catpred_input(test_csv_path, output_path):
    """
    Prepare input in CatPred format
    
    CatPred expects (from demo_run.py):
    - SMILES column (capitalized)
    - sequence column (lowercase)
    """
    df = pd.read_csv(test_csv_path)
    logger.info(f"Loaded {len(df)} samples from {test_csv_path}")
    
    # CatPred format: SMILES and sequence
    catpred_df = pd.DataFrame({
        'SMILES': df['smiles'],
        'sequence': df['sequence']
    })
    
    # Add sample_id for tracking (if available)
    if 'sample_id' in df.columns:
        catpred_df['sample_id'] = df['sample_id']
    
    catpred_df.to_csv(output_path, index=False)
    logger.info(f"Saved CatPred input to {output_path}")
    return output_path


def run_catpred_inference(input_path, output_path, catpred_dir, checkpoint_dir=None):
    """
    Run CatPred inference using predict.py
    
    Based on demo_run.py, CatPred uses:
    1. create_pdbrecords.py to create protein records
    2. predict.py to run inference
    """
    import subprocess
    
    logger.info(f"Running CatPred inference...")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    
    # Find checkpoint directory
    if checkpoint_dir is None:
        capsule_data = os.path.join(catpred_dir, 'capsule_data')
        if os.path.exists(capsule_data):
            # Look for checkpoint directories
            possible_checkpoints = [
                os.path.join(capsule_data, 'checkpoints', 'kcat'),
                os.path.join(capsule_data, 'models', 'kcat'),
            ]
            for cp in possible_checkpoints:
                if os.path.exists(cp):
                    checkpoint_dir = cp
                    break
        
        if checkpoint_dir is None:
            logger.error("Could not find CatPred checkpoint directory")
            logger.info(f"Please specify --checkpoint_dir or ensure capsule_data is extracted")
            return False
    
    logger.info(f"Using checkpoint: {checkpoint_dir}")
    
    # Prepare input file (normalize SMILES)
    from rdkit import Chem
    df = pd.read_csv(input_path)
    smiles_list_new = []
    for i, smi in enumerate(df['SMILES']):
        try:
            mol = Chem.MolFromSmiles(smi)
            smi = Chem.MolToSmiles(mol)
            if '.' in smi:
                smi = '.'.join(sorted(smi.split('.')))
            smiles_list_new.append(smi)
        except Exception as e:
            logger.warning(f"Invalid SMILES at row {i}: {e}")
            smiles_list_new.append(smi)
    
    df['SMILES'] = smiles_list_new
    input_file_new = input_path.replace('.csv', '_input.csv')
    df.to_csv(input_file_new, index=False)
    
    # Create protein records
    test_file_prefix = input_file_new[:-4]
    records_file = f"{test_file_prefix}.json"
    
    scripts_dir = os.path.join(catpred_dir, 'scripts')
    create_records_script = os.path.join(scripts_dir, 'create_pdbrecords.py')
    
    if os.path.exists(create_records_script):
        cmd1 = [
            'python', create_records_script,
            '--data_file', input_file_new,
            '--out_file', records_file
        ]
        logger.info(f"Creating protein records: {' '.join(cmd1)}")
        result1 = subprocess.run(cmd1, cwd=catpred_dir, capture_output=True, text=True)
        if result1.returncode != 0:
            logger.error(f"Failed to create records: {result1.stderr}")
            return False
    else:
        logger.warning(f"create_pdbrecords.py not found at {create_records_script}")
    
    # Run prediction
    predict_script = os.path.join(catpred_dir, 'predict.py')
    if not os.path.exists(predict_script):
        logger.error(f"predict.py not found at {predict_script}")
        return False
    
    cmd2 = [
        'python', predict_script,
        '--test_path', input_file_new,
        '--preds_path', output_path,
        '--checkpoint_dir', checkpoint_dir,
        '--uncertainty_method', 'mve',
        '--smiles_column', 'SMILES',
        '--individual_ensemble_predictions',
    ]
    
    if os.path.exists(records_file):
        cmd2.extend(['--protein_records_path', records_file])
    
    logger.info(f"Running prediction: {' '.join(cmd2)}")
    result2 = subprocess.run(cmd2, cwd=catpred_dir, capture_output=True, text=True, timeout=3600)
    
    if result2.returncode != 0:
        logger.error(f"Prediction failed: {result2.stderr}")
        return False
    
    logger.info(f"✅ Predictions saved to {output_path}")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run CatPred on test set")
    parser.add_argument('--test_csv', type=str,
                        default='data/sota_benchmark/catpred_test_input.csv',
                        help='Test CSV file')
    parser.add_argument('--output_dir', type=str,
                        default='results/catpred_real',
                        help='Output directory')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='CatPred checkpoint directory (auto-detect if not specified)')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU for prediction')
    
    args = parser.parse_args()
    
    project_root = get_project_root()
    test_csv = os.path.join(project_root, args.test_csv)
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check environment
    if not check_catpred_environment():
        logger.error("CatPred environment not ready. Please:")
        logger.error("1. Extract capsule_data_update.tar.gz")
        logger.error("2. Install CatPred: cd benchmark_tools/CatPred && pip install -e .")
        return
    
    # Prepare input
    catpred_input = os.path.join(output_dir, 'catpred_input.csv')
    prepare_catpred_input(test_csv, catpred_input)
    
    # Run inference
    output_path = os.path.join(output_dir, 'catpred_predictions.csv')
    catpred_dir = os.path.join(project_root, 'benchmark_tools', 'CatPred')
    success = run_catpred_inference(catpred_input, output_path, catpred_dir, 
                                    args.checkpoint_dir)
    
    if success:
        logger.info(f"✅ CatPred predictions saved to {output_path}")
    else:
        logger.error("❌ CatPred inference failed")
        logger.info("Need to check CatPred's actual API from demo files")


if __name__ == '__main__':
    main()

