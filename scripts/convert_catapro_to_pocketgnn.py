#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert CataPro dataset to PocketGNN format

This script converts the CataPro kcat dataset to PocketGNN format by:
1. Getting protein structures (ESMFold)
2. Running DiffDock for docking
3. Extracting pockets
4. Building graph data

Usage:
    python scripts/convert_catapro_to_pocketgnn.py --n_samples 100 --fold 0
    python scripts/convert_catapro_to_pocketgnn.py --all  # Process all data
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import hashlib
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from build_graph_dataset import enhanced_build_graph, parse_pocket_pdb
from docking import run_preprocess, smiles_to_sdf, DockingConfig, set_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_project_root():
    """Get project root directory"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def predict_structure_esmfold(sequence: str, output_path: str, timeout: int = 300) -> bool:
    """
    Predict protein structure using ESMFold API
    
    Args:
        sequence: Protein sequence
        output_path: Output PDB file path
        timeout: Request timeout in seconds
    
    Returns:
        bool: Success or not
    """
    import requests
    
    try:
        # ESMFold API endpoint
        url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
        
        response = requests.post(url, data=sequence, timeout=timeout)
        
        if response.status_code == 200:
            with open(output_path, 'w') as f:
                f.write(response.text)
            logger.info(f"✅ ESMFold prediction saved: {output_path}")
            return True
        else:
            logger.error(f"❌ ESMFold API error: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ ESMFold prediction failed: {e}")
        return False


def get_structure(uniprot_id: str, sequence: str, cache_dir: str) -> str:
    """
    Get protein structure (from cache or predict)
    
    Priority:
    1. Local cache
    2. AlphaFold DB (if UniProt ID available)
    3. ESMFold prediction
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate unique filename based on sequence hash
    seq_hash = hashlib.md5(sequence.encode()).hexdigest()[:8]
    cache_path = os.path.join(cache_dir, f"{uniprot_id}_{seq_hash}.pdb")
    
    # Check cache
    if os.path.exists(cache_path):
        logger.info(f"Using cached structure: {cache_path}")
        return cache_path
    
    # Try AlphaFold DB
    if uniprot_id and uniprot_id != 'nan':
        alphafold_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
        try:
            import requests
            response = requests.get(alphafold_url, timeout=30)
            if response.status_code == 200:
                with open(cache_path, 'w') as f:
                    f.write(response.text)
                logger.info(f"✅ Downloaded from AlphaFold DB: {uniprot_id}")
                return cache_path
        except Exception as e:
            logger.warning(f"AlphaFold DB not available for {uniprot_id}: {e}")
    
    # Fallback to ESMFold
    if predict_structure_esmfold(sequence, cache_path):
        return cache_path
    
    return None


def process_sample(row: pd.Series, output_dir: str, structure_cache: str, 
                   pocket_dir: str, index: int) -> dict:
    """
    Process a single sample
    
    Args:
        row: DataFrame row with sample data
        output_dir: Output directory for intermediate files
        structure_cache: Directory to cache protein structures
        pocket_dir: Directory to save pocket PDB files
        index: Sample index
    
    Returns:
        dict with processed data or None if failed
    """
    sample_id = row.name if isinstance(row.name, str) else f"kcat_{index}"
    sequence = row['Sequence']
    smiles = row['Smiles']
    kcat = row['kcat(s^-1)']
    uniprot_id = row.get('UniProtID', 'unknown')
    ec = row.get('EC', 'unknown')
    fold = row.get('fold', -1)
    
    logger.info(f"Processing {sample_id} (fold={fold})...")
    
    try:
        # 1. Get protein structure
        structure_path = get_structure(uniprot_id, sequence, structure_cache)
        if structure_path is None:
            logger.error(f"Failed to get structure for {sample_id}")
            return None
        
        # 2. Generate pocket hash for output filename
        pocket_hash = int(hashlib.sha256(smiles.encode()).hexdigest(), 16) & 0xffff
        pocket_path = os.path.join(pocket_dir, f"{sample_id}_{pocket_hash}_pocket.pdb")
        
        # 3. Run docking and pocket extraction
        if not os.path.exists(pocket_path):
            success = run_preprocess(
                uniprot_id=sample_id,
                smiles=smiles,
                prot_pdb_path=structure_path,
                output_pocket_path=pocket_path,
                index=index
            )
            
            if not success:
                logger.error(f"Docking/pocket extraction failed for {sample_id}")
                return None
        else:
            logger.info(f"Using cached pocket: {pocket_path}")
        
        # 4. Build graph
        atoms = parse_pocket_pdb(pocket_path)
        if atoms is None or len(atoms) == 0:
            logger.error(f"Failed to parse pocket PDB for {sample_id}")
            return None
        
        data = enhanced_build_graph(atoms, temperature=25.0)  # Default temperature
        
        # 5. Add metadata
        data.y = torch.tensor([np.log10(kcat)], dtype=torch.float)
        data.sample_id = sample_id
        data.ec = str(ec)
        data.uniprot_id = str(uniprot_id)
        data.fold = int(fold) if fold != -1 else -1
        data.smiles = smiles
        data.sequence_length = len(sequence)
        
        logger.info(f"✅ {sample_id}: nodes={data.x.shape[0]}, edges={data.edge_index.shape[1]}, kcat={kcat:.2e}")
        
        return {
            'sample_id': sample_id,
            'data': data,
            'fold': fold
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing {sample_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Convert CataPro to PocketGNN format")
    parser.add_argument('--input', type=str, 
                        default='benchmark_tools/CataPro/datasets/kcat-data_0.4simi-10fold.csv',
                        help='Input CataPro CSV file')
    parser.add_argument('--output_dir', type=str, default='data/catapro_benchmark',
                        help='Output directory')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Number of samples to process (default: all)')
    parser.add_argument('--fold', type=int, default=None,
                        help='Only process specific fold (0-9)')
    parser.add_argument('--wild_only', action='store_true',
                        help='Only process wild-type enzymes')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID for docking')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    project_root = get_project_root()
    input_path = os.path.join(project_root, args.input)
    output_dir = os.path.join(project_root, args.output_dir)
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    structure_cache = os.path.join(output_dir, 'structures')
    pocket_dir = os.path.join(output_dir, 'pockets')
    os.makedirs(structure_cache, exist_ok=True)
    os.makedirs(pocket_dir, exist_ok=True)
    
    # Configure docking
    config = DockingConfig(gpu_id=args.gpu)
    set_config(config)
    
    # Load data
    logger.info(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path, index_col=0)
    logger.info(f"Total samples: {len(df)}")
    
    # Filter data
    if args.wild_only:
        df = df[df['EnzymeType'] == 'wild']
        logger.info(f"Wild-type only: {len(df)} samples")
    
    if args.fold is not None:
        df = df[df['fold'] == args.fold]
        logger.info(f"Fold {args.fold}: {len(df)} samples")
    
    if args.n_samples is not None:
        df = df.head(args.n_samples)
        logger.info(f"Limited to {args.n_samples} samples")
    
    # Checkpoint file
    checkpoint_file = os.path.join(output_dir, 'processing_checkpoint.txt')
    processed_ids = set()
    
    if args.resume and os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            processed_ids = set(line.strip() for line in f)
        logger.info(f"Resuming from checkpoint: {len(processed_ids)} already processed")
    
    # Process samples
    all_data = []
    failed_samples = []
    
    for idx, (sample_id, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Processing")):
        if sample_id in processed_ids:
            continue
        
        result = process_sample(row, output_dir, structure_cache, pocket_dir, idx)
        
        if result is not None:
            all_data.append(result['data'])
            
            # Save checkpoint
            with open(checkpoint_file, 'a') as f:
                f.write(f"{sample_id}\n")
        else:
            failed_samples.append(sample_id)
        
        # Save intermediate results every 100 samples
        if len(all_data) > 0 and len(all_data) % 100 == 0:
            intermediate_path = os.path.join(output_dir, f'kcat_intermediate_{len(all_data)}.pt')
            torch.save(all_data, intermediate_path)
            logger.info(f"Saved intermediate: {intermediate_path}")
    
    # Save final results
    if len(all_data) > 0:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save all data
        output_path = os.path.join(output_dir, f'kcat_catapro_{timestamp}.pt')
        torch.save(all_data, output_path)
        logger.info(f"✅ Saved {len(all_data)} samples to {output_path}")
        
        # Save by fold
        if args.fold is None:
            for fold_id in range(10):
                fold_data = [d for d in all_data if hasattr(d, 'fold') and d.fold == fold_id]
                if len(fold_data) > 0:
                    fold_path = os.path.join(output_dir, f'kcat_fold{fold_id}.pt')
                    torch.save(fold_data, fold_path)
                    logger.info(f"Saved fold {fold_id}: {len(fold_data)} samples")
    
    # Save failed samples
    if failed_samples:
        failed_path = os.path.join(output_dir, 'failed_samples.txt')
        with open(failed_path, 'w') as f:
            f.write('\n'.join(failed_samples))
        logger.warning(f"⚠️ {len(failed_samples)} samples failed, saved to {failed_path}")
    
    # Summary
    logger.info(f"""
    =====================================
    Conversion Summary
    =====================================
    Total processed: {len(all_data)}
    Failed: {len(failed_samples)}
    Output: {output_dir}
    =====================================
    """)


if __name__ == '__main__':
    main()





