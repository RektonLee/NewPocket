#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert PocketGNN test set to CatPred/CataPro format

This script extracts SMILES and protein sequences from original CSV files
and matches them with the test set samples.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def load_test_set(test_pt_path):
    """Load PocketGNN test set"""
    test_data = torch.load(test_pt_path, weights_only=False)
    print(f"Loaded {len(test_data)} test samples")
    
    # Extract sample IDs
    sample_ids = []
    kcat_values = []
    for data in test_data:
        sample_id = getattr(data, 'sample_id', None)
        if sample_id:
            sample_ids.append(str(sample_id))
            kcat_values.append(float(data.y.item()) if hasattr(data, 'y') else None)
    
    return sample_ids, kcat_values


def find_matching_csv(csv_files, sample_ids):
    """Find CSV file that contains the test samples"""
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, nrows=1000)  # Check first 1000 rows
            print(f"\nChecking {csv_path}...")
            print(f"Columns: {df.columns.tolist()}")
            
            # Check if it has sample_id column
            if 'sample_id' in df.columns:
                matched = df[df['sample_id'].isin(sample_ids)]
                if len(matched) > 0:
                    print(f"✅ Found {len(matched)} matching samples in {csv_path}")
                    return csv_path
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue
    
    return None


def extract_test_data(csv_path, sample_ids):
    """Extract test data from CSV"""
    print(f"\nLoading full CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df)}")
    
    # Find matching samples
    if 'sample_id' in df.columns:
        test_df = df[df['sample_id'].isin(sample_ids)].copy()
        print(f"Matched {len(test_df)} samples")
    else:
        print("Warning: No sample_id column found")
        return None
    
    return test_df


def convert_to_catpred_format(test_df):
    """Convert to CatPred input format"""
    # CatPred needs: protein sequence, substrate SMILES
    # Check available columns
    print("\nAvailable columns:", test_df.columns.tolist())
    
    catpred_data = []
    for idx, row in test_df.iterrows():
        sample = {}
        
        # Try to find protein sequence
        for col in ['protein_sequence', 'sequence', 'Sequence', 'seq']:
            if col in row and pd.notna(row[col]):
                sample['sequence'] = str(row[col])
                break
        
        # Try to find SMILES
        for col in ['substrate_smiles', 'smiles', 'Smiles', 'SMILES', 'substrate_SMILES']:
            if col in row and pd.notna(row[col]):
                smiles = str(row[col])
                # Handle multiple SMILES separated by semicolon
                if ';' in smiles:
                    smiles = smiles.split(';')[0].strip()
                sample['smiles'] = smiles
                break
        
        # Try to find UniProt ID
        for col in ['uniprot', 'uniprot_id', 'UniProtID', 'UniProt']:
            if col in row and pd.notna(row[col]):
                sample['uniprot_id'] = str(row[col])
                break
        
        # Get sample_id
        if 'sample_id' in row:
            sample['sample_id'] = str(row['sample_id'])
        
        # Get kcat value
        for col in ['kcat_value', 'kcat', 'kcat Wildtype', 'kcat(s^-1)']:
            if col in row and pd.notna(row[col]):
                sample['kcat'] = float(row[col])
                break
        
        if 'sequence' in sample and 'smiles' in sample:
            catpred_data.append(sample)
    
    return pd.DataFrame(catpred_data)


def convert_to_catapro_format(test_df):
    """Convert to CataPro input format"""
    # CataPro needs: Enzyme_id, type, sequence, smiles
    catapro_data = []
    
    for idx, row in test_df.iterrows():
        sample = {}
        
        # Enzyme_id (use sample_id or uniprot_id)
        if 'sample_id' in row and pd.notna(row['sample_id']):
            sample['Enzyme_id'] = str(row['sample_id'])
        elif 'uniprot_id' in row and pd.notna(row['uniprot_id']):
            sample['Enzyme_id'] = str(row['uniprot_id'])
        else:
            continue
        
        # type (default to 'wild')
        sample['type'] = 'wild'
        if 'EnzymeType' in row and pd.notna(row['EnzymeType']):
            sample['type'] = str(row['EnzymeType'])
        
        # sequence
        for col in ['protein_sequence', 'sequence', 'Sequence', 'seq']:
            if col in row and pd.notna(row[col]):
                sample['sequence'] = str(row[col])
                break
        
        # smiles
        for col in ['substrate_smiles', 'smiles', 'Smiles', 'SMILES']:
            if col in row and pd.notna(row[col]):
                smiles = str(row[col])
                if ';' in smiles:
                    smiles = smiles.split(';')[0].strip()
                sample['smiles'] = smiles
                break
        
        if 'sequence' in sample and 'smiles' in sample:
            catapro_data.append(sample)
    
    return pd.DataFrame(catapro_data)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert test set for SOTA methods")
    parser.add_argument('--test_pt', type=str, 
                        default='data/processed/kcat_merged_hom40_test.pt',
                        help='PocketGNN test set .pt file')
    parser.add_argument('--output_dir', type=str, 
                        default='data/sota_benchmark',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test set
    print("="*60)
    print("Step 1: Loading PocketGNN test set")
    print("="*60)
    sample_ids, kcat_values = load_test_set(args.test_pt)
    print(f"Found {len(sample_ids)} sample IDs")
    
    # Find matching CSV
    print("\n" + "="*60)
    print("Step 2: Finding matching CSV file")
    print("="*60)
    csv_files = [
        'data/processed/kcat_full_1213.csv',
        'data/processed/kcat_test_new.csv',
        'data/raw/kcat_data.csv',
        'data/raw/successful_full_train_1213.csv',
    ]
    
    csv_path = find_matching_csv(csv_files, sample_ids[:100])  # Check first 100
    
    if csv_path is None:
        print("❌ Could not find matching CSV file")
        return
    
    # Extract test data
    print("\n" + "="*60)
    print("Step 3: Extracting test data from CSV")
    print("="*60)
    test_df = extract_test_data(csv_path, sample_ids)
    
    if test_df is None or len(test_df) == 0:
        print("❌ No matching data found")
        return
    
    # Convert to CatPred format
    print("\n" + "="*60)
    print("Step 4: Converting to CatPred format")
    print("="*60)
    catpred_df = convert_to_catpred_format(test_df)
    if len(catpred_df) > 0:
        catpred_path = os.path.join(args.output_dir, 'catpred_test_input.csv')
        catpred_df.to_csv(catpred_path, index=False)
        print(f"✅ Saved {len(catpred_df)} samples to {catpred_path}")
    else:
        print("⚠️  No valid CatPred samples")
    
    # Convert to CataPro format
    print("\n" + "="*60)
    print("Step 5: Converting to CataPro format")
    print("="*60)
    catapro_df = convert_to_catapro_format(test_df)
    if len(catapro_df) > 0:
        catapro_path = os.path.join(args.output_dir, 'catapro_test_input.csv')
        catapro_df.to_csv(catapro_path, index=False)
        print(f"✅ Saved {len(catapro_df)} samples to {catapro_path}")
    else:
        print("⚠️  No valid CataPro samples")
    
    # Save metadata
    metadata = {
        'test_pt_path': args.test_pt,
        'source_csv': csv_path,
        'total_samples': len(sample_ids),
        'matched_samples': len(test_df),
        'catpred_samples': len(catpred_df),
        'catapro_samples': len(catapro_df),
    }
    
    metadata_path = os.path.join(args.output_dir, 'conversion_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✅ Metadata saved to {metadata_path}")
    
    print("\n" + "="*60)
    print("Conversion Complete!")
    print("="*60)


if __name__ == '__main__':
    main()





