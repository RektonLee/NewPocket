#!/usr/bin/env python3
"""
Select representative cases for interpretability analysis.

This script selects diverse samples from the test set for detailed
mechanistic analysis and attention visualization.

Selection Criteria:
1. High-accuracy samples (prediction error < 0.5 log units)
2. Outlier samples (prediction error > 2.0 log units)
3. Known-mechanism samples (from literature with documented catalytic residues)
4. Diverse-size samples (small molecules vs large substrates)

Author: Claude Code
Date: 2026-02-25
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy.stats import pearsonr


def load_predictions(model_path: str, test_dataset_path: str, device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load model and generate predictions on test set.

    Returns:
        true_values: Ground truth log10(kcat) values
        predictions: Model predictions
        sample_ids: Sample IDs
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

    from GNN_model import PocketGNNKcatOnly

    print(f"🚀 Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Infer config from checkpoint
    hidden_dim = state_dict['node_encoder.weight'].shape[0]
    num_layers = sum(1 for k in state_dict.keys() if k.startswith('att_layers.') and '.lin.weight' in k)

    if 'att_layers.0.att_src' in state_dict:
        heads = state_dict['att_layers.0.att_src'].shape[1]
    else:
        heads = 4

    use_mlp_layernorm = len(state_dict['mlp.0.weight'].shape) == 1

    config = {
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'heads': heads,
        'pooling_type': 'mean',
        'dropout': 0.1,
        'use_seq_embedding': False,
        'use_mlp_layernorm': use_mlp_layernorm
    }

    print(f"Inferred config: hidden_dim={hidden_dim}, num_layers={num_layers}, heads={heads}, use_mlp_layernorm={use_mlp_layernorm}")

    model = PocketGNNKcatOnly(
        node_input_dim=52, edge_input_dim=24,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        heads=config['heads'],
        pooling_type=config['pooling_type'],
        dropout=config['dropout'],
        use_seq_embedding=config['use_seq_embedding'],
        use_mlp_layernorm=config['use_mlp_layernorm']
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"✅ Model loaded: {config}")

    # Load test dataset
    print(f"\n📊 Loading test dataset: {test_dataset_path}")
    test_dataset = torch.load(test_dataset_path)
    print(f"✅ Test dataset loaded: {len(test_dataset)} samples")

    # Generate predictions
    true_values = []
    predictions = []
    sample_ids = []

    with torch.no_grad():
        for i, data in enumerate(test_dataset):
            sample_id = getattr(data, 'sample_id', f'sample_{i}')
            sample_ids.append(sample_id)

            data = data.to(device)
            pred = model(data)

            true_values.append(data.y.item())
            predictions.append(pred.item())

            if (i + 1) % 100 == 0:
                print(f"   Processed {i+1}/{len(test_dataset)} samples...")

    print(f"✅ Predictions generated: {len(predictions)} samples")

    return np.array(true_values), np.array(predictions), sample_ids


def compute_errors(true_values: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Compute absolute prediction errors."""
    return np.abs(true_values - predictions)


def compute_molecular_properties(smiles: str) -> Dict[str, float]:
    """Compute molecular properties from SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                'mol_weight': 0.0,
                'num_heavy_atoms': 0,
                'num_h_donors': 0,
                'num_h_acceptors': 0,
                'num_rings': 0,
                'logp': 0.0
            }

        return {
            'mol_weight': Descriptors.MolWt(mol),
            'num_heavy_atoms': mol.GetNumHeavyAtoms(),
            'num_h_donors': Descriptors.NumHDonors(mol),
            'num_h_acceptors': Descriptors.NumHAcceptors(mol),
            'num_rings': Descriptors.RingCount(mol),
            'logp': Descriptors.MolLogP(mol)
        }
    except:
        return {
            'mol_weight': 0.0,
            'num_heavy_atoms': 0,
            'num_h_donors': 0,
            'num_h_acceptors': 0,
            'num_rings': 0,
            'logp': 0.0
        }


def select_cases(
    test_dataset,
    true_values: np.ndarray,
    predictions: np.ndarray,
    sample_ids: List[str],
    num_high_acc: int = 2,
    num_outliers: int = 2,
    num_diverse: int = 2
) -> Dict[str, List[Dict]]:
    """
    Select representative cases for interpretability analysis.

    Returns:
        Dictionary with categories: 'high_accuracy', 'outliers', 'diverse_size'
    """
    errors = compute_errors(true_values, predictions)

    # Category 1: High-accuracy samples (error < 0.5)
    high_acc_indices = np.where(errors < 0.5)[0]
    if len(high_acc_indices) > num_high_acc:
        # Select samples with diverse kcat values
        high_acc_selected = []
        for idx in high_acc_indices:
            high_acc_selected.append({
                'index': int(idx),
                'sample_id': sample_ids[idx],
                'true_kcat': float(true_values[idx]),
                'pred_kcat': float(predictions[idx]),
                'error': float(errors[idx])
            })
        # Sort by kcat value and pick extremes
        high_acc_selected.sort(key=lambda x: x['true_kcat'])
        high_acc_cases = [high_acc_selected[0], high_acc_selected[-1]][:num_high_acc]
    else:
        high_acc_cases = [{
            'index': int(idx),
            'sample_id': sample_ids[idx],
            'true_kcat': float(true_values[idx]),
            'pred_kcat': float(predictions[idx]),
            'error': float(errors[idx])
        } for idx in high_acc_indices[:num_high_acc]]

    # Category 2: Outlier samples (error > 2.0)
    outlier_indices = np.where(errors > 2.0)[0]
    if len(outlier_indices) > num_outliers:
        # Select most extreme outliers
        outlier_sorted = sorted(outlier_indices, key=lambda i: errors[i], reverse=True)
        outlier_indices = outlier_sorted[:num_outliers]

    outlier_cases = [{
        'index': int(idx),
        'sample_id': sample_ids[idx],
        'true_kcat': float(true_values[idx]),
        'pred_kcat': float(predictions[idx]),
        'error': float(errors[idx])
    } for idx in outlier_indices[:num_outliers]]

    # Category 3: Diverse molecular size
    # Compute molecular properties from SMILES
    mol_properties = []
    for i, data in enumerate(test_dataset):
        smiles = getattr(data, 'substrate_smiles', '')
        props = compute_molecular_properties(smiles)
        props['index'] = i
        props['sample_id'] = sample_ids[i]
        props['smiles'] = smiles
        props['error'] = float(errors[i])
        mol_properties.append(props)

    # Sort by molecular weight
    mol_properties.sort(key=lambda x: x['mol_weight'])

    # Select small and large molecules with reasonable error (0.5 < error < 2.0)
    moderate_error_props = [p for p in mol_properties if 0.5 <= p['error'] <= 2.0]

    diverse_cases = []
    if len(moderate_error_props) >= num_diverse:
        # Pick smallest and largest
        diverse_cases.append({
            'index': moderate_error_props[0]['index'],
            'sample_id': moderate_error_props[0]['sample_id'],
            'true_kcat': float(true_values[moderate_error_props[0]['index']]),
            'pred_kcat': float(predictions[moderate_error_props[0]['index']]),
            'error': moderate_error_props[0]['error'],
            'mol_weight': moderate_error_props[0]['mol_weight'],
            'num_heavy_atoms': moderate_error_props[0]['num_heavy_atoms'],
            'smiles': moderate_error_props[0]['smiles']
        })
        diverse_cases.append({
            'index': moderate_error_props[-1]['index'],
            'sample_id': moderate_error_props[-1]['sample_id'],
            'true_kcat': float(true_values[moderate_error_props[-1]['index']]),
            'pred_kcat': float(predictions[moderate_error_props[-1]['index']]),
            'error': moderate_error_props[-1]['error'],
            'mol_weight': moderate_error_props[-1]['mol_weight'],
            'num_heavy_atoms': moderate_error_props[-1]['num_heavy_atoms'],
            'smiles': moderate_error_props[-1]['smiles']
        })

    # Category 4: Known catalytic mechanism (placeholder - would need literature annotation)
    # For now, we'll select samples with specific EC numbers that are well-studied
    known_mechanism_cases = []

    return {
        'high_accuracy': high_acc_cases,
        'outliers': outlier_cases,
        'diverse_size': diverse_cases,
        'known_mechanism': known_mechanism_cases
    }


def visualize_selection(
    cases: Dict[str, List[Dict]],
    true_values: np.ndarray,
    predictions: np.ndarray,
    errors: np.ndarray,
    output_dir: Path
):
    """Create visualization of selected cases."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create scatter plot with selected cases highlighted
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: True vs Predicted with highlighted cases
    ax = axes[0]
    ax.scatter(true_values, predictions, alpha=0.3, s=20, c='gray', label='All samples')

    # Plot selected cases
    colors = {'high_accuracy': 'green', 'outliers': 'red', 'diverse_size': 'blue'}
    for category, color in colors.items():
        if cases[category]:
            indices = [c['index'] for c in cases[category]]
            ax.scatter(true_values[indices], predictions[indices],
                      s=100, c=color, marker='*', edgecolors='black',
                      linewidths=1.5, label=category.replace('_', ' ').title(), zorder=10)

    # Plot diagonal
    min_val = min(true_values.min(), predictions.min())
    max_val = max(true_values.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)

    ax.set_xlabel('True log₁₀(kcat)')
    ax.set_ylabel('Predicted log₁₀(kcat)')
    ax.set_title('Selected Cases for Interpretability Analysis')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: Error distribution with highlighted cases
    ax = axes[1]
    ax.hist(errors, bins=50, alpha=0.5, color='gray', label='All samples')

    for category, color in colors.items():
        if cases[category]:
            category_errors = [c['error'] for c in cases[category]]
            ax.scatter(category_errors, [5] * len(category_errors),
                      s=100, c=color, marker='*', edgecolors='black',
                      linewidths=1.5, zorder=10)

    ax.set_xlabel('Absolute Error (log₁₀ units)')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution with Selected Cases')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'case_selection_overview.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved case selection overview: {output_dir / 'case_selection_overview.png'}")


def main():
    parser = argparse.ArgumentParser(description='Select representative cases for interpretability analysis')
    parser.add_argument('--model', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--test-dataset', required=True, help='Path to test dataset (.pt)')
    parser.add_argument('--output-dir', default='results/interpretability/case_selection',
                       help='Output directory')
    parser.add_argument('--num-high-acc', type=int, default=2,
                       help='Number of high-accuracy cases to select')
    parser.add_argument('--num-outliers', type=int, default=2,
                       help='Number of outlier cases to select')
    parser.add_argument('--num-diverse', type=int, default=2,
                       help='Number of diverse-size cases to select')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Case Selection for Interpretability Analysis")
    print("=" * 60)

    # Load test dataset for additional metadata
    test_dataset = torch.load(args.test_dataset)

    # Generate predictions
    true_values, predictions, sample_ids = load_predictions(
        args.model, args.test_dataset, args.device
    )

    # Compute errors
    errors = compute_errors(true_values, predictions)

    # Compute performance metrics
    pearson_r, _ = pearsonr(true_values, predictions)
    r2 = 1 - np.sum((true_values - predictions) ** 2) / np.sum((true_values - true_values.mean()) ** 2)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean((true_values - predictions) ** 2))

    print(f"\n📊 Overall Performance:")
    print(f"   Pearson r: {pearson_r:.4f}")
    print(f"   R²: {r2:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")

    # Select cases
    print(f"\n🔍 Selecting representative cases...")
    cases = select_cases(
        test_dataset,
        true_values,
        predictions,
        sample_ids,
        num_high_acc=args.num_high_acc,
        num_outliers=args.num_outliers,
        num_diverse=args.num_diverse
    )

    # Print selected cases
    print(f"\n📋 Selected Cases:")
    for category, case_list in cases.items():
        if case_list:
            print(f"\n{category.upper().replace('_', ' ')} ({len(case_list)} cases):")
            for i, case in enumerate(case_list, 1):
                print(f"   {i}. {case['sample_id']}")
                print(f"      True: {case['true_kcat']:.3f}, Pred: {case['pred_kcat']:.3f}, Error: {case['error']:.3f}")
                if 'mol_weight' in case:
                    print(f"      MW: {case['mol_weight']:.1f}, Heavy atoms: {case['num_heavy_atoms']}")

    # Save selected cases
    cases_file = output_dir / 'selected_cases.json'
    with open(cases_file, 'w') as f:
        json.dump(cases, f, indent=2)
    print(f"\n✅ Saved selected cases: {cases_file}")

    # Create sample IDs file for visualization script
    all_sample_ids = []
    for category, case_list in cases.items():
        for case in case_list:
            all_sample_ids.append(case['sample_id'])

    sample_ids_file = output_dir / 'selected_sample_ids.txt'
    with open(sample_ids_file, 'w') as f:
        for sid in all_sample_ids:
            f.write(f"{sid}\n")
    print(f"✅ Saved sample IDs: {sample_ids_file}")

    # Visualize selection
    print(f"\n📊 Creating visualization...")
    visualize_selection(cases, true_values, predictions, errors, output_dir)

    # Create summary statistics
    summary = {
        'overall_performance': {
            'pearson_r': float(pearson_r),
            'r2': float(r2),
            'mae': float(mae),
            'rmse': float(rmse)
        },
        'case_counts': {
            'high_accuracy': len(cases['high_accuracy']),
            'outliers': len(cases['outliers']),
            'diverse_size': len(cases['diverse_size']),
            'known_mechanism': len(cases['known_mechanism'])
        },
        'error_thresholds': {
            'high_accuracy': '< 0.5 log units',
            'outliers': '> 2.0 log units',
            'diverse_size': '0.5-2.0 log units'
        }
    }

    summary_file = output_dir / 'case_selection_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved summary: {summary_file}")

    print("\n" + "=" * 60)
    print("✅ Case selection complete!")
    print(f"📁 Results saved to: {output_dir}")
    print("=" * 60)

    print(f"\n📝 Next Steps:")
    print(f"   1. Review selected cases in: {cases_file}")
    print(f"   2. Wait for DiffDock to complete generating PDB files")
    print(f"   3. Run attention visualization:")
    print(f"      python scripts/visualize_attention_weights.py \\")
    print(f"        --model {args.model} \\")
    print(f"        --test-dataset {args.test_dataset} \\")
    print(f"        --sample-ids {sample_ids_file} \\")
    print(f"        --output-dir results/interpretability/attention")


if __name__ == '__main__':
    main()
