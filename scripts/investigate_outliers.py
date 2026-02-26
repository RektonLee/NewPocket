#!/usr/bin/env python3
"""
Investigate outliers in enzyme kinetics predictions.

This script analyzes prediction outliers to identify:
1. Correlation between docking quality and prediction error
2. EC class enrichment in outliers
3. Substrate complexity effects
4. Failure mode categorization

Author: Claude Code
Date: 2026-02-25
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy.stats import pearsonr, spearmanr


def load_model_and_predict(model_path: str, test_dataset_path: str, device: str = 'cuda'):
    """Load model and generate predictions."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from GNN_model import PocketGNNKcatOnly

    print(f"🚀 Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Infer config
    hidden_dim = state_dict['node_encoder.weight'].shape[0]
    num_layers = sum(1 for k in state_dict.keys() if k.startswith('att_layers.') and '.lin.weight' in k)
    heads = state_dict['att_layers.0.att_src'].shape[1] if 'att_layers.0.att_src' in state_dict else 4
    use_mlp_layernorm = len(state_dict['mlp.0.weight'].shape) == 1

    model = PocketGNNKcatOnly(
        node_input_dim=52, edge_input_dim=24,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
        pooling_type='mean',
        dropout=0.1,
        use_seq_embedding=False,
        use_mlp_layernorm=use_mlp_layernorm
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"✅ Model loaded")

    # Load dataset and predict
    print(f"📊 Loading test dataset: {test_dataset_path}")
    test_dataset = torch.load(test_dataset_path)
    print(f"✅ Loaded {len(test_dataset)} samples")

    true_values = []
    predictions = []
    sample_ids = []
    ec_numbers = []
    pdb_ids = []
    num_nodes = []
    num_edges = []

    with torch.no_grad():
        for i, data in enumerate(test_dataset):
            sample_id = getattr(data, 'sample_id', f'sample_{i}')
            ec = getattr(data, 'ec', 'unknown')
            pdb_id = getattr(data, 'pdb_id', 'unknown')

            sample_ids.append(sample_id)
            ec_numbers.append(ec)
            pdb_ids.append(pdb_id)

            data_device = data.to(device)
            pred = model(data_device)

            true_values.append(data.y.item())
            predictions.append(pred.item())
            num_nodes.append(data.x.shape[0])
            num_edges.append(data.edge_index.shape[1])

            if (i + 1) % 100 == 0:
                print(f"   Processed {i+1}/{len(test_dataset)}")

    return {
        'true_values': np.array(true_values),
        'predictions': np.array(predictions),
        'sample_ids': sample_ids,
        'ec_numbers': ec_numbers,
        'pdb_ids': pdb_ids,
        'num_nodes': np.array(num_nodes),
        'num_edges': np.array(num_edges)
    }


def compute_substrate_properties(smiles: str) -> Dict:
    """Compute substrate molecular properties."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                'mol_weight': np.nan,
                'num_heavy_atoms': np.nan,
                'num_rings': np.nan,
                'num_h_donors': np.nan,
                'num_h_acceptors': np.nan,
                'logp': np.nan,
                'tpsa': np.nan
            }

        return {
            'mol_weight': Descriptors.MolWt(mol),
            'num_heavy_atoms': mol.GetNumHeavyAtoms(),
            'num_rings': Descriptors.RingCount(mol),
            'num_h_donors': Descriptors.NumHDonors(mol),
            'num_h_acceptors': Descriptors.NumHAcceptors(mol),
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol)
        }
    except:
        return {k: np.nan for k in ['mol_weight', 'num_heavy_atoms', 'num_rings',
                                     'num_h_donors', 'num_h_acceptors', 'logp', 'tpsa']}


def analyze_ec_enrichment(results: Dict, error_threshold: float = 2.0) -> pd.DataFrame:
    """Analyze EC class enrichment in outliers."""
    errors = np.abs(results['true_values'] - results['predictions'])
    is_outlier = errors > error_threshold

    # Count EC classes
    ec_outlier_counts = Counter()
    ec_total_counts = Counter()

    for ec, outlier in zip(results['ec_numbers'], is_outlier):
        if ec and ec != 'unknown':
            # Convert to string if needed
            ec_str = str(ec)
            # Use EC level 1 (first digit)
            ec_level1 = ec_str.split('.')[0] if '.' in ec_str else ec_str
            ec_total_counts[ec_level1] += 1
            if outlier:
                ec_outlier_counts[ec_level1] += 1

    # Compute enrichment
    enrichment_data = []
    total_outliers = is_outlier.sum()
    total_samples = len(results['sample_ids'])

    for ec_class in ec_total_counts.keys():
        outlier_count = ec_outlier_counts[ec_class]
        total_count = ec_total_counts[ec_class]

        outlier_rate = outlier_count / total_count if total_count > 0 else 0
        expected_rate = total_outliers / total_samples

        enrichment = outlier_rate / expected_rate if expected_rate > 0 else 0

        enrichment_data.append({
            'ec_class': f'EC {ec_class}',
            'outlier_count': outlier_count,
            'total_count': total_count,
            'outlier_rate': outlier_rate,
            'enrichment': enrichment
        })

    df = pd.DataFrame(enrichment_data)
    df = df.sort_values('enrichment', ascending=False)

    return df


def analyze_graph_complexity(results: Dict) -> Dict:
    """Analyze correlation between graph complexity and error."""
    errors = np.abs(results['true_values'] - results['predictions'])

    # Correlations
    corr_nodes, p_nodes = spearmanr(results['num_nodes'], errors)
    corr_edges, p_edges = spearmanr(results['num_edges'], errors)

    return {
        'nodes_vs_error': {'correlation': corr_nodes, 'p_value': p_nodes},
        'edges_vs_error': {'correlation': corr_edges, 'p_value': p_edges}
    }


def categorize_failures(results: Dict, error_thresholds: Tuple[float, float, float] = (0.5, 2.0, 4.0)):
    """Categorize samples by prediction quality."""
    errors = np.abs(results['true_values'] - results['predictions'])

    low_threshold, mid_threshold, high_threshold = error_thresholds

    categories = np.select(
        [errors < low_threshold,
         (errors >= low_threshold) & (errors < mid_threshold),
         (errors >= mid_threshold) & (errors < high_threshold),
         errors >= high_threshold],
        ['excellent', 'good', 'poor', 'failure'],
        default='unknown'
    )

    category_counts = Counter(categories)

    return {
        'categories': categories,
        'counts': dict(category_counts),
        'thresholds': {
            'excellent': f'< {low_threshold}',
            'good': f'{low_threshold} - {mid_threshold}',
            'poor': f'{mid_threshold} - {high_threshold}',
            'failure': f'>= {high_threshold}'
        }
    }


def visualize_outliers(results: Dict, output_dir: Path):
    """Create comprehensive outlier visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    errors = np.abs(results['true_values'] - results['predictions'])

    # Figure S14: 4-panel outlier analysis
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Panel A: Error distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(errors, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(2.0, color='red', linestyle='--', linewidth=2, label='Outlier threshold (2.0)')
    ax1.set_xlabel('Absolute Error (log₁₀ units)', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('A. Prediction Error Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Panel B: Graph complexity vs error
    ax2 = fig.add_subplot(gs[0, 1])
    scatter = ax2.scatter(results['num_nodes'], errors,
                         c=errors, cmap='YlOrRd', alpha=0.5, s=30)
    ax2.set_xlabel('Number of Pocket Atoms', fontsize=11)
    ax2.set_ylabel('Absolute Error (log₁₀ units)', fontsize=11)
    ax2.set_title('B. Graph Complexity vs Error', fontsize=12, fontweight='bold')

    corr, p_val = spearmanr(results['num_nodes'], errors)
    ax2.text(0.05, 0.95, f'Spearman ρ = {corr:.3f}\np = {p_val:.1e}',
            transform=ax2.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.grid(alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Error', fontsize=10)

    # Panel C: EC class enrichment
    ax3 = fig.add_subplot(gs[1, 0])
    ec_enrichment = analyze_ec_enrichment(results, error_threshold=2.0)

    if not ec_enrichment.empty:
        colors = ['red' if e > 1.5 else 'orange' if e > 1.0 else 'green'
                  for e in ec_enrichment['enrichment']]
        ax3.barh(ec_enrichment['ec_class'], ec_enrichment['enrichment'], color=colors, alpha=0.7)
        ax3.axvline(1.0, color='black', linestyle='--', linewidth=1, label='No enrichment')
        ax3.set_xlabel('Enrichment (Outlier Rate / Expected Rate)', fontsize=11)
        ax3.set_ylabel('EC Class', fontsize=11)
        ax3.set_title('C. EC Class Outlier Enrichment', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(axis='x', alpha=0.3)

    # Panel D: Failure mode categorization
    ax4 = fig.add_subplot(gs[1, 1])
    failure_cats = categorize_failures(results)

    category_order = ['excellent', 'good', 'poor', 'failure']
    counts = [failure_cats['counts'].get(cat, 0) for cat in category_order]
    colors_cat = ['green', 'yellow', 'orange', 'red']

    wedges, texts, autotexts = ax4.pie(counts, labels=category_order, autopct='%1.1f%%',
                                         colors=colors_cat, startangle=90)
    ax4.set_title('D. Prediction Quality Categories', fontsize=12, fontweight='bold')

    # Add legend with thresholds
    legend_labels = [f"{cat.capitalize()}: {failure_cats['thresholds'][cat]}"
                     for cat in category_order]
    ax4.legend(legend_labels, loc='lower left', bbox_to_anchor=(1.0, 0.0), fontsize=9)

    plt.savefig(output_dir / 'figS14_outlier_investigation.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved outlier investigation figure: {output_dir / 'figS14_outlier_investigation.png'}")


def generate_report(results: Dict, output_dir: Path):
    """Generate comprehensive outlier analysis report."""
    errors = np.abs(results['true_values'] - results['predictions'])

    failure_cat_result = categorize_failures(results)

    report = {
        'overall_statistics': {
            'total_samples': int(len(results['sample_ids'])),
            'mean_error': float(np.mean(errors)),
            'median_error': float(np.median(errors)),
            'std_error': float(np.std(errors)),
            'max_error': float(np.max(errors)),
            'outliers_count': int(np.sum(errors > 2.0)),
            'outlier_rate': float(np.mean(errors > 2.0))
        },
        'ec_enrichment': analyze_ec_enrichment(results, error_threshold=2.0).to_dict(orient='records'),
        'graph_complexity_analysis': analyze_graph_complexity(results),
        'failure_categorization': {
            'counts': failure_cat_result['counts'],
            'thresholds': failure_cat_result['thresholds']
        }
    }

    # Save JSON report
    report_file = output_dir / 'outlier_investigation_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✅ Saved outlier investigation report: {report_file}")

    # Save detailed CSV
    df = pd.DataFrame({
        'sample_id': results['sample_ids'],
        'ec_number': results['ec_numbers'],
        'pdb_id': results['pdb_ids'],
        'true_kcat': results['true_values'],
        'pred_kcat': results['predictions'],
        'error': errors,
        'num_nodes': results['num_nodes'],
        'num_edges': results['num_edges'],
        'is_outlier': errors > 2.0,
        'quality_category': failure_cat_result['categories']
    })

    csv_file = output_dir / 'outlier_details.csv'
    df.to_csv(csv_file, index=False)
    print(f"✅ Saved outlier details: {csv_file}")

    return report


def main():
    parser = argparse.ArgumentParser(description='Investigate prediction outliers')
    parser.add_argument('--model', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--test-dataset', required=True, help='Path to test dataset (.pt)')
    parser.add_argument('--output-dir', default='results/interpretability/outliers',
                       help='Output directory')
    parser.add_argument('--error-threshold', type=float, default=2.0,
                       help='Error threshold for outlier definition (log10 units)')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Outlier Investigation Analysis")
    print("=" * 60)

    # Load model and predict
    results = load_model_and_predict(args.model, args.test_dataset, args.device)

    # Compute overall performance
    pearson_r, _ = pearsonr(results['true_values'], results['predictions'])
    r2 = 1 - np.sum((results['true_values'] - results['predictions']) ** 2) / \
        np.sum((results['true_values'] - results['true_values'].mean()) ** 2)

    print(f"\n📊 Overall Performance:")
    print(f"   Pearson r: {pearson_r:.4f}")
    print(f"   R²: {r2:.4f}")

    # Generate visualizations
    print(f"\n📊 Creating visualizations...")
    visualize_outliers(results, output_dir)

    # Generate comprehensive report
    print(f"\n📝 Generating report...")
    report = generate_report(results, output_dir)

    # Print summary
    print(f"\n📋 Summary:")
    print(f"   Total samples: {report['overall_statistics']['total_samples']}")
    print(f"   Mean error: {report['overall_statistics']['mean_error']:.3f}")
    print(f"   Outliers (>2.0): {report['overall_statistics']['outliers_count']} ({report['overall_statistics']['outlier_rate']*100:.1f}%)")

    print(f"\n   Graph Complexity Correlations:")
    print(f"      Nodes vs Error: ρ={report['graph_complexity_analysis']['nodes_vs_error']['correlation']:.3f} (p={report['graph_complexity_analysis']['nodes_vs_error']['p_value']:.3e})")
    print(f"      Edges vs Error: ρ={report['graph_complexity_analysis']['edges_vs_error']['correlation']:.3f} (p={report['graph_complexity_analysis']['edges_vs_error']['p_value']:.3e})")

    print(f"\n   Failure Categories:")
    for cat, count in report['failure_categorization']['counts'].items():
        pct = 100 * count / report['overall_statistics']['total_samples']
        print(f"      {cat.capitalize()}: {count} ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("✅ Outlier investigation complete!")
    print(f"📁 Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
