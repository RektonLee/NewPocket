#!/usr/bin/env python3
"""
Feature Importance Analysis for PocketGNN
==========================================

Quantify the contribution of different feature types (RBF distances, angles, dihedrals)
to model predictions using Integrated Gradients and permutation-based importance.

Core Question: Do geometric features (angles + dihedrals) contribute more than just distances?

Usage:
    python scripts/analyze_feature_importance.py \
        --model outputs/kcat_hom40_train_20251213_221248/best_model.pt \
        --test_dataset data/processed/kcat_merged_hom40_test.pt \
        --output_dir results/interpretability/feature_importance \
        --num_samples 100

Outputs:
    - figS12_feature_importance_barplot.png: Top 20 most important features
    - figS12b_feature_type_importance.png: RBF vs Angles vs Dihedrals comparison
    - figS12c_permutation_importance.png: Permutation-based feature group importance
    - node_feature_importance.csv: Per-feature attribution scores (52 node features)
    - edge_feature_importance.csv: Per-feature attribution scores (24 edge features)
    - feature_importance_summary.json: Aggregated statistics
"""

import argparse
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

import torch
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_geometric.data import Batch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from GNN_model import PocketGNNKcatOnly


def load_model(model_path, device='cuda'):
    """
    Load trained PocketGNN model with automatic config inference

    Args:
        model_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        model: Loaded model in eval mode
        config: Model configuration dict
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract or infer config
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Infer from checkpoint keys
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        hidden_dim = state_dict['node_encoder.weight'].shape[0]
        num_layers = sum(1 for k in state_dict.keys() if k.startswith('att_layers.') and '.lin.weight' in k)

        # Infer heads from att_src shape [1, heads, out_per_head]
        if 'att_layers.0.att_src' in state_dict:
            heads = state_dict['att_layers.0.att_src'].shape[1]
        else:
            heads = 4

        # Infer use_mlp_layernorm from mlp.0 shape
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

    # Initialize model
    model = PocketGNNKcatOnly(
        node_input_dim=52,
        edge_input_dim=24,
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 3),
        heads=config.get('heads', 4),
        pooling_type=config.get('pooling_type', 'mean'),
        dropout=config.get('dropout', 0.1),
        use_seq_embedding=config.get('use_seq_embedding', False),
        seq_embedding_dim=config.get('seq_embedding_dim', 1280),
        use_mlp_layernorm=config.get('use_mlp_layernorm', True)
    )

    # Load weights
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    return model, config


def integrated_gradients(model, data, baseline='zero', steps=50, feature_type='node', device='cuda'):
    """
    Compute Integrated Gradients for feature attribution

    Reference: Sundararajan et al. "Axiomatic Attribution for Deep Networks" (ICML 2017)

    Args:
        model: Trained model
        data: PyG Data object
        baseline: Baseline type ('zero', 'random', or 'mean')
        steps: Number of interpolation steps
        feature_type: 'node' or 'edge'
        device: Device

    Returns:
        attributions: Feature attribution scores [num_features]
    """
    data = data.to(device)
    model.eval()

    # Select features to analyze
    if feature_type == 'node':
        features = data.x.clone()  # [num_nodes, 52]
    elif feature_type == 'edge':
        features = data.edge_attr.clone()  # [num_edges, 24]
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    # Create baseline
    if baseline == 'zero':
        baseline_features = torch.zeros_like(features)
    elif baseline == 'random':
        baseline_features = torch.randn_like(features) * 0.1
    elif baseline == 'mean':
        baseline_features = features.mean(dim=0, keepdim=True).expand_as(features)
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    # Interpolate between baseline and input
    alphas = torch.linspace(0, 1, steps, device=device)

    gradients = []
    for alpha in alphas:
        # Interpolated features
        interpolated = baseline_features + alpha * (features - baseline_features)
        interpolated.requires_grad = True

        # Create data copy with interpolated features
        if feature_type == 'node':
            data_copy = data.clone()
            data_copy.x = interpolated
        else:  # edge
            data_copy = data.clone()
            data_copy.edge_attr = interpolated

        # Forward pass
        output = model(data_copy)

        # Backward pass
        model.zero_grad()
        output.sum().backward()

        # Collect gradients
        gradients.append(interpolated.grad.detach())

    # Average gradients over interpolation steps
    avg_gradients = torch.stack(gradients).mean(dim=0)  # [num_nodes/edges, num_features]

    # Integrated gradients = (input - baseline) * avg_gradients
    ig = (features - baseline_features) * avg_gradients

    # Aggregate across nodes/edges (sum importance per feature dimension)
    attributions = ig.abs().sum(dim=0).cpu().numpy()  # [num_features]

    return attributions


def permutation_feature_importance(model, test_dataset, feature_indices, feature_type='edge', device='cuda', batch_size=32):
    """
    Compute permutation-based feature importance by shuffling feature groups

    Args:
        model: Trained model
        test_dataset: List of Data objects
        feature_indices: List of feature indices to permute together
        feature_type: 'node' or 'edge'
        device: Device
        batch_size: Batch size for processing

    Returns:
        importance_drop: Drop in Pearson correlation after permutation
    """
    model.eval()

    # Baseline predictions (no permutation)
    baseline_preds = []
    true_values = []

    with torch.no_grad():
        for i in range(0, len(test_dataset), batch_size):
            batch_data = test_dataset[i:min(i+batch_size, len(test_dataset))]

            # Ensure all data on CPU and remove metadata before batching
            batch_data_cpu = []
            for d in batch_data:
                d_clean = d.clone().cpu()
                # Remove string metadata that can't be batched
                for attr in ['sample_id', 'pdb_id', 'ec']:
                    if hasattr(d_clean, attr):
                        delattr(d_clean, attr)
                batch_data_cpu.append(d_clean)

            # Manual batching to ensure device consistency
            from torch_geometric.data import Batch
            batch = Batch.from_data_list(batch_data_cpu).to(device)

            preds = model(batch)
            baseline_preds.extend(preds.cpu().numpy().flatten())
            true_values.extend(batch.y.cpu().numpy().flatten())

    baseline_corr, _ = pearsonr(true_values, baseline_preds)

    # Permuted predictions
    permuted_preds = []

    with torch.no_grad():
        for i in range(0, len(test_dataset), batch_size):
            batch_data = test_dataset[i:min(i+batch_size, len(test_dataset))]

            # Ensure all data on CPU and remove metadata before batching
            batch_data_cpu = []
            for d in batch_data:
                d_clean = d.clone().cpu()
                # Remove string metadata that can't be batched
                for attr in ['sample_id', 'pdb_id', 'ec']:
                    if hasattr(d_clean, attr):
                        delattr(d_clean, attr)
                batch_data_cpu.append(d_clean)

            # Manual batching
            from torch_geometric.data import Batch
            batch = Batch.from_data_list(batch_data_cpu).to(device)

            # Permute specified features
            if feature_type == 'node':
                features = batch.x.clone()
                features[:, feature_indices] = features[torch.randperm(features.size(0)), :][:, feature_indices]
                batch.x = features
            elif feature_type == 'edge':
                features = batch.edge_attr.clone()
                features[:, feature_indices] = features[torch.randperm(features.size(0)), :][:, feature_indices]
                batch.edge_attr = features

            preds = model(batch)
            permuted_preds.extend(preds.cpu().numpy().flatten())

    permuted_corr, _ = pearsonr(true_values, permuted_preds)

    importance_drop = baseline_corr - permuted_corr

    return importance_drop, baseline_corr, permuted_corr


def plot_feature_importance_barplot(attributions, feature_names, output_path, top_k=20):
    """
    Plot top-K most important features as barplot

    Args:
        attributions: Feature attribution scores [num_features]
        feature_names: List of feature names
        output_path: Output path for figure
        top_k: Number of top features to show
    """
    # Sort by importance
    sorted_indices = np.argsort(attributions)[-top_k:][::-1]
    top_attrs = attributions[sorted_indices]
    top_names = [feature_names[i] for i in sorted_indices]

    # Normalize to percentages
    top_attrs_pct = 100 * top_attrs / top_attrs.sum()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_k))
    bars = ax.barh(range(top_k), top_attrs_pct, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.set_xlabel('Importance (%)', fontsize=11, weight='bold')
    ax.set_title(f'Top {top_k} Most Important Features (Integrated Gradients)', fontsize=13, weight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_attrs_pct)):
        ax.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved feature importance barplot: {output_path}")


def plot_feature_type_comparison(edge_attributions, output_path):
    """
    Compare importance of RBF, Angles, and Dihedrals

    Edge features structure (24-dim):
        - [0:16]: RBF distance encoding (16-dim)
        - [16:20]: Bond angles (4-dim)
        - [20:24]: Dihedral angles (4-dim)

    Args:
        edge_attributions: Edge feature attributions [24]
        output_path: Output path
    """
    # Aggregate by feature type
    rbf_importance = edge_attributions[:16].sum()
    angle_importance = edge_attributions[16:20].sum()
    dihedral_importance = edge_attributions[20:24].sum()

    total = rbf_importance + angle_importance + dihedral_importance

    # Percentages
    rbf_pct = 100 * rbf_importance / total
    angle_pct = 100 * angle_importance / total
    dihedral_pct = 100 * dihedral_importance / total

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    feature_types = ['RBF\nDistance\n(16-dim)', 'Bond\nAngles\n(4-dim)', 'Dihedral\nAngles\n(4-dim)']
    importances = [rbf_importance, angle_importance, dihedral_importance]
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    bars = ax1.bar(feature_types, importances, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.set_ylabel('Importance Score', fontsize=11, weight='bold')
    ax1.set_title('Feature Type Importance (Absolute)', fontsize=13, weight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, importances):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + height*0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, weight='bold')

    # Pie chart
    percentages = [rbf_pct, angle_pct, dihedral_pct]
    wedges, texts, autotexts = ax2.pie(percentages, labels=feature_types, colors=colors,
                                        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11},
                                        explode=[0.05, 0.05, 0.05], shadow=True)

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')
        autotext.set_fontsize(12)

    ax2.set_title('Feature Type Importance (Relative)', fontsize=13, weight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved feature type comparison: {output_path}")
    print(f"   RBF: {rbf_pct:.1f}% | Angles: {angle_pct:.1f}% | Dihedrals: {dihedral_pct:.1f}%")

    return {
        'rbf': {'importance': float(rbf_importance), 'percentage': float(rbf_pct)},
        'angles': {'importance': float(angle_importance), 'percentage': float(angle_pct)},
        'dihedrals': {'importance': float(dihedral_importance), 'percentage': float(dihedral_pct)}
    }


def plot_permutation_importance(perm_results, output_path):
    """
    Plot permutation-based importance for feature groups

    Args:
        perm_results: Dict with feature group results
        output_path: Output path
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    groups = list(perm_results.keys())
    importance_drops = [perm_results[g]['importance_drop'] for g in groups]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    bars = ax.bar(groups, importance_drops, color=colors[:len(groups)],
                  edgecolor='black', linewidth=1.5, alpha=0.8)

    ax.set_ylabel('Pearson Correlation Drop', fontsize=11, weight='bold')
    ax.set_title('Permutation-Based Feature Group Importance', fontsize=13, weight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

    # Add value labels
    for bar, val, group in zip(bars, importance_drops, groups):
        height = bar.get_height()
        baseline = perm_results[group]['baseline_corr']
        permuted = perm_results[group]['permuted_corr']

        ax.text(bar.get_x() + bar.get_width()/2, height + 0.005,
               f'Δ={val:.3f}\n({baseline:.3f}→{permuted:.3f})',
               ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved permutation importance: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze feature importance for PocketGNN")
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--test_dataset', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--output_dir', type=str, default='results/interpretability/feature_importance',
                       help='Output directory')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to analyze (for IG computation)')
    parser.add_argument('--ig_steps', type=int, default=50,
                       help='Number of interpolation steps for Integrated Gradients')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for permutation importance')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Loading model from: {args.model}")
    model, config = load_model(args.model, device=args.device)
    print(f"✅ Model loaded: {config}")

    print(f"\n📊 Loading test dataset: {args.test_dataset}")
    test_dataset = torch.load(args.test_dataset, weights_only=False)
    print(f"✅ Test dataset loaded: {len(test_dataset)} samples")

    # Select subset for IG computation
    np.random.seed(42)
    ig_indices = np.random.choice(len(test_dataset), min(args.num_samples, len(test_dataset)), replace=False)
    ig_samples = [test_dataset[i] for i in ig_indices]

    print(f"\n{'='*60}")
    print(f"Phase 1: Integrated Gradients (on {len(ig_samples)} samples)")
    print(f"{'='*60}")

    # Compute IG for node features
    print("\n🔍 Computing node feature attributions...")
    node_attributions_all = []
    for i, data in enumerate(ig_samples):
        if (i + 1) % 20 == 0:
            print(f"   Processed {i+1}/{len(ig_samples)} samples...")
        attrs = integrated_gradients(model, data, baseline='zero', steps=args.ig_steps,
                                    feature_type='node', device=args.device)
        node_attributions_all.append(attrs)

    node_attributions = np.mean(node_attributions_all, axis=0)  # Average across samples
    print(f"✅ Node feature attributions computed: {node_attributions.shape}")

    # Compute IG for edge features
    print("\n🔍 Computing edge feature attributions...")
    edge_attributions_all = []
    for i, data in enumerate(ig_samples):
        if (i + 1) % 20 == 0:
            print(f"   Processed {i+1}/{len(ig_samples)} samples...")
        attrs = integrated_gradients(model, data, baseline='zero', steps=args.ig_steps,
                                    feature_type='edge', device=args.device)
        edge_attributions_all.append(attrs)

    edge_attributions = np.mean(edge_attributions_all, axis=0)  # Average across samples
    print(f"✅ Edge feature attributions computed: {edge_attributions.shape}")

    # Save attributions to CSV
    node_feature_names = [
        # Element type (10-dim)
        'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H',
        # Residue type (21-dim)
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'LIG',
        # Other (21-dim)
        'ligand_flag', 'min_distance',
        'elec_0', 'elec_1', 'elec_2', 'elec_3', 'elec_4', 'elec_5', 'elec_6', 'elec_7',
        'elec_8', 'elec_9', 'elec_10', 'elec_11', 'elec_12', 'elec_13', 'elec_14', 'elec_15',
        'mass', 'electronegativity', 'radius'
    ]

    edge_feature_names = [f'RBF_{i}' for i in range(16)] + \
                        [f'Angle_{i}' for i in range(4)] + \
                        [f'Dihedral_{i}' for i in range(4)]

    node_df = pd.DataFrame({
        'feature_name': node_feature_names,
        'importance': node_attributions
    }).sort_values('importance', ascending=False)
    node_df.to_csv(output_dir / 'node_feature_importance.csv', index=False)
    print(f"✅ Saved node feature importance: {output_dir / 'node_feature_importance.csv'}")

    edge_df = pd.DataFrame({
        'feature_name': edge_feature_names,
        'importance': edge_attributions
    }).sort_values('importance', ascending=False)
    edge_df.to_csv(output_dir / 'edge_feature_importance.csv', index=False)
    print(f"✅ Saved edge feature importance: {output_dir / 'edge_feature_importance.csv'}")

    # Plot top-20 node features
    plot_feature_importance_barplot(
        node_attributions, node_feature_names,
        output_dir / 'figS12_node_feature_importance.png', top_k=20
    )

    # Plot top-20 edge features
    plot_feature_importance_barplot(
        edge_attributions, edge_feature_names,
        output_dir / 'figS12_edge_feature_importance.png', top_k=20
    )

    # Plot feature type comparison (RBF vs Angles vs Dihedrals)
    feature_type_stats = plot_feature_type_comparison(
        edge_attributions,
        output_dir / 'figS12b_feature_type_importance.png'
    )

    print(f"\n{'='*60}")
    print(f"Phase 2: Permutation-Based Importance")
    print(f"{'='*60}")

    # Test feature groups directly on dataset
    perm_results = {}

    print("\n🔄 Permuting RBF features (0:16)...")
    drop, baseline, permuted = permutation_feature_importance(
        model, test_dataset, list(range(16)), feature_type='edge',
        device=args.device, batch_size=args.batch_size
    )
    perm_results['RBF (16-dim)'] = {
        'importance_drop': drop,
        'baseline_corr': baseline,
        'permuted_corr': permuted
    }
    print(f"   Baseline Pearson: {baseline:.4f} → Permuted: {permuted:.4f} (Drop: {drop:.4f})")

    print("\n🔄 Permuting Angle features (16:20)...")
    drop, baseline, permuted = permutation_feature_importance(
        model, test_dataset, list(range(16, 20)), feature_type='edge',
        device=args.device, batch_size=args.batch_size
    )
    perm_results['Angles (4-dim)'] = {
        'importance_drop': drop,
        'baseline_corr': baseline,
        'permuted_corr': permuted
    }
    print(f"   Baseline Pearson: {baseline:.4f} → Permuted: {permuted:.4f} (Drop: {drop:.4f})")

    print("\n🔄 Permuting Dihedral features (20:24)...")
    drop, baseline, permuted = permutation_feature_importance(
        model, test_dataset, list(range(20, 24)), feature_type='edge',
        device=args.device, batch_size=args.batch_size
    )
    perm_results['Dihedrals (4-dim)'] = {
        'importance_drop': drop,
        'baseline_corr': baseline,
        'permuted_corr': permuted
    }
    print(f"   Baseline Pearson: {baseline:.4f} → Permuted: {permuted:.4f} (Drop: {drop:.4f})")

    print("\n🔄 Permuting ALL edge features (0:24)...")
    drop, baseline, permuted = permutation_feature_importance(
        model, test_dataset, list(range(24)), feature_type='edge',
        device=args.device, batch_size=args.batch_size
    )
    perm_results['All Edge (24-dim)'] = {
        'importance_drop': drop,
        'baseline_corr': baseline,
        'permuted_corr': permuted
    }
    print(f"   Baseline Pearson: {baseline:.4f} → Permuted: {permuted:.4f} (Drop: {drop:.4f})")

    # Plot permutation importance
    plot_permutation_importance(perm_results, output_dir / 'figS12c_permutation_importance.png')

    # Save summary
    summary = {
        'integrated_gradients': {
            'node_features': {
                'top_5': [
                    {'name': node_feature_names[i], 'importance': float(node_attributions[i])}
                    for i in np.argsort(node_attributions)[-5:][::-1]
                ],
                'total_importance': float(node_attributions.sum())
            },
            'edge_features': {
                'top_5': [
                    {'name': edge_feature_names[i], 'importance': float(edge_attributions[i])}
                    for i in np.argsort(edge_attributions)[-5:][::-1]
                ],
                'total_importance': float(edge_attributions.sum()),
                'by_type': feature_type_stats
            }
        },
        'permutation_importance': perm_results
    }

    with open(output_dir / 'feature_importance_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Saved summary: {output_dir / 'feature_importance_summary.json'}")

    print(f"\n{'='*60}")
    print(f"✅ Feature importance analysis complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"{'='*60}")

    # Print key findings
    print(f"\n📊 Key Findings:")
    print(f"   Edge Feature Importance:")
    print(f"      RBF (distances):    {feature_type_stats['rbf']['percentage']:.1f}%")
    print(f"      Angles:             {feature_type_stats['angles']['percentage']:.1f}%")
    print(f"      Dihedrals:          {feature_type_stats['dihedrals']['percentage']:.1f}%")
    print(f"   Permutation Impact:")
    for group, result in perm_results.items():
        print(f"      {group:20s}: Pearson drop = {result['importance_drop']:.4f}")


if __name__ == '__main__':
    main()
