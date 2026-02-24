#!/usr/bin/env python3
"""
Compare pocket representations: Simple 5Å extraction vs PHPTransformer
Evaluate using clustering quality metrics (no downstream task)
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

sys.path.insert(0, 'src')

def load_pocket_data(dataset_path: str, max_samples: int = 500):
    """Load pocket dataset"""
    print(f"Loading {dataset_path}...")
    data_list = torch.load(dataset_path)
    if len(data_list) > max_samples:
        indices = np.random.choice(len(data_list), max_samples, replace=False)
        data_list = [data_list[i] for i in indices]
    print(f"  Loaded {len(data_list)} samples")
    return data_list

def extract_simple_representation(data_list):
    """Simple representation: mean pooling of node features"""
    representations = []
    for data in data_list:
        if hasattr(data, 'x') and data.x is not None:
            # Simple mean pooling
            rep = data.x.mean(dim=0).numpy()
            representations.append(rep)
    return np.array(representations)

def extract_graph_stats(data_list):
    """Extract graph-level statistics as features"""
    stats = []
    for data in data_list:
        x = data.x.numpy() if hasattr(data, 'x') and data.x is not None else None
        edge_attr = data.edge_attr.numpy() if hasattr(data, 'edge_attr') and data.edge_attr is not None else None

        if x is None:
            continue

        # Node features stats
        node_mean = x.mean(axis=0)
        node_std = x.std(axis=0)

        # Edge features stats (if available)
        if edge_attr is not None and len(edge_attr) > 0:
            edge_mean = edge_attr.mean(axis=0)
            edge_std = edge_attr.std(axis=0)
        else:
            edge_mean = np.zeros(24)
            edge_std = np.zeros(24)

        # Graph-level stats
        num_nodes = len(x)
        num_edges = len(edge_attr) if edge_attr is not None else 0

        # Combine
        stat = np.concatenate([
            node_mean, node_std[:10],  # Node stats
            edge_mean, edge_std[:8],   # Edge stats
            [num_nodes, num_edges]     # Size stats
        ])
        stats.append(stat)

    return np.array(stats)

def evaluate_clustering(representations, ec_labels=None, name=""):
    """Evaluate representation quality using clustering metrics"""
    print(f"\n{name} Representation Evaluation:")
    print("-" * 40)

    # Remove any NaN/inf
    valid_mask = ~(np.isnan(representations).any(axis=1) | np.isinf(representations).any(axis=1))
    representations = representations[valid_mask]

    if len(representations) < 10:
        print("  Not enough valid samples")
        return {}

    # Normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(representations)

    # Clustering with different K
    results = {}
    for k in [3, 5, 7, 10]:
        if len(X) < k * 3:
            continue

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Metrics
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)

        print(f"  K={k}:")
        print(f"    Silhouette: {silhouette:.3f} (higher better)")
        print(f"    Davies-Bouldin: {davies_bouldin:.3f} (lower better)")
        print(f"    Calinski-Harabasz: {calinski:.1f} (higher better)")

        results[f'k{k}_silhouette'] = silhouette
        results[f'k{k}_davies_bouldin'] = davies_bouldin
        results[f'k{k}_calinski'] = calinski

    # t-SNE visualization
    if len(X) > 50:
        print("  Generating t-SNE plot...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
        X_tsne = tsne.fit_transform(X)
        results['tsne'] = X_tsne

    return results

def compare_representations(data_list, output_dir: Path):
    """Compare different representation methods"""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Pocket Representation Quality Comparison")
    print("=" * 60)
    print(f"Samples: {len(data_list)}")

    # 1. Simple mean pooling (baseline)
    simple_rep = extract_simple_representation(data_list)
    simple_results = evaluate_clustering(simple_rep, name="Simple Mean Pooling (5Å)")

    # 2. Graph statistics (enhanced features)
    stats_rep = extract_graph_stats(data_list)
    stats_results = evaluate_clustering(stats_rep, name="Graph Statistics (24D edges)")

    # Generate comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # t-SNE plots
    if 'tsne' in simple_results:
        axes[0].scatter(simple_results['tsne'][:, 0], simple_results['tsne'][:, 1],
                       c='steelblue', alpha=0.6, s=20)
        axes[0].set_title('Simple 5Å Extraction\n(Mean Pooling)', fontsize=12)
        axes[0].set_xlabel('t-SNE 1')
        axes[0].set_ylabel('t-SNE 2')

    if 'tsne' in stats_results:
        axes[1].scatter(stats_results['tsne'][:, 0], stats_results['tsne'][:, 1],
                       c='coral', alpha=0.6, s=20)
        axes[1].set_title('Enhanced Features\n(24D Edge + Graph Stats)', fontsize=12)
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')

    plt.tight_layout()
    fig_path = output_dir / "representation_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {fig_path}")
    plt.close()

    # Summary comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    comparison = []
    for k in [5, 7]:
        if f'k{k}_silhouette' in simple_results and f'k{k}_silhouette' in stats_results:
            comparison.append({
                'K': k,
                'Simple_Silhouette': simple_results[f'k{k}_silhouette'],
                'Enhanced_Silhouette': stats_results[f'k{k}_silhouette'],
                'Simple_DBI': simple_results[f'k{k}_davies_bouldin'],
                'Enhanced_DBI': stats_results[f'k{k}_davies_bouldin'],
            })

    if comparison:
        comp_df = pd.DataFrame(comparison)
        print("\nSilhouette Score (higher = better clustering):")
        for _, row in comp_df.iterrows():
            diff = row['Enhanced_Silhouette'] - row['Simple_Silhouette']
            print(f"  K={int(row['K'])}: Simple={row['Simple_Silhouette']:.3f}, "
                  f"Enhanced={row['Enhanced_Silhouette']:.3f} ({'+' if diff > 0 else ''}{diff:.3f})")

        # Save
        comp_df.to_csv(output_dir / "representation_comparison.csv", index=False)

    print("\n✅ Comparison complete!")

def main():
    # Use existing processed dataset
    dataset_path = "data/processed/kcat_merged_hom40_test.pt"
    output_dir = Path("results/representation_comparison")

    if not Path(dataset_path).exists():
        # Try alternative
        dataset_path = "data/processed/kcat_test_new_backup.pt"

    if not Path(dataset_path).exists():
        print(f"Dataset not found: {dataset_path}")
        return

    # Load data
    data_list = load_pocket_data(dataset_path, max_samples=300)

    # Compare representations
    compare_representations(data_list, output_dir)

if __name__ == "__main__":
    main()
