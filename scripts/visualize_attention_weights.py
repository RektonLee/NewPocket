#!/usr/bin/env python3
"""
Attention Weight Visualization for PocketGNN
=============================================

Extract and visualize GAT attention weights to understand which atomic interactions
the model focuses on during prediction.

Usage:
    python scripts/visualize_attention_weights.py \
        --model outputs/kcat_hom40_train_20251213_221248/best_model.pt \
        --test_dataset data/processed/kcat_hom40_test.pt \
        --output_dir results/interpretability/attention \
        --sample_ids kcat_001234,kcat_005678  # Optional: specific samples to analyze

Outputs:
    - figS11_attention_heatmap_{sample_id}.png: 3D structure with attention overlay
    - figS11b_attention_flow_layers.png: Attention distribution across layers
    - attention_weights_{sample_id}.csv: Per-atom attention scores
    - attention_summary.csv: Aggregated statistics across all samples
"""

import argparse
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import torch
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from GNN_model import PocketGNNKcatOnly
from Bio.PDB import PDBParser


def load_model(model_path, device='cuda'):
    """
    Load trained PocketGNN model

    Args:
        model_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract model config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Infer config from checkpoint keys
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Infer hidden_dim from node_encoder
        hidden_dim = state_dict['node_encoder.weight'].shape[0]

        # Count GAT layers
        num_layers = sum(1 for k in state_dict.keys() if k.startswith('att_layers.') and '.lin.weight' in k)

        # Infer heads from att_layers shape
        # att_layers.0.att_src has shape [1, heads, out_channels_per_head]
        if 'att_layers.0.att_src' in state_dict:
            heads = state_dict['att_layers.0.att_src'].shape[1]  # Middle dimension is heads
        else:
            heads = 4  # default

        # Infer use_mlp_layernorm from checkpoint
        # If mlp.0 has key like "mlp.0.weight" with 1D shape, it's LayerNorm (use_mlp_layernorm=True)
        # If mlp.0 has 2D shape, it's Linear (use_mlp_layernorm=False)
        use_mlp_layernorm = len(state_dict['mlp.0.weight'].shape) == 1

        config = {
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'heads': heads,
            'pooling_type': 'mean',  # inferred
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

    # Try to load, handling potential mismatches
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"⚠️  Warning: Loading with strict=False due to: {e}")
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    return model, config


def extract_attention_weights(model, data, device='cuda'):
    """
    Extract attention weights from all GAT layers

    Args:
        model: Trained PocketGNN model
        data: PyG Data object (single sample)
        device: Device

    Returns:
        all_layer_attns: List of attention tensors [num_layers]
                        Each: [num_edges, num_heads] after aggregation
        edge_index: Edge connectivity [2, num_edges]
        prediction: Model prediction (scalar)
    """
    data = data.to(device)

    with torch.no_grad():
        # Model returns: (output, all_attention_weights)
        # all_attention_weights is a list of attention tensors [num_layers]
        output, all_attention_weights = model(data, return_attention_weights=True)

        # Process attention weights
        # Each attention tensor shape: [num_edges, num_heads] (already in correct format from GAT)
        all_layer_attns = []
        for layer_idx, attn_weights in enumerate(all_attention_weights):
            # attn_weights: [num_edges, num_heads] (from GATConv)
            # Ensure correct shape
            if attn_weights.dim() == 1:
                # Edge case: single head, reshape to [num_edges, 1]
                attn_weights = attn_weights.unsqueeze(-1)

            all_layer_attns.append(attn_weights)

        return all_layer_attns, data.edge_index, output.item()


def aggregate_edge_to_node_attention(edge_attn, edge_index, num_nodes):
    """
    Aggregate edge attention scores to node-level importance

    Args:
        edge_attn: Edge attention scores [num_edges]
        edge_index: Edge connectivity [2, num_edges]
        num_nodes: Number of nodes

    Returns:
        node_attn: Node-level attention [num_nodes]
    """
    # Sum incoming attention for each node
    node_attn = torch.zeros(num_nodes, device=edge_attn.device)

    for e in range(edge_index.shape[1]):
        src, dst = edge_index[0, e], edge_index[1, e]
        node_attn[dst] += edge_attn[e]

    # Normalize by in-degree
    in_degree = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.float32)
    node_attn = node_attn / (in_degree + 1e-8)

    return node_attn


def visualize_3d_attention(pdb_path, node_attn, sample_id, output_path,
                          title="Attention Heatmap", layer_name="Layer 3"):
    """
    Visualize attention on 3D protein structure

    Args:
        pdb_path: Path to PDB file
        node_attn: Node attention scores [num_nodes]
        sample_id: Sample ID
        output_path: Where to save figure
        title: Figure title
        layer_name: Which layer attention is from
    """
    # Parse PDB structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pocket', pdb_path)

    # Extract atom coordinates
    coords = []
    atom_names = []
    residues = []

    for atom in structure.get_atoms():
        coords.append(atom.get_coord())
        atom_names.append(atom.get_name())
        res = atom.get_parent()
        residues.append(f"{res.get_resname()}{res.id[1]}")

    coords = np.array(coords)

    # Ensure node_attn matches number of atoms
    if len(node_attn) != len(coords):
        print(f"Warning: Attention length ({len(node_attn)}) != num atoms ({len(coords)})")
        # Pad or truncate
        if len(node_attn) < len(coords):
            padding = np.zeros(len(coords) - len(node_attn))
            node_attn = np.concatenate([node_attn, padding])
        else:
            node_attn = node_attn[:len(coords)]

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with attention-based coloring
    colors = node_attn
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                   c=colors, cmap='hot', s=80, alpha=0.8,
                   vmin=0, vmax=np.percentile(colors, 95), edgecolors='black', linewidths=0.5)

    # Annotate top-5 attention atoms
    top5_idx = np.argsort(colors)[-5:]
    for rank, idx in enumerate(reversed(top5_idx)):
        label = f"{residues[idx]}-{atom_names[idx]}"
        ax.text(coords[idx, 0], coords[idx, 1], coords[idx, 2],
               f"  {label}\n  ({colors[idx]:.3f})",
               fontsize=9, color='red', weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, label='Attention Score', shrink=0.6, pad=0.1)
    cbar.ax.tick_params(labelsize=10)

    # Labels and title
    ax.set_xlabel('X (Å)', fontsize=11)
    ax.set_ylabel('Y (Å)', fontsize=11)
    ax.set_zlabel('Z (Å)', fontsize=11)
    ax.set_title(f'{title} - {sample_id}\n{layer_name}', fontsize=13, weight='bold')

    # Set equal aspect ratio for all axes
    max_range = np.array([coords[:, 0].max()-coords[:, 0].min(),
                         coords[:, 1].max()-coords[:, 1].min(),
                         coords[:, 2].max()-coords[:, 2].min()]).max() / 2.0

    mid_x = (coords[:, 0].max()+coords[:, 0].min()) * 0.5
    mid_y = (coords[:, 1].max()+coords[:, 1].min()) * 0.5
    mid_z = (coords[:, 2].max()+coords[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved attention heatmap: {output_path}")


def plot_attention_flow_across_layers(all_layer_attns, edge_index, num_nodes,
                                      sample_id, output_path):
    """
    Visualize how attention distribution changes across GAT layers

    Args:
        all_layer_attns: List of attention tensors [num_layers x [num_edges, num_heads]]
        edge_index: Edge connectivity
        num_nodes: Number of nodes
        sample_id: Sample ID
        output_path: Output path
    """
    num_layers = len(all_layer_attns)

    fig, axes = plt.subplots(1, num_layers, figsize=(6*num_layers, 5))

    if num_layers == 1:
        axes = [axes]

    for layer_idx, (ax, attn) in enumerate(zip(axes, all_layer_attns)):
        # Average across heads
        attn_avg = attn.mean(dim=1).cpu().numpy()  # [num_edges]

        # Aggregate to node level
        node_attn = aggregate_edge_to_node_attention(
            torch.from_numpy(attn_avg), edge_index, num_nodes
        ).cpu().numpy()

        # Histogram of attention distribution
        ax.hist(node_attn, bins=50, alpha=0.7,
               color=['blue', 'orange', 'red'][layer_idx % 3], edgecolor='black')
        ax.set_title(f'Layer {layer_idx+1}', fontsize=12, weight='bold')
        ax.set_xlabel('Node Attention Score', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.grid(alpha=0.3)

        # Compute Gini coefficient (sparsity measure)
        sorted_attn = np.sort(node_attn)
        n = len(sorted_attn)
        cumsum = np.cumsum(sorted_attn)
        gini = (2 * np.sum((np.arange(n) + 1) * sorted_attn)) / (n * cumsum[-1]) - (n + 1) / n

        # Add statistics
        stats_text = f"Mean: {node_attn.mean():.4f}\n"
        stats_text += f"Std: {node_attn.std():.4f}\n"
        stats_text += f"Gini: {gini:.3f}"
        ax.text(0.65, 0.88, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle(f'Attention Distribution Across Layers - {sample_id}',
                fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved attention flow plot: {output_path}")


def save_attention_csv(node_attn, atom_info, sample_id, output_path):
    """
    Save attention scores to CSV with atom metadata

    Args:
        node_attn: Node attention scores [num_nodes]
        atom_info: Dict with atom metadata (names, residues, etc.)
        sample_id: Sample ID
        output_path: Output CSV path
    """
    df = pd.DataFrame({
        'sample_id': sample_id,
        'atom_idx': range(len(node_attn)),
        'atom_name': atom_info.get('atom_names', [''] * len(node_attn)),
        'residue': atom_info.get('residues', [''] * len(node_attn)),
        'attention_score': node_attn
    })

    # Sort by attention (descending)
    df = df.sort_values('attention_score', ascending=False)

    df.to_csv(output_path, index=False)
    print(f"✅ Saved attention CSV: {output_path}")


def parse_pdb_metadata(pdb_path):
    """Extract atom names and residue info from PDB"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pocket', pdb_path)

    atom_names = []
    residues = []

    for atom in structure.get_atoms():
        atom_names.append(atom.get_name())
        res = atom.get_parent()
        residues.append(f"{res.get_resname()}{res.id[1]}")

    return {'atom_names': atom_names, 'residues': residues}


def main():
    parser = argparse.ArgumentParser(description="Visualize GAT attention weights")
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--test_dataset', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--output_dir', type=str, default='results/interpretability/attention',
                       help='Output directory')
    parser.add_argument('--sample_ids', type=str, default=None,
                       help='Comma-separated sample IDs to analyze (optional)')
    parser.add_argument('--num_samples', type=int, default=8,
                       help='Number of random samples if sample_ids not specified')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--pdb_dir', type=str, default='sample_data/samples',
                       help='Directory containing PDB files')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Loading model from: {args.model}")
    model, config = load_model(args.model, device=args.device)
    print(f"✅ Model loaded: {config}")

    print(f"\n📊 Loading test dataset: {args.test_dataset}")
    test_dataset = torch.load(args.test_dataset)
    print(f"✅ Test dataset loaded: {len(test_dataset)} samples")

    # Select samples to analyze
    if args.sample_ids:
        sample_ids = args.sample_ids.split(',')
        print(f"\n🎯 Analyzing specified samples: {sample_ids}")
    else:
        # Random sampling
        np.random.seed(42)
        indices = np.random.choice(len(test_dataset), args.num_samples, replace=False)
        sample_ids = [test_dataset[i].sample_id for i in indices]
        print(f"\n🎲 Randomly selected {args.num_samples} samples")

    # Process each sample
    summary_data = []

    for sample_id in sample_ids:
        print(f"\n{'='*60}")
        print(f"Processing: {sample_id}")
        print(f"{'='*60}")

        # Find sample in dataset
        data = None
        for d in test_dataset:
            if hasattr(d, 'sample_id') and d.sample_id == sample_id:
                data = d
                break

        if data is None:
            print(f"⚠️  Sample {sample_id} not found in dataset, skipping...")
            continue

        # Extract attention weights
        try:
            all_layer_attns, edge_index, prediction = extract_attention_weights(
                model, data, device=args.device
            )
            print(f"✅ Extracted attention from {len(all_layer_attns)} layers")
        except Exception as e:
            print(f"❌ Error extracting attention: {e}")
            continue

        # Use last layer (most refined attention)
        last_layer_attn = all_layer_attns[-1]  # [num_edges, num_heads]

        # Average across heads
        edge_attn_avg = last_layer_attn.mean(dim=1)  # [num_edges]

        # Aggregate to node level
        node_attn = aggregate_edge_to_node_attention(
            edge_attn_avg, edge_index, data.x.shape[0]
        ).cpu().numpy()

        print(f"📊 Node attention stats: mean={node_attn.mean():.4f}, std={node_attn.std():.4f}, max={node_attn.max():.4f}")

        # Find PDB file
        pdb_path = Path(args.pdb_dir) / sample_id / 'pocket.pdb'
        if not pdb_path.exists():
            pdb_path = Path(args.pdb_dir) / sample_id / f'{sample_id}_pocket.pdb'

        if not pdb_path.exists():
            print(f"⚠️  PDB file not found: {pdb_path}, skipping visualization...")
            continue

        # Parse PDB metadata
        atom_info = parse_pdb_metadata(pdb_path)

        # Visualize attention on 3D structure
        fig_path = output_dir / f'figS11_attention_heatmap_{sample_id}.png'
        visualize_3d_attention(pdb_path, node_attn, sample_id, fig_path,
                              layer_name=f"Layer {len(all_layer_attns)}")

        # Plot attention flow across layers
        flow_path = output_dir / f'figS11b_attention_flow_{sample_id}.png'
        plot_attention_flow_across_layers(all_layer_attns, edge_index, data.x.shape[0],
                                         sample_id, flow_path)

        # Save attention CSV
        csv_path = output_dir / f'attention_weights_{sample_id}.csv'
        save_attention_csv(node_attn, atom_info, sample_id, csv_path)

        # Aggregate summary statistics
        summary_data.append({
            'sample_id': sample_id,
            'num_atoms': len(node_attn),
            'mean_attention': node_attn.mean(),
            'std_attention': node_attn.std(),
            'max_attention': node_attn.max(),
            'top5_mean': np.sort(node_attn)[-5:].mean(),
            'prediction': prediction,
            'true_value': data.y.item() if hasattr(data, 'y') else None
        })

    # Save summary
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = output_dir / 'attention_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✅ Saved summary: {summary_path}")
        print(f"\n📊 Attention Summary:")
        print(summary_df.to_string(index=False))

    print(f"\n{'='*60}")
    print(f"✅ Attention visualization complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
