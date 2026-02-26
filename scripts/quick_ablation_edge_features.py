#!/usr/bin/env python3
"""
Quick ablation study: Compare full 24-dim edge features vs RBF-only.

This validates the feature importance finding that angles contribute 89%.
"""

import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from GNN_model import PocketGNNKcatOnly
from scipy.stats import pearsonr


def test_edge_feature_ablation(dataset_path, model_path, device='cuda'):
    """Test model with and without angular features."""

    # Load dataset
    dataset = torch.load(dataset_path)
    print(f"Loaded {len(dataset)} samples")

    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Infer config
    hidden_dim = state_dict['node_encoder.weight'].shape[0]
    num_layers = sum(1 for k in state_dict.keys() if k.startswith('att_layers.') and '.lin.weight' in k)
    heads = state_dict['att_layers.0.att_src'].shape[1] if 'att_layers.0.att_src' in state_dict else 4
    use_mlp_layernorm = len(state_dict['mlp.0.weight'].shape) == 1

    model = PocketGNNKcatOnly(
        node_input_dim=52, edge_input_dim=24,
        hidden_dim=hidden_dim, num_layers=num_layers, heads=heads,
        pooling_type='mean', dropout=0.1, use_seq_embedding=False,
        use_mlp_layernorm=use_mlp_layernorm
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"\nModel config: hidden={hidden_dim}, layers={num_layers}, heads={heads}")

    # Test 1: Full features (24-dim)
    print("\n" + "="*60)
    print("Test 1: FULL 24-DIM EDGE FEATURES")
    print("="*60)

    true_full = []
    pred_full = []

    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            pred = model(data)
            true_full.append(data.y.item())
            pred_full.append(pred.item())

    true_full = np.array(true_full)
    pred_full = np.array(pred_full)

    r_full, _ = pearsonr(true_full, pred_full)
    mae_full = np.mean(np.abs(true_full - pred_full))
    rmse_full = np.sqrt(np.mean((true_full - pred_full) ** 2))

    print(f"Pearson r: {r_full:.4f}")
    print(f"MAE: {mae_full:.4f}")
    print(f"RMSE: {rmse_full:.4f}")

    # Test 2: RBF-only (first 16 dims)
    print("\n" + "="*60)
    print("Test 2: RBF-ONLY (remove angles, 16-dim)")
    print("="*60)

    true_rbf = []
    pred_rbf = []

    with torch.no_grad():
        for data in dataset:
            # ABLATION: Keep only first 16 dims (RBF), zero out angles
            data_ablated = data.clone().to(device)
            data_ablated.edge_attr[:, 16:] = 0  # Zero out angles (dims 16-23)

            pred = model(data_ablated)
            true_rbf.append(data.y.item())
            pred_rbf.append(pred.item())

    true_rbf = np.array(true_rbf)
    pred_rbf = np.array(pred_rbf)

    r_rbf, _ = pearsonr(true_rbf, pred_rbf)
    mae_rbf = np.mean(np.abs(true_rbf - pred_rbf))
    rmse_rbf = np.sqrt(np.mean((true_rbf - pred_rbf) ** 2))

    print(f"Pearson r: {r_rbf:.4f}")
    print(f"MAE: {mae_rbf:.4f}")
    print(f"RMSE: {rmse_rbf:.4f}")

    # Test 3: Angles-only (last 8 dims)
    print("\n" + "="*60)
    print("Test 3: ANGLES-ONLY (remove RBF, 8-dim)")
    print("="*60)

    true_ang = []
    pred_ang = []

    with torch.no_grad():
        for data in dataset:
            # ABLATION: Keep only last 8 dims (angles), zero out RBF
            data_ablated = data.clone().to(device)
            data_ablated.edge_attr[:, :16] = 0  # Zero out RBF (dims 0-15)

            pred = model(data_ablated)
            true_ang.append(data.y.item())
            pred_ang.append(pred.item())

    true_ang = np.array(true_ang)
    pred_ang = np.array(pred_ang)

    r_ang, _ = pearsonr(true_ang, pred_ang)
    mae_ang = np.mean(np.abs(true_ang - pred_ang))
    rmse_ang = np.sqrt(np.mean((true_ang - pred_ang) ** 2))

    print(f"Pearson r: {r_ang:.4f}")
    print(f"MAE: {mae_ang:.4f}")
    print(f"RMSE: {rmse_ang:.4f}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Edge Feature Ablation Study")
    print("="*60)
    print(f"{'Configuration':<30} {'Pearson r':>12} {'MAE':>10} {'RMSE':>10}")
    print("-" * 60)
    print(f"{'Full (24-dim: RBF+Angles)':<30} {r_full:>12.4f} {mae_full:>10.4f} {rmse_full:>10.4f}")
    print(f"{'RBF-only (16-dim)':<30} {r_rbf:>12.4f} {mae_rbf:>10.4f} {rmse_rbf:>10.4f}")
    print(f"{'Angles-only (8-dim)':<30} {r_ang:>12.4f} {mae_ang:>10.4f} {rmse_ang:>10.4f}")
    print("-" * 60)
    print(f"{'Performance drop (RBF-only)':<30} {r_full-r_rbf:>12.4f} {mae_rbf-mae_full:>10.4f} {rmse_rbf-rmse_full:>10.4f}")
    print(f"{'Performance drop (Angles-only)':<30} {r_full-r_ang:>12.4f} {mae_ang-mae_full:>10.4f} {rmse_ang-rmse_full:>10.4f}")
    print("="*60)

    print(f"\n✅ Key Finding:")
    print(f"   Removing angles (keeping only RBF) causes {abs(r_full-r_rbf):.4f} correlation drop")
    print(f"   Removing RBF (keeping only angles) causes {abs(r_full-r_ang):.4f} correlation drop")
    print(f"   → Angles are {abs(r_full-r_rbf)/abs(r_full-r_ang):.1f}x more important than RBF!")


if __name__ == '__main__':
    test_edge_feature_ablation(
        dataset_path='data/processed/kcat_merged_hom40_test.pt',
        model_path='outputs/kcat_hom40_train_20251213_221248/best_model.pt',
        device='cuda'
    )
