#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量口袋编码器训练脚本（EGNN）
用于生成可迁移的Pocket Embedding，不影响主训练流程
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

import GNN_model as MD


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_dataset(dataset):
    flat = []
    for item in dataset:
        if hasattr(item, 'graphs'):
            for g in item.graphs:
                g.y = item.y
                if hasattr(item, 'sample_id'):
                    g.sample_id = item.sample_id
                flat.append(g)
        else:
            flat.append(item)
    return flat


def split_dataset(dataset, seed=42, ratios=(0.8, 0.1, 0.1)):
    indices = np.arange(len(dataset))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n = len(indices)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return [dataset[i] for i in train_idx], [dataset[i] for i in val_idx], [dataset[i] for i in test_idx]


def export_embeddings(encoder, data_list, save_path, device):
    encoder.eval()
    loader = DataLoader(data_list, batch_size=64, shuffle=False)
    records = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb = encoder.get_graph_embedding(batch).detach().cpu()
            sample_ids = getattr(batch, 'sample_id', None)
            if sample_ids is None:
                for i in range(emb.size(0)):
                    records.append({"index": len(records), "embedding": emb[i]})
            else:
                for i in range(emb.size(0)):
                    records.append({"sample_id": str(sample_ids[i]), "embedding": emb[i]})
    torch.save(records, save_path)


def main():
    parser = argparse.ArgumentParser(description="Train Pocket EGNN Encoder")
    parser.add_argument('--dataset', type=str, required=True, help='Path to .pt dataset')
    parser.add_argument('--save_dir', type=str, default='outputs/pocket_encoder', help='Output directory')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--pooling_type', type=str, default='mean', choices=['mean', 'global_attention', 'set2set'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_ratios', type=str, default='0.8,0.1,0.1')
    parser.add_argument('--export_embeddings', action='store_true')
    parser.add_argument('--embeddings_path', type=str, default='outputs/pocket_encoder/embeddings.pt')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    dataset = torch.load(args.dataset, weights_only=False)
    dataset = flatten_dataset(dataset)

    ratios = [float(x.strip()) for x in args.split_ratios.split(',')]
    train_list, val_list, test_list = split_dataset(dataset, seed=args.seed, ratios=ratios)

    train_loader = DataLoader(train_list, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    node_dim = train_list[0].x.shape[1]
    edge_dim = train_list[0].edge_attr.shape[1]

    encoder = MD.PocketEGNNEncoder(
        node_input_dim=node_dim,
        edge_input_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pooling_type=args.pooling_type
    ).to(device)

    head = nn.Sequential(
        nn.LayerNorm(encoder.readout_dim),
        nn.Linear(encoder.readout_dim, args.hidden_dim),
        nn.ReLU(),
        nn.Dropout(args.dropout),
        nn.Linear(args.hidden_dim, 1)
    ).to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_val = float('inf')
    best_path = os.path.join(args.save_dir, 'best_encoder.pt')

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        head.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            graph_x = encoder(batch)
            pred = head(graph_x).view(-1)
            loss = criterion(pred, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs

        train_loss /= max(1, len(train_list))

        encoder.eval()
        head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                graph_x = encoder(batch)
                pred = head(graph_x).view(-1)
                loss = criterion(pred, batch.y.view(-1))
                val_loss += loss.item() * batch.num_graphs
        val_loss /= max(1, len(val_list))

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "encoder_state_dict": encoder.state_dict(),
                "head_state_dict": head.state_dict(),
                "config": vars(args)
            }, best_path)

    print(f"✅ Best encoder saved to {best_path}")

    if args.export_embeddings:
        export_embeddings(encoder, dataset, args.embeddings_path, device=device)
        print(f"✅ Embeddings saved to {args.embeddings_path}")

    config_path = os.path.join(args.save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)


if __name__ == '__main__':
    main()
