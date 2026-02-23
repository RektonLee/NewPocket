#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量自监督预训练：Masked Atom/Residue + 距离重建 + 对比学习
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
                if hasattr(item, 'sample_id'):
                    g.sample_id = item.sample_id
                flat.append(g)
        else:
            flat.append(item)
    return flat


def split_dataset(dataset, seed=42, ratios=(0.9, 0.1)):
    indices = np.arange(len(dataset))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n = len(indices)
    n_train = int(n * ratios[0])
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    return [dataset[i] for i in train_idx], [dataset[i] for i in val_idx]


def make_view(batch, feature_drop=0.1, coord_noise=0.2):
    view = batch.clone()
    if feature_drop > 0:
        drop_mask = torch.rand_like(view.x) < feature_drop
        view.x = view.x.masked_fill(drop_mask, 0.0)
    if coord_noise > 0:
        view.pos = view.pos + torch.randn_like(view.pos) * coord_noise
    return view


def apply_node_mask(view, mask_ratio, element_slice, residue_slice):
    num_nodes = view.x.size(0)
    mask = torch.rand(num_nodes, device=view.x.device) < mask_ratio
    if mask.sum() == 0:
        return view, None
    elem_feat = view.x[mask, element_slice]
    res_feat = view.x[mask, residue_slice]
    view.x[mask, element_slice] = 0.0
    view.x[mask, residue_slice] = 0.0
    elem_label = elem_feat.argmax(dim=-1)
    res_label = res_feat.argmax(dim=-1)
    return view, (mask, elem_label, res_label)


def info_nce(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = torch.matmul(z1, z2.t()) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) * 0.5
    return loss


class PocketPretrainModel(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim=256, num_layers=4, dropout=0.1,
                 pooling_type='mean', element_classes=10, residue_classes=21):
        super().__init__()
        self.encoder = MD.PocketEGNNEncoder(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pooling_type=pooling_type
        )
        self.element_head = nn.Linear(hidden_dim, element_classes)
        self.residue_head = nn.Linear(hidden_dim, residue_classes)
        self.dist_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data, return_graph=True):
        node_emb, graph_emb = self.encoder(data, return_node_embeddings=True, return_graph_embedding=return_graph)
        return node_emb, graph_emb


def main():
    parser = argparse.ArgumentParser(description="Pretrain Pocket EGNN Encoder")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='outputs/pocket_pretrain')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--pooling_type', type=str, default='mean', choices=['mean', 'global_attention', 'set2set'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mask_ratio', type=float, default=0.15)
    parser.add_argument('--feature_drop', type=float, default=0.1)
    parser.add_argument('--coord_noise', type=float, default=0.2)
    parser.add_argument('--max_edges', type=int, default=8000)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--w_mask', type=float, default=1.0)
    parser.add_argument('--w_dist', type=float, default=0.5)
    parser.add_argument('--w_contrast', type=float, default=1.0)
    parser.add_argument('--element_slice', type=str, default='0:10')
    parser.add_argument('--residue_slice', type=str, default='10:31')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    dataset = torch.load(args.dataset, weights_only=False)
    dataset = flatten_dataset(dataset)
    train_list, val_list = split_dataset(dataset, seed=args.seed)

    train_loader = DataLoader(train_list, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    node_dim = train_list[0].x.shape[1]
    edge_dim = train_list[0].edge_attr.shape[1]

    elem_start, elem_end = [int(x) for x in args.element_slice.split(':')]
    res_start, res_end = [int(x) for x in args.residue_slice.split(':')]

    model = PocketPretrainModel(
        node_input_dim=node_dim,
        edge_input_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pooling_type=args.pooling_type,
        element_classes=elem_end - elem_start,
        residue_classes=res_end - res_start
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float('inf')
    best_path = os.path.join(args.save_dir, 'best_pretrain.pt')

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            view1 = make_view(batch, feature_drop=args.feature_drop, coord_noise=args.coord_noise)
            view2 = make_view(batch, feature_drop=args.feature_drop, coord_noise=args.coord_noise)

            view1, mask_info = apply_node_mask(view1, args.mask_ratio, slice(elem_start, elem_end), slice(res_start, res_end))

            optimizer.zero_grad()
            node_emb1, graph_emb1 = model(view1, return_graph=True)
            _, graph_emb2 = model(view2, return_graph=True)

            loss_mask = torch.tensor(0.0, device=device)
            if mask_info is not None:
                mask, elem_label, res_label = mask_info
                elem_logits = model.element_head(node_emb1[mask])
                res_logits = model.residue_head(node_emb1[mask])
                loss_mask = F.cross_entropy(elem_logits, elem_label) + F.cross_entropy(res_logits, res_label)

            loss_dist = torch.tensor(0.0, device=device)
            if args.w_dist > 0:
                row, col = view1.edge_index
                num_edges = row.size(0)
                if num_edges > args.max_edges:
                    idx = torch.randperm(num_edges, device=device)[:args.max_edges]
                    row = row[idx]
                    col = col[idx]
                pair_emb = torch.cat([node_emb1[row], node_emb1[col]], dim=-1)
                dist_pred = model.dist_head(pair_emb).view(-1)
                dist_true = torch.norm(view1.pos[row] - view1.pos[col], dim=-1)
                loss_dist = F.mse_loss(dist_pred, dist_true)

            loss_contrast = info_nce(graph_emb1, graph_emb2, temperature=args.temperature)

            loss = args.w_mask * loss_mask + args.w_dist * loss_dist + args.w_contrast * loss_contrast
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        total_loss /= max(1, len(train_list))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                view1 = make_view(batch, feature_drop=args.feature_drop, coord_noise=args.coord_noise)
                view2 = make_view(batch, feature_drop=args.feature_drop, coord_noise=args.coord_noise)
                view1, mask_info = apply_node_mask(view1, args.mask_ratio, slice(elem_start, elem_end), slice(res_start, res_end))

                node_emb1, graph_emb1 = model(view1, return_graph=True)
                _, graph_emb2 = model(view2, return_graph=True)

                loss_mask = torch.tensor(0.0, device=device)
                if mask_info is not None:
                    mask, elem_label, res_label = mask_info
                    elem_logits = model.element_head(node_emb1[mask])
                    res_logits = model.residue_head(node_emb1[mask])
                    loss_mask = F.cross_entropy(elem_logits, elem_label) + F.cross_entropy(res_logits, res_label)

                loss_dist = torch.tensor(0.0, device=device)
                if args.w_dist > 0:
                    row, col = view1.edge_index
                    num_edges = row.size(0)
                    if num_edges > args.max_edges:
                        idx = torch.randperm(num_edges, device=device)[:args.max_edges]
                        row = row[idx]
                        col = col[idx]
                    pair_emb = torch.cat([node_emb1[row], node_emb1[col]], dim=-1)
                    dist_pred = model.dist_head(pair_emb).view(-1)
                    dist_true = torch.norm(view1.pos[row] - view1.pos[col], dim=-1)
                    loss_dist = F.mse_loss(dist_pred, dist_true)

                loss_contrast = info_nce(graph_emb1, graph_emb2, temperature=args.temperature)
                loss = args.w_mask * loss_mask + args.w_dist * loss_dist + args.w_contrast * loss_contrast
                val_loss += loss.item() * batch.num_graphs

        val_loss /= max(1, len(val_list))
        print(f"Epoch {epoch:03d} | train_loss={total_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": vars(args)
            }, best_path)

    print(f"✅ Best pretrain model saved to {best_path}")
    config_path = os.path.join(args.save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)


if __name__ == '__main__':
    main()
