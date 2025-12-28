import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import os
import argparse
import GNN_model as MD
from torch_geometric.loader import DataLoader

def predict(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    print(f"Loading test dataset: {args.dataset}...")
    data_list = torch.load(args.dataset, weights_only=False)
    
    # Preprocessing (remove metadata)
    for data in data_list:
        if hasattr(data, 'ec'): delattr(data, 'ec')
        if hasattr(data, 'pdb_id'): delattr(data, 'pdb_id')
        if hasattr(data, 'sample_id'): delattr(data, 'sample_id')
        if hasattr(data, 'uniprot_id'): delattr(data, 'uniprot_id')
        
    # Load Embeddings
    seq_embedding_dim = 0
    if args.use_seq_embedding:
        print(f"Loading embeddings from {args.embedding_path}...")
        embeddings_map = torch.load(args.embedding_path, weights_only=False)
        # 简单匹配逻辑 (这里假设测试集的 sample_id 已经对应好了，或者需要重新匹配)
        # 注意：这里我们加载的 data_list 已经没有 sample_id 了... 
        # 哎呀，这是个问题。我们需要在删除 sample_id 之前匹配。
        # 重新加载一遍吧
        data_list = torch.load(args.dataset, weights_only=False)
        
        matched_count = 0
        for data in data_list:
            key = str(getattr(data, 'sample_id', ''))
            if key in embeddings_map:
                emb = embeddings_map[key]
                if not isinstance(emb, torch.Tensor):
                    emb = torch.tensor(emb, dtype=torch.float)
                if seq_embedding_dim == 0: seq_embedding_dim = emb.shape[0]
                data.seq_embedding = emb.unsqueeze(0)
                matched_count += 1
            else:
                # 尝试 pdb_id
                key = str(getattr(data, 'pdb_id', ''))
                if key in embeddings_map:
                    emb = embeddings_map[key]
                    if not isinstance(emb, torch.Tensor):
                        emb = torch.tensor(emb, dtype=torch.float)
                    if seq_embedding_dim == 0: seq_embedding_dim = emb.shape[0]
                    data.seq_embedding = emb.unsqueeze(0)
                    matched_count += 1
        
        print(f"Matched embeddings: {matched_count}/{len(data_list)}")
        
        # 再次删除 metadata
        for data in data_list:
            if hasattr(data, 'ec'): delattr(data, 'ec')
            if hasattr(data, 'pdb_id'): delattr(data, 'pdb_id')
            if hasattr(data, 'sample_id'): delattr(data, 'sample_id')
            if hasattr(data, 'uniprot_id'): delattr(data, 'uniprot_id')
            if not hasattr(data, 'seq_embedding') and seq_embedding_dim > 0:
                 data.seq_embedding = torch.zeros((1, seq_embedding_dim), dtype=torch.float)

    loader = DataLoader(data_list, batch_size=32, shuffle=False)
    
    # 2. Load Model
    print("Initializing model...")
    sample = data_list[0]
    node_dim = sample.x.shape[1]
    edge_dim = sample.edge_attr.shape[1]
    
    model = MD.PocketGNNKcatOnly(
        node_input_dim=node_dim,
        edge_input_dim=edge_dim,
        hidden_dim=128,
        num_layers=3,
        heads=4,
        dropout=0.1,
        pooling_type=args.pooling_type,
        use_seq_embedding=args.use_seq_embedding,
        seq_embedding_dim=seq_embedding_dim
    ).to(device)
    
    print(f"Loading weights from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # 3. Predict
    y_true = []
    y_pred = []
    
    print("Predicting...")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            
            # Label handling
            if batch.y.shape[0] == batch.num_graphs:
                y = batch.y.reshape(-1, 1)
            else:
                y = batch.y.reshape(batch.num_graphs, 2)[:, 0:1]
                
            y_true.append(y.cpu())
            y_pred.append(out.cpu())
            
    y_true = torch.cat(y_true, dim=0).numpy().flatten()
    y_pred = torch.cat(y_pred, dim=0).numpy().flatten()
    
    # 4. Metrics & Plot
    r2 = r2_score(y_true, y_pred)
    pearson = pearsonr(y_true, y_pred)[0]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"Test Results:")
    print(f"  R2: {r2:.4f}")
    print(f"  Pearson: {pearson:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Range True: [{y_true.min():.2f}, {y_true.max():.2f}]")
    print(f"  Range Pred: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    
    # Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect')
    plt.xlabel('True kcat')
    plt.ylabel('Predicted kcat')
    plt.title(f'Test Set Prediction\nR2={r2:.3f}, r={pearson:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(os.path.dirname(args.model_path), 'test_scatter_check.png'))
    print(f"Plot saved to {os.path.join(os.path.dirname(args.model_path), 'test_scatter_check.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--embedding_path', default='data/processed/esm_embeddings.pt')
    parser.add_argument('--pooling_type', default='set2set')
    parser.add_argument('--use_seq_embedding', action='store_true')
    args = parser.parse_args()
    predict(args)

