#!/usr/bin/env python3
"""
Quick test on kcat_test_new_diffdock.csv samples
"""
import sys
import pandas as pd
import torch
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_kcat_graph_from_pdb
from GNN_model import PocketGNNKcatOnly
import warnings
warnings.filterwarnings('ignore')

def main():
    # Load CSV
    csv_path = Path("data/processed/kcat_test_new_diffdock.csv")
    df = pd.read_csv(csv_path)
    print(f"📊 Loaded {len(df)} samples from {csv_path.name}")

    # Find model
    model_path = Path("outputs/kcat_after_new/best_model.pt")
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return

    print(f"🔧 Loading model from {model_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)

    # Create model
    model = PocketGNNKcatOnly(
        node_input_dim=52,
        edge_input_dim=16,  # RBF only
        hidden_dim=checkpoint.get('hidden_dim', 128),
        num_layers=checkpoint.get('num_layers', 3),
        heads=checkpoint.get('heads', 4),
        dropout=0.0,
        pooling_type=checkpoint.get('pooling_type', 'mean'),
        use_seq_embedding=False
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Model loaded")

    # Predict
    predictions = []
    ground_truths = []
    failed = []

    print(f"\n🔬 Running predictions...")
    with torch.no_grad():
        for idx, row in df.iterrows():
            sample_id = row['sample_id']
            gt_value = row['experimental value[log10]']

            # Find PDB
            pdb_path = Path(f"sample_data/samples/{sample_id}/docking/{sample_id}_*.pdb")
            pdb_files = list(Path("sample_data/samples").glob(f"{sample_id}/docking/*.pdb"))

            if not pdb_files:
                failed.append(sample_id)
                continue

            pdb_file = pdb_files[0]

            try:
                # Load graph
                data = load_kcat_graph_from_pdb(str(pdb_file), kcat_value=gt_value)
                if data is None:
                    failed.append(sample_id)
                    continue

                # Remove metadata
                for key in ['sample_id', 'pdb_id', 'ec']:
                    if hasattr(data, key):
                        delattr(data, key)

                data = data.to(device)

                # Predict
                pred = model(data).item()
                predictions.append(pred)
                ground_truths.append(gt_value)

                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx+1}/{len(df)}...")

            except Exception as e:
                failed.append(sample_id)
                continue

    # Compute metrics
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    print(f"\n📈 Results:")
    print(f"  Successful: {len(predictions)}/{len(df)} ({100*len(predictions)/len(df):.1f}%)")
    print(f"  Failed: {len(failed)}")

    if len(predictions) > 0:
        r, _ = pearsonr(ground_truths, predictions)
        r2 = r2_score(ground_truths, predictions)
        mae = mean_absolute_error(ground_truths, predictions)
        rmse = np.sqrt(mean_squared_error(ground_truths, predictions))

        print(f"\n✨ Metrics:")
        print(f"  Pearson r: {r:.3f}")
        print(f"  R²: {r2:.3f}")
        print(f"  MAE: {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")

        # Save results
        results_df = pd.DataFrame({
            'ground_truth': ground_truths,
            'prediction': predictions
        })
        output_path = Path("results/kcat_test_new_predictions.csv")
        output_path.parent.mkdir(exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\n💾 Saved predictions to {output_path}")

if __name__ == '__main__':
    main()
