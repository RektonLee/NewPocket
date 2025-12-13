import torch
import esm
import pandas as pd
import argparse
import os
from tqdm import tqdm

def generate_embeddings(csv_path, output_path, model_name="esm2_t33_150M_UR50D", device=None):
    """
    Generate ESM-2 embeddings for protein sequences in a CSV file.
    
    Args:
        csv_path: Path to CSV containing 'protein_sequence' and identifiers.
        output_path: Path to save the .pt dictionary {id: embedding}.
        model_name: ESM-2 model name.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model: {model_name} on {device}")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    print(f"Reading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Check columns
    if 'protein_sequence' not in df.columns:
        raise ValueError("CSV must contain 'protein_sequence' column")
    
    # Determine ID column
    id_col = None
    if 'uniprot' in df.columns:
        id_col = 'uniprot'
    elif 'pdb_id' in df.columns:
        id_col = 'pdb_id'
    elif 'uniprot_id' in df.columns:
        id_col = 'uniprot_id'
    else:
        print("Warning: No 'uniprot' or 'pdb_id' column found. Using index as ID.")
        id_col = 'index'
        df['index'] = df.index.astype(str)

    embeddings_dict = {}
    
    print("Generating embeddings...")
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            seq = row['protein_sequence']
            identifier = row[id_col]
            
            # Skip invalid sequences
            if not isinstance(seq, str) or len(seq) == 0:
                continue
                
            # Truncate if too long (ESM-2 usually handles up to 1024, but check memory)
            # Standard ESM-2 max length is often 1022 + 2 special tokens
            if len(seq) > 1022:
                seq = seq[:1022]

            batch_labels, batch_strs, batch_tokens = batch_converter([(identifier, seq)])
            batch_tokens = batch_tokens.to(device)
            
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]
            
            # Generate per-sequence representation via averaging (excluding start/end tokens)
            # token_representations is [batch, seq_len, dim]
            # NOTE: batch_tokens includes CLS and EOS. 
            # We want mean over residue tokens.
            
            # Iterate over batch (here batch size is 1)
            for i, (_, seq_str) in enumerate([(identifier, seq)]):
                # tokens: [CLS, seq..., EOS, padding...]
                # seq_len = len(seq_str)
                # embedding = token_representations[i, 1 : 1 + len(seq_str)].mean(0)
                
                # Alternatively use CLS token:
                # embedding = token_representations[i, 0]
                
                # Using mean pooling as requested in Enhance.md "mean pooling 或 CLS"
                # Let's use mean pooling over the sequence
                embedding = token_representations[i, 1 : 1 + len(seq_str)].mean(0)
                
                embeddings_dict[identifier] = embedding.cpu()

    print(f"Saving {len(embeddings_dict)} embeddings to {output_path}")
    torch.save(embeddings_dict, output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ESM-2 embeddings for dataset")
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=str, default="data/esm_embeddings.pt", help="Path to output .pt file")
    parser.add_argument("--model", type=str, default="esm2_t33_150M_UR50D", help="ESM-2 model name")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generate_embeddings(args.csv, args.output, args.model)
