import torch
import esm
import pandas as pd
import argparse
import os
from tqdm import tqdm

def load_esm_model_from_huggingface(model_name, device):
    """
    从HuggingFace加载ESM-2模型
    """
    try:
        from transformers import EsmModel, EsmTokenizer
    except ImportError:
        raise ImportError("transformers库未安装。请运行: pip install transformers")
    
    print(f"从HuggingFace加载模型: {model_name}")
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    # 创建wrapper以匹配esm库的API
    class HuggingFaceAlphabet:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
        
        def get_batch_converter(self):
            def batch_converter(batch_labels_and_strs):
                labels, strs = zip(*batch_labels_and_strs)
                # Tokenize sequences
                encoded = self.tokenizer(
                    list(strs),
                    padding=True,
                    return_tensors="pt",
                    add_special_tokens=True
                )
                input_ids = encoded["input_ids"]
                return labels, strs, input_ids
            return batch_converter
    
    class HuggingFaceModel:
        def __init__(self, model):
            self.model = model
            self.num_layers = model.config.num_hidden_layers
        
        def __call__(self, tokens, repr_layers=None, return_contacts=False):
            # tokens shape: [batch, seq_len]
            outputs = self.model(tokens, output_hidden_states=True)
            # Convert to esm format
            # HuggingFace的hidden_states结构:
            #   hidden_states[0] = embedding层
            #   hidden_states[1] = transformer第1层
            #   ...
            #   hidden_states[n] = transformer第n层 (n = num_hidden_layers)
            # esm库的repr_layers通常是1-based (1, 2, ..., n)，但也可以是0-based
            representations = {}
            for layer_idx in repr_layers:
                # 如果请求的层号大于模型层数，使用最后一层
                if layer_idx > self.num_layers:
                    # 使用最后一层transformer层
                    hf_layer_idx = self.num_layers
                elif layer_idx <= 0:
                    # 如果请求0或负数，使用最后一层
                    hf_layer_idx = self.num_layers
                else:
                    # layer_idx对应transformer的第layer_idx层
                    # 在hidden_states中索引为layer_idx
                    hf_layer_idx = min(layer_idx, self.num_layers)
                
                # 确保索引在有效范围内 (hidden_states有num_layers+1个元素: 0到num_layers)
                if 0 <= hf_layer_idx <= self.num_layers and hf_layer_idx < len(outputs.hidden_states):
                    representations[layer_idx] = outputs.hidden_states[hf_layer_idx]
                else:
                    # 使用最后一层
                    representations[layer_idx] = outputs.hidden_states[-1]
            return {"representations": representations}
    
    alphabet = HuggingFaceAlphabet(tokenizer)
    wrapped_model = HuggingFaceModel(model)
    batch_converter = alphabet.get_batch_converter()
    
    print(f"✓ 成功从HuggingFace加载模型 (共{model.config.num_hidden_layers}层)")
    return wrapped_model, alphabet, batch_converter

def load_esm_model(model_name, device):
    """
    Load ESM-2 model with support for both esm library and HuggingFace.
    
    如果model_name以"facebook/"开头，则从HuggingFace加载；
    否则使用esm.pretrained加载。
    """
    print(f"Loading model: {model_name} on {device}")
    
    # 检查是否是HuggingFace模型
    if model_name.startswith("facebook/"):
        return load_esm_model_from_huggingface(model_name, device)
    
    # 否则使用esm库加载
    # Check local cache first
    cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
    model_file = os.path.join(cache_dir, f"{model_name}.pt")
    
    if os.path.exists(model_file):
        print(f"Found cached model at: {model_file}")
    
    try:
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        model.to(device)
        model.eval()
        batch_converter = alphabet.get_batch_converter()
        print("✓ Successfully loaded model from esm.pretrained")
        return model, alphabet, batch_converter
    except Exception as e:
        error_msg = str(e)
        print(f"\n✗ Failed to load model: {error_msg}")
        
        # Check if it's a network/download error
        if "403" in error_msg or "Forbidden" in error_msg or "Could not load" in error_msg or "HTTP Error" in error_msg:
            print("\n" + "="*70)
            print("NETWORK ERROR: Cannot download model from Facebook servers")
            print("="*70)
            print("\nSOLUTION: 使用HuggingFace模型（推荐）")
            print("运行以下命令下载模型:")
            print(f"   python scripts/download_esm_from_hf.py --model facebook/esm2_t30_150M_UR50D")
            print("\n然后使用HuggingFace模型:")
            print(f"   python src/generate_esm_embeddings.py --csv <your_csv> --output <output> --model facebook/esm2_t30_150M_UR50D")
            print("="*70)
            raise Exception(f"Network error: Cannot download model. See solutions above.")
        else:
            # Re-raise if it's not a network error
            raise

def generate_embeddings(csv_path, output_path, model_name="esm2_t33_150M_UR50D", device=None):
    """
    Generate ESM-2 embeddings for protein sequences in a CSV file.
    
    Args:
        csv_path: Path to CSV containing 'protein_sequence' or 'sequence' column and identifiers.
        output_path: Path to save the .pt dictionary {id: embedding}.
        model_name: ESM-2 model name.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, alphabet, batch_converter = load_esm_model(model_name, device)

    print(f"Reading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Determine sequence column (support both 'sequence' and 'protein_sequence')
    seq_col = None
    if 'protein_sequence' in df.columns:
        seq_col = 'protein_sequence'
    elif 'sequence' in df.columns:
        seq_col = 'sequence'
    else:
        raise ValueError("CSV must contain 'protein_sequence' or 'sequence' column")
    
    # Determine ID column
    # 优先级：sample_id > uniprot > pdb_id > uniprot_id
    # sample_id 优先级最高，因为训练和测试时都使用 sample_id 进行匹配
    id_col = None
    if 'sample_id' in df.columns:
        id_col = 'sample_id'
    elif 'uniprot' in df.columns:
        id_col = 'uniprot'
    elif 'pdb_id' in df.columns:
        id_col = 'pdb_id'
    elif 'uniprot_id' in df.columns:
        id_col = 'uniprot_id'
    else:
        print("Warning: No identifier column found. Using index as ID.")
        id_col = 'index'
        df['index'] = df.index.astype(str)

    embeddings_dict = {}
    
    print("Generating embeddings...")
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            seq = row[seq_col]
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
            
            # Determine which layer to use
            # 对于esm2_t33模型使用33层，对于esm2_t30使用30层，其他使用最后一层
            if hasattr(model, 'num_layers'):
                # 优先使用最后一层（通常效果最好）
                target_layer = model.num_layers - 1
            else:
                # 默认尝试33层（esm2_t33），如果不存在则使用最后一层
                target_layer = 33
            
            try:
                repr_layers = [target_layer]
                results = model(batch_tokens, repr_layers=repr_layers, return_contacts=False)
                # Check if target layer exists
                if target_layer in results["representations"]:
                    token_representations = results["representations"][target_layer]
                else:
                    # Use the last available layer
                    available_layers = list(results["representations"].keys())
                    last_layer = max(available_layers) if available_layers else 0
                    if idx == 0:  # 只在第一次打印警告
                        print(f"Warning: Layer {target_layer} not available, using layer {last_layer}")
                    token_representations = results["representations"][last_layer]
            except Exception as e:
                # Fallback: use the last layer
                if hasattr(model, 'num_layers'):
                    last_layer = model.num_layers - 1
                else:
                    last_layer = 12  # Default fallback
                repr_layers = [last_layer]
                results = model(batch_tokens, repr_layers=repr_layers, return_contacts=False)
                token_representations = results["representations"][last_layer]
            
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
    parser.add_argument("--model", type=str, default="facebook/esm2_t30_150M_UR50D", 
                       help="ESM-2 model name. 支持两种格式:\n"
                            "  1. esm库格式: esm2_t33_150M_UR50D\n"
                            "  2. HuggingFace格式: facebook/esm2_t30_150M_UR50D (推荐，更可靠)\n"
                            "  如果下载失败，建议使用: facebook/esm2_t30_150M_UR50D")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generate_embeddings(args.csv, args.output, args.model)
