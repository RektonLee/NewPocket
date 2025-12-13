"""
ESM-2 序列嵌入生成工具

该脚本用于为数据集中的蛋白质序列生成 ESM-2 嵌入，用于 late fusion。
支持离线提取，避免在训练时重复计算。
"""

import torch
import esm
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np


def generate_esm_embeddings(dataset_path, output_path, esm_model_name='esm2_t33_650M_UR50D', batch_size=1):
    """
    为数据集生成 ESM-2 序列嵌入
    
    参数:
        dataset_path: PyG 图数据集路径 (.pt 文件)
        output_path: 输出嵌入文件路径 (.pt 文件)
        esm_model_name: ESM 模型名称
        batch_size: 批次大小（建议为1以避免OOM）
    """
    print(f"📦 加载数据集: {dataset_path}")
    data_list = torch.load(dataset_path, weights_only=False)
    
    print(f"🧬 加载 ESM-2 模型: {esm_model_name}")
    model, alphabet = esm.pretrained.load_model_and_alphabet(esm_model_name)
    model.eval()
    
    # 使用 GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"💻 使用设备: {device}")
    
    batch_converter = alphabet.get_batch_converter()
    
    embeddings = []
    failed_indices = []
    
    print(f"\n🔄 开始生成嵌入（共 {len(data_list)} 个样本）...")
    
    for idx, data in enumerate(tqdm(data_list)):
        try:
            # 从 data 对象中提取序列（假设存储在 data.sequence 或 data.protein_sequence 中）
            if hasattr(data, 'sequence'):
                sequence = data.sequence
            elif hasattr(data, 'protein_sequence'):
                sequence = data.protein_sequence
            else:
                # 如果数据中没有序列信息，尝试从 pdb_id 推断或使用占位符
                print(f"⚠️ 样本 {idx} 缺少序列信息，使用零向量")
                embeddings.append(torch.zeros(1280))  # ESM-2 650M 的嵌入维度
                failed_indices.append(idx)
                continue
            
            # 准备输入
            batch_labels, batch_strs, batch_tokens = batch_converter([("protein", sequence)])
            batch_tokens = batch_tokens.to(device)
            
            # 生成嵌入
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            
            # 提取嵌入（使用序列的平均池化）
            token_representations = results["representations"][33]
            
            # 移除 BOS 和 EOS token，然后平均池化
            sequence_embedding = token_representations[0, 1:len(sequence)+1].mean(0)
            
            embeddings.append(sequence_embedding.cpu())
            
        except Exception as e:
            print(f"\n❌ 样本 {idx} 处理失败: {e}")
            embeddings.append(torch.zeros(1280))
            failed_indices.append(idx)
    
    # 堆叠所有嵌入
    embeddings_tensor = torch.stack(embeddings)
    
    # 保存嵌入
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'embeddings': embeddings_tensor,
        'model_name': esm_model_name,
        'failed_indices': failed_indices
    }, output_path)
    
    print(f"\n✅ 嵌入已保存到: {output_path}")
    print(f"📊 嵌入形状: {embeddings_tensor.shape}")
    print(f"❌ 失败样本数: {len(failed_indices)}")
    
    if failed_indices:
        print(f"⚠️ 失败样本索引: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}")
    
    return embeddings_tensor


def main():
    parser = argparse.ArgumentParser(description='为数据集生成 ESM-2 序列嵌入')
    
    parser.add_argument('--dataset', type=str, required=True,
                        help='PyG 图数据集路径 (.pt 文件)')
    parser.add_argument('--output', type=str, required=True,
                        help='输出嵌入文件路径 (.pt 文件)')
    parser.add_argument('--model', type=str, default='esm2_t33_650M_UR50D',
                        choices=['esm2_t33_650M_UR50D', 'esm2_t33_150M', 'esm2_t30_150M_UR50D'],
                        help='ESM-2 模型名称')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批次大小（建议为1）')
    
    args = parser.parse_args()
    
    generate_esm_embeddings(
        dataset_path=args.dataset,
        output_path=args.output,
        esm_model_name=args.model,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
