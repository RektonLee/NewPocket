#!/usr/bin/env python3
"""
从HuggingFace下载ESM-2模型的脚本

注意：HuggingFace上没有esm2_t33_150M_UR50D，最接近的模型是：
- facebook/esm2_t30_150M_UR50D (150M参数，30层)
- facebook/esm2_t33_650M_UR50D (650M参数，33层)

本脚本会下载esm2_t30_150M_UR50D作为替代。
"""

import os
import argparse
from transformers import AutoModel, AutoTokenizer

def download_esm_model(model_name="facebook/esm2_t30_150M_UR50D", save_dir=None):
    """
    从HuggingFace下载ESM-2模型
    
    Args:
        model_name: HuggingFace模型名称
        save_dir: 保存目录，如果为None则使用HuggingFace默认缓存
    """
    print(f"正在从HuggingFace下载模型: {model_name}")
    print("这可能需要几分钟时间，请耐心等待...")
    
    try:
        # 下载tokenizer
        print("\n1. 下载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=save_dir
        )
        print("✓ Tokenizer下载完成")
        
        # 下载模型
        print("\n2. 下载模型（这可能需要较长时间）...")
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=save_dir
        )
        print("✓ 模型下载完成")
        
        # 显示保存位置
        if save_dir:
            print(f"\n模型已保存到: {save_dir}")
        else:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            print(f"\n模型已保存到HuggingFace默认缓存: {cache_dir}")
            print(f"模型路径: {cache_dir}/models--{model_name.replace('/', '--')}")
        
        print("\n✓ 下载完成！现在可以使用以下命令生成embeddings:")
        print(f"   python src/generate_esm_embeddings.py --csv <your_csv> --output <output> --model {model_name}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        print("\n可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 确保已安装transformers: pip install transformers")
        print("3. 尝试使用VPN或更换网络")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从HuggingFace下载ESM-2模型")
    parser.add_argument(
        "--model", 
        type=str, 
        default="facebook/esm2_t30_150M_UR50D",
        help="HuggingFace模型名称。可选: facebook/esm2_t6_8M_UR50D, facebook/esm2_t12_35M_UR50D, "
             "facebook/esm2_t30_150M_UR50D, facebook/esm2_t33_650M_UR50D"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="保存目录（可选，默认使用HuggingFace缓存）"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("ESM-2 模型下载工具")
    print("="*70)
    print(f"\n注意: HuggingFace上没有esm2_t33_150M_UR50D")
    print("最接近的模型:")
    print("  - facebook/esm2_t30_150M_UR50D (150M参数，30层) ← 推荐")
    print("  - facebook/esm2_t33_650M_UR50D (650M参数，33层)")
    print(f"\n将下载: {args.model}")
    print("="*70)
    
    download_esm_model(args.model, args.save_dir)

