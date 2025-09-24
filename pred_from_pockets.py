#!/usr/bin/env python3
"""
从已有的pocket PDB文件快速构建图数据并进行模型预测
适用于已经完成PDB生成和docking步骤的情况
"""

import torch
import pandas as pd
import numpy as np
import os
import logging
import argparse
from pathlib import Path
import json
from datetime import datetime
import hashlib

from GNN_model import PocketGNNWithAttention, PocketGNNWithAttentionNoTemp
from graph_builder_rbf import build_graph, parse_pocket, NODE_INPUT_DIM, EDGE_INPUT_DIM

def find_pocket_files(pocket_dir, sample_info_df=None):
    """
    在指定目录中查找pocket PDB文件
    
    参数:
    pocket_dir: pocket文件目录
    sample_info_df: 样本信息DataFrame（可选）
    
    返回:
    pocket_files: pocket文件路径列表
    sample_mapping: 样本ID到pocket文件的映射
    """
    pocket_files = []
    sample_mapping = {}
    
    if sample_info_df is not None:
        # 根据样本信息查找对应的pocket文件
        for _, row in sample_info_df.iterrows():
            sample_id = row.get('sample_id', f"sample_{len(pocket_files)}")
            uniprot_id = row.get('uniprot', sample_id)
            smiles = row['smiles']
            
            # 计算pocket文件名（与docking.py中的逻辑一致）
            pocket_hash = int(hashlib.sha256(smiles.encode()).hexdigest(), 16) & 0xffff
            base_name = f"{uniprot_id}_{pocket_hash}"
            pocket_file = os.path.join(pocket_dir, f"{base_name}_10A.pdb")
            
            if os.path.exists(pocket_file):
                pocket_files.append(pocket_file)
                sample_mapping[sample_id] = {
                    'pocket_file': pocket_file,
                    'sequence': row['sequence'],
                    'smiles': smiles,
                    'uniprot': uniprot_id,
                    'experimental_km_log10': row.get('experimental_km_log10', None)
                }
            else:
                logging.warning(f"⚠️ 未找到pocket文件: {pocket_file}")
    else:
        # 直接扫描目录中的所有pocket文件
        for file_path in Path(pocket_dir).glob("*_10A.pdb"):
            pocket_files.append(str(file_path))
            # 从文件名推断样本ID
            sample_id = file_path.stem.replace("_10A", "")
            sample_mapping[sample_id] = {
                'pocket_file': str(file_path),
                'sequence': None,  # 需要额外提供
                'smiles': None,    # 需要额外提供
                'uniprot': None,
                'experimental_km_log10': None
            }
    
    logging.info(f"找到 {len(pocket_files)} 个pocket文件")
    return pocket_files, sample_mapping

def build_graphs_from_pockets(pocket_files, sample_mapping, temperature=303.15):
    """
    从pocket文件构建图数据
    
    参数:
    pocket_files: pocket文件路径列表
    sample_mapping: 样本映射信息
    temperature: 温度
    
    返回:
    graph_data_list: 图数据列表
    valid_samples: 有效样本信息
    """
    graph_data_list = []
    valid_samples = []
    
    for i, pocket_file in enumerate(pocket_files):
        try:
            logging.info(f"处理pocket文件 {i+1}/{len(pocket_files)}: {os.path.basename(pocket_file)}")
            
            # 解析pocket文件
            atoms = parse_pocket(pocket_file)
            
            if len(atoms) == 0:
                logging.warning(f"⚠️ 跳过空pocket文件: {pocket_file}")
                continue
            
            # 构建图数据
            graph_data = build_graph(atoms, temperature)
            
            # 获取样本信息
            sample_id = os.path.basename(pocket_file).replace("_10A.pdb", "")
            sample_info = sample_mapping.get(sample_id, {})
            
            # 添加样本元数据
            graph_data.sample_id = sample_id
            graph_data.pocket_file = pocket_file
            graph_data.sequence = sample_info.get('sequence', '')
            graph_data.smiles = sample_info.get('smiles', '')
            graph_data.uniprot = sample_info.get('uniprot', '')
            graph_data.experimental_km_log10 = sample_info.get('experimental_km_log10', None)
            
            graph_data_list.append(graph_data)
            valid_samples.append(sample_info)
            
            logging.info(f"✅ 成功构建图数据，包含 {len(atoms)} 个原子")
            
        except Exception as e:
            logging.error(f"❌ 处理pocket文件失败 {pocket_file}: {str(e)}")
            continue
    
    logging.info(f"✅ 成功构建 {len(graph_data_list)} 个图数据")
    return graph_data_list, valid_samples

def predict_with_model(model_path, graph_data_list, output_dir, model_name=None, batch_size=32):
    """
    使用指定模型对图数据进行预测
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")
    
    # 首先检查模型是否包含温度模块
    try:
        state_dict = torch.load(model_path, map_location=device)
        has_temp_module = any(key.startswith('temp_mlp') for key in state_dict.keys())
        logging.info(f"模型包含温度模块: {has_temp_module}")
    except Exception as e:
        logging.error(f"❌ 无法加载模型状态字典: {str(e)}")
        raise
    
    # 根据模型类型创建相应的模型
    if has_temp_module:
        model = PocketGNNWithAttention(
            node_input_dim=NODE_INPUT_DIM,
            edge_input_dim=EDGE_INPUT_DIM,
            hidden_dim=256,
            num_layers=6,
            heads=8,
            dropout=0.1
        ).to(device)
        logging.info("✅ 创建包含温度模块的模型")
    else:
        model = PocketGNNWithAttentionNoTemp(
            node_input_dim=NODE_INPUT_DIM,
            edge_input_dim=EDGE_INPUT_DIM,
            hidden_dim=256,
            num_layers=6,
            heads=8,
            dropout=0.1
        ).to(device)
        logging.info("✅ 创建不包含温度模块的模型")
    
    try:
        model.load_state_dict(state_dict)
        logging.info(f"✅ 成功加载模型: {model_path}")
    except Exception as e:
        logging.error(f"❌ 加载模型失败: {str(e)}")
        raise
    
    model.eval()
    
    # 进行预测
    all_predictions = []
    
    logging.info("开始预测...")
    with torch.no_grad():
        for i, graph_data in enumerate(graph_data_list):
            # 单个样本预测
            graph_data = graph_data.to(device)
            
            # 如果模型不包含温度模块，移除温度信息
            if not has_temp_module and hasattr(graph_data, 'temperature'):
                delattr(graph_data, 'temperature')
            
            predictions = model(graph_data)
            
            # 转换回原始尺度
            kcat_pred = 10 ** predictions[0][0].item()
            km_pred = 10 ** predictions[0][1].item()
            
            # 收集预测结果
            result = {
                'sample_id': getattr(graph_data, 'sample_id', f'sample_{i}'),
                'pocket_file': getattr(graph_data, 'pocket_file', ''),
                'sequence': getattr(graph_data, 'sequence', ''),
                'smiles': getattr(graph_data, 'smiles', ''),
                'uniprot': getattr(graph_data, 'uniprot', ''),
                'kcat_pred': kcat_pred,
                'km_pred': km_pred,
                'km_pred_log10': np.log10(km_pred),
                'experimental_km_log10': getattr(graph_data, 'experimental_km_log10', None),
                'temperature': 303.15
            }
            
            # 如果有实验值，计算误差
            if result['experimental_km_log10'] is not None:
                result['km_error_log10'] = abs(np.log10(km_pred) - result['experimental_km_log10'])
                result['km_error_relative'] = abs(km_pred - 10**result['experimental_km_log10']) / (10**result['experimental_km_log10'])
            
            all_predictions.append(result)
            
            if (i + 1) % 10 == 0:
                logging.info(f"已处理 {i + 1} 个样本")
    
    # 转换为DataFrame
    predictions_df = pd.DataFrame(all_predictions)
    
    # 保存结果
    model_suffix = f"_{model_name}" if model_name else ""
    output_file = os.path.join(output_dir, f'predictions{model_suffix}.csv')
    predictions_df.to_csv(output_file, index=False)
    logging.info(f"✅ 预测结果已保存到: {output_file}")
    
    # 生成统计报告
    if len(predictions_df) > 0:
        stats = {
            'model_name': model_name or os.path.basename(model_path),
            'model_path': model_path,
            'total_samples': len(predictions_df),
            'kcat_pred_range': f"{predictions_df['kcat_pred'].min():.2f} - {predictions_df['kcat_pred'].max():.2f}",
            'km_pred_range': f"{predictions_df['km_pred'].min():.2f} - {predictions_df['km_pred'].max():.2f}",
            'timestamp': datetime.now().isoformat()
        }
        
        # 如果有实验值，添加误差统计
        if 'km_error_log10' in predictions_df.columns:
            comparison_data = predictions_df.dropna(subset=['km_error_log10'])
            if len(comparison_data) > 0:
                stats.update({
                    'samples_with_experimental_km': len(comparison_data),
                    'km_mae_log10': comparison_data['km_error_log10'].mean(),
                    'km_median_error_log10': comparison_data['km_error_log10'].median(),
                    'km_mae_relative': comparison_data['km_error_relative'].mean(),
                    'km_median_error_relative': comparison_data['km_error_relative'].median()
                })
        
        # 保存统计信息
        stats_file = os.path.join(output_dir, f'stats{model_suffix}.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logging.info(f"✅ 统计信息已保存到: {stats_file}")
    
    return predictions_df

def main():
    parser = argparse.ArgumentParser(description='从pocket PDB文件进行模型预测')
    parser.add_argument('--pocket-dir', type=str, required=True,
                       help='pocket PDB文件目录')
    parser.add_argument('--model', type=str, nargs='+', required=True,
                       help='模型文件路径（可以指定多个模型）')
    parser.add_argument('--output', type=str, default='pocket_predictions',
                       help='输出目录')
    parser.add_argument('--sample-info', type=str, default=None,
                       help='样本信息CSV文件（包含sample_id, sequence, smiles等列）')
    parser.add_argument('--temperature', type=float, default=303.15,
                       help='温度（K）')
    parser.add_argument('--compare', action='store_true',
                       help='比较多个模型的性能')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output, 'pocket_prediction.log')),
            logging.StreamHandler()
        ]
    )
    
    try:
        # 加载样本信息（如果提供）
        sample_info_df = None
        if args.sample_info and os.path.exists(args.sample_info):
            sample_info_df = pd.read_csv(args.sample_info)
            logging.info(f"✅ 加载样本信息: {len(sample_info_df)} 个样本")
        
        # 查找pocket文件
        pocket_files, sample_mapping = find_pocket_files(args.pocket_dir, sample_info_df)
        
        if len(pocket_files) == 0:
            raise ValueError(f"在目录 {args.pocket_dir} 中未找到任何pocket文件")
        
        # 构建图数据
        graph_data_list, valid_samples = build_graphs_from_pockets(
            pocket_files, sample_mapping, args.temperature
        )
        
        if len(graph_data_list) == 0:
            raise ValueError("未能构建任何有效的图数据")
        
        # 进行预测
        if args.compare and len(args.model) > 1:
            # 比较多个模型
            all_results = []
            for i, model_path in enumerate(args.model):
                model_name = f"model_{i+1}_{os.path.basename(model_path).replace('.pt', '')}"
                logging.info(f"正在测试模型 {i+1}/{len(args.model)}: {model_name}")
                
                try:
                    predictions_df = predict_with_model(
                        model_path=model_path,
                        graph_data_list=graph_data_list,
                        output_dir=args.output,
                        model_name=model_name
                    )
                    predictions_df['model_name'] = model_name
                    predictions_df['model_path'] = model_path
                    all_results.append(predictions_df)
                    
                except Exception as e:
                    logging.error(f"❌ 模型 {model_name} 预测失败: {str(e)}")
                    continue
            
            if all_results:
                # 合并比较结果
                comparison_df = pd.concat(all_results, ignore_index=True)
                comparison_file = os.path.join(args.output, 'model_comparison.csv')
                comparison_df.to_csv(comparison_file, index=False)
                logging.info(f"✅ 模型比较结果已保存到: {comparison_file}")
        else:
            # 单个或多个模型分别预测
            for i, model_path in enumerate(args.model):
                model_name = f"model_{i+1}" if len(args.model) > 1 else None
                predictions_df = predict_with_model(
                    model_path=model_path,
                    graph_data_list=graph_data_list,
                    output_dir=args.output,
                    model_name=model_name
                )
                logging.info(f"✅ 模型 {i+1} 预测完成")
        
        logging.info("🎉 所有预测任务完成！")
        
    except Exception as e:
        logging.error(f"❌ 预测过程中出现错误: {str(e)}")
        raise

if __name__ == '__main__':
    main()
