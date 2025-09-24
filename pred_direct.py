#!/usr/bin/env python3
"""
直接从已构建的图数据进行模型预测的脚本
适用于已经完成PDB生成、docking和构图步骤的情况
支持测试多个不同的模型
"""

import torch
import pandas as pd
import numpy as np
import os
import logging
from torch_geometric.loader import DataLoader
from GNN_model import PocketGNNWithAttention, PocketGNNWithAttentionNoTemp
import argparse
from pathlib import Path
import json
from datetime import datetime

def load_graph_data(graph_data_path):
    """
    加载已构建的图数据
    
    参数:
    graph_data_path: 图数据文件路径（.pt文件）
    
    返回:
    data_list: 图数据列表
    """
    if not os.path.exists(graph_data_path):
        raise FileNotFoundError(f"图数据文件不存在: {graph_data_path}")
    
    logging.info(f"正在加载图数据: {graph_data_path}")
    data_list = torch.load(graph_data_path, weights_only=False)
    logging.info(f"✅ 成功加载 {len(data_list)} 个图数据样本")
    
    return data_list

def predict_with_model(model_path, data_list, output_dir, model_name=None, batch_size=32):
    """
    使用指定模型对图数据进行预测
    
    参数:
    model_path: 模型文件路径
    data_list: 图数据列表
    output_dir: 输出目录
    model_name: 模型名称（用于文件命名）
    batch_size: 批处理大小
    
    返回:
    predictions_df: 预测结果DataFrame
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")
    
    # 获取图数据的维度信息
    if len(data_list) == 0:
        raise ValueError("图数据列表为空")
    
    sample_data = data_list[0]
    node_input_dim = sample_data.x.shape[1]
    edge_input_dim = sample_data.edge_attr.shape[1] if hasattr(sample_data, 'edge_attr') else 0
    
    logging.info(f"图数据维度 - 节点特征: {node_input_dim}, 边特征: {edge_input_dim}")
    
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
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=256,
            num_layers=6,
            heads=8,
            dropout=0.1
        ).to(device)
        logging.info("✅ 创建包含温度模块的模型")
    else:
        model = PocketGNNWithAttentionNoTemp(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
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
    
    # 创建数据加载器
    data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    
    # 进行预测
    all_predictions = []
    all_metadata = []
    
    logging.info("开始预测...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            batch = batch.to(device)
            batch_size_actual = batch.num_graphs
            
            # 模型预测
            predictions = model(batch)
            
            # 转换回原始尺度
            kcat_pred = 10 ** predictions[:, 0].cpu().numpy()
            km_pred = 10 ** predictions[:, 1].cpu().numpy()
            
            # 收集预测结果
            for i in range(batch_size_actual):
                sample_idx = batch_idx * batch_size + i
                if sample_idx < len(data_list):
                    sample_data = data_list[sample_idx]
                    
                    # 获取样本元数据
                    metadata = {
                        'sample_idx': sample_idx,
                        'pdb_id': getattr(sample_data, 'pdb_id', f'sample_{sample_idx}'),
                        'kcat_pred': kcat_pred[i],
                        'km_pred': km_pred[i],
                        'km_pred_log10': np.log10(km_pred[i])
                    }
                    
                    # 如果有真实标签，添加比较信息
                    if hasattr(sample_data, 'y') and sample_data.y is not None:
                        y_true = sample_data.y.cpu().numpy()
                        if len(y_true) >= 2:
                            kcat_true = 10 ** y_true[0]
                            km_true = 10 ** y_true[1]
                            metadata.update({
                                'kcat_true': kcat_true,
                                'km_true': km_true,
                                'km_true_log10': y_true[1],
                                'kcat_error': abs(kcat_pred[i] - kcat_true) / kcat_true if kcat_true > 0 else None,
                                'km_error': abs(km_pred[i] - km_true) / km_true if km_true > 0 else None,
                                'km_error_log10': abs(np.log10(km_pred[i]) - y_true[1])
                            })
                    
                    all_predictions.append(metadata)
            
            if (batch_idx + 1) % 10 == 0:
                logging.info(f"已处理 {batch_idx + 1} 个批次")
    
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
        
        # 如果有真实值，添加误差统计
        if 'kcat_error' in predictions_df.columns:
            valid_kcat = predictions_df.dropna(subset=['kcat_error'])
            valid_km = predictions_df.dropna(subset=['km_error'])
            
            if len(valid_kcat) > 0:
                stats.update({
                    'kcat_mae': valid_kcat['kcat_error'].mean(),
                    'kcat_median_error': valid_kcat['kcat_error'].median(),
                    'kcat_samples_with_truth': len(valid_kcat)
                })
            
            if len(valid_km) > 0:
                stats.update({
                    'km_mae': valid_km['km_error'].mean(),
                    'km_median_error': valid_km['km_error'].median(),
                    'km_mae_log10': valid_km['km_error_log10'].mean(),
                    'km_samples_with_truth': len(valid_km)
                })
        
        # 保存统计信息
        stats_file = os.path.join(output_dir, f'stats{model_suffix}.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logging.info(f"✅ 统计信息已保存到: {stats_file}")
    
    return predictions_df

def compare_models(model_paths, data_list, output_dir, batch_size=32):
    """
    比较多个模型的预测结果
    
    参数:
    model_paths: 模型路径列表
    data_list: 图数据列表
    output_dir: 输出目录
    batch_size: 批处理大小
    
    返回:
    comparison_df: 比较结果DataFrame
    """
    logging.info(f"开始比较 {len(model_paths)} 个模型")
    
    all_results = []
    
    for i, model_path in enumerate(model_paths):
        model_name = f"model_{i+1}_{os.path.basename(model_path).replace('.pt', '')}"
        logging.info(f"正在测试模型 {i+1}/{len(model_paths)}: {model_name}")
        
        try:
            predictions_df = predict_with_model(
                model_path=model_path,
                data_list=data_list,
                output_dir=output_dir,
                model_name=model_name,
                batch_size=batch_size
            )
            
            # 添加模型标识
            predictions_df['model_name'] = model_name
            predictions_df['model_path'] = model_path
            all_results.append(predictions_df)
            
        except Exception as e:
            logging.error(f"❌ 模型 {model_name} 预测失败: {str(e)}")
            continue
    
    if not all_results:
        raise RuntimeError("所有模型预测都失败了")
    
    # 合并所有结果
    comparison_df = pd.concat(all_results, ignore_index=True)
    
    # 保存比较结果
    comparison_file = os.path.join(output_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_file, index=False)
    logging.info(f"✅ 模型比较结果已保存到: {comparison_file}")
    
    # 生成比较统计
    if 'kcat_error' in comparison_df.columns:
        comparison_stats = comparison_df.groupby('model_name').agg({
            'kcat_error': ['mean', 'median', 'std'],
            'km_error': ['mean', 'median', 'std'],
            'km_error_log10': ['mean', 'median', 'std']
        }).round(4)
        
        stats_file = os.path.join(output_dir, 'comparison_stats.csv')
        comparison_stats.to_csv(stats_file)
        logging.info(f"✅ 比较统计已保存到: {stats_file}")
    
    return comparison_df

def main():
    parser = argparse.ArgumentParser(description='直接从图数据进行模型预测')
    parser.add_argument('--graph-data', type=str, required=True, 
                       help='图数据文件路径（.pt文件）')
    parser.add_argument('--model', type=str, nargs='+', required=True,
                       help='模型文件路径（可以指定多个模型进行比较）')
    parser.add_argument('--output', type=str, default='direct_predictions',
                       help='输出目录')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批处理大小')
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
            logging.FileHandler(os.path.join(args.output, 'direct_prediction.log')),
            logging.StreamHandler()
        ]
    )
    
    try:
        # 加载图数据
        data_list = load_graph_data(args.graph_data)
        
        if args.compare and len(args.model) > 1:
            # 比较多个模型
            comparison_df = compare_models(
                model_paths=args.model,
                data_list=data_list,
                output_dir=args.output,
                batch_size=args.batch_size
            )
            logging.info(f"✅ 模型比较完成，共比较 {len(args.model)} 个模型")
        else:
            # 单个模型预测
            for i, model_path in enumerate(args.model):
                model_name = f"model_{i+1}" if len(args.model) > 1 else None
                predictions_df = predict_with_model(
                    model_path=model_path,
                    data_list=data_list,
                    output_dir=args.output,
                    model_name=model_name,
                    batch_size=args.batch_size
                )
                logging.info(f"✅ 模型 {i+1} 预测完成")
        
        logging.info("🎉 所有预测任务完成！")
        
    except Exception as e:
        logging.error(f"❌ 预测过程中出现错误: {str(e)}")
        raise

if __name__ == '__main__':
    main()
