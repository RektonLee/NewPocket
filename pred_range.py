#!/usr/bin/env python3
"""
范围预测脚本 - 支持指定样本范围进行预测
用于并行化任务分配
"""

import torch
import pandas as pd
import numpy as np
from data_loader import ProteinStructureProcessor, MoleculeEmbeddingGenerator
from GNN_model import PocketGNNWithAttention
from sample_manager import SampleManager
import logging
import os
from graph_builder_rbf import build_graph, parse_pocket, DIST_CUTOFF, RBF_CENTERS, RBF_DMIN, RBF_DMAX, RBF_GAMMA
from docking import run_preprocess
import hashlib
import argparse

# 从graph_builder_rbf.py中获取正确的维度
NODE_INPUT_DIM = 52
EDGE_INPUT_DIM = RBF_CENTERS

def predict_kinetics_range(input_data, model_path, output_dir='predictions', 
                          start_idx=0, end_idx=None, temperature=303.15, 
                          use_sample_manager=True, sample_data_dir='sample_data'):
    """
    预测指定范围的酶动力学参数
    
    参数:
    input_data: CSV文件路径或DataFrame
    model_path: 训练好的模型路径
    output_dir: 输出目录
    start_idx: 开始索引
    end_idx: 结束索引（None表示到末尾）
    temperature: 温度
    use_sample_manager: 是否使用SampleManager
    sample_data_dir: SampleManager基础目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(output_dir, 'prediction.log')
    
    # 清除现有的日志配置
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 重新配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )
    
    logging.info(f"开始范围预测 [{start_idx}:{end_idx}]，日志文件: {log_file}")
    
    # 解析输入数据
    if isinstance(input_data, str):
        input_df = pd.read_csv(input_data)
    elif isinstance(input_data, pd.DataFrame):
        input_df = input_data
    else:
        raise ValueError("input_data必须是CSV路径或DataFrame")
    
    # 应用范围过滤
    if end_idx is None:
        end_idx = len(input_df)
    
    input_df = input_df.iloc[start_idx:end_idx].copy()
    logging.info(f"处理样本范围: {start_idx}-{end_idx} (共 {len(input_df)} 个样本)")
    
    # 检查必要的列
    required_columns = ['sequence', 'smiles']
    if use_sample_manager:
        required_columns.append('sample_id')
    
    missing_columns = [col for col in required_columns if col not in input_df.columns]
    if missing_columns:
        raise ValueError(f"输入数据缺少必要的列: {missing_columns}")
    
    # 兼容原始数据格式
    if 'sample_id' not in input_df.columns and not use_sample_manager:
        input_df['sample_id'] = [f"sample_{start_idx + i + 1}" for i in range(len(input_df))]
    
    # 初始化SampleManager（如果启用）
    sample_manager = None
    if use_sample_manager:
        sample_manager = SampleManager(sample_data_dir)
        logging.info(f"SampleManager已初始化，管理 {len(sample_manager.sample_registry)} 个样本")
    
    # 初始化处理器
    structure_processor = ProteinStructureProcessor(sample_manager=sample_manager)
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PocketGNNWithAttention(
        node_input_dim=NODE_INPUT_DIM,
        edge_input_dim=EDGE_INPUT_DIM,
        hidden_dim=256,
        num_layers=6,
        heads=8,
        dropout=0.1
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f"✅ 成功加载模型: {model_path}")
    except Exception as e:
        logging.error(f"❌ 加载模型失败: {str(e)}")
        raise
    
    model.eval()
    
    results = []
    total_samples = len(input_df)
    
    for i, row in input_df.iterrows():
        sample_id = row.get('sample_id', f'sample_{start_idx + i + 1}')
        seq = row['sequence']
        smiles = row['smiles']
        uniprot_id = row.get('uniprot', None)
        experimental_km_log10 = row.get('experimental value[log10]', None)  # 注意列名
        
        try:
            logging.info(f"处理样本 {i+1}/{total_samples}: {sample_id}")
            
            # 1. 获取蛋白质结构
            if use_sample_manager and sample_manager:
                protein_path, is_shared = sample_manager.get_protein_path(sample_id)
                
                if protein_path.exists() and protein_path.stat().st_size > 0:
                    pdb_path = str(protein_path)
                    logging.info(f"使用已存在的PDB文件: {pdb_path}")
                else:
                    logging.warning(f"⚠️ 跳过样本 {sample_id}: PDB文件不存在 ({protein_path})")
                    if use_sample_manager and sample_manager:
                        sample_manager.log_failure(sample_id, "missing_pdb", f"PDB文件不存在: {protein_path}", len(seq))
                    
                    results.append({
                        'sample_id': sample_id,
                        'sequence': seq,
                        'smiles': smiles,
                        'kcat_pred': None,
                        'km_pred': None,
                        'km_pred_log10': None,
                        'experimental_km_log10': experimental_km_log10,
                        'km_error_log10': None,
                        'km_error_relative': None,
                        'temperature': temperature,
                        'error': f'PDB文件不存在: {protein_path}'
                    })
                    continue
            else:
                pdb_content = structure_processor.predict_structure(seq, uniprot_id)
                temp_pdb_dir = os.path.join(output_dir, 'temp_pdbs')
                os.makedirs(temp_pdb_dir, exist_ok=True)
                pdb_path = os.path.join(temp_pdb_dir, f"{sample_id}.pdb")
                with open(pdb_path, 'w') as f:
                    f.write(pdb_content)
                logging.info(f"创建临时PDB文件: {pdb_path}")
            
            logging.info("✅ 蛋白质结构预测完成")

            # 2. docking 步骤
            if use_sample_manager:
                pocket_dir = os.path.join(sample_data_dir, "samples", sample_id)
            else:
                pocket_dir = output_dir
            os.makedirs(pocket_dir, exist_ok=True)
            
            pocket_hash = int(hashlib.sha256(smiles.encode()).hexdigest(), 16) & 0xffff
            base_name = f"{uniprot_id or sample_id}_{pocket_hash}"
            pocket_pdb = os.path.join(pocket_dir, f"{base_name}_10A.pdb")
            
            try:
                result = run_preprocess(uniprot_id or sample_id, smiles, pdb_path, pocket_pdb, i)
                if not result:
                    raise RuntimeError("docking预处理失败")
                
                if not os.path.exists(pocket_pdb):
                    raise FileNotFoundError(f"Pocket文件未生成: {pocket_pdb}")
                
                logging.info(f"✅ docking完成，pocket文件: {pocket_pdb}")
            except Exception as e:
                logging.error(f"❌ docking失败: {str(e)}")
                raise

            # 3. 提取口袋原子
            atoms = parse_pocket(pocket_pdb)
            logging.info(f"✅ 口袋原子提取完成，共 {len(atoms)} 个原子")
            
            if len(atoms) == 0:
                raise ValueError("口袋文件中没有有效原子，无法进行预测")

            # 4. 生成图数据
            graph_data = build_graph(atoms, temperature)
            logging.info("✅ 图数据构建完成")

            # 5. 预测
            with torch.no_grad():
                graph_data = graph_data.to(device)
                predictions = model(graph_data)

            if predictions is None or predictions.numel() == 0:
                raise ValueError("模型返回了空的预测结果")
            
            if predictions.shape[0] == 0:
                raise ValueError("模型返回的预测结果第0维大小为0")
            
            if predictions.shape[1] < 2:
                raise ValueError(f"模型返回的预测结果维度不足，期望至少2维，实际得到{predictions.shape[1]}维")

            # 6. 转换回原始尺度
            kcat_pred = 10 ** predictions[0][0].item()
            km_pred = 10 ** predictions[0][1].item()

            result = {
                'sample_id': sample_id,
                'sequence': seq,
                'smiles': smiles,
                'kcat_pred': kcat_pred,
                'km_pred': km_pred,
                'km_pred_log10': np.log10(km_pred),
                'experimental_km_log10': experimental_km_log10,
                'temperature': temperature
            }
            
            # 如果有实验值，计算误差
            if experimental_km_log10 is not None:
                result['km_error_log10'] = abs(np.log10(km_pred) - experimental_km_log10)
                result['km_error_relative'] = abs(km_pred - 10**experimental_km_log10) / (10**experimental_km_log10)

            # 如果使用SampleManager，添加额外信息
            if use_sample_manager and sample_manager:
                sample_info = sample_manager.sample_registry.get(sample_id)
                if sample_info:
                    result['protein_hash'] = sample_info.protein_hash
                    result['pdb_path'] = sample_info.pdb_path

            results.append(result)
            
            # 修复Km显示格式
            if km_pred < 0.01:
                km_display = f"{km_pred:.2e}"
            else:
                km_display = f"{km_pred:.4f}"
            
            logging.info(f"✅ 预测完成 - kcat: {kcat_pred:.2f}, Km: {km_display}")

        except Exception as e:
            logging.error(f"❌ 处理样本 {sample_id} 时出错: {str(e)}")
            if use_sample_manager and sample_manager:
                sample_manager.log_failure(sample_id, "prediction", str(e), len(seq))

            results.append({
                'sample_id': sample_id,
                'sequence': seq,
                'smiles': smiles,
                'kcat_pred': None,
                'km_pred': None,
                'km_pred_log10': None,
                'experimental_km_log10': experimental_km_log10,
                'km_error_log10': None,
                'km_error_relative': None,
                'temperature': temperature,
                'error': str(e)
            })
        
        finally:
            # 清理临时PDB文件
            try:
                if 'pdb_path' in locals() and os.path.exists(pdb_path):
                    if 'temp_pdbs' in pdb_path:
                        os.remove(pdb_path)
                        logging.debug(f"清理临时PDB文件: {pdb_path}")
            except Exception as e:
                logging.debug(f"清理临时文件时出现警告: {e}")
    
    # 清理临时目录
    try:
        temp_pdb_dir = os.path.join(output_dir, 'temp_pdbs')
        if os.path.exists(temp_pdb_dir):
            import shutil
            shutil.rmtree(temp_pdb_dir)
            logging.info("✅ 临时PDB文件已清理")
    except Exception as e:
        logging.warning(f"清理临时文件时出现警告: {e}")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    logging.info(f"✅ 预测结果已保存到: {os.path.join(output_dir, 'predictions.csv')}")
    
    # 生成统计报告
    valid_predictions = results_df.dropna(subset=['kcat_pred', 'km_pred'])
    failed_predictions = results_df[results_df['kcat_pred'].isna() | results_df['km_pred'].isna()]
    
    # 保存成功预测的样本
    if len(valid_predictions) > 0:
        success_columns = ['sample_id', 'sequence', 'smiles', 'kcat_pred', 'km_pred', 'km_pred_log10', 'experimental_km_log10', 'km_error_log10', 'km_error_relative', 'temperature']
        available_columns = [col for col in success_columns if col in valid_predictions.columns]
        success_df = valid_predictions[available_columns].copy()
        success_df.to_csv(os.path.join(output_dir, 'successful_predictions.csv'), index=False)
        logging.info(f"✅ 成功预测的样本已保存: {len(success_df)} 个")
    
    # 保存失败样本
    if len(failed_predictions) > 0:
        failed_df = failed_predictions[['sample_id', 'sequence', 'smiles', 'error']].copy()
        failed_df.to_csv(os.path.join(output_dir, 'failed_predictions.csv'), index=False)
        logging.info(f"✅ 失败样本已保存: {len(failed_df)} 个")
    
    # 生成统计
    if len(valid_predictions) > 0:
        stats = {
            '样本范围': f"{start_idx}-{end_idx}",
            '总样本数': len(results_df),
            '成功预测数': len(valid_predictions),
            '失败/跳过数': len(failed_predictions),
            '成功率': f"{len(valid_predictions)/len(results_df)*100:.1f}%",
            'kcat预测范围': f"{valid_predictions['kcat_pred'].min():.2f} - {valid_predictions['kcat_pred'].max():.2f}",
            'Km预测范围': f"{valid_predictions['km_pred'].min():.2e} - {valid_predictions['km_pred'].max():.2e}"
        }
        pd.DataFrame([stats]).to_csv(os.path.join(output_dir, 'stats.csv'), index=False)
        logging.info("✅ 统计报告已生成")
    
    logging.info("🎉 范围预测任务完成！")
    return results_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="范围预测脚本")
    parser.add_argument('--input', type=str, required=True, help='输入CSV文件路径')
    parser.add_argument('--model', type=str, 
                       default="/home/lizihao/Work/enzyme_prediction/src/simple2/outputs/nopqr_attention_rbf/best_model.pt",
                       help='模型文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--start', type=int, default=0, help='开始索引')
    parser.add_argument('--end', type=int, help='结束索引（不包含）')
    parser.add_argument('--temperature', type=float, default=303.15, help='温度（K）')
    parser.add_argument('--use-sample-manager', action='store_true', help='使用SampleManager')
    parser.add_argument('--sample-data-dir', type=str, default='sample_data', help='SampleManager基础目录')
    
    args = parser.parse_args()
    
    # 进行范围预测
    results = predict_kinetics_range(
        input_data=args.input,
        model_path=args.model,
        output_dir=args.output,
        start_idx=args.start,
        end_idx=args.end,
        temperature=args.temperature,
        use_sample_manager=args.use_sample_manager,
        sample_data_dir=args.sample_data_dir
    )
    
    print(f"范围预测完成 [{args.start}:{args.end}]，结果保存在: {args.output}")
    
    # 打印简要统计
    valid_count = len(results.dropna(subset=['kcat_pred', 'km_pred']))
    total_count = len(results)
    failed_count = total_count - valid_count
    print(f"预测结果: {valid_count}/{total_count} 个样本成功，{failed_count} 个跳过/失败")
