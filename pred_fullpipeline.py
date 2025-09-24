import torch
import pandas as pd
import numpy as np
from data_loader import ProteinStructureProcessor, MoleculeEmbeddingGenerator
from GNN_model import PocketGNNWithAttention, PocketGNNWithAttentionNoTemp
from sample_manager import SampleManager  # 新增导入
import logging
import os
from graph_builder_rbf import build_graph, parse_pocket, DIST_CUTOFF, RBF_CENTERS, RBF_DMIN, RBF_DMAX, RBF_GAMMA
# docking集成 - 使用原始docking.py（修复了路径问题）
from docking import run_preprocess
import hashlib
import signal
import time
from contextlib import contextmanager

# 从graph_builder_rbf.py中获取正确的维度
NODE_INPUT_DIM = 52  # 10(el_feat) + 21(res_feat) + 1(is_lig) + 1(min_dists) + 16(elec) + 3(props)
EDGE_INPUT_DIM = RBF_CENTERS  # 16，从RBF_CENTERS获取

# 超时处理类
class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds):
    """超时上下文管理器"""
    def signal_handler(signum, frame):
        raise TimeoutError(f"操作超时 ({seconds}秒)")
    
    # 设置信号处理器
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # 恢复原来的信号处理器
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def predict_kinetics(input_data, model_path, output_dir='predictions', temperature=303.15, use_sample_manager=True, sample_data_dir='sample_data', docking_timeout=100):
    """
    预测酶动力学参数
    
    参数:
    input_data: 可以是CSV文件路径，或者包含sample_id/sequence/smiles的DataFrame，或者(sequences, smiles)元组
    model_path: 训练好的模型路径
    output_dir: 输出目录
    temperature: 温度（默认303.15K，即30℃）
    use_sample_manager: 是否使用SampleManager进行文件管理
    sample_data_dir: SampleManager基础目录
    docking_timeout: docking超时时间（秒），默认300秒（5分钟）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'prediction.log')),
            logging.StreamHandler()
        ]
    )
    
    # 解析输入数据
    if isinstance(input_data, str):
        # CSV文件路径
        input_df = pd.read_csv(input_data)
    elif isinstance(input_data, pd.DataFrame):
        input_df = input_data
    elif isinstance(input_data, tuple) and len(input_data) == 2:
        # (sequences, smiles)元组 - 向后兼容
        protein_sequences, substrate_smiles = input_data
        input_df = pd.DataFrame({
            'sample_id': [f"sample_{i+1}" for i in range(len(protein_sequences))],
            'sequence': protein_sequences,
            'smiles': substrate_smiles
        })
        use_sample_manager = False  # 自动禁用SampleManager
    else:
        raise ValueError("input_data必须是CSV路径、DataFrame或(sequences, smiles)元组")
    
    # 检查必要的列
    required_columns = ['sequence', 'smiles']
    if use_sample_manager:
        required_columns.append('sample_id')
    
    missing_columns = [col for col in required_columns if col not in input_df.columns]
    if missing_columns:
        raise ValueError(f"输入数据缺少必要的列: {missing_columns}")
    
    # 兼容原始数据格式，如果没有sample_id但不使用sample_manager，自动生成
    if 'sample_id' not in input_df.columns and not use_sample_manager:
        input_df['sample_id'] = [f"sample_{i+1}" for i in range(len(input_df))]
    
    # 初始化SampleManager（如果启用）
    sample_manager = None
    if use_sample_manager:
        sample_manager = SampleManager(sample_data_dir)
        # 注册样本
        new_samples = sample_manager.register_samples_from_csv(input_data if isinstance(input_data, str) else None)
        if new_samples == 0 and isinstance(input_data, pd.DataFrame):
            # 手动注册DataFrame中的样本
            for _, row in input_df.iterrows():
                sample_manager.register_samples_from_csv(input_df)
                break
        logging.info(f"SampleManager已初始化，管理 {len(sample_manager.sample_registry)} 个样本")
    
    # 初始化处理器
    structure_processor = ProteinStructureProcessor(sample_manager=sample_manager)
    # molecule_generator = MoleculeEmbeddingGenerator()
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 首先尝试加载模型状态字典来检查是否包含温度模块
    try:
        state_dict = torch.load(model_path, map_location=device)
        has_temp_module = any(key.startswith('temp_mlp') for key in state_dict.keys())
        logging.info(f"模型包含温度模块: {has_temp_module}")
    except Exception as e:
        logging.error(f"❌ 无法加载模型状态字典: {str(e)}")
        raise
    
    # 根据模型是否包含温度模块来创建相应的模型
    if has_temp_module:
        # 包含温度模块的模型
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
        # 不包含温度模块的模型
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

    
    results = []
    total_samples = len(input_df)
    
    for i, row in input_df.iterrows():
        sample_id = row.get('sample_id', f'sample_{i+1}')
        seq = row['sequence']
        smiles = row['smiles']
        uniprot_id = row.get('uniprot', None)
        experimental_km_log10 = row.get('experimental_km_log10', None)  # 获取实验值
        try:
            logging.info(f"处理样本 {i+1}/{total_samples}: {sample_id}")
            # 1. 获取蛋白质结构
            if use_sample_manager and sample_manager:
                # 获取蛋白质路径（支持去重）
                protein_path, is_shared = sample_manager.get_protein_path(sample_id)
                
                if protein_path.exists() and protein_path.stat().st_size > 0:
                    # 直接使用已存在的PDB文件路径
                    pdb_path = str(protein_path)
                    logging.info(f"使用已存在的PDB文件: {pdb_path}")
                else:
                    # 如果文件不存在，跳过该样本
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

            # 2. docking 步骤，生成 pocket pdb
            if use_sample_manager:
                # 使用SampleManager时，口袋文件保存在sample_data/samples/sample_id/目录中
                pocket_dir = os.path.join(sample_data_dir, "samples", sample_id)
            else:
                # 不使用SampleManager时，口袋文件保存在output_dir中
                pocket_dir = output_dir
            os.makedirs(pocket_dir, exist_ok=True)
            
            # 计算口袋文件名
            pocket_hash = int(hashlib.sha256(smiles.encode()).hexdigest(), 16) & 0xffff
            base_name = f"{uniprot_id or sample_id}_{pocket_hash}"
            pocket_pdb = os.path.join(pocket_dir, f"{base_name}_10A.pdb")
            
            # 检查pocket文件是否已经存在
            if os.path.exists(pocket_pdb) and os.path.getsize(pocket_pdb) > 0:
                logging.info(f"✅ 使用已存在的pocket文件: {pocket_pdb}")
            else:
                try:
                    logging.info(f"🔄 开始docking处理: {sample_id} (超时限制: {docking_timeout}秒)")
                    start_time = time.time()
                    
                    # 使用超时机制运行docking预处理
                    with timeout(docking_timeout):
                        result = run_preprocess(uniprot_id or sample_id, smiles, pdb_path, pocket_pdb, i)
                        if not result:
                            raise RuntimeError("docking预处理失败")
                    
                    # 检查pocket文件是否真的生成了
                    if not os.path.exists(pocket_pdb):
                        raise FileNotFoundError(f"Pocket文件未生成: {pocket_pdb}")
                    
                    elapsed_time = time.time() - start_time
                    logging.info(f"✅ docking完成，pocket文件: {pocket_pdb} (耗时: {elapsed_time:.1f}秒)")
                    
                except TimeoutError as e:
                    logging.warning(f"⏰ docking超时跳过样本 {sample_id}: {str(e)}")
                    if use_sample_manager and sample_manager:
                        sample_manager.log_failure(sample_id, "docking_timeout", str(e), len(seq))
                    
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
                        'error': f'docking超时: {str(e)}'
                    })
                    continue
                except Exception as e:
                    logging.error(f"❌ docking失败: {str(e)}")
                    raise

            # 3. 提取口袋原子（用pocket_pdb）
            atoms = parse_pocket(pocket_pdb)
            logging.info(f"✅ 口袋原子提取完成，共 {len(atoms)} 个原子")
            
            # 检查原子数量
            if len(atoms) == 0:
                raise ValueError("口袋文件中没有有效原子，无法进行预测")

            # 4. 生成图数据
            graph_data = build_graph(atoms, temperature)
            logging.info("✅ 图数据构建完成")

            # 5. 预测
            with torch.no_grad():
                graph_data = graph_data.to(device)
                
                # 如果模型不包含温度模块，移除温度信息
                if not has_temp_module and hasattr(graph_data, 'temperature'):
                    delattr(graph_data, 'temperature')
                    logging.info("✅ 移除温度信息以匹配模型")
                
                predictions = model(graph_data)

            # 检查预测结果的有效性
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
                'km_pred_log10': np.log10(km_pred),  # 添加log10值便于比较
                'experimental_km_log10': experimental_km_log10,  # 添加实验值
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
            
            # Fix Km display format
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
            # 清理临时PDB文件（只清理临时创建的，不删除SampleManager管理的文件）
            try:
                if 'pdb_path' in locals() and os.path.exists(pdb_path):
                    # 只删除临时目录中的文件
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
    
    # 保存成功预测的样本到单独的CSV文件
    if len(valid_predictions) > 0:
        # 选择关键列，包含比较信息
        success_columns = ['sample_id', 'sequence', 'smiles', 'kcat_pred', 'km_pred', 'km_pred_log10', 'experimental_km_log10', 'km_error_log10', 'km_error_relative', 'temperature']
        # 只保留存在的列
        available_columns = [col for col in success_columns if col in valid_predictions.columns]
        success_df = valid_predictions[available_columns].copy()
        success_df.to_csv(os.path.join(output_dir, 'successful_predictions.csv'), index=False)
        logging.info(f"✅ 成功预测的样本已保存到: {os.path.join(output_dir, 'successful_predictions.csv')}")
        logging.info(f"   包含 {len(success_df)} 个成功预测的样本")
    
    # 保存失败/跳过的样本到单独的CSV文件
    if len(failed_predictions) > 0:
        failed_df = failed_predictions[['sample_id', 'sequence', 'smiles', 'error']].copy()
        failed_df.to_csv(os.path.join(output_dir, 'failed_predictions.csv'), index=False)
        logging.info(f"✅ 失败/跳过的样本已保存到: {os.path.join(output_dir, 'failed_predictions.csv')}")
        logging.info(f"   包含 {len(failed_df)} 个失败/跳过的样本")
    
    if len(valid_predictions) > 0:
        stats = {
            '总样本数': len(results_df),
            '成功预测数': len(valid_predictions),
            '失败/跳过数': len(failed_predictions),
            '成功率': f"{len(valid_predictions)/len(results_df)*100:.1f}%",
            'kcat预测范围': f"{valid_predictions['kcat_pred'].min():.2f} - {valid_predictions['kcat_pred'].max():.2f}",
            'Km预测范围': f"{valid_predictions['km_pred'].min():.2e} - {valid_predictions['km_pred'].max():.2e}"
        }
        pd.DataFrame([stats]).to_csv(os.path.join(output_dir, 'prediction_stats.csv'), index=False)
        logging.info("✅ 统计报告已生成")
        
        # 记录失败原因统计
        if len(failed_predictions) > 0:
            failure_reasons = failed_predictions['error'].value_counts()
            failure_stats = pd.DataFrame({
                '失败原因': failure_reasons.index,
                '数量': failure_reasons.values
            })
            failure_stats.to_csv(os.path.join(output_dir, 'failure_reasons.csv'), index=False)
            logging.info(f"✅ 失败原因统计已生成: {len(failed_predictions)} 个失败样本")
        
        # 如果有实验值，生成比较分析报告
        if 'experimental_km_log10' in valid_predictions.columns:
            comparison_data = valid_predictions.dropna(subset=['experimental_km_log10'])
            if len(comparison_data) > 0:
                comparison_stats = {
                    '有实验值的样本数': len(comparison_data),
                    '平均绝对误差(log10)': comparison_data['km_error_log10'].mean(),
                    '中位数绝对误差(log10)': comparison_data['km_error_log10'].median(),
                    '平均相对误差': comparison_data['km_error_relative'].mean(),
                    '中位数相对误差': comparison_data['km_error_relative'].median(),
                    'R²相关系数': comparison_data['km_pred_log10'].corr(comparison_data['experimental_km_log10'])**2
                }
                pd.DataFrame([comparison_stats]).to_csv(os.path.join(output_dir, 'comparison_analysis.csv'), index=False)
                logging.info(f"✅ 比较分析报告已生成: {len(comparison_data)} 个有实验值的样本")
    
    # 如果使用SampleManager，生成更详细的报告
    if use_sample_manager and sample_manager:
        # 保存SampleManager统计
        sm_stats = sample_manager.get_stats()
        with open(os.path.join(output_dir, 'sample_manager_stats.json'), 'w') as f:
            import json
            json.dump(sm_stats, f, indent=2)
        
        # 保存失败摘要
        failures = sample_manager.get_failed_samples_summary()
        if not failures.empty:
            failures.to_csv(os.path.join(output_dir, 'failures_summary.csv'), index=False)
        
        logging.info("✅ SampleManager报告已生成")
    
    return results_df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='输入CSV文件路径，包含sample_id, sequence和smiles列')
    parser.add_argument('--model', type=str, default="/home/lizihao/Work/enzyme_prediction/src/simple2/scripts/outputs/nopqr_attention_rbf_no_temp/best_model.pt", help='模型文件路径')
    parser.add_argument('--output', type=str, default='predictions', help='输出目录')
    parser.add_argument('--temperature', type=float, default=303.15, help='温度（K），默认303.15K（30℃）')
    parser.add_argument('--use-sample-manager', action='store_true', help='使用SampleManager进行文件管理（推荐）')
    parser.add_argument('--sample-data-dir', type=str, default='sample_data', help='SampleManager基础目录')
    parser.add_argument('--docking-timeout', type=int, default=100, help='docking超时时间（秒），默认300秒（5分钟）')
    args = parser.parse_args()
    
    # 进行预测
    results = predict_kinetics(
        input_data=args.input,
        model_path=args.model,
        output_dir=args.output,
        temperature=args.temperature,
        use_sample_manager=args.use_sample_manager,
        sample_data_dir=args.sample_data_dir,
        docking_timeout=args.docking_timeout
    )
    
    print(f"预测完成，结果已保存到 {args.output}/")
    print(f"  📊 完整结果: predictions.csv")
    print(f"  ✅ 成功样本: successful_predictions.csv")
    print(f"  ❌ 失败样本: failed_predictions.csv")
    print(f"  📈 统计报告: prediction_stats.csv")
    print(f"  🔍 比较分析: comparison_analysis.csv")
    
    # 打印简要统计
    valid_count = len(results.dropna(subset=['kcat_pred', 'km_pred']))
    total_count = len(results)
    failed_count = total_count - valid_count
    print(f"\n预测完成: {valid_count}/{total_count} 个样本成功，{failed_count} 个跳过/失败")
    
    if args.use_sample_manager:
        print(f"详细报告请查看: {args.sample_data_dir}/metadata/ 目录")
    else:
        print("建议使用 --use-sample-manager 获得更好的文件管理和失败跟踪")