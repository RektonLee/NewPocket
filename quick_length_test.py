#!/usr/bin/env python3
"""
快速序列长度测试 - 使用现有的 ESMFoldPredictor 类
"""

import os
import time
import logging
import tempfile
import pandas as pd
from generate_pdb import ESMFoldPredictor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_test_sequence(length: int) -> str:
    """生成指定长度的测试序列"""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    import random
    random.seed(42)  # 固定种子确保可重复
    return ''.join(random.choices(amino_acids, k=length))

def test_sequence_lengths():
    """测试不同长度的序列"""
    
    # 测试长度列表
    test_lengths = [100, 200, 400, 500, 800, 1000, 1200]
    
    logging.info("初始化 ESMFold 预测器...")
    try:
        # 使用较小的batch token限制来避免OOM
        predictor = ESMFoldPredictor(gpus='0', fp16=True, max_batch_tokens=1024)
        logging.info("预测器初始化成功")
    except Exception as e:
        logging.error(f"预测器初始化失败: {e}")
        return
    
    results = []
    
    for length in test_lengths:
        logging.info(f"\n测试序列长度: {length}")
        
        # 生成测试序列
        test_seq = generate_test_sequence(length)
        
        start_time = time.time()
        try:
            # 使用现有的预测方法
            batch_results = predictor.predict_batch([test_seq])
            pdb_content = batch_results[0][1]
            
            elapsed = time.time() - start_time
            
            # 检查是否生成了有效的PDB
            is_dummy = "HEADER    DUMMY STRUCTURE" in pdb_content
            success = not is_dummy
            
            result = {
                'length': length,
                'success': success,
                'is_dummy': is_dummy,
                'time_seconds': elapsed,
                'pdb_lines': len(pdb_content.split('\n')),
                'error': None
            }
            
            if success:
                logging.info(f"✓ 长度 {length}: 成功生成PDB ({elapsed:.2f}s)")
            else:
                logging.warning(f"✗ 长度 {length}: 生成了dummy PDB ({elapsed:.2f}s)")
                
        except Exception as e:
            elapsed = time.time() - start_time
            result = {
                'length': length,
                'success': False,
                'is_dummy': True,
                'time_seconds': elapsed,
                'pdb_lines': 0,
                'error': str(e)
            }
            logging.error(f"✗ 长度 {length}: 失败 - {e} ({elapsed:.2f}s)")
        
        results.append(result)
        
        # 如果连续失败，提前结束
        if not result['success'] and length >= 1000:
            logging.warning(f"长度 {length} 失败，可能已达到极限")
            break
    
    # 保存结果
    results_df = pd.DataFrame(results)
    output_file = 'quick_length_test_results.csv'
    results_df.to_csv(output_file, index=False)
    
    # 打印总结
    print("\n" + "="*70)
    print("序列长度测试结果总结:")
    print("="*70)
    print(f"{'长度':<6} {'状态':<8} {'时间(s)':<8} {'PDB行数':<8} {'备注'}")
    print("-" * 70)
    
    for _, row in results_df.iterrows():
        status = "✓ 成功" if row['success'] else "✗ 失败"
        note = "dummy" if row['is_dummy'] else "正常"
        if row['error']:
            note = f"错误: {row['error'][:20]}..."
        print(f"{row['length']:<6} {status:<8} {row['time_seconds']:<8.2f} {row['pdb_lines']:<8} {note}")
    
    print(f"\n结果已保存到: {output_file}")
    
    # 找到最大成功长度
    successful_results = results_df[results_df['success'] == True]
    if not successful_results.empty:
        max_successful_length = successful_results['length'].max()
        print(f"最大成功处理长度: {max_successful_length}")
    else:
        print("没有成功处理的序列")

if __name__ == '__main__':
    test_sequence_lengths()
