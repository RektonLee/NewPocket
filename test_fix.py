#!/usr/bin/env python3
"""
测试修复后的预测管道
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pred_fullpipeline import predict_kinetics
import pandas as pd

def test_prediction_fix():
    """测试修复后的预测功能"""
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'sample_id': ['test_fix_001'],
        'sequence': ['MKVLLLILLCLGLAFA'],  # 短序列用于测试
        'smiles': ['CC1=NC=C(C(=C1O)CO)CO']
    })
    
    # 模型路径（需要根据实际情况调整）
    model_path = "path/to/your/model.pt"  # 请替换为实际的模型路径
    
    try:
        print("开始测试预测管道...")
        results = predict_kinetics(
            input_data=test_data,
            model_path=model_path,
            output_dir='test_fix_results',
            temperature=303.15,
            use_sample_manager=True
        )
        
        print("✅ 预测成功完成！")
        print(f"结果数量: {len(results)}")
        
        for result in results:
            print(f"样本ID: {result['sample_id']}")
            print(f"kcat预测: {result.get('kcat_pred', 'N/A')}")
            print(f"Km预测: {result.get('km_pred', 'N/A')}")
            if 'error' in result:
                print(f"错误: {result['error']}")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction_fix()
