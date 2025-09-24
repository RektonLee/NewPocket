#!/usr/bin/env python3
"""
测试单GPU处理是否正常工作
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_pdb_fixed import ESMFoldPredictor
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_single_sequence():
    """测试单个序列处理"""
    # 创建预测器（单GPU）
    predictor = ESMFoldPredictor(gpus="0", fp16=True)
    
    # 测试序列
    test_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    
    try:
        # 测试单序列预测
        pdb_content = predictor.predict_single_sequence(test_sequence, 0)
        
        if pdb_content and len(pdb_content) > 100:  # 检查PDB内容是否合理
            print("✅ 单序列预测成功！")
            print(f"PDB长度: {len(pdb_content)} 字符")
            print("前100个字符:")
            print(pdb_content[:100])
            return True
        else:
            print("❌ PDB内容为空或太短")
            return False
            
    except Exception as e:
        print(f"❌ 单序列预测失败: {e}")
        return False

if __name__ == "__main__":
    print("测试单GPU单序列处理...")
    success = test_single_sequence()
    if success:
        print("测试通过！")
    else:
        print("测试失败！")


