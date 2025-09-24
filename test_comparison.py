#!/usr/bin/env python3
"""
测试包含实验值比较的预测管道
"""

import os
import pandas as pd
from pred_fullpipeline import predict_kinetics

def test_comparison():
    """测试包含实验值的预测功能"""
    
    # 使用包含实验值的数据文件
    input_file = "/home/lizihao/Work/enzyme_prediction/PGNN/km_test_data_with_values.csv"
    
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return
    
    # 检查文件内容
    df = pd.read_csv(input_file)
    print(f"输入文件包含 {len(df)} 个样本")
    print(f"列名: {list(df.columns)}")
    
    # 检查有多少样本有实验值
    has_experimental = df['experimental_km_log10'].notna().sum()
    print(f"有实验值的样本: {has_experimental}/{len(df)}")
    
    # 检查哪些样本有PDB文件
    print("\n检查PDB文件存在性 (前10个样本):")
    for i, row in df.head(10).iterrows():
        sample_id = row['sample_id']
        pdb_path = f"sample_data/samples/{sample_id}/{sample_id}_protein.pdb"
        exists = os.path.exists(pdb_path)
        exp_val = row.get('experimental_km_log10', 'N/A')
        print(f"  {sample_id}: PDB={'✅' if exists else '❌'}, 实验值={exp_val}")
    
    # 模型路径
    model_path = "/home/lizihao/Work/enzyme_prediction/src/simple2/outputs/nopqr_attention_rbf/best_model.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    try:
        print(f"\n开始测试预测管道...")
        results = predict_kinetics(
            input_data=input_file,
            model_path=model_path,
            output_dir='test_comparison_results',
            temperature=303.15,
            use_sample_manager=True,
            sample_data_dir='sample_data'
        )
        
        print("✅ 预测测试完成！")
        
        # 分析结果
        valid_results = results.dropna(subset=['kcat_pred', 'km_pred'])
        failed_results = results[results['kcat_pred'].isna() | results['km_pred'].isna()]
        
        print(f"\n结果分析:")
        print(f"  成功预测: {len(valid_results)} 个样本")
        print(f"  跳过/失败: {len(failed_results)} 个样本")
        
        # 检查比较分析
        if 'experimental_km_log10' in valid_results.columns:
            comparison_data = valid_results.dropna(subset=['experimental_km_log10'])
            print(f"  有实验值的成功样本: {len(comparison_data)} 个")
            
            if len(comparison_data) > 0:
                print(f"\n比较分析 (前5个样本):")
                for _, result in comparison_data.head(5).iterrows():
                    pred_log10 = result['km_pred_log10']
                    exp_log10 = result['experimental_km_log10']
                    error_log10 = result['km_error_log10']
                    print(f"  {result['sample_id']}: 预测={pred_log10:.3f}, 实验={exp_log10:.3f}, 误差={error_log10:.3f}")
        
        # 检查输出文件
        print(f"\n输出文件:")
        output_files = [
            'test_comparison_results/predictions.csv',
            'test_comparison_results/successful_predictions.csv', 
            'test_comparison_results/failed_predictions.csv',
            'test_comparison_results/prediction_stats.csv',
            'test_comparison_results/comparison_analysis.csv'
        ]
        
        for file_path in output_files:
            if os.path.exists(file_path):
                print(f"  ✅ {file_path}")
                # 如果是比较分析文件，显示内容
                if 'comparison_analysis.csv' in file_path:
                    comp_df = pd.read_csv(file_path)
                    print(f"     比较统计: {comp_df.to_dict('records')[0]}")
            else:
                print(f"  ❌ {file_path} (未生成)")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_comparison()

