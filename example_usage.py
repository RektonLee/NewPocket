#!/usr/bin/env python3
"""
使用示例：展示如何使用新的预测脚本
"""

import os
import subprocess
import sys

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"执行命令: {cmd}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 命令执行成功")
            if result.stdout:
                print("输出:")
                print(result.stdout)
        else:
            print("❌ 命令执行失败")
            if result.stderr:
                print("错误信息:")
                print(result.stderr)
    except Exception as e:
        print(f"❌ 执行命令时出现异常: {e}")

def main():
    print("🎯 酶动力学预测脚本使用示例")
    print("=" * 60)
    
    # 示例1: 使用已有的图数据文件进行预测
    print("\n📊 示例1: 使用已有的图数据文件进行预测")
    print("如果你有已保存的图数据文件（.pt文件），可以使用 pred_direct.py")
    
    # 检查是否有图数据文件
    graph_data_paths = [
        "/home/lizihao/Work/enzyme_prediction/src/simple2/data/processed/dataset_DLKcat_S.pt",
        "/home/lizihao/Work/enzyme_prediction/src/simple2/data/processed/dataset_DLKcat.pt"
    ]
    
    available_graph_data = None
    for path in graph_data_paths:
        if os.path.exists(path):
            available_graph_data = path
            break
    
    if available_graph_data:
        print(f"✅ 找到图数据文件: {available_graph_data}")
        
        # 示例命令
        model_paths = [
            "/home/lizihao/Work/enzyme_prediction/src/simple2/outputs/nopqr_attention_rbf/best_model.pt",
            "/home/lizihao/Work/enzyme_prediction/src/simple2/outputs/nopqr_attention/best_model.pt"
        ]
        
        cmd1 = f"""python pred_direct.py \\
    --graph-data {available_graph_data} \\
    --model {' '.join(model_paths)} \\
    --output results/direct_prediction_test \\
    --compare"""
        
        print("示例命令:")
        print(cmd1)
        print("\n这个命令会:")
        print("- 加载已有的图数据")
        print("- 使用多个模型进行预测")
        print("- 比较不同模型的性能")
        print("- 生成详细的比较报告")
    else:
        print("❌ 未找到图数据文件")
    
    # 示例2: 从pocket PDB文件进行预测
    print("\n\n🧬 示例2: 从pocket PDB文件进行预测")
    print("如果你有pocket PDB文件，可以使用 pred_from_pockets.py")
    
    # 检查是否有pocket目录
    pocket_dirs = [
        "sample_data/samples",
        "predictions",
        "data/processed/pockets"
    ]
    
    available_pocket_dir = None
    for dir_path in pocket_dirs:
        if os.path.exists(dir_path):
            # 检查是否有pocket文件
            pocket_files = [f for f in os.listdir(dir_path) if f.endswith("_10A.pdb")]
            if pocket_files:
                available_pocket_dir = dir_path
                break
    
    if available_pocket_dir:
        print(f"✅ 找到pocket目录: {available_pocket_dir}")
        
        # 示例命令
        cmd2 = f"""python pred_from_pockets.py \\
    --pocket-dir {available_pocket_dir} \\
    --model /home/lizihao/Work/enzyme_prediction/src/simple2/outputs/nopqr_attention_rbf/best_model.pt \\
    --output results/pocket_prediction_test \\
    --temperature 303.15"""
        
        print("示例命令:")
        print(cmd2)
        print("\n这个命令会:")
        print("- 扫描pocket目录中的所有PDB文件")
        print("- 为每个pocket构建图数据")
        print("- 使用指定模型进行预测")
        print("- 生成预测结果和统计报告")
        
        # 如果有样本信息文件
        sample_info_files = [
            "km_test_data.csv",
            "kcat_data.csv",
            "test_minimal.csv"
        ]
        
        available_sample_info = None
        for file_path in sample_info_files:
            if os.path.exists(file_path):
                available_sample_info = file_path
                break
        
        if available_sample_info:
            print(f"\n✅ 找到样本信息文件: {available_sample_info}")
            cmd3 = f"""python pred_from_pockets.py \\
    --pocket-dir {available_pocket_dir} \\
    --model /home/lizihao/Work/enzyme_prediction/src/simple2/outputs/nopqr_attention_rbf/best_model.pt \\
    --sample-info {available_sample_info} \\
    --output results/pocket_prediction_with_info \\
    --compare"""
            
            print("带样本信息的示例命令:")
            print(cmd3)
            print("\n这个命令会:")
            print("- 使用样本信息文件匹配pocket文件")
            print("- 包含实验值用于误差计算")
            print("- 生成更详细的比较分析")
    else:
        print("❌ 未找到包含pocket文件的目录")
    
    # 示例3: 模型比较
    print("\n\n🔬 示例3: 模型性能比较")
    print("比较多个模型的预测性能")
    
    model_paths = [
        "/home/lizihao/Work/enzyme_prediction/src/simple2/outputs/nopqr_attention_rbf/best_model.pt",
        "/home/lizihao/Work/enzyme_prediction/src/simple2/outputs/nopqr_attention/best_model.pt",
        "/home/lizihao/Work/enzyme_prediction/src/simple2/outputs/nopqr/best_model.pt"
    ]
    
    # 检查哪些模型文件存在
    existing_models = [path for path in model_paths if os.path.exists(path)]
    
    if existing_models and available_graph_data:
        print(f"✅ 找到 {len(existing_models)} 个模型文件")
        
        cmd4 = f"""python pred_direct.py \\
    --graph-data {available_graph_data} \\
    --model {' '.join(existing_models)} \\
    --output results/model_comparison \\
    --compare"""
        
        print("模型比较命令:")
        print(cmd4)
        print("\n这个命令会:")
        print("- 使用相同的图数据测试所有模型")
        print("- 生成详细的性能比较报告")
        print("- 包含误差统计和可视化结果")
    
    # 使用建议
    print("\n\n💡 使用建议:")
    print("=" * 60)
    print("1. 如果你有完整的图数据文件（.pt），使用 pred_direct.py")
    print("   - 速度最快，跳过构图步骤")
    print("   - 适合快速测试不同模型")
    print()
    print("2. 如果你只有pocket PDB文件，使用 pred_from_pockets.py")
    print("   - 从pocket文件重新构图")
    print("   - 支持样本信息匹配")
    print("   - 适合验证新的pocket数据")
    print()
    print("3. 模型比较功能:")
    print("   - 使用 --compare 参数比较多个模型")
    print("   - 生成详细的性能统计")
    print("   - 支持误差分析和可视化")
    print()
    print("4. 输出文件说明:")
    print("   - predictions.csv: 详细预测结果")
    print("   - stats.json: 统计信息")
    print("   - model_comparison.csv: 模型比较结果")
    print("   - *.log: 详细日志文件")
    
    # 快速测试命令
    print("\n\n⚡ 快速测试命令:")
    print("=" * 60)
    
    if available_graph_data and existing_models:
        quick_test_cmd = f"""python pred_direct.py \\
    --graph-data {available_graph_data} \\
    --model {existing_models[0]} \\
    --output results/quick_test \\
    --batch-size 16"""
        
        print("快速测试单个模型:")
        print(quick_test_cmd)
    
    print("\n🎉 现在你可以开始测试不同的模型了！")

if __name__ == '__main__':
    main()

