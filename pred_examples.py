#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预测脚本使用示例
展示如何使用改进后的pred.py进行不同模型的预测
"""

import os
import subprocess
import sys

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"💻 命令: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 命令执行成功")
            if result.stdout:
                print("📊 输出:")
                print(result.stdout)
        else:
            print("❌ 命令执行失败")
            if result.stderr:
                print("🔍 错误信息:")
                print(result.stderr)
    except Exception as e:
        print(f"❌ 执行异常: {e}")

def main():
    """主函数 - 展示不同的预测模式"""
    
    print("📋 预测脚本使用示例")
    print("=" * 60)
    
    # 示例1: 原始模型预测
    cmd1 = """python pred.py --dataset kcat_test_new.pt --model outputs/kcat_after_new/best_model.pt --save_dir outputs/predictions/original_model --mode original"""
    run_command(cmd1, "原始模型预测")
    
    # 示例2: 改进模型预测（自动检测）
    cmd2 = """python pred.py --dataset kcat_test_new.pt --model outputs/improved_training/best_model.pt --save_dir outputs/predictions/improved_model --mode auto"""
    run_command(cmd2, "改进模型预测（自动检测）")
    
    # 示例3: 改进模型预测（显式指定）
    cmd3 = """python pred.py --dataset kcat_test_new.pt --model outputs/improved_training/best_model.pt --save_dir outputs/predictions/improved_explicit --mode improved"""
    run_command(cmd3, "改进模型预测（显式指定）")
    
    # 示例4: 自定义模型配置预测
    cmd4 = """python pred.py --dataset kcat_test_new.pt --model outputs/kcat_after_new/best_model.pt --save_dir outputs/predictions/custom_model --mode custom --hidden_dim 256 --num_layers 4 --heads 8 --dropout 0.2"""
    run_command(cmd4, "自定义模型配置预测")
    
    # 示例5: 从配置文件加载
    cmd5 = """python pred.py --dataset kcat_test_new.pt --model outputs/improved_training/best_model.pt --save_dir outputs/predictions/config_file --mode improved --load_config"""
    run_command(cmd5, "从配置文件加载模型参数")
    
    print(f"\n{'='*60}")
    print("📊 使用说明:")
    print("1. --mode original: 使用原始模型配置 (hidden_dim=128, num_layers=3, heads=4, dropout=0.1)")
    print("2. --mode improved: 使用改进模型配置 (hidden_dim=256, num_layers=4, heads=8, dropout=0.2)")
    print("3. --mode auto: 自动检测模型类型（推荐）")
    print("4. --mode custom: 自定义模型参数")
    print("5. --load_config: 从模型目录的model_config.json文件加载配置")
    print("6. --config_file: 指定配置文件路径")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()


