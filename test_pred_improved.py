#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试改进后的预测脚本功能
"""

import os
import subprocess
import sys

def test_pred_modes():
    """测试不同的预测模式"""
    print("🧪 测试改进后的预测脚本功能")
    print("=" * 60)
    
    # 检查必要文件是否存在
    test_dataset = "kcat_test_new.pt"
    original_model = "outputs/kcat_after_new/best_model.pt"
    
    if not os.path.exists(test_dataset):
        print(f"❌ 测试数据集不存在: {test_dataset}")
        return False
    
    if not os.path.exists(original_model):
        print(f"❌ 原始模型不存在: {original_model}")
        print("💡 请先运行原始训练脚本")
        return False
    
    print("✅ 必要文件检查通过")
    
    # 测试1: 原始模式
    print(f"\n🔍 测试1: 原始模式")
    cmd1 = f"python pred.py --dataset {test_dataset} --model {original_model} --save_dir outputs/test_pred/original --mode original --batch_size 16"
    print(f"命令: {cmd1}")
    
    try:
        result = subprocess.run(cmd1, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ 原始模式测试成功")
        else:
            print(f"❌ 原始模式测试失败: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("⏰ 原始模式测试超时")
    except Exception as e:
        print(f"❌ 原始模式测试异常: {e}")
    
    # 测试2: 自动检测模式
    print(f"\n🔍 测试2: 自动检测模式")
    cmd2 = f"python pred.py --dataset {test_dataset} --model {original_model} --save_dir outputs/test_pred/auto --mode auto --batch_size 16"
    print(f"命令: {cmd2}")
    
    try:
        result = subprocess.run(cmd2, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ 自动检测模式测试成功")
        else:
            print(f"❌ 自动检测模式测试失败: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("⏰ 自动检测模式测试超时")
    except Exception as e:
        print(f"❌ 自动检测模式测试异常: {e}")
    
    # 测试3: 自定义模式
    print(f"\n🔍 测试3: 自定义模式")
    cmd3 = f"python pred.py --dataset {test_dataset} --model {original_model} --save_dir outputs/test_pred/custom --mode custom --hidden_dim 128 --num_layers 3 --heads 4 --dropout 0.1 --batch_size 16"
    print(f"命令: {cmd3}")
    
    try:
        result = subprocess.run(cmd3, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ 自定义模式测试成功")
        else:
            print(f"❌ 自定义模式测试失败: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("⏰ 自定义模式测试超时")
    except Exception as e:
        print(f"❌ 自定义模式测试异常: {e}")
    
    # 测试4: 改进模式（如果存在改进模型）
    improved_model = "outputs/improved_training/best_model.pt"
    if os.path.exists(improved_model):
        print(f"\n🔍 测试4: 改进模式")
        cmd4 = f"python pred.py --dataset {test_dataset} --model {improved_model} --save_dir outputs/test_pred/improved --mode improved --batch_size 16"
        print(f"命令: {cmd4}")
        
        try:
            result = subprocess.run(cmd4, shell=True, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print("✅ 改进模式测试成功")
            else:
                print(f"❌ 改进模式测试失败: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("⏰ 改进模式测试超时")
        except Exception as e:
            print(f"❌ 改进模式测试异常: {e}")
    else:
        print(f"\n⚠️ 改进模型不存在: {improved_model}")
        print("💡 请先运行改进训练脚本")
    
    print(f"\n{'='*60}")
    print("🎯 测试完成！")
    print("📁 查看测试结果:")
    print("  - outputs/test_pred/original/")
    print("  - outputs/test_pred/auto/")
    print("  - outputs/test_pred/custom/")
    if os.path.exists(improved_model):
        print("  - outputs/test_pred/improved/")

def show_usage_examples():
    """显示使用示例"""
    print(f"\n{'='*60}")
    print("📋 使用示例:")
    print("=" * 60)
    
    examples = [
        ("原始模型预测", "python pred.py --dataset kcat_test_new.pt --model outputs/kcat_after_new/best_model.pt --mode original"),
        ("自动检测模式", "python pred.py --dataset kcat_test_new.pt --model outputs/improved_training/best_model.pt --mode auto"),
        ("改进模型预测", "python pred.py --dataset kcat_test_new.pt --model outputs/improved_training/best_model.pt --mode improved"),
        ("自定义配置", "python pred.py --dataset kcat_test_new.pt --model outputs/kcat_after_new/best_model.pt --mode custom --hidden_dim 256 --num_layers 4 --heads 8 --dropout 0.2"),
        ("从配置文件", "python pred.py --dataset kcat_test_new.pt --model outputs/improved_training/best_model.pt --mode improved --load_config")
    ]
    
    for i, (desc, cmd) in enumerate(examples, 1):
        print(f"{i}. {desc}:")
        print(f"   {cmd}")
        print()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--examples":
        show_usage_examples()
    else:
        test_pred_modes()
        show_usage_examples()


