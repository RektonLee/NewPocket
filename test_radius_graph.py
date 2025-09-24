#!/usr/bin/env python3
"""
测试radius_graph的导入和功能
"""

import torch
import sys

def test_radius_graph_import():
    """测试radius_graph的导入"""
    print("测试radius_graph导入...")
    
    # 测试不同的导入方式
    try:
        from torch_cluster import radius_graph
        print("✅ 成功从 torch_cluster 导入 radius_graph")
        return radius_graph, "torch_cluster"
    except ImportError as e:
        print(f"❌ 从 torch_cluster 导入失败: {e}")
    
    try:
        from torch_geometric.nn import radius_graph
        print("✅ 成功从 torch_geometric.nn 导入 radius_graph")
        return radius_graph, "torch_geometric.nn"
    except ImportError as e:
        print(f"❌ 从 torch_geometric.nn 导入失败: {e}")
    
    print("❌ 所有导入方式都失败了")
    return None, None

def test_radius_graph_function(radius_graph_func, source):
    """测试radius_graph函数的功能"""
    if radius_graph_func is None:
        print("❌ radius_graph函数不可用，无法测试")
        return False
    
    print(f"使用来自 {source} 的 radius_graph 函数进行测试...")
    
    try:
        # 创建测试数据
        pos = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]
        ], dtype=torch.float)
        
        print(f"测试数据形状: {pos.shape}")
        
        # 调用radius_graph
        edge_index = radius_graph_func(pos, r=1.5)
        print(f"✅ radius_graph调用成功，边索引形状: {edge_index.shape}")
        print(f"边索引: {edge_index}")
        
        return True
        
    except Exception as e:
        print(f"❌ radius_graph调用失败: {e}")
        return False

def main():
    print("=" * 50)
    print("radius_graph 导入和功能测试")
    print("=" * 50)
    
    # 显示环境信息
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    
    # 测试导入
    radius_graph_func, source = test_radius_graph_import()
    
    if radius_graph_func is not None:
        # 测试功能
        success = test_radius_graph_function(radius_graph_func, source)
        
        if success:
            print("\n✅ 所有测试通过！radius_graph可以正常使用")
        else:
            print("\n❌ 功能测试失败")
    else:
        print("\n❌ 导入测试失败，无法进行功能测试")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
