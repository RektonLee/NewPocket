"""
向后兼容性测试

该脚本验证增强版代码与原始版本的向后兼容性。
"""

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import GNN_model as MD
import sys


def test_model_initialization():
    """测试模型初始化的向后兼容性"""
    print("=" * 60)
    print("测试 1: 模型初始化（原始参数）")
    print("=" * 60)
    
    try:
        # 使用原始参数初始化模型
        model = MD.PocketGNNKcatOnly(
            node_input_dim=52,
            edge_input_dim=24,
            hidden_dim=128,
            num_layers=3,
            heads=4,
            dropout=0.1
        )
        print("✅ 模型初始化成功（使用原始参数）")
        
        # 检查模型参数数量
        num_params = sum(p.numel() for p in model.parameters())
        print(f"   模型参数数量: {num_params:,}")
        
        return True
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return False


def test_model_with_new_features():
    """测试带新特性的模型初始化"""
    print("\n" + "=" * 60)
    print("测试 2: 模型初始化（新特性）")
    print("=" * 60)
    
    tests = [
        ("GlobalAttention Pooling", {'pooling_type': 'attention'}),
        ("Set2Set Pooling", {'pooling_type': 'set2set'}),
        ("序列嵌入支持", {'use_seq_embedding': True, 'seq_embedding_dim': 1280}),
        ("所有新特性", {'pooling_type': 'attention', 'use_seq_embedding': True})
    ]
    
    all_passed = True
    for test_name, kwargs in tests:
        try:
            model = MD.PocketGNNKcatOnly(
                node_input_dim=52,
                edge_input_dim=24,
                hidden_dim=128,
                num_layers=3,
                heads=4,
                dropout=0.3,
                **kwargs
            )
            print(f"✅ {test_name}: 成功")
        except Exception as e:
            print(f"❌ {test_name}: 失败 - {e}")
            all_passed = False
    
    return all_passed


def test_forward_pass():
    """测试前向传播的向后兼容性"""
    print("\n" + "=" * 60)
    print("测试 3: 前向传播")
    print("=" * 60)
    
    try:
        from torch_geometric.data import Data
        
        # 创建虚拟数据
        num_nodes = 10
        num_edges = 20
        node_dim = 52
        edge_dim = 24
        
        x = torch.randn(num_nodes, node_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, edge_dim)
        batch = torch.zeros(num_nodes, dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        
        # 测试原始模型
        model_original = MD.PocketGNNKcatOnly(
            node_input_dim=node_dim,
            edge_input_dim=edge_dim,
            hidden_dim=128,
            num_layers=3,
            heads=4,
            dropout=0.1
        )
        
        with torch.no_grad():
            output = model_original(data)
        
        assert output.shape == (1, 1), f"输出形状错误: {output.shape}"
        print(f"✅ 原始模型前向传播成功，输出形状: {output.shape}")
        
        # 测试带 GlobalAttention 的模型
        model_attention = MD.PocketGNNKcatOnly(
            node_input_dim=node_dim,
            edge_input_dim=edge_dim,
            hidden_dim=128,
            num_layers=3,
            heads=4,
            dropout=0.1,
            pooling_type='attention'
        )
        
        with torch.no_grad():
            output_attention = model_attention(data)
        
        assert output_attention.shape == (1, 1), f"输出形状错误: {output_attention.shape}"
        print(f"✅ Attention Pooling 前向传播成功，输出形状: {output_attention.shape}")
        
        # 测试带序列嵌入的模型
        model_with_seq = MD.PocketGNNKcatOnly(
            node_input_dim=node_dim,
            edge_input_dim=edge_dim,
            hidden_dim=128,
            num_layers=3,
            heads=4,
            dropout=0.1,
            use_seq_embedding=True,
            seq_embedding_dim=1280
        )
        
        seq_embedding = torch.randn(1, 1280)  # batch_size=1, embedding_dim=1280
        
        with torch.no_grad():
            output_with_seq = model_with_seq(data, seq_embedding=seq_embedding)
        
        assert output_with_seq.shape == (1, 1), f"输出形状错误: {output_with_seq.shape}"
        print(f"✅ 序列嵌入模型前向传播成功，输出形状: {output_with_seq.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_script_compatibility():
    """测试训练脚本的参数兼容性"""
    print("\n" + "=" * 60)
    print("测试 4: 训练脚本参数兼容性")
    print("=" * 60)
    
    # 检查 train.py 是否存在
    try:
        import train
        print("✅ train.py 模块导入成功")
        
        # 检查 train 函数签名
        import inspect
        sig = inspect.signature(train.train)
        params = list(sig.parameters.keys())
        
        expected_params = [
            'dataset_path', 'save_dir', 'batch_size', 'lr', 'max_epochs',
            'weight_decay', 'dropout', 'loss_type', 'scheduler_type',
            'pooling_type', 'use_seq_embedding', 'seq_embedding_path'
        ]
        
        missing_params = [p for p in expected_params if p not in params]
        if missing_params:
            print(f"⚠️ 缺少参数: {missing_params}")
        else:
            print(f"✅ 所有预期参数都存在")
        
        # 检查默认值
        print("\n参数默认值:")
        for param_name, param in sig.parameters.items():
            if param.default != inspect.Parameter.empty:
                print(f"  {param_name}: {param.default}")
        
        return len(missing_params) == 0
        
    except Exception as e:
        print(f"❌ 训练脚本测试失败: {e}")
        return False


def test_loss_functions():
    """测试损失函数"""
    print("\n" + "=" * 60)
    print("测试 5: 损失函数")
    print("=" * 60)
    
    try:
        # MSE Loss
        mse_loss = nn.MSELoss()
        pred = torch.tensor([[1.0], [2.0], [3.0]])
        target = torch.tensor([[1.1], [2.1], [2.9]])
        mse = mse_loss(pred, target)
        print(f"✅ MSE Loss: {mse.item():.4f}")
        
        # Huber Loss
        huber_loss = nn.HuberLoss(delta=1.0)
        huber = huber_loss(pred, target)
        print(f"✅ Huber Loss: {huber.item():.4f}")
        
        return True
    except Exception as e:
        print(f"❌ 损失函数测试失败: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "🧪" * 30)
    print("开始向后兼容性测试")
    print("🧪" * 30 + "\n")
    
    results = {
        "模型初始化（原始）": test_model_initialization(),
        "模型初始化（新特性）": test_model_with_new_features(),
        "前向传播": test_forward_pass(),
        "训练脚本兼容性": test_training_script_compatibility(),
        "损失函数": test_loss_functions()
    }
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name:.<50} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！向后兼容性验证成功。")
    else:
        print("⚠️ 部分测试失败，请检查上述错误信息。")
    print("=" * 60 + "\n")
    
    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
