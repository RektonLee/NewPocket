#!/usr/bin/env python3
import torch
import torch_geometric

# 添加安全全局变量
torch.serialization.add_safe_globals([torch_geometric.data.data.Data])

# 加载数据
data = torch.load('kcat_dataset_enhanced1.pt', weights_only=False)

print(f'数据集大小: {len(data)}')
print(f'第一个样本的y值: {data[0].y}')
print(f'y值形状: {data[0].y.shape}')
print(f'前5个样本的y值:')
for i in range(5):
    print(f'  样本{i}: {data[i].y}')

# 检查y值的分布
y_values = [d.y for d in data]
print(f'\ny值统计:')
print(f'  最小值: {min([y.min().item() for y in y_values]):.3f}')
print(f'  最大值: {max([y.max().item() for y in y_values]):.3f}')
print(f'  平均值: {sum([y.mean().item() for y in y_values]) / len(y_values):.3f}')

