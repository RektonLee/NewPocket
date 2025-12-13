#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 .pt 文件中的 kcat 值是否经过 log10 处理
"""

import torch
import pandas as pd
import numpy as np

# 检查 .pt 文件
pt_path = "data/processed/kcat_test_new.pt"
print(f"加载: {pt_path}")
dataset = torch.load(pt_path, weights_only=False)

print(f"\n数据集大小: {len(dataset)}")
print(f"\n前5个样本的 kcat 值:")

for i in range(min(5, len(dataset))):
    data = dataset[i]
    y = getattr(data, 'y', None)
    sample_id = getattr(data, 'sample_id', None)
    
    if y is not None:
        if isinstance(y, torch.Tensor):
            y_val = y.item() if y.numel() == 1 else y[0].item()
        else:
            y_val = y[0] if isinstance(y, (list, np.ndarray)) else y
        
        # 判断是否经过 log10 处理
        # 如果值在合理范围内（比如 -5 到 6），可能是 log10
        # 如果值很大（比如 > 10），可能是原始值
        print(f"  [{i}] sample_id: {sample_id}")
        print(f"      y 值: {y_val}")
        print(f"      10^y = {10**y_val:.4f}")
        print(f"      如果是原始 kcat，log10(kcat) = {np.log10(abs(y_val)) if y_val != 0 else 'N/A'}")
        print()

