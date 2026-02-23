import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def quantile_loss(pred, target, quantiles=[0.05, 0.5, 0.95], weights=None):
    """
    Quantile Loss (Pinball Loss) for uncertainty quantification
    
    Args:
        pred: [batch_size, num_quantiles] - 预测的分位数值
        target: [batch_size, 1] - 真实值
        quantiles: list of quantile levels (e.g., [0.05, 0.5, 0.95] for 5%, median, 95%)
        weights: optional list of weights for each quantile (default: None, equal weights)
                 If provided, should have same length as quantiles.
                 Example: [0.2, 0.6, 0.2] to emphasize median prediction
    
    Returns:
        loss: scalar tensor
    """
    target = target.expand_as(pred)  # [batch_size, num_quantiles]
    errors = target - pred  # [batch_size, num_quantiles]
    
    losses = []
    for i, q in enumerate(quantiles):
        # Pinball loss: max(q * error, (q-1) * error)
        loss_q = torch.max(q * errors[:, i], (q - 1) * errors[:, i])
        losses.append(loss_q)
    
    # 应用权重（如果提供）
    if weights is not None:
        assert len(weights) == len(quantiles), "Weights must have same length as quantiles"
        weights_tensor = torch.tensor(weights, device=losses[0].device, dtype=losses[0].dtype)
        weighted_losses = [losses[i] * weights_tensor[i] for i in range(len(losses))]
        total_loss = torch.stack(weighted_losses).sum() / weights_tensor.sum()
    else:
        # 总loss = 所有分位数loss的平均
        total_loss = torch.stack(losses).mean()
    
    return total_loss

def compute_quantile_metrics(pred_quantiles, target, quantiles=[0.05, 0.5, 0.95]):
    """
    计算分位数预测的评估指标
    
    Returns:
        dict with metrics including:
        - coverage: 真实值落在预测区间内的比例
        - interval_width: 预测区间的平均宽度
        - median_mae: 中位数预测的MAE
    """
    pred_quantiles = pred_quantiles.numpy()  # [N, num_quantiles]
    target = target.numpy().flatten()  # [N]
    
    q_low = pred_quantiles[:, 0]   # 5% 分位数
    q_median = pred_quantiles[:, 1]  # 50% 分位数（中位数）
    q_high = pred_quantiles[:, 2]  # 95% 分位数
    
    # Coverage: 真实值落在 [q_low, q_high] 区间内的比例
    in_interval = (target >= q_low) & (target <= q_high)
    coverage = in_interval.mean()
    
    # Interval width: 预测区间的平均宽度
    interval_width = (q_high - q_low).mean()
    
    # Median prediction MAE
    from sklearn.metrics import mean_absolute_error
    median_mae = mean_absolute_error(target, q_median)
    
    return {
        'coverage': coverage,
        'interval_width': interval_width,
        'median_mae': median_mae,
        'q_low_mean': q_low.mean(),
        'q_high_mean': q_high.mean()
    }



