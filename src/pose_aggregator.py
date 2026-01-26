#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pose聚合器模块
用于将多个pose的预测结果聚合成最终预测

支持多种聚合策略：
1. Score-weighted mean: 基于confidence分数的加权平均
2. Attention aggregation: 基于注意力的聚合
3. Mixture-of-experts: 混合专家模型（支持不确定性量化）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreWeightedAggregator(nn.Module):
    """
    基于confidence分数的加权聚合器
    
    使用softmax对confidence分数进行归一化，然后加权平均
    """
    def __init__(self, temperature=1.0):
        """
        Args:
            temperature: softmax温度参数，控制权重分布的尖锐程度
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, predictions, pose_scores):
        """
        Args:
            predictions: [B, K, 1] 或 [B*K, 1] - K个pose的预测结果
            pose_scores: [B, K] 或 [K] - pose的confidence分数
        
        Returns:
            aggregated: [B, 1] - 聚合后的预测
            weights: [B, K] - 每个pose的权重
        """
        # 处理输入维度
        if predictions.dim() == 2:
            # [B*K, 1] -> [B, K, 1]
            B = pose_scores.shape[0] if pose_scores.dim() == 2 else 1
            K = predictions.shape[0] // B
            predictions = predictions.view(B, K, -1)
        
        if pose_scores.dim() == 1:
            # [K] -> [1, K]
            pose_scores = pose_scores.unsqueeze(0)
        
        B, K = pose_scores.shape
        
        # 使用softmax计算权重
        weights = F.softmax(pose_scores / self.temperature, dim=1)  # [B, K]
        
        # 加权平均
        aggregated = torch.sum(predictions * weights.unsqueeze(-1), dim=1)  # [B, 1]
        
        return aggregated, weights


class AttentionAggregator(nn.Module):
    """
    基于注意力的聚合器
    
    使用MLP学习每个pose的重要性权重
    """
    def __init__(self, input_dim, hidden_dim=64):
        """
        Args:
            input_dim: 输入特征维度（通常是图embedding维度）
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for pose_score
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, graph_embeddings, pose_scores):
        """
        Args:
            graph_embeddings: [B, K, D] - K个pose的图embedding
            pose_scores: [B, K] - pose的confidence分数
        
        Returns:
            aggregated: [B, D] - 聚合后的embedding
            weights: [B, K] - 每个pose的注意力权重
        """
        B, K, D = graph_embeddings.shape
        
        # 拼接embedding和score
        if pose_scores.dim() == 1:
            pose_scores = pose_scores.unsqueeze(0)
        
        # [B, K, D] + [B, K, 1] -> [B, K, D+1]
        combined = torch.cat([graph_embeddings, pose_scores.unsqueeze(-1)], dim=-1)
        
        # 计算注意力分数
        attention_scores = self.attention(combined).squeeze(-1)  # [B, K]
        weights = F.softmax(attention_scores, dim=1)  # [B, K]
        
        # 加权聚合
        aggregated = torch.sum(graph_embeddings * weights.unsqueeze(-1), dim=1)  # [B, D]
        
        return aggregated, weights


class MixtureOfExpertsAggregator(nn.Module):
    """
    混合专家模型聚合器
    
    将每个pose的预测视为一个专家，输出混合分布
    支持不确定性量化：输出均值和方差
    """
    def __init__(self, use_uncertainty=True):
        """
        Args:
            use_uncertainty: 是否输出不确定性（方差）
        """
        super().__init__()
        self.use_uncertainty = use_uncertainty
    
    def forward(self, predictions, uncertainties=None, pose_scores=None):
        """
        Args:
            predictions: [B, K, 1] - K个pose的预测均值
            uncertainties: [B, K, 1] - K个pose的预测方差（可选）
            pose_scores: [B, K] - pose的confidence分数（用于计算权重）
        
        Returns:
            mean: [B, 1] - 聚合后的预测均值
            var: [B, 1] - 聚合后的预测方差（如果use_uncertainty=True）
            weights: [B, K] - 每个pose的权重
        """
        B, K, _ = predictions.shape
        
        # 计算权重
        if pose_scores is not None:
            if pose_scores.dim() == 1:
                pose_scores = pose_scores.unsqueeze(0)
            weights = F.softmax(pose_scores, dim=1)  # [B, K]
        else:
            # 均匀权重
            weights = torch.ones(B, K, device=predictions.device) / K
        
        # 聚合均值
        mean = torch.sum(predictions * weights.unsqueeze(-1), dim=1)  # [B, 1]
        
        if self.use_uncertainty:
            if uncertainties is not None:
                # 混合分布的方差公式：
                # Var[Y] = Σ w_k (σ_k² + μ_k²) - (Σ w_k μ_k)²
                # 其中 w_k 是权重，μ_k 是均值，σ_k² 是方差
                weighted_variance = torch.sum(
                    weights.unsqueeze(-1) * (uncertainties + predictions ** 2),
                    dim=1
                )  # [B, 1]
                var = weighted_variance - mean ** 2
                var = torch.clamp(var, min=1e-6)  # 防止负方差
            else:
                # 如果没有提供不确定性，使用预测的方差
                # 计算预测的方差（epistemic uncertainty）
                var = torch.sum(
                    weights.unsqueeze(-1) * (predictions - mean.unsqueeze(1)) ** 2,
                    dim=1
                )  # [B, 1]
            
            return mean, var, weights
        else:
            return mean, weights


def create_aggregator(aggregator_type='score_weighted', **kwargs):
    """
    创建聚合器
    
    Args:
        aggregator_type: 聚合器类型
            - 'score_weighted': 基于分数的加权平均
            - 'attention': 注意力聚合
            - 'mixture': 混合专家模型
        **kwargs: 聚合器特定参数
    
    Returns:
        aggregator实例
    """
    if aggregator_type == 'score_weighted':
        temperature = kwargs.get('temperature', 1.0)
        return ScoreWeightedAggregator(temperature=temperature)
    
    elif aggregator_type == 'attention':
        input_dim = kwargs.get('input_dim', 128)
        hidden_dim = kwargs.get('hidden_dim', 64)
        return AttentionAggregator(input_dim=input_dim, hidden_dim=hidden_dim)
    
    elif aggregator_type == 'mixture':
        use_uncertainty = kwargs.get('use_uncertainty', True)
        return MixtureOfExpertsAggregator(use_uncertainty=use_uncertainty)
    
    else:
        raise ValueError(f"Unknown aggregator type: {aggregator_type}")
