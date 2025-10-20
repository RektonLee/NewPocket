"""
训练元数据管理工具
用于记录和管理深度学习训练实验的元数据
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd


class ExperimentTracker:
    """实验跟踪器 - 管理所有训练实验的元数据"""
    
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # 实验记录文件
        self.experiments_file = self.base_dir / "experiments_log.json"
        self.summary_file = self.base_dir / "experiments_summary.csv"
        
        # 加载现有实验记录
        self.experiments = self._load_experiments()
    
    def _load_experiments(self) -> Dict[str, Dict]:
        """加载现有实验记录"""
        if self.experiments_file.exists():
            with open(self.experiments_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_experiments(self):
        """保存实验记录"""
        with open(self.experiments_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiments, f, indent=2, ensure_ascii=False)
    
    def _update_summary_csv(self):
        """更新汇总CSV文件"""
        if not self.experiments:
            return
        
        summary_data = []
        for exp_id, exp_data in self.experiments.items():
            summary_data.append({
                'experiment_id': exp_id,
                'timestamp': exp_data.get('timestamp', ''),
                'save_dir': exp_data.get('save_dir', ''),
                'dataset_path': exp_data.get('dataset_path', ''),
                'model_version': exp_data.get('model_version', ''),
                'graph_builder_version': exp_data.get('graph_builder_version', ''),
                'comments': exp_data.get('comments', ''),
                'status': exp_data.get('status', 'running'),
                'best_val_loss': exp_data.get('best_val_loss', None),
                'final_r2': exp_data.get('final_r2', None),
                'final_pearson': exp_data.get('final_pearson', None),
                'training_time_minutes': exp_data.get('training_time_minutes', None)
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(self.summary_file, index=False, encoding='utf-8')
    
    def start_experiment(self, 
                        save_dir: str,
                        dataset_path: str,
                        model_version: str = "unknown",
                        graph_builder_version: str = "unknown", 
                        comments: str = "",
                        **kwargs) -> str:
        """开始新实验并返回实验ID"""
        
        # 生成实验ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"exp_{timestamp}"
        
        # 创建实验记录
        experiment_data = {
            'experiment_id': exp_id,
            'timestamp': timestamp,
            'start_time': datetime.now().isoformat(),
            'save_dir': save_dir,
            'dataset_path': dataset_path,
            'model_version': model_version,
            'graph_builder_version': graph_builder_version,
            'comments': comments,
            'status': 'running',
            'parameters': kwargs
        }
        
        self.experiments[exp_id] = experiment_data
        self._save_experiments()
        self._update_summary_csv()
        
        print(f"🚀 开始实验 {exp_id}")
        print(f"   保存目录: {save_dir}")
        print(f"   数据集: {dataset_path}")
        print(f"   模型版本: {model_version}")
        
        return exp_id
    
    def update_experiment(self, 
                         exp_id: str,
                         status: str = None,
                         best_val_loss: float = None,
                         final_r2: float = None,
                         final_pearson: float = None,
                         training_time_minutes: float = None,
                         **kwargs):
        """更新实验状态和结果"""
        
        if exp_id not in self.experiments:
            print(f"⚠️  实验 {exp_id} 不存在")
            return
        
        if status:
            self.experiments[exp_id]['status'] = status
            if status == 'completed':
                self.experiments[exp_id]['end_time'] = datetime.now().isoformat()
        
        if best_val_loss is not None:
            self.experiments[exp_id]['best_val_loss'] = best_val_loss
        
        if final_r2 is not None:
            self.experiments[exp_id]['final_r2'] = final_r2
            
        if final_pearson is not None:
            self.experiments[exp_id]['final_pearson'] = final_pearson
            
        if training_time_minutes is not None:
            self.experiments[exp_id]['training_time_minutes'] = training_time_minutes
        
        # 更新其他参数
        for key, value in kwargs.items():
            self.experiments[exp_id][key] = value
        
        self._save_experiments()
        self._update_summary_csv()
    
    def get_experiment(self, exp_id: str) -> Optional[Dict]:
        """获取特定实验的详细信息"""
        return self.experiments.get(exp_id)
    
    def list_experiments(self, status: str = None) -> Dict[str, Dict]:
        """列出所有实验，可选按状态过滤"""
        if status:
            return {k: v for k, v in self.experiments.items() if v.get('status') == status}
        return self.experiments
    
    def print_summary(self):
        """打印实验汇总"""
        if not self.experiments:
            print("📊 暂无实验记录")
            return
        
        print("📊 实验汇总:")
        print("-" * 80)
        
        for exp_id, exp_data in sorted(self.experiments.items(), key=lambda x: x[1]['timestamp'], reverse=True):
            status_emoji = {
                'running': '🔄',
                'completed': '✅', 
                'failed': '❌'
            }.get(exp_data.get('status', 'unknown'), '❓')
            
            print(f"{status_emoji} {exp_id}")
            print(f"   时间: {exp_data.get('timestamp', 'N/A')}")
            print(f"   状态: {exp_data.get('status', 'N/A')}")
            print(f"   保存目录: {exp_data.get('save_dir', 'N/A')}")
            print(f"   模型: {exp_data.get('model_version', 'N/A')}")
            
            if exp_data.get('final_r2') is not None:
                print(f"   最终R²: {exp_data['final_r2']:.3f}")
            if exp_data.get('best_val_loss') is not None:
                print(f"   最佳验证损失: {exp_data['best_val_loss']:.4f}")
            
            print(f"   备注: {exp_data.get('comments', 'N/A')}")
            print()


def save_metadata(save_dir: str,
                  dataset_path: str,
                  graph_builder_version: str = "unknown",
                  gnn_model_version: str = "unknown", 
                  comments: str = "",
                  **kwargs):
    """
    保存训练元数据（向后兼容函数）
    
    Args:
        save_dir: 模型保存目录
        dataset_path: 数据集路径
        graph_builder_version: 图构建器版本
        gnn_model_version: GNN模型版本
        comments: 备注信息
        **kwargs: 其他参数
    """
    tracker = ExperimentTracker()
    
    # 开始实验
    exp_id = tracker.start_experiment(
        save_dir=save_dir,
        dataset_path=dataset_path,
        model_version=gnn_model_version,
        graph_builder_version=graph_builder_version,
        comments=comments,
        **kwargs
    )
    
    # 在save_dir中保存实验ID，方便后续更新
    metadata_file = Path(save_dir) / "experiment_id.txt"
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, 'w') as f:
        f.write(exp_id)
    
    return exp_id


def update_training_results(save_dir: str,
                           best_val_loss: float = None,
                           final_r2: float = None,
                           final_pearson: float = None,
                           training_time_minutes: float = None,
                           status: str = "completed",
                           **kwargs):
    """
    更新训练结果
    
    Args:
        save_dir: 模型保存目录
        best_val_loss: 最佳验证损失
        final_r2: 最终R²分数
        final_pearson: 最终Pearson相关系数
        training_time_minutes: 训练时间（分钟）
        status: 训练状态
        **kwargs: 其他结果参数
    """
    tracker = ExperimentTracker()
    
    # 读取实验ID
    metadata_file = Path(save_dir) / "experiment_id.txt"
    if not metadata_file.exists():
        print(f"⚠️  未找到实验ID文件: {metadata_file}")
        return
    
    with open(metadata_file, 'r') as f:
        exp_id = f.read().strip()
    
    # 更新实验结果
    tracker.update_experiment(
        exp_id=exp_id,
        status=status,
        best_val_loss=best_val_loss,
        final_r2=final_r2,
        final_pearson=final_pearson,
        training_time_minutes=training_time_minutes,
        **kwargs
    )
    
    print(f"✅ 已更新实验 {exp_id} 的结果")


if __name__ == "__main__":
    # 测试功能
    tracker = ExperimentTracker()
    tracker.print_summary()
