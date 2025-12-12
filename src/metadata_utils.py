"""
训练元数据管理工具
用于记录和管理深度学习训练实验的元数据

新的实验记录结构：
experiments/
├── kcat_attn_v1/
│   ├── run_01/
│   │   ├── metadata.yaml
│   │   ├── metrics.json
│   │   └── ckpt.pt
│   ├── run_02/
│   └── README.md
"""

import os
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd

try:
    import yaml
except ImportError:
    print("⚠️  PyYAML not installed. Installing...")
    subprocess.check_call(["pip", "install", "pyyaml"])
    import yaml


def get_git_commit() -> str:
    """获取当前 git commit hash"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "not_git_repo"


def get_next_run_number(exp_dir: Path) -> int:
    """获取下一个 run 编号"""
    if not exp_dir.exists():
        return 1
    
    existing_runs = [d for d in exp_dir.iterdir() 
                     if d.is_dir() and d.name.startswith('run_')]
    
    if not existing_runs:
        return 1
    
    run_numbers = []
    for run_dir in existing_runs:
        try:
            num = int(run_dir.name.split('_')[1])
            run_numbers.append(num)
        except (ValueError, IndexError):
            continue
    
    return max(run_numbers) + 1 if run_numbers else 1


class ExperimentTracker:
    """实验跟踪器 - 管理所有训练实验的元数据（新版本：语义化文件夹 + YAML）"""
    
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # 保留旧的汇总文件（向后兼容）
        self.summary_file = self.base_dir / "experiments_summary.csv"
    
    def start_experiment(self,
                        exp_name: str,
                        save_dir: str,
                        dataset_path: str,
                        model_version: str = "unknown",
                        graph_builder_version: str = "unknown",
                        comments: str = "",
                        config_path: Optional[str] = None,
                        **kwargs) -> Tuple[str, str]:
        """
        开始新实验并返回 (exp_name, run_id)
        
        Args:
            exp_name: 实验名称（语义化，如 "kcat_attn_v1"）
            save_dir: 模型保存目录（outputs/...）
            dataset_path: 数据集路径
            model_version: 模型版本
            graph_builder_version: 图构建器版本
            comments: 备注信息
            config_path: 配置文件路径（可选）
            **kwargs: 其他参数
        
        Returns:
            (exp_name, run_id) 元组，如 ("kcat_attn_v1", "run_01")
        """
        # 创建实验目录
        exp_dir = self.base_dir / exp_name
        exp_dir.mkdir(exist_ok=True)
        
        # 获取下一个 run 编号
        run_num = get_next_run_number(exp_dir)
        run_id = f"run_{run_num:02d}"
        run_dir = exp_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        # 创建 metadata.yaml
        start_time = datetime.now()
        metadata = {
            'run_id': run_id,
            'exp_name': exp_name,
            'start_time': start_time.isoformat(),
            'git_commit': get_git_commit(),
            'save_dir': str(save_dir),
            'dataset_path': dataset_path,
            'model_version': model_version,
            'graph_builder_version': graph_builder_version,
            'status': 'running',
            'note': comments,
            **kwargs
        }
        
        if config_path:
            metadata['config'] = config_path
        
        # 保存 metadata.yaml
        metadata_file = run_dir / "metadata.yaml"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
        
        # 在 save_dir 中保存 run_dir 路径，方便后续更新
        run_dir_file = Path(save_dir) / "run_dir.txt"
        run_dir_file.parent.mkdir(parents=True, exist_ok=True)
        with open(run_dir_file, 'w') as f:
            f.write(str(run_dir))
        
        print(f"🚀 开始实验: {exp_name} / {run_id}")
        print(f"   实验目录: {run_dir}")
        print(f"   保存目录: {save_dir}")
        print(f"   Git commit: {metadata['git_commit']}")
        
        return exp_name, run_id
    
    def update_experiment(self,
                         run_dir: Path,
                         status: str = None,
                         best_val_loss: float = None,
                         final_r2: float = None,
                         final_pearson: float = None,
                         training_time_minutes: float = None,
                         **kwargs):
        """
        更新实验状态和结果
        
        Args:
            run_dir: run 目录路径（experiments/exp_name/run_XX/）
            status: 训练状态
            best_val_loss: 最佳验证损失
            final_r2: 最终R²分数
            final_pearson: 最终Pearson相关系数
            training_time_minutes: 训练时间（分钟）
            **kwargs: 其他结果参数
        """
        metadata_file = run_dir / "metadata.yaml"
        
        if not metadata_file.exists():
            print(f"⚠️  未找到 metadata.yaml: {metadata_file}")
            return
        
        # 读取现有 metadata
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = yaml.safe_load(f)
        
        # 更新字段
        if status:
            metadata['status'] = status
            if status == 'completed':
                metadata['end_time'] = datetime.now().isoformat()
        
        if best_val_loss is not None:
            metadata['best_val_loss'] = best_val_loss
        
        if final_r2 is not None:
            metadata['final_r2'] = final_r2
        
        if final_pearson is not None:
            metadata['final_pearson'] = final_pearson
        
        if training_time_minutes is not None:
            metadata['training_time_minutes'] = training_time_minutes
        
        # 更新其他参数
        for key, value in kwargs.items():
            metadata[key] = value
        
        # 保存 metrics.json
        metrics = {
            'best_val_loss': metadata.get('best_val_loss'),
            'final_r2': metadata.get('final_r2'),
            'final_pearson': metadata.get('final_pearson'),
            'training_time_minutes': metadata.get('training_time_minutes')
        }
        metrics_file = run_dir / "metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # 保存更新后的 metadata.yaml
        with open(metadata_file, 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
    
    def get_experiment(self, exp_name: str, run_id: str) -> Optional[Dict]:
        """获取特定实验的详细信息"""
        run_dir = self.base_dir / exp_name / run_id
        metadata_file = run_dir / "metadata.yaml"
        
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def list_experiments(self) -> Dict[str, list]:
        """列出所有实验"""
        experiments = {}
        
        if not self.base_dir.exists():
            return experiments
        
        for exp_dir in sorted(self.base_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            
            exp_name = exp_dir.name
            runs = []
            
            for run_dir in sorted(exp_dir.iterdir()):
                if not run_dir.is_dir() or not run_dir.name.startswith('run_'):
                    continue
                
                metadata_file = run_dir / "metadata.yaml"
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        runs.append(yaml.safe_load(f))
            
            if runs:
                experiments[exp_name] = runs
        
        return experiments
    
    def print_summary(self):
        """打印实验汇总"""
        experiments = self.list_experiments()
        
        if not experiments:
            print("📊 暂无实验记录")
            return
        
        print("📊 实验汇总:")
        print("-" * 80)
        
        for exp_name, runs in sorted(experiments.items()):
            print(f"\n📁 {exp_name}")
            for run_data in runs:
                status_emoji = {
                    'running': '🔄',
                    'completed': '✅',
                    'failed': '❌'
                }.get(run_data.get('status', 'unknown'), '❓')
                
                run_id = run_data.get('run_id', 'N/A')
                print(f"  {status_emoji} {run_id}")
                print(f"    开始时间: {run_data.get('start_time', 'N/A')}")
                print(f"    Git commit: {run_data.get('git_commit', 'N/A')}")
                
                if run_data.get('final_r2') is not None:
                    print(f"    最终R²: {run_data['final_r2']:.3f}")
                if run_data.get('best_val_loss') is not None:
                    print(f"    最佳验证损失: {run_data['best_val_loss']:.4f}")
                
                if run_data.get('note'):
                    print(f"    备注: {run_data['note']}")
                print()


def save_metadata(save_dir: str,
                  dataset_path: str,
                  exp_name: str = "kcat_default",
                  graph_builder_version: str = "unknown",
                  gnn_model_version: str = "unknown",
                  comments: str = "",
                  config_path: Optional[str] = None,
                  **kwargs):
    """
    保存训练元数据（新版本：使用语义化实验名称）
    
    Args:
        save_dir: 模型保存目录（outputs/...）
        dataset_path: 数据集路径
        exp_name: 实验名称（语义化，如 "kcat_attn_v1"）
        graph_builder_version: 图构建器版本
        gnn_model_version: GNN模型版本
        comments: 备注信息
        config_path: 配置文件路径（可选）
        **kwargs: 其他参数
    
    Returns:
        (exp_name, run_id) 元组
    """
    tracker = ExperimentTracker()
    
    # 开始实验
    exp_name, run_id = tracker.start_experiment(
        exp_name=exp_name,
        save_dir=save_dir,
        dataset_path=dataset_path,
        model_version=gnn_model_version,
        graph_builder_version=graph_builder_version,
        comments=comments,
        config_path=config_path,
        **kwargs
    )
    
    return exp_name, run_id


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
        save_dir: 模型保存目录（outputs/...）
        best_val_loss: 最佳验证损失
        final_r2: 最终R²分数
        final_pearson: 最终Pearson相关系数
        training_time_minutes: 训练时间（分钟）
        status: 训练状态
        **kwargs: 其他结果参数
    """
    tracker = ExperimentTracker()
    
    # 读取 run_dir 路径
    run_dir_file = Path(save_dir) / "run_dir.txt"
    if not run_dir_file.exists():
        print(f"⚠️  未找到 run_dir.txt 文件: {run_dir_file}")
        return
    
    with open(run_dir_file, 'r') as f:
        run_dir = Path(f.read().strip())
    
    # 更新实验结果
    tracker.update_experiment(
        run_dir=run_dir,
        status=status,
        best_val_loss=best_val_loss,
        final_r2=final_r2,
        final_pearson=final_pearson,
        training_time_minutes=training_time_minutes,
        **kwargs
    )
    
    print(f"✅ 已更新实验结果: {run_dir}")


if __name__ == "__main__":
    # 测试功能
    tracker = ExperimentTracker()
    tracker.print_summary()
