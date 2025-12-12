"""
样本管理器 - 基于sample_id的数据组织和去重
替代hash编码，使用清晰的sample_id进行数据管理
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
from dataclasses import dataclass, asdict
import fcntl
import time


@dataclass
class SampleInfo:
    """样本信息数据类"""
    sample_id: str
    sequence: str
    smiles: str
    protein_hash: str  # 用于蛋白质去重
    substrate_hash: str  # 用于底物去重
    temperature: float = 303.15
    status: str = "pending"  # pending, processing, completed, failed
    error_msg: str = ""
    pdb_path: str = ""
    ligand_path: str = ""
    docking_path: str = ""
    

class SampleManager:
    """
    样本管理器 - 统一管理sample_id和文件组织
    
    特性:
    1. 基于sample_id创建文件夹结构
    2. 蛋白质序列去重(相同序列共享PDB)
    3. 失败记录和状态跟踪
    4. 支持断点续传
    """
    
    def __init__(self, base_dir: str = "sample_data", enable_deduplication: bool = True):
        self.base_dir = Path(base_dir)
        self.enable_deduplication = enable_deduplication
        
        # 创建基础目录结构
        self.samples_dir = self.base_dir / "samples"
        self.shared_dir = self.base_dir / "shared"
        self.metadata_dir = self.base_dir / "metadata"
        
        for dir_path in [self.samples_dir, self.shared_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 子目录
        self.shared_proteins_dir = self.shared_dir / "proteins"
        self.shared_ligands_dir = self.shared_dir / "ligands"
        self.shared_proteins_dir.mkdir(exist_ok=True)
        self.shared_ligands_dir.mkdir(exist_ok=True)
        
        # 元数据文件
        self.registry_file = self.metadata_dir / "sample_registry.json"
        self.protein_index_file = self.metadata_dir / "protein_index.json"
        self.ligand_index_file = self.metadata_dir / "ligand_index.json"
        self.failure_log_file = self.metadata_dir / "failures.json"
        
        # 加载现有数据
        self.sample_registry: Dict[str, SampleInfo] = self._load_registry()
        self.protein_index: Dict[str, str] = self._load_index(self.protein_index_file)
        self.ligand_index: Dict[str, str] = self._load_index(self.ligand_index_file)
        self.failure_log: List[Dict] = self._load_failures()
        
        logging.info(f"SampleManager初始化完成: {len(self.sample_registry)} 个样本")
    
    def _compute_hash(self, content: str) -> str:
        """计算内容的hash值"""
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _load_registry(self) -> Dict[str, SampleInfo]:
        """加载样本注册表"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    content = f.read().strip()
                    if not content:  # 文件为空
                        return {}
                    data = json.loads(content)
                    return {k: SampleInfo(**v) for k, v in data.items()}
            except (json.JSONDecodeError, ValueError) as e:
                logging.warning(f"Registry file {self.registry_file} is corrupted: {e}. Starting with empty registry.")
                return {}
        return {}
    
    def _load_index(self, index_file: Path) -> Dict[str, str]:
        """加载索引文件"""
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    content = f.read().strip()
                    if not content:  # 文件为空
                        return {}
                    return json.loads(content)
            except (json.JSONDecodeError, ValueError) as e:
                logging.warning(f"Index file {index_file} is corrupted: {e}. Starting with empty index.")
                return {}
        return {}
    
    def _load_failures(self) -> List[Dict]:
        """加载失败记录"""
        if self.failure_log_file.exists():
            try:
                with open(self.failure_log_file, 'r') as f:
                    content = f.read().strip()
                    if not content:  # 文件为空
                        return []
                    return json.loads(content)
            except (json.JSONDecodeError, ValueError) as e:
                logging.warning(f"Failure log file {self.failure_log_file} is corrupted: {e}. Starting with empty failure log.")
                return []
        return []
    
    def save_metadata(self):
        """保存所有元数据"""
        # 保存样本注册表
        with open(self.registry_file, 'w') as f:
            json.dump({k: asdict(v) for k, v in self.sample_registry.items()}, f, indent=2)
        
        # 保存索引
        with open(self.protein_index_file, 'w') as f:
            json.dump(self.protein_index, f, indent=2)
        
        with open(self.ligand_index_file, 'w') as f:
            json.dump(self.ligand_index, f, indent=2)
        
        # 保存失败记录
        with open(self.failure_log_file, 'w') as f:
            json.dump(self.failure_log, f, indent=2)
    
    def register_samples_from_csv(self, csv_path: str, temperature: float = 303.15) -> int:
        """从CSV注册样本"""
        df = pd.read_csv(csv_path)
        required_cols = ['sample_id', 'sequence', 'smiles']
        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"CSV缺少必要列: {col}")
        
        new_count = 0
        for _, row in df.iterrows():
            sample_id = str(row['sample_id'])
            sequence = str(row['sequence'])
            smiles = str(row['smiles'])
            
            if sample_id not in self.sample_registry:
                sample_info = SampleInfo(
                    sample_id=sample_id,
                    sequence=sequence,
                    smiles=smiles,
                    protein_hash=self._compute_hash(sequence),
                    substrate_hash=self._compute_hash(smiles),
                    temperature=temperature
                )
                self.sample_registry[sample_id] = sample_info
                new_count += 1
        
        self.save_metadata()
        logging.info(f"注册了 {new_count} 个新样本")
        return new_count
    
    def get_sample_dir(self, sample_id: str) -> Path:
        """获取样本专用目录"""
        sample_dir = self.samples_dir / sample_id
        sample_dir.mkdir(exist_ok=True)
        return sample_dir
    
    def get_protein_path(self, sample_id: str, force_unique: bool = False) -> Tuple[Path, bool]:
        """
        获取蛋白质PDB路径（线程安全版本）
        返回: (路径, 是否为共享文件)
        """
        if sample_id not in self.sample_registry:
            raise ValueError(f"未知sample_id: {sample_id}")
        
        sample_info = self.sample_registry[sample_id]
        protein_hash = sample_info.protein_hash
        
        # 使用文件锁确保并发安全
        lock_file = self.metadata_dir / f"protein_{protein_hash}.lock"
        
        try:
            with open(lock_file, 'w') as lock_fd:
                # 获取独占锁
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
                
                # 重新加载最新的索引（其他进程可能已更新）
                self.protein_index = self._load_index(self.protein_index_file)
                
                # 检查是否可以使用共享文件
                if self.enable_deduplication and not force_unique and protein_hash in self.protein_index:
                    shared_path = Path(self.protein_index[protein_hash])
                    if shared_path.exists():
                        # 创建共享链接文件
                        sample_dir = self.get_sample_dir(sample_id)
                        link_file = sample_dir / f"{sample_id}_protein.txt"
                        with open(link_file, 'w') as f:
                            f.write(str(shared_path))
                        logging.info(f"使用共享PDB: {sample_id} -> {shared_path}")
                        return shared_path, True
                
                # 使用样本专用路径
                sample_dir = self.get_sample_dir(sample_id)
                unique_path = sample_dir / f"{sample_id}_protein.pdb"
                
                # 如果启用去重，记录到共享索引
                if self.enable_deduplication:
                    self.protein_index[protein_hash] = str(unique_path)
                    self.save_metadata()
                    logging.info(f"创建新PDB: {sample_id} -> {unique_path}")
                
                return unique_path, False
                
        except Exception as e:
            logging.error(f"获取蛋白质路径时出错: {e}")
            # 降级到非锁定模式
            sample_dir = self.get_sample_dir(sample_id)
            unique_path = sample_dir / f"{sample_id}_protein.pdb"
            return unique_path, False
        finally:
            # 清理锁文件
            try:
                if lock_file.exists():
                    lock_file.unlink()
            except:
                pass
    
    def get_ligand_path(self, sample_id: str, format: str = "sdf") -> Path:
        """获取配体结构路径"""
        sample_dir = self.get_sample_dir(sample_id)
        return sample_dir / f"{sample_id}_ligand.{format}"
    
    def get_docking_dir(self, sample_id: str) -> Path:
        """获取对接结果目录"""
        sample_dir = self.get_sample_dir(sample_id)
        docking_dir = sample_dir / "docking"
        docking_dir.mkdir(exist_ok=True)
        return docking_dir
    
    def update_sample_status(self, sample_id: str, status: str, error_msg: str = "", **extra_fields):
        """更新样本状态"""
        if sample_id in self.sample_registry:
            self.sample_registry[sample_id].status = status
            self.sample_registry[sample_id].error_msg = error_msg
            
            # 更新额外字段
            for field, value in extra_fields.items():
                if hasattr(self.sample_registry[sample_id], field):
                    setattr(self.sample_registry[sample_id], field, value)
            
            self.save_metadata()
    
    def log_failure(self, sample_id: str, stage: str, error_msg: str, sequence_length: int = 0):
        """记录失败"""
        failure_record = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "sample_id": sample_id,
            "stage": stage,  # structure_prediction, docking, etc.
            "error_msg": error_msg,
            "sequence_length": sequence_length
        }
        self.failure_log.append(failure_record)
        self.update_sample_status(sample_id, "failed", error_msg)
        logging.warning(f"样本 {sample_id} 在 {stage} 阶段失败: {error_msg}")
    
    def get_samples_by_status(self, status: str) -> List[str]:
        """按状态获取样本ID列表"""
        return [sid for sid, info in self.sample_registry.items() if info.status == status]
    
    def get_failed_samples_summary(self) -> pd.DataFrame:
        """获取失败样本摘要"""
        if not self.failure_log:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.failure_log)
        summary = df.groupby(['stage', 'error_msg']).agg({
            'sample_id': 'count',
            'sequence_length': 'mean'
        }).reset_index()
        summary.columns = ['失败阶段', '错误信息', '样本数量', '平均序列长度']
        return summary
    
    def find_similar_proteins(self, sample_id: str) -> List[str]:
        """找到使用相同蛋白质的其他样本"""
        if sample_id not in self.sample_registry:
            return []
        
        target_hash = self.sample_registry[sample_id].protein_hash
        similar = []
        
        for sid, info in self.sample_registry.items():
            if sid != sample_id and info.protein_hash == target_hash:
                similar.append(sid)
        
        return similar
    
    def cleanup_unused_files(self, dry_run: bool = True) -> List[str]:
        """清理未使用的文件"""
        unused_files = []
        
        # 检查共享蛋白质文件
        for protein_file in self.shared_proteins_dir.glob("*.pdb"):
            # 检查是否在索引中被引用
            if not any(Path(path) == protein_file for path in self.protein_index.values()):
                unused_files.append(str(protein_file))
                if not dry_run:
                    protein_file.unlink()
        
        if unused_files:
            logging.info(f"{'发现' if dry_run else '清理了'} {len(unused_files)} 个未使用文件")
        
        return unused_files
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        total_samples = len(self.sample_registry)
        status_counts = {}
        for info in self.sample_registry.values():
            status_counts[info.status] = status_counts.get(info.status, 0) + 1
        
        unique_proteins = len(set(info.protein_hash for info in self.sample_registry.values()))
        unique_ligands = len(set(info.substrate_hash for info in self.sample_registry.values()))
        
        return {
            "总样本数": total_samples,
            "状态分布": status_counts,
            "唯一蛋白质数": unique_proteins,
            "唯一配体数": unique_ligands,
            "去重率": f"{(1 - unique_proteins/total_samples)*100:.1f}%" if total_samples > 0 else "0%",
            "失败记录数": len(self.failure_log)
        }


def demo_usage():
    """演示用法"""
    # 初始化管理器
    manager = SampleManager("sample_data")
    
    # 注册样本
    manager.register_samples_from_csv("pred_DLKcat_S.csv")
    
    # 获取路径
    protein_path, is_shared = manager.get_protein_path("DLKcat_001")
    ligand_path = manager.get_ligand_path("DLKcat_001")
    docking_dir = manager.get_docking_dir("DLKcat_001")
    
    print(f"蛋白质路径: {protein_path} (共享: {is_shared})")
    print(f"配体路径: {ligand_path}")
    print(f"对接目录: {docking_dir}")
    
    # 更新状态
    manager.update_sample_status("DLKcat_001", "completed")
    
    # 记录失败
    manager.log_failure("DLKcat_002", "structure_prediction", "序列过长", sequence_length=1200)
    
    # 查看统计
    print("\n统计信息:")
    for k, v in manager.get_stats().items():
        print(f"{k}: {v}")
    
    # 查看失败摘要
    failures = manager.get_failed_samples_summary()
    if not failures.empty:
        print("\n失败摘要:")
        print(failures)


if __name__ == "__main__":
    demo_usage()