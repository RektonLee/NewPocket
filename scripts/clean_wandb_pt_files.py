#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理 wandb 目录下的 .pt 模型文件

这些文件通常是训练过程中上传到 wandb 的模型检查点，会占用大量空间。
原始模型文件已经保存在 outputs/ 目录下，wandb 中的是重复的。
"""

import os
import sys
from pathlib import Path


def find_pt_files_in_wandb(wandb_dir="wandb"):
    """查找 wandb 目录下的所有 .pt 文件"""
    wandb_path = Path(wandb_dir)
    if not wandb_path.exists():
        print(f"❌ wandb 目录不存在: {wandb_dir}")
        return []
    
    pt_files = []
    for pt_file in wandb_path.rglob("*.pt"):
        pt_files.append(pt_file)
    
    return pt_files


def clean_wandb_pt_files(wandb_dir="wandb", dry_run=True):
    """清理 wandb 目录下的 .pt 文件"""
    pt_files = find_pt_files_in_wandb(wandb_dir)
    
    if not pt_files:
        print(f"✅ wandb 目录下没有找到 .pt 文件")
        return
    
    print(f"找到 {len(pt_files)} 个 .pt 文件")
    
    if dry_run:
        print(f"\n[DRY RUN] 以下文件将被删除:")
        total_size = 0
        for pt_file in pt_files:
            size = pt_file.stat().st_size
            total_size += size
            print(f"  {pt_file} ({size / 1024 / 1024:.2f} MB)")
        print(f"\n总大小: {total_size / 1024 / 1024:.2f} MB")
        print(f"\n要实际删除，请使用 --execute 参数")
    else:
        print(f"\n开始删除...")
        deleted_count = 0
        deleted_size = 0
        failed_count = 0
        
        for pt_file in pt_files:
            try:
                size = pt_file.stat().st_size
                pt_file.unlink()
                deleted_count += 1
                deleted_size += size
                print(f"  ✅ 已删除: {pt_file} ({size / 1024 / 1024:.2f} MB)")
            except Exception as e:
                failed_count += 1
                print(f"  ❌ 删除失败: {pt_file} - {e}")
        
        print(f"\n{'='*60}")
        print(f"删除完成:")
        print(f"  成功删除: {deleted_count} 个文件")
        print(f"  释放空间: {deleted_size / 1024 / 1024:.2f} MB")
        if failed_count > 0:
            print(f"  失败: {failed_count} 个文件")
        print(f"{'='*60}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="清理 wandb 目录下的 .pt 模型文件"
    )
    parser.add_argument(
        "--wandb_dir", default="wandb",
        help="wandb 目录路径（默认: wandb）"
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="实际执行删除操作（默认是 dry run，只显示将要删除的文件）"
    )
    
    args = parser.parse_args()
    
    clean_wandb_pt_files(
        wandb_dir=args.wandb_dir,
        dry_run=not args.execute
    )


if __name__ == "__main__":
    main()

