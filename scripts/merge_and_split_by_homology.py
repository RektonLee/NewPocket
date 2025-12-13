#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并两个 .pt 文件，然后基于同源聚类（40% identity）进行划分

使用方法:
    python scripts/merge_and_split_by_homology.py \
        --train_pt data/processed/kcat_full_1213.pt \
        --test_pt data/processed/kcat_test_new.pt \
        --train_csv data/raw/successful_full_train_1213.csv \
        --test_csv data/raw/kcat_test_results.csv \
        --output_prefix data/processed/kcat_merged_hom40 \
        --identity_threshold 0.4
"""

import argparse
import os
import sys
import subprocess
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch


def merge_pt_files(pt_files, output_path):
    """合并多个 .pt 文件"""
    print(f"合并 .pt 文件...")
    all_data = []
    
    for pt_file in pt_files:
        print(f"  加载: {pt_file}")
        dataset = torch.load(pt_file, weights_only=False)
        print(f"    包含 {len(dataset)} 个样本")
        all_data.extend(dataset)
    
    print(f"  总计: {len(all_data)} 个样本")
    print(f"  保存到: {output_path}")
    torch.save(all_data, output_path)
    
    return all_data


def create_fasta_from_pt(pt_path, train_csv, test_csv, output_fasta):
    """从 .pt 文件提取 sample_id，然后从对应的 CSV 中提取序列生成 FASTA"""
    print(f"\n从 .pt 文件提取序列...")
    dataset = torch.load(pt_path, weights_only=False)
    
    # 加载两个 CSV 文件
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    
    # 合并 CSV（如果需要）
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    
    # 创建 sample_id -> sequence 的映射
    id_to_seq = dict(zip(df_all["sample_id"], df_all["sequence"]))
    
    # 提取序列
    sequences = []
    missing_ids = []
    
    for data in dataset:
        sid = getattr(data, "sample_id", None)
        if sid is None:
            print(f"  警告: 发现没有 sample_id 的数据条目，跳过")
            continue
        
        if sid in id_to_seq:
            seq = id_to_seq[sid]
            if pd.notna(seq) and len(str(seq).strip()) > 0:
                sequences.append((sid, str(seq).strip()))
            else:
                missing_ids.append(sid)
        else:
            missing_ids.append(sid)
    
    print(f"  成功提取 {len(sequences)} 个序列")
    if missing_ids:
        print(f"  警告: {len(missing_ids)} 个 sample_id 在 CSV 中未找到或序列为空")
        if len(missing_ids) <= 10:
            print(f"  缺失的 sample_id: {missing_ids[:10]}")
    
    # 写入 FASTA
    with open(output_fasta, "w") as f:
        for sid, seq in sequences:
            f.write(f">{sid}\n{seq}\n")
    
    print(f"  FASTA 文件已保存: {output_fasta}")
    return len(sequences), missing_ids


def run_mmseqs_cluster(fasta_path, output_dir, identity_threshold=0.4, coverage=0.8):
    """使用 MMseqs2 对序列进行聚类"""
    print(f"\n运行 MMseqs2 聚类 (identity >= {identity_threshold}, coverage >= {coverage})...")
    
    # 检查 MMseqs2 是否安装
    try:
        result = subprocess.run(["mmseqs", "easy-cluster"], capture_output=True, text=True, timeout=5)
        if "easy-cluster" not in result.stdout and "easy-cluster" not in result.stderr:
            raise FileNotFoundError
    except (FileNotFoundError, subprocess.TimeoutExpired):
        raise RuntimeError(
            "MMseqs2 未安装或不在 PATH 中。\n"
            "请先激活 conda 环境: conda activate env2\n"
            "或安装方法: conda install -c conda-forge mmseqs2\n"
            "或访问: https://github.com/soedinglab/MMseqs2"
        )
    
    # 创建临时目录
    tmp_dir = os.path.join(output_dir, "mmseqs_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    
    cluster_prefix = os.path.join(output_dir, "clusters")
    
    # 运行聚类
    cmd = [
        "mmseqs", "easy-cluster",
        fasta_path,
        cluster_prefix,
        tmp_dir,
        "--min-seq-id", str(identity_threshold),
        "-c", str(coverage),
        "--threads", "4"
    ]
    
    print(f"  执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  MMseqs2 错误输出:\n{result.stderr}")
        raise RuntimeError(f"MMseqs2 聚类失败: {result.stderr}")
    
    # 查找输出文件
    cluster_tsv = f"{cluster_prefix}_cluster.tsv"
    if not os.path.exists(cluster_tsv):
        possible_names = [
            f"{cluster_prefix}_cluster.tsv",
            f"{cluster_prefix}.cluster.tsv",
            os.path.join(output_dir, "clusters_cluster.tsv")
        ]
        for name in possible_names:
            if os.path.exists(name):
                cluster_tsv = name
                break
        else:
            raise FileNotFoundError(f"找不到聚类输出文件。可能的文件: {possible_names}")
    
    print(f"  聚类完成，结果文件: {cluster_tsv}")
    return cluster_tsv


def read_mmseqs_tsv(tsv_path):
    """读取 MMseqs2 聚类结果，返回 member_id -> cluster_rep 的映射"""
    member_to_rep = {}
    with open(tsv_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                rep = parts[0]
                mem = parts[1]
                member_to_rep[mem] = rep
    return member_to_rep


def make_cluster_ids(member_to_rep):
    """将 cluster_rep 转换为数字 cluster_id"""
    reps = sorted(set(member_to_rep.values()))
    rep_to_cid = {rep: i for i, rep in enumerate(reps)}
    member_to_cid = {m: rep_to_cid[rep] for m, rep in member_to_rep.items()}
    return member_to_cid, rep_to_cid


def split_clusters(rep_to_cid, seed=42, ratios=(0.8, 0.1, 0.1)):
    """按聚类划分数据集"""
    assert abs(sum(ratios) - 1.0) < 1e-6
    cids = list(rep_to_cid.values())
    random.Random(seed).shuffle(cids)
    n = len(cids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_c = set(cids[:n_train])
    val_c = set(cids[n_train:n_train+n_val])
    test_c = set(cids[n_train+n_val:])
    return train_c, val_c, test_c


def split_dataset_by_clusters(dataset, member_to_cid, train_c, val_c, test_c):
    """根据聚类结果划分数据集"""
    train, val, test, missing = [], [], [], 0
    
    for d in dataset:
        sid = getattr(d, "sample_id", None)
        if sid is None:
            raise ValueError("Data object missing sample_id; cannot do homology split.")
        if sid not in member_to_cid:
            missing += 1
            continue  # drop to avoid leakage
        cid = member_to_cid[sid]
        if cid in train_c:
            train.append(d)
        elif cid in val_c:
            val.append(d)
        else:
            test.append(d)
    
    return train, val, test, missing


def main():
    parser = argparse.ArgumentParser(
        description="合并 .pt 文件并基于同源聚类进行划分"
    )
    parser.add_argument(
        "--train_pt", required=True,
        help="训练集 .pt 文件路径"
    )
    parser.add_argument(
        "--test_pt", required=True,
        help="测试集 .pt 文件路径"
    )
    parser.add_argument(
        "--train_csv", required=True,
        help="训练集 CSV 文件路径，包含 sample_id 和 sequence"
    )
    parser.add_argument(
        "--test_csv", required=True,
        help="测试集 CSV 文件路径，包含 sample_id 和 sequence"
    )
    parser.add_argument(
        "--output_prefix", required=True,
        help="输出文件前缀，例如: data/processed/kcat_merged_hom40"
    )
    parser.add_argument(
        "--identity_threshold", type=float, default=0.4,
        help="序列相似度阈值（默认: 0.4，即40%% identity）"
    )
    parser.add_argument(
        "--coverage", type=float, default=0.8,
        help="覆盖度阈值（默认: 0.8）"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（默认: 42）"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8,
        help="训练集比例（默认: 0.8）"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1,
        help="验证集比例（默认: 0.1）"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.1,
        help="测试集比例（默认: 0.1）"
    )
    parser.add_argument(
        "--skip_clustering", action="store_true",
        help="跳过聚类步骤，使用已有的聚类结果（需要 --cluster_tsv）"
    )
    parser.add_argument(
        "--cluster_tsv",
        help="已有的聚类结果文件（当使用 --skip_clustering 时）"
    )
    
    args = parser.parse_args()
    
    # 检查比例
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError(f"比例之和必须为 1.0，当前: {args.train_ratio + args.val_ratio + args.test_ratio}")
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 1. 合并 .pt 文件
    merged_pt = args.output_prefix + "_merged.pt"
    dataset = merge_pt_files([args.train_pt, args.test_pt], merged_pt)
    
    # 2. 生成 FASTA 文件
    fasta_path = args.output_prefix + "_sequences.fasta"
    create_fasta_from_pt(merged_pt, args.train_csv, args.test_csv, fasta_path)
    
    # 3. 运行 MMseqs2 聚类
    if args.skip_clustering:
        if not args.cluster_tsv or not os.path.exists(args.cluster_tsv):
            raise ValueError("使用 --skip_clustering 时必须提供有效的 --cluster_tsv")
        cluster_tsv = args.cluster_tsv
        print(f"\n跳过聚类，使用已有结果: {cluster_tsv}")
    else:
        cluster_tsv = run_mmseqs_cluster(
            fasta_path, output_dir if output_dir else ".",
            args.identity_threshold, args.coverage
        )
    
    # 4. 解析聚类结果
    print(f"\n解析聚类结果...")
    member_to_rep = read_mmseqs_tsv(cluster_tsv)
    member_to_cid, rep_to_cid = make_cluster_ids(member_to_rep)
    print(f"  发现 {len(rep_to_cid)} 个聚类")
    print(f"  总序列数: {len(member_to_cid)}")
    
    # 5. 划分聚类
    train_c, val_c, test_c = split_clusters(
        rep_to_cid,
        seed=args.seed,
        ratios=(args.train_ratio, args.val_ratio, args.test_ratio)
    )
    print(f"\n聚类划分:")
    print(f"  训练集聚类: {len(train_c)} 个")
    print(f"  验证集聚类: {len(val_c)} 个")
    print(f"  测试集聚类: {len(test_c)} 个")
    
    # 6. 根据聚类划分数据集
    print(f"\n划分数据集...")
    train, val, test, missing = split_dataset_by_clusters(
        dataset, member_to_cid, train_c, val_c, test_c
    )
    
    print(f"\n{'='*60}")
    print(f"划分结果")
    print(f"{'='*60}")
    print(f"总样本数: {len(dataset)}")
    print(f"缺失 sample_id 在聚类表中: {missing}")
    print(f"训练集: {len(train)} 个样本 ({len(train)/len(dataset)*100:.2f}%)")
    print(f"验证集: {len(val)} 个样本 ({len(val)/len(dataset)*100:.2f}%)")
    print(f"测试集: {len(test)} 个样本 ({len(test)/len(dataset)*100:.2f}%)")
    print(f"{'='*60}\n")
    
    # 7. 保存划分后的数据集
    train_path = args.output_prefix + "_train.pt"
    val_path = args.output_prefix + "_val.pt"
    test_path = args.output_prefix + "_test.pt"
    
    print(f"保存数据集...")
    torch.save(train, train_path)
    print(f"  训练集: {train_path}")
    torch.save(val, val_path)
    print(f"  验证集: {val_path}")
    torch.save(test, test_path)
    print(f"  测试集: {test_path}")
    
    # 8. 验证无泄露
    print(f"\n验证数据泄露...")
    train_ids = {getattr(d, "sample_id", None) for d in train}
    val_ids = {getattr(d, "sample_id", None) for d in val}
    test_ids = {getattr(d, "sample_id", None) for d in test}
    
    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print(f"  ⚠️  警告: 发现直接重复的 sample_id！")
        if train_val_overlap:
            print(f"    训练集-验证集重叠: {len(train_val_overlap)} 个")
        if train_test_overlap:
            print(f"    训练集-测试集重叠: {len(train_test_overlap)} 个")
        if val_test_overlap:
            print(f"    验证集-测试集重叠: {len(val_test_overlap)} 个")
    else:
        print(f"  ✅ 未发现直接重复的 sample_id")
    
    # 检查聚类级别的泄露
    train_clusters = {member_to_cid.get(getattr(d, "sample_id", None)) for d in train if getattr(d, "sample_id", None) in member_to_cid}
    val_clusters = {member_to_cid.get(getattr(d, "sample_id", None)) for d in val if getattr(d, "sample_id", None) in member_to_cid}
    test_clusters = {member_to_cid.get(getattr(d, "sample_id", None)) for d in test if getattr(d, "sample_id", None) in member_to_cid}
    
    train_val_cluster_overlap = train_clusters & val_clusters
    train_test_cluster_overlap = train_clusters & test_clusters
    val_test_cluster_overlap = val_clusters & test_clusters
    
    if train_val_cluster_overlap or train_test_cluster_overlap or val_test_cluster_overlap:
        print(f"  ⚠️  警告: 发现聚类级别的泄露！")
        if train_val_cluster_overlap:
            print(f"    训练集-验证集聚类重叠: {len(train_val_cluster_overlap)} 个")
        if train_test_cluster_overlap:
            print(f"    训练集-测试集聚类重叠: {len(train_test_cluster_overlap)} 个")
        if val_test_cluster_overlap:
            print(f"    验证集-测试集聚类重叠: {len(val_test_cluster_overlap)} 个")
    else:
        print(f"  ✅ 未发现聚类级别的泄露")
    
    print(f"\n✅ 完成！")


if __name__ == "__main__":
    main()

