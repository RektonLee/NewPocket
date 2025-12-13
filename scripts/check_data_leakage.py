#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查训练集和测试集之间的序列相似度，检测数据泄露问题。

使用方法:
    # 多阈值检查模式（推荐）
    python scripts/check_data_leakage.py \
        --train_pt data/processed/kcat_full_1213.pt \
        --test_pt data/processed/kcat_test_new.pt \
        --train_csv data/raw/successful_full_train_1213.csv \
        --test_csv data/raw/kcat_test_results.csv \
        --output_dir results/leakage_check \
        --multi_threshold
    
    # 单阈值聚类模式（原有方式）
    python scripts/check_data_leakage.py \
        --train_pt data/processed/kcat_full_1213.pt \
        --test_pt data/processed/kcat_test_new.pt \
        --train_csv data/raw/successful_full_train_1213.csv \
        --test_csv data/raw/kcat_test_results.csv \
        --output_dir results/leakage_check \
        --identity_threshold 0.4
"""

import argparse
import os
import sys
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch


def load_sample_ids_from_pt(pt_path):
    """从 .pt 文件中提取所有 sample_id"""
    print(f"加载数据集: {pt_path}")
    # 使用 weights_only=False 以支持 PyG Data 对象
    dataset = torch.load(pt_path, weights_only=False)
    
    sample_ids = set()
    for data in dataset:
        sid = getattr(data, "sample_id", None)
        if sid is None:
            print(f"警告: 发现没有 sample_id 的数据条目，跳过")
            continue
        sample_ids.add(sid)
    
    print(f"  提取到 {len(sample_ids)} 个唯一的 sample_id")
    return sample_ids


def create_fasta_from_sample_ids(sample_ids, csv_path, output_fasta, label="unknown"):
    """根据 sample_id 从 CSV 中提取序列，生成 FASTA 文件"""
    print(f"\n从 CSV 提取序列 ({label})...")
    df = pd.read_csv(csv_path)
    
    # 检查必要的列
    if "sample_id" not in df.columns:
        raise ValueError(f"CSV 文件缺少 'sample_id' 列")
    if "sequence" not in df.columns:
        raise ValueError(f"CSV 文件缺少 'sequence' 列")
    
    # 创建 sample_id -> sequence 的映射
    id_to_seq = dict(zip(df["sample_id"], df["sequence"]))
    
    # 提取序列
    sequences = []
    missing_ids = []
    for sid in sample_ids:
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
            print(f"  缺失的 sample_id: {missing_ids}")
    
    # 写入 FASTA
    with open(output_fasta, "w") as f:
        for sid, seq in sequences:
            f.write(f">{sid}\n{seq}\n")
    
    print(f"  FASTA 文件已保存: {output_fasta}")
    return len(sequences), missing_ids


def run_mmseqs_search(query_fasta, target_fasta, output_dir, identity_threshold=0.4, coverage=0.8):
    """使用 MMseqs2 搜索 query 序列在 target 中的相似序列"""
    print(f"\n运行 MMseqs2 搜索 (identity >= {identity_threshold}, coverage >= {coverage})...")
    
    # 检查 MMseqs2 是否安装
    try:
        result = subprocess.run(["mmseqs", "easy-search"], capture_output=True, text=True, timeout=5)
        if "easy-search" not in result.stdout and "easy-search" not in result.stderr:
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
    
    result_file = os.path.join(output_dir, f"search_results_{int(identity_threshold*100)}.tsv")
    
    # 运行搜索
    cmd = [
        "mmseqs", "easy-search",
        query_fasta,
        target_fasta,
        result_file,
        tmp_dir,
        "--min-seq-id", str(identity_threshold),
        "-c", str(coverage),
        "--threads", "4"
    ]
    
    print(f"  执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  MMseqs2 错误输出:\n{result.stderr}")
        raise RuntimeError(f"MMseqs2 搜索失败: {result.stderr}")
    
    print(f"  搜索完成，结果文件: {result_file}")
    return result_file


def run_mmseqs_cluster(fasta_path, output_dir, identity_threshold=0.4, coverage=0.8):
    """使用 MMseqs2 对序列进行聚类"""
    print(f"\n运行 MMseqs2 聚类 (identity >= {identity_threshold}, coverage >= {coverage})...")
    
    # 检查 MMseqs2 是否安装
    try:
        # 尝试直接运行 mmseqs
        result = subprocess.run(["mmseqs", "easy-cluster"], capture_output=True, text=True, timeout=5)
        # 如果返回非0但输出了帮助信息，说明命令存在
        if "easy-cluster" in result.stdout or "easy-cluster" in result.stderr:
            pass  # 命令存在
        else:
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
        # 尝试其他可能的文件名
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


def parse_cluster_tsv(cluster_tsv):
    """解析 MMseqs2 聚类结果，返回 member -> cluster_rep 的映射"""
    print(f"\n解析聚类结果...")
    member_to_rep = {}
    rep_to_members = defaultdict(set)
    
    with open(cluster_tsv, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                rep = parts[0]
                member = parts[1]
                member_to_rep[member] = rep
                rep_to_members[rep].add(member)
    
    print(f"  发现 {len(rep_to_members)} 个聚类")
    print(f"  总序列数: {len(member_to_rep)}")
    
    return member_to_rep, rep_to_members


def parse_search_results(search_tsv):
    """解析 MMseqs2 搜索结果，返回 query_id -> [target_ids] 的映射"""
    query_to_targets = defaultdict(set)
    
    if not os.path.exists(search_tsv):
        return query_to_targets
    
    with open(search_tsv, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                query_id = parts[0]
                target_id = parts[1]
                query_to_targets[query_id].add(target_id)
    
    return query_to_targets


def check_leakage_by_thresholds(train_sample_ids, test_sample_ids, train_csv, test_csv, output_dir, thresholds=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9], coverage=0.8):
    """
    使用多个阈值检查数据泄露，返回每个阈值下的统计信息
    """
    print(f"\n{'='*80}")
    print(f"多阈值数据泄露检查")
    print(f"{'='*80}")
    print(f"测试阈值: {[f'{t*100:.0f}%' for t in thresholds]}")
    print()
    
    # 生成训练集和测试集的 FASTA
    train_fasta = os.path.join(output_dir, "train_sequences.fasta")
    test_fasta = os.path.join(output_dir, "test_sequences.fasta")
    
    train_count, train_missing = create_fasta_from_sample_ids(
        train_sample_ids, train_csv, train_fasta, "训练集"
    )
    test_count, test_missing = create_fasta_from_sample_ids(
        test_sample_ids, test_csv, test_fasta, "测试集"
    )
    
    # 为每个阈值进行搜索
    threshold_results = []
    
    for threshold in thresholds:
        print(f"\n处理阈值 {threshold*100:.0f}%...")
        try:
            # 搜索测试集序列在训练集中的相似序列
            search_result = run_mmseqs_search(
                test_fasta, train_fasta, output_dir,
                identity_threshold=threshold, coverage=coverage
            )
            
            # 解析搜索结果
            query_to_targets = parse_search_results(search_result)
            
            # 统计泄露情况
            test_with_hits = set(query_to_targets.keys())  # 在训练集中找到相似序列的测试序列
            test_without_hits = test_sample_ids - test_with_hits  # 在训练集中没有相似序列的测试序列
            
            # 计算比率
            total_test = len(test_sample_ids)
            leaked_count = len(test_with_hits)
            safe_count = len(test_without_hits)
            leak_rate = leaked_count / total_test if total_test > 0 else 0
            safe_rate = safe_count / total_test if total_test > 0 else 0
            
            threshold_results.append({
                "threshold": threshold,
                "threshold_percent": f"{threshold*100:.0f}%",
                "total_test": total_test,
                "leaked_count": leaked_count,
                "safe_count": safe_count,
                "leak_rate": leak_rate,
                "safe_rate": safe_rate,
                "test_with_hits": test_with_hits,
                "test_without_hits": test_without_hits
            })
            
            print(f"  阈值 {threshold*100:.0f}%: 泄露 {leaked_count}/{total_test} ({leak_rate*100:.2f}%), "
                  f"安全 {safe_count}/{total_test} ({safe_rate*100:.2f}%)")
            
        except Exception as e:
            print(f"  ⚠️  阈值 {threshold*100:.0f}% 处理失败: {e}")
            threshold_results.append({
                "threshold": threshold,
                "threshold_percent": f"{threshold*100:.0f}%",
                "error": str(e)
            })
    
    # 生成可视化报告
    print(f"\n{'='*80}")
    print(f"数据泄露检查结果汇总")
    print(f"{'='*80}")
    print(f"\n训练集样本数: {len(train_sample_ids)}")
    print(f"测试集样本数: {len(test_sample_ids)}")
    print(f"\n各阈值下的泄露情况:")
    print(f"{'阈值':<10} {'泄露数':<10} {'安全数':<10} {'泄露率':<12} {'安全率':<12}")
    print(f"{'-'*60}")
    
    for result in threshold_results:
        if "error" not in result:
            print(f"{result['threshold_percent']:<10} "
                  f"{result['leaked_count']:<10} "
                  f"{result['safe_count']:<10} "
                  f"{result['leak_rate']*100:>10.2f}%  "
                  f"{result['safe_rate']*100:>10.2f}%")
        else:
            print(f"{result['threshold_percent']:<10} 错误: {result['error']}")
    
    print(f"{'='*80}\n")
    
    return threshold_results


def check_leakage(train_sample_ids, test_sample_ids, member_to_rep, rep_to_members):
    """检查训练集和测试集之间是否存在数据泄露"""
    print(f"\n检查数据泄露...")
    
    # 创建 sample_id -> cluster_rep 的映射
    train_rep_to_samples = defaultdict(set)
    test_rep_to_samples = defaultdict(set)
    
    train_in_clusters = 0
    test_in_clusters = 0
    
    for sid in train_sample_ids:
        if sid in member_to_rep:
            rep = member_to_rep[sid]
            train_rep_to_samples[rep].add(sid)
            train_in_clusters += 1
    
    for sid in test_sample_ids:
        if sid in member_to_rep:
            rep = member_to_rep[sid]
            test_rep_to_samples[rep].add(sid)
            test_in_clusters += 1
    
    # 找出同时包含 train 和 test 的 cluster
    leaking_clusters = []
    for rep in rep_to_members:
        has_train = rep in train_rep_to_samples
        has_test = rep in test_rep_to_samples
        if has_train and has_test:
            leaking_clusters.append({
                "cluster_rep": rep,
                "train_samples": train_rep_to_samples[rep],
                "test_samples": test_rep_to_samples[rep],
                "total_members": rep_to_members[rep]
            })
    
    # 统计泄露的样本
    leaking_train_samples = set()
    leaking_test_samples = set()
    for leak in leaking_clusters:
        leaking_train_samples.update(leak["train_samples"])
        leaking_test_samples.update(leak["test_samples"])
    
    # 输出结果
    print(f"\n{'='*60}")
    print(f"数据泄露检查结果")
    print(f"{'='*60}")
    print(f"训练集样本数: {len(train_sample_ids)}")
    print(f"测试集样本数: {len(test_sample_ids)}")
    print(f"训练集在聚类中的样本数: {train_in_clusters}")
    print(f"测试集在聚类中的样本数: {test_in_clusters}")
    print(f"\n泄露的聚类数量: {len(leaking_clusters)}")
    print(f"泄露的训练集样本数: {len(leaking_train_samples)}")
    print(f"泄露的测试集样本数: {len(leaking_test_samples)}")
    
    if len(leaking_clusters) > 0:
        print(f"\n⚠️  警告: 发现数据泄露！")
        print(f"   有 {len(leaking_clusters)} 个聚类同时包含训练集和测试集的序列")
        print(f"   泄露比例: 训练集 {len(leaking_train_samples)/len(train_sample_ids)*100:.2f}%, "
              f"测试集 {len(leaking_test_samples)/len(test_sample_ids)*100:.2f}%")
        
        # 显示前10个泄露的聚类详情
        print(f"\n前10个泄露的聚类详情:")
        for i, leak in enumerate(leaking_clusters[:10], 1):
            print(f"  聚类 {i} (代表序列: {leak['cluster_rep']}):")
            print(f"    训练集样本: {len(leak['train_samples'])} 个")
            print(f"    测试集样本: {len(leak['test_samples'])} 个")
            print(f"    总成员数: {len(leak['total_members'])} 个")
            if len(leak['train_samples']) <= 5 and len(leak['test_samples']) <= 5:
                print(f"    训练集: {list(leak['train_samples'])}")
                print(f"    测试集: {list(leak['test_samples'])}")
    else:
        print(f"\n✅ 未发现数据泄露")
        print(f"   所有聚类都只包含训练集或只包含测试集的序列")
    
    print(f"{'='*60}\n")
    
    return {
        "leaking_clusters": len(leaking_clusters),
        "leaking_train_samples": len(leaking_train_samples),
        "leaking_test_samples": len(leaking_test_samples),
        "leakage_rate_train": len(leaking_train_samples) / len(train_sample_ids) if train_sample_ids else 0,
        "leakage_rate_test": len(leaking_test_samples) / len(test_sample_ids) if test_sample_ids else 0,
        "leaking_clusters_detail": leaking_clusters
    }


def main():
    parser = argparse.ArgumentParser(
        description="检查训练集和测试集之间的序列相似度，检测数据泄露"
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
        "--output_dir", default="results/leakage_check",
        help="输出目录（默认: results/leakage_check）"
    )
    parser.add_argument(
        "--identity_threshold", type=float, default=0.4,
        help="序列相似度阈值（默认: 0.4，即40%% identity）。如果使用 --multi_threshold，此参数将被忽略"
    )
    parser.add_argument(
        "--multi_threshold", action="store_true",
        help="启用多阈值检查模式，将在 40%%-90%% 的多个阈值下检查数据泄露"
    )
    parser.add_argument(
        "--thresholds", type=str, default="0.4,0.5,0.6,0.7,0.8,0.9",
        help="自定义阈值列表（逗号分隔，例如: 0.4,0.5,0.6）。仅在 --multi_threshold 模式下有效"
    )
    parser.add_argument(
        "--coverage", type=float, default=0.8,
        help="覆盖度阈值（默认: 0.8）"
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
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 从 .pt 文件提取 sample_id
    train_sample_ids = load_sample_ids_from_pt(args.train_pt)
    test_sample_ids = load_sample_ids_from_pt(args.test_pt)
    
    # 检查是否有重叠的 sample_id（直接泄露）
    direct_overlap = train_sample_ids & test_sample_ids
    if direct_overlap:
        print(f"\n⚠️  警告: 发现 {len(direct_overlap)} 个直接重复的 sample_id！")
        print(f"   这是最严重的数据泄露形式。")
        if len(direct_overlap) <= 20:
            print(f"   重复的 sample_id: {sorted(direct_overlap)}")
    else:
        print(f"\n✅ 未发现直接重复的 sample_id")
    
    # 2. 检查模式：多阈值模式 vs 单阈值聚类模式
    if args.multi_threshold:
        # 多阈值检查模式
        thresholds = [float(t.strip()) for t in args.thresholds.split(",")]
        threshold_results = check_leakage_by_thresholds(
            train_sample_ids, test_sample_ids,
            args.train_csv, args.test_csv,
            args.output_dir,
            thresholds=thresholds,
            coverage=args.coverage
        )
        
        # 保存结果
        import json
        result_file = os.path.join(args.output_dir, "multi_threshold_leakage_report.json")
        with open(result_file, "w") as f:
            # 转换 set 为 list 以便 JSON 序列化
            serializable_results = []
            for r in threshold_results:
                sr = r.copy()
                if "test_with_hits" in sr:
                    sr["test_with_hits"] = list(sr["test_with_hits"])
                if "test_without_hits" in sr:
                    sr["test_without_hits"] = list(sr["test_without_hits"])
                serializable_results.append(sr)
            
            json.dump({
                "train_samples": len(train_sample_ids),
                "test_samples": len(test_sample_ids),
                "threshold_results": serializable_results
            }, f, indent=2, default=str)
        
        print(f"\n结果已保存: {result_file}")
        
        # 返回状态码
        max_leak_rate = max([r.get("leak_rate", 0) for r in threshold_results if "error" not in r], default=0)
        if max_leak_rate > 0:
            print(f"\n⚠️  检测到数据泄露（最大泄露率: {max_leak_rate*100:.2f}%）")
            return 1
        else:
            print(f"\n✅ 未检测到数据泄露")
            return 0
    
    else:
        # 单阈值聚类模式（原有逻辑）
        train_fasta = os.path.join(args.output_dir, "train_sequences.fasta")
        test_fasta = os.path.join(args.output_dir, "test_sequences.fasta")
        combined_fasta = os.path.join(args.output_dir, "all_sequences.fasta")
        
        train_count, train_missing = create_fasta_from_sample_ids(
            train_sample_ids, args.train_csv, train_fasta, "训练集"
        )
        test_count, test_missing = create_fasta_from_sample_ids(
            test_sample_ids, args.test_csv, test_fasta, "测试集"
        )
        
        # 合并所有序列用于聚类
        print(f"\n合并序列用于聚类...")
        with open(combined_fasta, "w") as out:
            with open(train_fasta, "r") as f:
                out.write(f.read())
            with open(test_fasta, "r") as f:
                out.write(f.read())
        print(f"  合并后的 FASTA: {combined_fasta} ({train_count + test_count} 个序列)")
        
        # 3. 运行 MMseqs2 聚类（如果可用）
        cluster_tsv = None
        member_to_rep = None
        rep_to_members = None
        
        if args.skip_clustering:
            if not args.cluster_tsv or not os.path.exists(args.cluster_tsv):
                raise ValueError("使用 --skip_clustering 时必须提供有效的 --cluster_tsv")
            cluster_tsv = args.cluster_tsv
            print(f"\n跳过聚类，使用已有结果: {cluster_tsv}")
            member_to_rep, rep_to_members = parse_cluster_tsv(cluster_tsv)
        else:
            try:
                cluster_tsv = run_mmseqs_cluster(
                    combined_fasta, args.output_dir,
                    args.identity_threshold, args.coverage
                )
                # 4. 解析聚类结果
                member_to_rep, rep_to_members = parse_cluster_tsv(cluster_tsv)
            except RuntimeError as e:
                print(f"\n⚠️  {e}")
                print(f"\n由于 MMseqs2 不可用，将只检查直接重复的 sample_id。")
                print(f"要检查序列相似度泄露，请先安装 MMseqs2:")
                print(f"  conda install -c conda-forge mmseqs2")
                print(f"然后重新运行此脚本。")
                member_to_rep = {}
                rep_to_members = {}
        
        # 5. 检查泄露
        if member_to_rep:
            leakage_result = check_leakage(
                train_sample_ids, test_sample_ids, member_to_rep, rep_to_members
            )
        else:
            # 如果没有聚类结果，只报告直接重复
            leakage_result = {
                "leaking_clusters": 0,
                "leaking_train_samples": 0,
                "leaking_test_samples": 0,
                "leakage_rate_train": 0,
                "leakage_rate_test": 0,
                "leaking_clusters_detail": [],
                "note": "序列相似度检查未执行（需要 MMseqs2）"
            }
    
        # 6. 保存结果（单阈值模式）
        import json
        result_file = os.path.join(args.output_dir, "leakage_report.json")
        with open(result_file, "w") as f:
            json.dump({
                "train_samples": len(train_sample_ids),
                "test_samples": len(test_sample_ids),
                "train_in_clusters": train_count - len(train_missing),
                "test_in_clusters": test_count - len(test_missing),
                "identity_threshold": args.identity_threshold,
                "coverage": args.coverage,
                "direct_overlap_count": len(direct_overlap),
                "leakage": leakage_result
            }, f, indent=2, default=str)
        
        print(f"\n结果已保存: {result_file}")
        
        # 返回状态码
        if leakage_result["leaking_clusters"] > 0 or len(direct_overlap) > 0:
            print(f"\n⚠️  检测到数据泄露，建议重新划分数据集！")
            return 1
        else:
            print(f"\n✅ 未检测到数据泄露")
            return 0


if __name__ == "__main__":
    sys.exit(main())

