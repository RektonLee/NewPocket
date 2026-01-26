#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在数据集上评估 DLKcat 和 UniKP 的预测效果
需要的列：sequence, substrate_smiles, kcat_value
"""

import os
import sys
import argparse
import subprocess
import tempfile
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime


def pearson_r(y_true, y_pred):
    """计算 Pearson 相关系数"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return np.nan
    r, _ = stats.pearsonr(y_true[mask], y_pred[mask])
    return r


def spearman_r(y_true, y_pred):
    """计算 Spearman 相关系数"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return np.nan
    r, _ = stats.spearmanr(y_true[mask], y_pred[mask])
    return r


def evaluate_predictions(y_true, y_pred, prefix=""):
    """计算评估指标"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]
    
    if len(y_true_valid) < 2:
        return {f"{prefix}n_valid": 0}
    
    metrics = {
        f"{prefix}n_valid": len(y_true_valid),
        f"{prefix}r2": r2_score(y_true_valid, y_pred_valid),
        f"{prefix}pearson_r": pearson_r(y_true_valid, y_pred_valid),
        f"{prefix}spearman_r": spearman_r(y_true_valid, y_pred_valid),
        f"{prefix}rmse": np.sqrt(mean_squared_error(y_true_valid, y_pred_valid)),
        f"{prefix}mae": mean_absolute_error(y_true_valid, y_pred_valid),
    }
    return metrics


# ============== DLKcat 相关函数 ==============

def prepare_dlkcat_input(df, output_file):
    """
    准备 DLKcat 输入文件
    DLKcat 需要 TSV 格式：substrate\tsequence 或者按其要求格式
    """
    # DLKcat 输入格式参考其 example/input.tsv
    # 通常是 substrate_name \t sequence \t substrate_smiles (或类似)
    dlkcat_df = df[['substrate_smiles', 'sequence']].copy()
    dlkcat_df.columns = ['Smiles', 'Sequence']
    dlkcat_df.to_csv(output_file, sep='\t', index=False)
    print(f"✅ DLKcat 输入文件已保存: {output_file}")
    return output_file


def run_dlkcat_prediction(input_file, dlkcat_path, output_file=None):
    """
    运行 DLKcat 预测
    需要安装 DLKcat: https://github.com/SysBioChalmers/DLKcat
    """
    if output_file is None:
        output_file = input_file.replace('.tsv', '_predictions.tsv')
    
    # DLKcat 的运行方式取决于其安装方式
    # 通常是调用 Python 脚本或 MATLAB
    # 这里提供一个示例命令框架
    
    predict_script = os.path.join(dlkcat_path, "DeeplearningApproach", "Code", "prediction.py")
    
    if not os.path.exists(predict_script):
        # 尝试其他可能的路径
        predict_script = os.path.join(dlkcat_path, "prediction.py")
    
    if not os.path.exists(predict_script):
        raise FileNotFoundError(
            f"找不到 DLKcat 预测脚本。请确保 DLKCAT_PATH 设置正确。\n"
            f"尝试的路径: {predict_script}\n"
            f"请参考 DLKcat GitHub: https://github.com/SysBioChalmers/DLKcat"
        )
    
    cmd = [
        sys.executable, predict_script,
        "--input", input_file,
        "--output", output_file
    ]
    
    print(f"🔬 运行 DLKcat 预测...")
    print(f"   命令: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ DLKcat 运行失败: {result.stderr}")
        return None
    
    print(f"✅ DLKcat 预测完成: {output_file}")
    return output_file


def load_dlkcat_predictions(predictions_file):
    """加载 DLKcat 预测结果"""
    df = pd.read_csv(predictions_file, sep='\t')
    # 根据 DLKcat 输出格式调整列名
    if 'predicted_kcat' in df.columns:
        return df['predicted_kcat'].values
    elif 'kcat_pred' in df.columns:
        return df['kcat_pred'].values
    else:
        # 尝试找到预测列
        for col in df.columns:
            if 'pred' in col.lower() or 'kcat' in col.lower():
                return df[col].values
        raise ValueError(f"无法找到 DLKcat 预测列。可用列: {df.columns.tolist()}")


# ============== UniKP 相关函数 ==============

def prepare_unikp_input(df, output_file):
    """
    准备 UniKP 输入文件
    UniKP 需要 sequence 和 SMILES
    """
    unikp_df = df[['sequence', 'substrate_smiles']].copy()
    unikp_df.columns = ['sequence', 'smiles']
    unikp_df.to_csv(output_file, index=False)
    print(f"✅ UniKP 输入文件已保存: {output_file}")
    return output_file


def run_unikp_prediction(input_file, unikp_path, output_file=None, task='kcat'):
    """
    运行 UniKP 预测
    需要安装 UniKP: https://github.com/Luo-SynBioLab/UniKP
    """
    if output_file is None:
        output_file = input_file.replace('.csv', f'_unikp_{task}_predictions.csv')
    
    # UniKP 的运行方式
    predict_script = os.path.join(unikp_path, "predict.py")
    
    if not os.path.exists(predict_script):
        # 尝试其他可能的路径
        predict_script = os.path.join(unikp_path, "scripts", "predict.py")
    
    if not os.path.exists(predict_script):
        raise FileNotFoundError(
            f"找不到 UniKP 预测脚本。请确保 UNIKP_PATH 设置正确。\n"
            f"尝试的路径: {predict_script}\n"
            f"请参考 UniKP GitHub: https://github.com/Luo-SynBioLab/UniKP"
        )
    
    cmd = [
        sys.executable, predict_script,
        "--input", input_file,
        "--output", output_file,
        "--task", task  # 'kcat', 'km', 或 'kcat_km'
    ]
    
    print(f"🔬 运行 UniKP 预测...")
    print(f"   命令: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ UniKP 运行失败: {result.stderr}")
        return None
    
    print(f"✅ UniKP 预测完成: {output_file}")
    return output_file


def load_unikp_predictions(predictions_file):
    """加载 UniKP 预测结果"""
    df = pd.read_csv(predictions_file)
    # 根据 UniKP 输出格式调整列名
    if 'predicted_kcat' in df.columns:
        return df['predicted_kcat'].values
    elif 'kcat_pred' in df.columns:
        return df['kcat_pred'].values
    elif 'prediction' in df.columns:
        return df['prediction'].values
    else:
        # 尝试找到预测列
        for col in df.columns:
            if 'pred' in col.lower() or 'kcat' in col.lower():
                return df[col].values
        raise ValueError(f"无法找到 UniKP 预测列。可用列: {df.columns.tolist()}")


# ============== 主函数 ==============

def main():
    parser = argparse.ArgumentParser(description="评估 DLKcat 和 UniKP 在数据集上的效果")
    parser.add_argument("--dataset", type=str, required=True, help="数据集 CSV 文件路径")
    parser.add_argument("--save_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--dlkcat_path", type=str, default=None, help="DLKcat 安装路径")
    parser.add_argument("--unikp_path", type=str, default=None, help="UniKP 安装路径")
    parser.add_argument("--run_dlkcat", action="store_true", help="运行 DLKcat 预测")
    parser.add_argument("--run_unikp", action="store_true", help="运行 UniKP 预测")
    parser.add_argument("--dlkcat_predictions", type=str, default=None, help="已有的 DLKcat 预测结果文件")
    parser.add_argument("--unikp_predictions", type=str, default=None, help="已有的 UniKP 预测结果文件")
    parser.add_argument("--log_transform", action="store_true", help="对 kcat 值进行 log10 变换")
    
    args = parser.parse_args()
    
    # 创建输出目录
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = f"results/benchmark_dlkcat_unikp_{timestamp}"
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载数据
    print(f"📊 加载数据集: {args.dataset}")
    df = pd.read_csv(args.dataset)
    print(f"   样本数: {len(df)}")
    
    # 检查必要的列
    required_cols = ['sequence', 'substrate_smiles', 'kcat_value']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要的列: {col}")
    
    # 清理数据
    df_clean = df.dropna(subset=required_cols).copy()
    print(f"   清理后样本数: {len(df_clean)}")
    
    # 准备真实值
    if args.log_transform:
        y_true = np.log10(df_clean['kcat_value'].values + 1e-10)
        print(f"   使用 log10(kcat) 作为真实值")
    else:
        y_true = df_clean['kcat_value'].values
        print(f"   使用原始 kcat 值")
    
    results = {"dataset": args.dataset, "n_samples": len(df_clean), "log_transform": args.log_transform}
    
    # ========== DLKcat ==========
    if args.run_dlkcat and args.dlkcat_path:
        print("\n" + "="*60)
        print("运行 DLKcat 预测")
        print("="*60)
        
        dlkcat_input = os.path.join(args.save_dir, "dlkcat_input.tsv")
        prepare_dlkcat_input(df_clean, dlkcat_input)
        
        dlkcat_output = run_dlkcat_prediction(dlkcat_input, args.dlkcat_path)
        if dlkcat_output:
            args.dlkcat_predictions = dlkcat_output
    
    if args.dlkcat_predictions and os.path.exists(args.dlkcat_predictions):
        print(f"\n📊 评估 DLKcat 预测结果: {args.dlkcat_predictions}")
        try:
            y_pred_dlkcat = load_dlkcat_predictions(args.dlkcat_predictions)
            if args.log_transform:
                y_pred_dlkcat = np.log10(y_pred_dlkcat + 1e-10)
            
            dlkcat_metrics = evaluate_predictions(y_true, y_pred_dlkcat, prefix="dlkcat_")
            results.update(dlkcat_metrics)
            
            print(f"\n📈 DLKcat 评估结果:")
            for k, v in dlkcat_metrics.items():
                if isinstance(v, float):
                    print(f"   {k}: {v:.4f}")
                else:
                    print(f"   {k}: {v}")
        except Exception as e:
            print(f"❌ 加载 DLKcat 预测结果失败: {e}")
    
    # ========== UniKP ==========
    if args.run_unikp and args.unikp_path:
        print("\n" + "="*60)
        print("运行 UniKP 预测")
        print("="*60)
        
        unikp_input = os.path.join(args.save_dir, "unikp_input.csv")
        prepare_unikp_input(df_clean, unikp_input)
        
        unikp_output = run_unikp_prediction(unikp_input, args.unikp_path)
        if unikp_output:
            args.unikp_predictions = unikp_output
    
    if args.unikp_predictions and os.path.exists(args.unikp_predictions):
        print(f"\n📊 评估 UniKP 预测结果: {args.unikp_predictions}")
        try:
            y_pred_unikp = load_unikp_predictions(args.unikp_predictions)
            if args.log_transform:
                y_pred_unikp = np.log10(y_pred_unikp + 1e-10)
            
            unikp_metrics = evaluate_predictions(y_true, y_pred_unikp, prefix="unikp_")
            results.update(unikp_metrics)
            
            print(f"\n📈 UniKP 评估结果:")
            for k, v in unikp_metrics.items():
                if isinstance(v, float):
                    print(f"   {k}: {v:.4f}")
                else:
                    print(f"   {k}: {v}")
        except Exception as e:
            print(f"❌ 加载 UniKP 预测结果失败: {e}")
    
    # ========== 保存结果 ==========
    results_df = pd.DataFrame([results])
    results_file = os.path.join(args.save_dir, "benchmark_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\n✅ 评估结果已保存: {results_file}")
    
    # 打印汇总
    print("\n" + "="*60)
    print("📊 评估汇总")
    print("="*60)
    print(f"数据集: {args.dataset}")
    print(f"样本数: {len(df_clean)}")
    print(f"Log变换: {args.log_transform}")
    
    if 'dlkcat_r2' in results:
        print(f"\nDLKcat:")
        print(f"  R²: {results['dlkcat_r2']:.4f}")
        print(f"  Pearson r: {results['dlkcat_pearson_r']:.4f}")
        print(f"  Spearman r: {results['dlkcat_spearman_r']:.4f}")
        print(f"  RMSE: {results['dlkcat_rmse']:.4f}")
        print(f"  MAE: {results['dlkcat_mae']:.4f}")
    
    if 'unikp_r2' in results:
        print(f"\nUniKP:")
        print(f"  R²: {results['unikp_r2']:.4f}")
        print(f"  Pearson r: {results['unikp_pearson_r']:.4f}")
        print(f"  Spearman r: {results['unikp_spearman_r']:.4f}")
        print(f"  RMSE: {results['unikp_rmse']:.4f}")
        print(f"  MAE: {results['unikp_mae']:.4f}")
    
    print("\n✅ 评估完成！")
    return results


if __name__ == "__main__":
    main()







