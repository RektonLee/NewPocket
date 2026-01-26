#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行 UniKP 预测
基于 UniKP: https://github.com/Luo-SynBioLab/UniKP

UniKP 运行逻辑:
1. 底物处理: SMILES → SMILES Transformer → [N, 256] 向量
2. 蛋白质处理: 序列 → ProtT5 → [N, 1024] 向量
3. 合并特征: concat → [N, 1280]
4. 预测: ExtraTreesRegressor → log10(kcat)
"""

import os
import sys
import subprocess
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_predict, KFold
from datetime import datetime
from tqdm import tqdm
import pickle
import gc
import re
import json
import tempfile

# 路径设置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
UNIKP_PATH = os.path.join(PROJECT_ROOT, "benchmark_tools", "UniKP")


def evaluate_predictions(y_true, y_pred, prefix=""):
    """计算评估指标"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]
    
    if len(y_true_valid) < 2:
        return {f"{prefix}n_valid": 0}
    
    pearson_r, _ = stats.pearsonr(y_true_valid, y_pred_valid)
    spearman_r, _ = stats.spearmanr(y_true_valid, y_pred_valid)
    
    metrics = {
        f"{prefix}n_valid": len(y_true_valid),
        f"{prefix}r2": r2_score(y_true_valid, y_pred_valid),
        f"{prefix}pearson_r": pearson_r,
        f"{prefix}spearman_r": spearman_r,
        f"{prefix}rmse": np.sqrt(mean_squared_error(y_true_valid, y_pred_valid)),
        f"{prefix}mae": mean_absolute_error(y_true_valid, y_pred_valid),
    }
    return metrics


# ============== 内联特征提取脚本 ==============
FEATURE_EXTRACTION_SCRIPT = '''
import os
import sys
import json
import pickle
import numpy as np
import torch
import re
import gc

# 设置路径
unikp_path = sys.argv[1]
input_file = sys.argv[2]
output_file = sys.argv[3]
device_str = sys.argv[4]

os.chdir(unikp_path)
sys.path.insert(0, unikp_path)

from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from utils import split
from transformers import T5EncoderModel, T5Tokenizer

# 加载输入数据
with open(input_file, 'r') as f:
    data = json.load(f)
sequences = data['sequences']
smiles_list = data['smiles']

device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}", file=sys.stderr)

# ========== SMILES 特征 ==========
print("提取 SMILES 特征...", file=sys.stderr)
vocab = WordVocab.load_vocab('vocab.pkl')
trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
trfm.load_state_dict(torch.load('trfm_12_23000.pkl', map_location='cpu'))
trfm.eval()

pad_index = 0
unk_index = 1
eos_index = 2
sos_index = 3

def get_inputs(sm):
    seq_len = 220
    sm = sm.split()
    if len(sm) > 218:
        sm = sm[:109] + sm[-109:]
    ids = [vocab.stoi.get(token, unk_index) for token in sm]
    ids = [sos_index] + ids + [eos_index]
    seg = [1] * len(ids)
    padding = [pad_index] * (seq_len - len(ids))
    ids.extend(padding)
    seg.extend(padding)
    return ids, seg

x_split = [split(sm) for sm in smiles_list]
x_id, x_seg = [], []
for sm in x_split:
    a, b = get_inputs(sm)
    x_id.append(a)
    x_seg.append(b)

xid = torch.tensor(x_id)
smiles_features = trfm.encode(torch.t(xid))  # 返回的已经是 numpy 数组
print(f"SMILES 特征形状: {smiles_features.shape}", file=sys.stderr)

# ========== 蛋白质特征 ==========
print("提取蛋白质特征...", file=sys.stderr)
# 设置 HuggingFace 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
print("使用 HuggingFace 镜像: https://hf-mirror.com", file=sys.stderr)
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
prot_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
prot_model = prot_model.to(device)
prot_model.eval()

# 处理序列
processed_seqs = []
for seq in sequences:
    if len(seq) > 1000:
        seq = seq[:500] + seq[-500:]
    seq_spaced = ' '.join(list(seq))
    seq_spaced = re.sub(r"[UZOB]", "X", seq_spaced)
    processed_seqs.append(seq_spaced)

seq_features = []
batch_size = 4

for i in range(0, len(processed_seqs), batch_size):
    if i % 20 == 0:
        print(f"处理序列 {i+1}/{len(processed_seqs)}...", file=sys.stderr)
    batch = processed_seqs[i:i+batch_size]
    
    ids = tokenizer.batch_encode_plus(
        batch, 
        add_special_tokens=True, 
        padding='longest',
        return_tensors='pt'
    )
    
    input_ids = ids['input_ids'].to(device)
    attention_mask = ids['attention_mask'].to(device)
    
    with torch.no_grad():
        embedding = prot_model(input_ids=input_ids, attention_mask=attention_mask)
    
    for j in range(len(batch)):
        seq_len = attention_mask[j].sum().item()
        feat = embedding.last_hidden_state[j, :seq_len-1, :].mean(dim=0).cpu().numpy()
        seq_features.append(feat)
    
    gc.collect()
    torch.cuda.empty_cache()

seq_features = np.array(seq_features)
print(f"序列特征形状: {seq_features.shape}", file=sys.stderr)

# 合并特征
features = np.concatenate([smiles_features, seq_features], axis=1)
print(f"合并特征形状: {features.shape}", file=sys.stderr)

# 保存结果
with open(output_file, 'wb') as f:
    pickle.dump(features, f)

print("特征提取完成!", file=sys.stderr)
'''


class UniKPPredictor:
    """UniKP 预测器封装"""
    
    def __init__(self, device='cuda:1'):
        self.device = device
        self.regressor = None
        print(f"🔧 UniKP 将使用设备: {device}")
    
    def extract_features(self, sequences, smiles_list):
        """提取特征（使用子进程在 UniKP 目录下运行）"""
        print("\n📊 提取特征（使用子进程）...")
        
        # 设置 HuggingFace 镜像
        env = os.environ.copy()
        env['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print("   使用 HuggingFace 镜像: https://hf-mirror.com")
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'sequences': sequences, 'smiles': smiles_list}, f)
            input_file = f.name
        
        output_file = input_file.replace('.json', '_features.pkl')
        script_file = input_file.replace('.json', '_extract.py')
        
        # 写入特征提取脚本
        with open(script_file, 'w') as f:
            f.write(FEATURE_EXTRACTION_SCRIPT)
        
        try:
            # 运行特征提取
            cmd = [
                sys.executable, script_file,
                UNIKP_PATH, input_file, output_file, self.device
            ]
            
            print(f"   运行命令: {' '.join(cmd[:3])} ...")
            result = subprocess.run(cmd, capture_output=False, text=True, env=env)
            
            if result.returncode != 0:
                raise RuntimeError(f"特征提取失败")
            
            # 加载特征
            with open(output_file, 'rb') as f:
                features = pickle.load(f)
            
            print(f"✅ 特征提取完成! 形状: {features.shape}")
            return features
            
        finally:
            # 清理临时文件
            for f in [input_file, output_file, script_file]:
                if os.path.exists(f):
                    os.remove(f)
    
    def train(self, features, labels):
        """训练回归模型"""
        print("\n🔧 训练 ExtraTreesRegressor...")
        self.regressor = ExtraTreesRegressor(n_estimators=100, n_jobs=-1, random_state=42)
        self.regressor.fit(features, labels)
        print("✅ 训练完成!")
    
    def predict(self, features):
        """预测"""
        if self.regressor is None:
            raise ValueError("模型未训练，请先调用 train() 方法")
        return self.regressor.predict(features)
    
    def cross_validate(self, features, labels, n_splits=5):
        """交叉验证预测"""
        print(f"\n🔄 进行 {n_splits} 折交叉验证...")
        print(f"   特征形状: {features.shape}, 标签形状: {labels.shape}")
        regressor = ExtraTreesRegressor(n_estimators=100, n_jobs=-1, random_state=42)
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # 手动进行交叉验证以便显示进度
        predictions = np.zeros_like(labels)
        for fold, (train_idx, val_idx) in enumerate(kfold.split(features), 1):
            print(f"   训练第 {fold}/{n_splits} 折...")
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            regressor.fit(X_train, y_train)
            predictions[val_idx] = regressor.predict(X_val)
            print(f"   第 {fold} 折完成")
        
        print("✅ 交叉验证完成!")
        return predictions


def main():
    parser = argparse.ArgumentParser(description="运行 UniKP 预测")
    parser.add_argument("--dataset", type=str, required=True, help="数据集 CSV 文件路径")
    parser.add_argument("--save_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--device", type=str, default="cuda:1", help="计算设备")
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数（用于测试）")
    parser.add_argument("--cv_folds", type=int, default=5, help="交叉验证折数")
    parser.add_argument("--smiles_col", type=str, default="substrate_smiles", help="SMILES列名")
    parser.add_argument("--seq_col", type=str, default="sequence", help="序列列名")
    parser.add_argument("--kcat_col", type=str, default="kcat_value", help="kcat列名")
    
    args = parser.parse_args()
    
    # 创建输出目录
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = os.path.join(PROJECT_ROOT, "results", f"unikp_benchmark_{timestamp}")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载数据
    print(f"📊 加载数据集: {args.dataset}")
    df = pd.read_csv(args.dataset)
    print(f"   样本数: {len(df)}")
    
    # 检查必要的列
    required_cols = [args.seq_col, args.smiles_col, args.kcat_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要的列: {col}")
    
    # 清理数据
    df_clean = df.dropna(subset=required_cols).copy()
    
    # 转换 kcat_value 为数值类型
    df_clean[args.kcat_col] = pd.to_numeric(df_clean[args.kcat_col], errors='coerce')
    df_clean = df_clean.dropna(subset=[args.kcat_col])
    df_clean = df_clean[df_clean[args.kcat_col] > 0]
    
    # 过滤包含 . 的 SMILES (多分子混合物)
    df_clean = df_clean[~df_clean[args.smiles_col].str.contains(r'\.', regex=True, na=False)]
    
    print(f"   清理后样本数: {len(df_clean)}")
    
    # 限制样本数（用于测试）
    if args.max_samples:
        df_clean = df_clean.head(args.max_samples)
        print(f"   使用样本数: {len(df_clean)}")
    
    # 准备数据
    sequences = df_clean[args.seq_col].tolist()
    smiles_list = df_clean[args.smiles_col].tolist()
    y_true = np.log10(df_clean[args.kcat_col].values + 1e-10)
    
    # 初始化预测器
    print(f"\n🔬 初始化 UniKP...")
    predictor = UniKPPredictor(device=args.device)
    
    # 提取特征
    features = predictor.extract_features(sequences, smiles_list)
    
    # 交叉验证预测
    y_pred = predictor.cross_validate(features, y_true, n_splits=args.cv_folds)
    
    # 评估
    print(f"\n📈 评估结果:")
    metrics = evaluate_predictions(y_true, y_pred, prefix="unikp_")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")
    
    # 保存结果
    results_df = pd.DataFrame({
        'sequence': sequences,
        'smiles': smiles_list,
        'kcat_true': df_clean[args.kcat_col].values,
        'log_kcat_true': y_true,
        'log_kcat_pred': y_pred,
    })
    results_file = os.path.join(args.save_dir, "unikp_predictions.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\n✅ 预测结果已保存: {results_file}")
    
    # 保存特征
    features_file = os.path.join(args.save_dir, "unikp_features.pkl")
    with open(features_file, 'wb') as f:
        pickle.dump(features, f)
    print(f"✅ 特征已保存: {features_file}")
    
    # 保存评估指标
    metrics_df = pd.DataFrame([metrics])
    metrics_file = os.path.join(args.save_dir, "unikp_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print(f"✅ 评估指标已保存: {metrics_file}")
    
    return metrics


if __name__ == "__main__":
    main()
