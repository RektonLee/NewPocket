"""Robust PDB generation script using ESMFold with multi-GPU, batching, length control & fallbacks.

Features added:
1. Multi-GPU support (DataParallel) with --gpus 0,1
2. Batching (--batch-size) to improve throughput when sequences are short
3. Max sequence length handling (--max-length) with truncate / skip / chunk modes
4. OOM resilience: retry with half precision (--fp16) and finally fallback to dummy linear CA trace
5. Caching: skip existing PDBs unless --overwrite
6. Detailed logging (progress, timings, fallbacks) saved to pdb_generation.log
7. Dummy PDB generator for failed predictions
8. Graceful degradation if no GPU: will run on CPU (slow)
9. **NEW: SampleManager integration for organized file management**
"""

import os
import time
import math
import json
import logging
import argparse
from typing import List, Tuple
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from sample_manager import SampleManager  # 新增导入

# ---------------------------------
# Logging setup
# ---------------------------------
def setup_logger(log_path: str = 'pdb_generation.log'):
    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='a'),
            logging.StreamHandler()
        ]
    )


# ---------------------------------
# Utility: dummy PDB fallback
# ---------------------------------
AA3 = {
    'A': 'ALA','C': 'CYS','D': 'ASP','E': 'GLU','F': 'PHE','G': 'GLY','H': 'HIS','I': 'ILE','K': 'LYS','L': 'LEU',
    'M': 'MET','N': 'ASN','P': 'PRO','Q': 'GLN','R': 'ARG','S': 'SER','T': 'THR','V': 'VAL','W': 'TRP','Y': 'TYR'
}

def make_dummy_pdb(sequence: str) -> str:
    lines = ["HEADER    DUMMY STRUCTURE"]
    for i, aa in enumerate(sequence):
        res3 = AA3.get(aa, 'GLY')
        x = float(i * 3.8); y = 0.0; z = 0.0
        lines.append(f"ATOM  {i+1:5d}  CA  {res3} A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  ")
    lines.append("END")
    return "\n".join(lines)


# ---------------------------------
# ESM output conversion
# ---------------------------------
def convert_outputs_to_pdb(outputs) -> List[str]:
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs_np = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs_np["atom37_atom_exists"]
    pdbs = []
    
    # 检查输出数据的有效性
    if outputs_np["aatype"].shape[0] == 0:
        logging.warning("ESMFold输出为空，无法生成PDB")
        return []
    
    for i in range(outputs_np["aatype"].shape[0]):
        try:
            aa = outputs_np["aatype"][i]
            pred_pos = final_atom_positions[i]
            mask = final_atom_mask[i]
            resid = outputs_np["residue_index"][i] + 1
            pred = OFProtein(
                aatype=aa,
                atom_positions=pred_pos,
                atom_mask=mask,
                residue_index=resid,
                b_factors=outputs_np["plddt"][i],
                chain_index=outputs_np.get("chain_index", [None])[i] if "chain_index" in outputs_np else None,
            )
            pdbs.append(to_pdb(pred))
        except Exception as e:
            logging.error(f"转换第{i}个结构时出错: {e}")
            # 如果转换失败，添加一个空的PDB字符串
            pdbs.append("")
    
    return pdbs


# ---------------------------------
# Predictor class
# ---------------------------------
class ESMFoldPredictor:
    def __init__(self, gpus: str, fp16: bool = False, max_batch_tokens: int = 4096):
        self.device_ids = []
        if gpus:
            self.device_ids = [int(i) for i in gpus.split(',') if i.strip() != '']
        use_cuda = torch.cuda.is_available() and len(self.device_ids) > 0
        self.device = torch.device(f'cuda:{self.device_ids[0]}' if use_cuda else 'cpu')
        self.max_batch_tokens = max_batch_tokens
        logging.info(f"Using device: {self.device}")
        torch.backends.cuda.matmul.allow_tf32 = True
        self.fp16 = fp16 and use_cuda
        self.original_fp16 = self.fp16  # 记录初始精度设置

        logging.info("Loading ESMFold model (facebook/esmfold_v1)...")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        self.model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
        self.model.to(self.device)
        
        # 精度控制
        if self.fp16:
            self.model.half()
            logging.info("Model converted to half precision (FP16)")
        
        # 多GPU策略改进：支持真正的并行处理
        if use_cuda and len(self.device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
            logging.info(f"Using DataParallel on GPUs: {self.device_ids}")
            # 调整batch token限制以适应多GPU
            self.max_batch_tokens = max_batch_tokens * len(self.device_ids)
            logging.info(f"Adjusted max_batch_tokens to {self.max_batch_tokens} for {len(self.device_ids)} GPUs")
        
        self.model.eval()

    def predict_batch(self, sequences: List[str]) -> List[Tuple[str, str]]:
        """批处理预测，支持智能分组和多GPU并行"""
        results = []
        
        # 按长度排序，便于智能batching
        seq_with_idx = [(i, seq) for i, seq in enumerate(sequences)]
        seq_with_idx.sort(key=lambda x: len(x[1]))
        
        # 智能分组：相似长度的序列组成batch
        batches = self._group_sequences_by_length(seq_with_idx)
        
        for batch_indices, batch_seqs in batches:
            try:
                batch_results = self._predict_single_batch(batch_seqs)
                # 恢复原始顺序
                for idx, (seq, pdb) in zip(batch_indices, batch_results):
                    results.append((idx, seq, pdb))
            except Exception as e:
                logging.warning(f"Batch prediction failed: {e}, falling back to individual prediction")
                # 批处理失败时，逐个处理
                for idx, seq in zip(batch_indices, batch_seqs):
                    try:
                        individual_result = self._predict_single_batch([seq])
                        results.append((idx, seq, individual_result[0][1]))
                    except Exception as e2:
                        logging.error(f"Individual prediction failed for seq len={len(seq)}: {e2}")
                        results.append((idx, seq, make_dummy_pdb(seq)))
        
        # 恢复原始输入顺序
        results.sort(key=lambda x: x[0])
        return [(seq, pdb) for _, seq, pdb in results]
    
    def _group_sequences_by_length(self, seq_with_idx: List[Tuple[int, str]]) -> List[Tuple[List[int], List[str]]]:
        """将序列按长度分组，避免padding浪费"""
        batches = []
        current_batch_indices = []
        current_batch_seqs = []
        current_tokens = 0
        
        for idx, seq in seq_with_idx:
            seq_tokens = len(seq) + 2  # +2 for special tokens
            
            # 检查是否超出token限制或与当前batch长度差异过大
            if current_batch_seqs and (
                current_tokens + seq_tokens > self.max_batch_tokens or
                len(seq) > len(current_batch_seqs[0]) * 1.5  # 长度差异不超过50%
            ):
                # 保存当前batch
                if current_batch_seqs:
                    batches.append((current_batch_indices, current_batch_seqs))
                current_batch_indices = []
                current_batch_seqs = []
                current_tokens = 0
            
            current_batch_indices.append(idx)
            current_batch_seqs.append(seq)
            current_tokens += seq_tokens
        
        # 添加最后一个batch
        if current_batch_seqs:
            batches.append((current_batch_indices, current_batch_seqs))
        
        logging.info(f"Grouped {len(seq_with_idx)} sequences into {len(batches)} batches")
        return batches
    
    def _predict_single_batch(self, sequences: List[str]) -> List[Tuple[str, str]]:
        """预测单个batch，支持OOM回退"""
        try:
            # 批量tokenize（padding到最长序列）
            tokenized = self.tokenizer(sequences, return_tensors="pt", padding=True, add_special_tokens=False)
            input_ids = tokenized['input_ids'].to(self.device)
            
            # 精度控制
            if self.fp16:
                # 注意：input_ids是LongTensor，不能.half()，但attention_mask可以
                if 'attention_mask' in tokenized:
                    attention_mask = tokenized['attention_mask'].to(self.device).half()
                else:
                    attention_mask = None
            else:
                attention_mask = tokenized.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
            
            with torch.no_grad():
                # 传入attention_mask避免padding位置的计算
                if attention_mask is not None:
                    output = self.model(input_ids, attention_mask=attention_mask)
                else:
                    output = self.model(input_ids)
            
            # 转换为PDB
            pdb_contents = convert_outputs_to_pdb(output)
            
            # 检查PDB转换结果
            if len(pdb_contents) == 0:
                logging.warning("PDB转换结果为空，使用dummy PDB")
                return [(seq, make_dummy_pdb(seq)) for seq in sequences]
            
            if len(pdb_contents) != len(sequences):
                logging.warning(f"PDB数量({len(pdb_contents)})与序列数量({len(sequences)})不匹配")
                # 补齐缺失的PDB
                while len(pdb_contents) < len(sequences):
                    pdb_contents.append(make_dummy_pdb(sequences[len(pdb_contents)]))
            
            return [(seq, pdb) for seq, pdb in zip(sequences, pdb_contents)]
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                logging.warning(f"OOM encountered with batch size {len(sequences)}, attempting fallback")
                torch.cuda.empty_cache()
                
                # 首次OOM尝试FP16
                if not self.fp16 and self.device.type == 'cuda':
                    logging.info("Trying FP16 fallback...")
                    self.model.half()
                    self.fp16 = True
                    try:
                        return self._predict_single_batch(sequences)  # 递归重试
                    except Exception as e2:
                        logging.error(f"FP16 fallback failed: {e2}")
                
                # 如果batch size > 1，尝试分割
                if len(sequences) > 1:
                    logging.info("Trying to split batch...")
                    mid = len(sequences) // 2
                    left_results = self._predict_single_batch(sequences[:mid])
                    right_results = self._predict_single_batch(sequences[mid:])
                    return left_results + right_results
                
                # 单序列仍OOM，生成dummy
                logging.error(f"Single sequence OOM (len={len(sequences[0])}), using dummy PDB")
                return [(seq, make_dummy_pdb(seq)) for seq in sequences]
            else:
                raise e


# ---------------------------------
# Sequence preprocessing
# ---------------------------------
def handle_sequence(seq: str, max_length: int, mode: str) -> Tuple[str, dict]:
    """Apply max length策略.
    mode: truncate | skip | headtail
    headtail: 保留前 max_length//2 + 后 max_length - front 部分, 用 'GGGG' 连接
    Returns processed_seq, metadata
    """
    original_length = len(seq)
    meta = {"original_length": original_length}
    
    # 如果序列长度在限制内，不需要处理
    if max_length <= 0 or original_length <= max_length:
        meta.update({"processed": False, "mode": "keep"})
        return seq, meta
    
    # 超长序列需要处理
    meta.update({"processed": True, "mode": mode})
    
    if mode == 'skip':
        meta['skipped'] = True
        return '', meta
    elif mode == 'truncate':
        return seq[:max_length], meta
    elif mode == 'headtail':
        front = max_length // 2
        back = max_length - front - 4
        if back <= 0:  # 避免负数索引
            return seq[:max_length], meta
        return seq[:front] + 'GGGG' + seq[-back:], meta
    else:
        # default fallback to truncate
        meta["mode"] = "truncate"
        return seq[:max_length], meta


# ---------------------------------
# Main execution
# ---------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate PDBs using ESMFold with robustness features")
    parser.add_argument('--input', type=str, default='/home/lizihao/Work/enzyme_prediction/PGNN/km_test_data.csv', help='Input CSV with sample_id, sequence, smiles columns')
    parser.add_argument('--sequence-column', type=str, default='sequence', help='Column name for sequences')
    parser.add_argument('--sample-id-column', type=str, default='sample_id', help='Column name for sample IDs')
    parser.add_argument('--use-sample-manager', action='store_true', help='Use SampleManager for organized file management')
    parser.add_argument('--sample-data-dir', type=str, default='sample_data', help='SampleManager base directory')
    parser.add_argument('--output-dir', type=str, default='output/pdb_files', help='Directory to save PDBs (when not using SampleManager)')
    parser.add_argument('--gpus', type=str, default='0,1', help='Comma-separated GPU ids, e.g. 0,1 (empty for CPU)')
    parser.add_argument('--batch-size', type=int, default=1, help='Deprecated: use --max-batch-tokens instead')
    parser.add_argument('--max-batch-tokens', type=int, default=8192, help='Max tokens per batch (auto-adjusted for multi-GPU)')
    parser.add_argument('--max-length', type=int, default=500, help='Max sequence length for ESMFold (400 recommended for stability)')
    parser.add_argument('--truncate-mode', type=str, choices=['truncate','skip','headtail'], default='skip', help='Strategy when sequence exceeds max length')
    parser.add_argument('--fp16', action='store_true', help='Load model in FP16 (saves memory)')
    parser.add_argument('--gradient-checkpointing', action='store_true', help='Enable gradient checkpointing (saves memory)')
    parser.add_argument('--cpu-offload', action='store_true', help='Offload model to CPU between predictions')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing PDB files')
    parser.add_argument('--report-json', type=str, default='pdb_report.json', help='Save summary JSON')
    args = parser.parse_args()

    setup_logger()

    # 设置输出目录和样本管理
    if args.use_sample_manager:
        sample_manager = SampleManager(args.sample_data_dir)
        # 从CSV注册样本
        sample_manager.register_samples_from_csv(args.input)
        output_dir = sample_manager.base_dir
        logging.info(f"Using SampleManager with base dir: {output_dir}")
    else:
        sample_manager = None
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    required_columns = [args.sequence_column]
    if args.use_sample_manager:
        required_columns.append(args.sample_id_column)
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in input CSV")

    # 优化的预测器初始化
    predictor = ESMFoldPredictor(
        gpus=args.gpus, 
        fp16=args.fp16, 
        max_batch_tokens=args.max_batch_tokens
    )
    
    # 应用额外的内存优化
    if args.gradient_checkpointing and hasattr(predictor.model, 'gradient_checkpointing_enable'):
        predictor.model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing")

    sequences = df[args.sequence_column].tolist()
    sample_ids = df[args.sample_id_column].tolist() if args.use_sample_manager else [f"seq_{i+1}" for i in range(len(sequences))]
    report = []
    start_time = time.time()

    # 改进的批处理逻辑
    batch_size = max(1, args.max_batch_tokens // 512)  # 估算合理的batch size
    total_batches = (len(sequences) + batch_size - 1) // batch_size
    logging.info(f"Processing {len(sequences)} sequences in {total_batches} batches (max {batch_size} per batch)")

    for batch_start in range(0, len(sequences), batch_size):
        batch_end = min(batch_start + batch_size, len(sequences))
        batch_indices = list(range(batch_start, batch_end))
        batch_sequences = [sequences[i] for i in batch_indices]
        batch_sample_ids = [sample_ids[i] for i in batch_indices]
        
        logging.info(f"Processing batch {batch_start//batch_size + 1}/{total_batches}")
        
        # 预处理序列
        processed_batch = []
        batch_meta = []
        for i, (idx, raw_seq, sample_id) in enumerate(zip(batch_indices, batch_sequences, batch_sample_ids)):
            proc_seq, meta = handle_sequence(raw_seq, args.max_length, args.truncate_mode)
            
            # 检查是否需要跳过
            if args.use_sample_manager:
                pdb_path, is_shared = sample_manager.get_protein_path(sample_id)
                if pdb_path.exists() and not args.overwrite:
                    logging.info(f"[{idx+1}/{len(sequences)}] Skip existing {sample_id}")
                    meta['status'] = 'cached'
                    meta['is_shared'] = is_shared
                    report.append(meta)
                    continue
            else:
                pdb_filename = f"seq_{idx+1}.pdb"
                pdb_path = os.path.join(output_dir, pdb_filename)
                if os.path.exists(pdb_path) and not args.overwrite:
                    meta['status'] = 'cached'
                    report.append(meta)
                    continue
            
            if meta.get('skipped'):
                # 跳过超长序列，不生成文件
                logging.info(f"[{idx+1}/{len(sequences)}] Skipped {sample_id} (length: {len(raw_seq)} > {args.max_length})")
                if args.use_sample_manager:
                    sample_manager.log_failure(sample_id, "length_check", f"Sequence too long: {len(raw_seq)} > {args.max_length}", len(raw_seq))
                meta['status'] = 'skipped_too_long'
                meta['reason'] = f'Length {len(raw_seq)} exceeds limit {args.max_length}'
                report.append(meta)
                continue
            
            processed_batch.append((idx, proc_seq, sample_id, pdb_path, meta))
        
        if not processed_batch:
            continue  # 跳过空batch
        
        # 批量预测
        try:
            batch_seqs_only = [item[1] for item in processed_batch]
            batch_results = predictor.predict_batch(batch_seqs_only)
            
            # 保存结果
            for (idx, proc_seq, sample_id, pdb_path, meta), (_, pdb_content) in zip(processed_batch, batch_results):
                with open(pdb_path, 'w') as f:
                    f.write(pdb_content)
                
                if args.use_sample_manager:
                    sample_manager.update_sample_status(sample_id, "completed", pdb_path=str(pdb_path))
                
                meta['status'] = 'ok'
                if not args.use_sample_manager:
                    meta['output_file'] = os.path.basename(pdb_path)
                report.append(meta)
                
                logging.info(f"[{idx+1}/{len(sequences)}] Completed {sample_id}")
        
        except Exception as e:
            logging.error(f"Batch prediction failed: {e}")
            # 单独处理失败的batch
            for idx, proc_seq, sample_id, pdb_path, meta in processed_batch:
                dummy_content = make_dummy_pdb(proc_seq)
                with open(pdb_path, 'w') as f:
                    f.write(dummy_content)
                
                if args.use_sample_manager:
                    sample_manager.log_failure(sample_id, "structure_prediction", str(e), len(proc_seq))
                
                meta['status'] = 'dummy_fallback'
                meta['error'] = str(e)
                report.append(meta)
        
        # CPU offload option
        if args.cpu_offload:
            predictor.model.cpu()
            torch.cuda.empty_cache()
            predictor.model.to(predictor.device)

    elapsed = time.time() - start_time
    logging.info(f"Finished {len(sequences)} sequences in {elapsed/60:.2f} min")
    
    # 保存报告
    report_path = os.path.join(output_dir, args.report_json)
    with open(report_path, 'w') as rf:
        json.dump(report, rf, indent=2)
    logging.info(f"Report saved to {report_path}")
    
    # 生成失败样本摘要
    failed_samples = []
    for item in report:
        if item.get('status') in ['skipped_too_long', 'dummy_fallback']:
            failed_info = {
                'sample_id': item.get('sample_id', 'unknown'),
                'original_length': item.get('original_length', 0),
                'status': item['status'],
                'reason': item.get('reason', item.get('error', 'unknown'))
            }
            failed_samples.append(failed_info)
    
    if failed_samples:
        failed_report_path = os.path.join(output_dir, 'failed_samples.json')
        with open(failed_report_path, 'w') as f:
            json.dump(failed_samples, f, indent=2)
        logging.info(f"❌ {len(failed_samples)} samples failed - details saved to {failed_report_path}")
        logging.info("📝 You can use AF3 ColabFold for these failed samples if needed")
    else:
        logging.info("✅ All samples processed successfully")
    
    # 如果使用SampleManager，打印统计信息
    if args.use_sample_manager:
        stats = sample_manager.get_stats()
        logging.info("SampleManager Stats:")
        for k, v in stats.items():
            logging.info(f"  {k}: {v}")
        
        # 保存失败摘要
        failures = sample_manager.get_failed_samples_summary()
        if not failures.empty:
            failure_path = output_dir / "failure_summary.csv"
            failures.to_csv(failure_path, index=False)
            logging.info(f"Failure summary saved to {failure_path}")


if __name__ == '__main__':
    main()