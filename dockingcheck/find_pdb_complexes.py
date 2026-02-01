#!/usr/bin/env python3
"""
筛选数据集中有 PDB 实验结构的酶-底物复合物

使用方法：
    python src/find_pdb_complexes.py --input data/processed/kcat_train.pt --output results/pdb_matched.csv
"""

import torch
import requests
import pandas as pd
import argparse
from tqdm import tqdm
from typing import List, Dict, Optional
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PDBMatcher:
    """查询 PDB 数据库，匹配酶-底物复合物"""

    def __init__(self):
        self.pdb_api_url = "https://search.rcsb.org/rcsbsearch/v2/query"
        self.cache = {}

    def query_by_ec_number(self, ec_number: str) -> List[str]:
        """通过 EC 号查询 PDB"""
        if ec_number in self.cache:
            return self.cache[ec_number]

        query = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_polymer_entity.rcsb_ec_lineage.id",
                    "operator": "exact_match",
                    "value": ec_number
                }
            },
            "return_type": "entry",
            "request_options": {
                "return_all_hits": True
            }
        }

        try:
            response = requests.post(self.pdb_api_url, json=query, timeout=10)
            if response.status_code == 200:
                data = response.json()
                pdb_ids = [hit['identifier'] for hit in data.get('result_set', [])]
                self.cache[ec_number] = pdb_ids
                return pdb_ids
            else:
                logger.warning(f"Query failed for EC {ec_number}: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error querying EC {ec_number}: {e}")
            return []

    def check_ligand_in_pdb(self, pdb_id: str) -> bool:
        """检查 PDB 是否包含小分子配体"""
        try:
            url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                # 检查是否有 ligand
                if 'rcsb_entry_info' in data:
                    num_ligands = data['rcsb_entry_info'].get('deposited_nonpolymer_entity_instance_count', 0)
                    return num_ligands > 0
            return False
        except Exception as e:
            logger.error(f"Error checking ligand for {pdb_id}: {e}")
            return False

    def get_pdb_metadata(self, pdb_id: str) -> Dict:
        """获取 PDB 元数据（分辨率、方法等）"""
        try:
            url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    'pdb_id': pdb_id,
                    'method': data.get('exptl', [{}])[0].get('method', 'unknown'),
                    'resolution': data.get('rcsb_entry_info', {}).get('resolution_combined', [None])[0],
                    'year': data.get('rcsb_accession_info', {}).get('deposit_date', '')[:4]
                }
        except Exception as e:
            logger.error(f"Error fetching metadata for {pdb_id}: {e}")
        return {'pdb_id': pdb_id, 'method': 'unknown', 'resolution': None, 'year': None}


def load_dataset(file_path: str):
    """加载 PyTorch Geometric 数据集"""
    logger.info(f"Loading dataset from {file_path}")
    # 显式允许加载 PyG Data（PyTorch 2.6+ 默认 weights_only=True 会报错）
    data = torch.load(file_path, weights_only=False)
    logger.info(f"Loaded {len(data)} samples")
    return data


def extract_metadata(dataset) -> pd.DataFrame:
    """从数据集中提取元数据（EC号、样本ID等）"""
    records = []
    for i, sample in enumerate(dataset):
        record = {
            'index': i,
            'sample_id': sample.get('sample_id', f'sample_{i}'),
            'ec': sample.get('ec', None),
            'pdb_id': sample.get('pdb_id', None),
            'substrate': sample.get('substrate', None),  # 如果有 SMILES
        }
        records.append(record)

    df = pd.DataFrame(records)
    logger.info(f"Extracted metadata for {len(df)} samples")
    return df


def match_pdb_structures(
    df: pd.DataFrame,
    matcher: PDBMatcher,
    max_pdb_per_ec: int = 5,
    limit_ec: Optional[int] = None
) -> pd.DataFrame:
    """按 EC 号匹配 PDB 结构（避免对每个样本重复查询）"""
    results = []
    df_ec = df.dropna(subset=['ec']).copy()
    grouped = df_ec.groupby('ec')
    if limit_ec is not None:
        # 仅用于快速试跑/抽样验证
        grouped = list(grouped)[:limit_ec]
        grouped_iter = grouped
        total = len(grouped)
    else:
        grouped_iter = grouped
        total = grouped.ngroups

    for ec_number, group in tqdm(grouped_iter, total=total, desc="Matching PDB structures by EC"):
        pdb_ids = matcher.query_by_ec_number(str(ec_number))
        if not pdb_ids:
            continue

        # 每个 EC 号只取前 max_pdb_per_ec 个 PDB（避免过慢）
        matched_pdbs = []
        for pdb_id in pdb_ids[:max_pdb_per_ec]:
            time.sleep(0.1)
            if matcher.check_ligand_in_pdb(pdb_id):
                metadata = matcher.get_pdb_metadata(pdb_id)
                matched_pdbs.append({
                    'ec': ec_number,
                    'matched_pdb': pdb_id,
                    'method': metadata['method'],
                    'resolution': metadata['resolution'],
                    'year': metadata['year']
                })

        if not matched_pdbs:
            continue

        # 聚合该 EC 下的样本信息
        sample_ids = group['sample_id'].dropna().astype(str).unique().tolist()
        for row in matched_pdbs:
            row.update({
                'sample_count': len(sample_ids),
                'sample_ids': ";".join(sample_ids)
            })
            results.append(row)

    result_df = pd.DataFrame(results)
    logger.info(f"Found {len(result_df)} EC-PDB matches")
    return result_df


def expand_ec_matches_to_samples(match_df: pd.DataFrame, df_samples: pd.DataFrame) -> pd.DataFrame:
    """把 EC-PDB 匹配结果展开到样本级别（可选）"""
    if match_df.empty:
        return match_df
    merged = df_samples.merge(match_df[['ec', 'matched_pdb', 'method', 'resolution', 'year']], on='ec', how='inner')
    merged = merged.rename(columns={'index': 'sample_index'})
    return merged[['sample_index', 'sample_id', 'ec', 'matched_pdb', 'method', 'resolution', 'year']]


def main():
    parser = argparse.ArgumentParser(description="Find PDB complexes in dataset")
    parser.add_argument('--input', type=str, required=True, help='Input .pt file')
    parser.add_argument('--output', type=str, default='results/pdb_matched_ec.csv', help='Output EC-level CSV file')
    parser.add_argument('--sample_output', type=str, default=None, help='Optional sample-level CSV output')
    parser.add_argument('--max_pdb_per_ec', type=int, default=5, help='Max PDBs to check per EC')
    parser.add_argument('--limit_ec', type=int, default=None, help='Limit number of ECs (for quick dry-run)')
    args = parser.parse_args()

    # 加载数据集
    dataset = load_dataset(args.input)

    # 提取元数据
    df = extract_metadata(dataset)

    # 匹配 PDB 结构（按 EC）
    matcher = PDBMatcher()
    result_df = match_pdb_structures(
        df,
        matcher,
        max_pdb_per_ec=args.max_pdb_per_ec,
        limit_ec=args.limit_ec
    )

    # 保存结果
    result_df.to_csv(args.output, index=False)
    logger.info(f"Results saved to {args.output}")

    # 可选：展开到样本级别
    if args.sample_output:
        sample_df = expand_ec_matches_to_samples(result_df, df)
        sample_df.to_csv(args.sample_output, index=False)
        logger.info(f"Sample-level results saved to {args.sample_output}")

    # 打印统计信息
    print("\n=== Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Matched EC-PDB pairs: {len(result_df)}")
    print(f"Unique EC numbers with PDB: {result_df['ec'].nunique()}")
    print(f"Unique PDB IDs: {result_df['matched_pdb'].nunique()}")

    if len(result_df) > 0:
        print("\n=== Top 10 Matched PDBs ===")
        print(result_df[['ec', 'matched_pdb', 'resolution', 'method', 'sample_count']].head(10))


if __name__ == '__main__':
    main()
