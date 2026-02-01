#!/usr/bin/env python3
"""
从外部基准集（PDBbind / Binding MOAD）抽取复合物 PDB ID 列表

使用方法：
    python dockingcheck/collect_external_complexes.py \
        --pdbbind-index /path/INDEX_general_PL.2020 \
        --output results/pdbbind_ids.csv

    python dockingcheck/collect_external_complexes.py \
        --moad-list /path/BindingMOAD.txt \
        --output results/moad_ids.csv
"""

import argparse
import re
from pathlib import Path
import pandas as pd


def parse_pdbbind_index(path: Path):
    """解析 PDBbind INDEX_general_PL.* 文件，提取 PDB ID"""
    pdb_ids = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # PDBbind index 的前 4 位通常是 PDB ID
            pdb_id = line[:4].strip()
            if len(pdb_id) == 4:
                pdb_ids.append(pdb_id.upper())
    return sorted(set(pdb_ids))


def parse_moad_list(path: Path):
    """从 MOAD/自定义列表中用正则提取 PDB ID"""
    pdb_ids = set()
    pattern = re.compile(r"\b[0-9][A-Za-z0-9]{3}\b")
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for match in pattern.findall(line):
                pdb_ids.add(match.upper())
    return sorted(pdb_ids)


def main():
    parser = argparse.ArgumentParser(description="Collect external complex PDB IDs")
    parser.add_argument("--pdbbind-index", type=str, default=None, help="PDBbind INDEX_general_PL.* path")
    parser.add_argument("--moad-list", type=str, default=None, help="Binding MOAD list/text file path")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    args = parser.parse_args()

    rows = []
    if args.pdbbind_index:
        ids = parse_pdbbind_index(Path(args.pdbbind_index))
        rows.extend([{"source": "PDBbind", "pdb_id": pid} for pid in ids])
    if args.moad_list:
        ids = parse_moad_list(Path(args.moad_list))
        rows.extend([{"source": "BindingMOAD", "pdb_id": pid} for pid in ids])

    if not rows:
        raise SystemExit("No input provided. Use --pdbbind-index or --moad-list.")

    df = pd.DataFrame(rows).drop_duplicates()
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} PDB IDs to {args.output}")


if __name__ == "__main__":
    main()
