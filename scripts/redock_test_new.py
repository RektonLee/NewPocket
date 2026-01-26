#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-dock kcat_test_new using DiffDock or Chai-1 and rebuild a .pt dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from typing import Optional, Tuple

import pandas as pd
import torch

from build_graph_dataset import enhanced_build_graph
from graph_builder_rbf import parse_pocket
from sample_manager import SampleManager
from data_loader import ProteinStructureProcessor
from docking import run_preprocess
from docking_chai1 import dock_with_chai1, check_chai1_available


def resolve_kcat_columns(df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    if "experimental value[log10]" in df.columns:
        return "experimental value[log10]", None
    if "log10_kcat" in df.columns:
        return "log10_kcat", None
    if "kcat_value" in df.columns:
        return None, "kcat_value"
    if "kcat" in df.columns:
        return None, "kcat"
    raise ValueError("No kcat/log10_kcat column found in CSV.")


def compute_pocket_name(sample_id: str, smiles: str, cutoff: float) -> str:
    pocket_hash = int(hashlib.sha256(smiles.encode()).hexdigest(), 16) & 0xffff
    cutoff_label = int(round(cutoff))
    return f"{sample_id}_{pocket_hash}_{cutoff_label}A.pdb"


def _find_existing_pdb(sample_id: str, pdb_dirs: Optional[str]) -> Optional[str]:
    if not pdb_dirs:
        return None
    for d in pdb_dirs.split(","):
        d = d.strip()
        if not d:
            continue
        candidate = os.path.join(d, sample_id, f"{sample_id}_protein.pdb")
        if os.path.exists(candidate):
            return candidate
    return None


def dock_with_diffdock(sample_id: str,
                       smiles: str,
                       sequence: str,
                       sample_manager: SampleManager,
                       structure_processor: ProteinStructureProcessor,
                       pocket_pdb: str,
                       gpu_id: Optional[int],
                       validate_pose: bool,
                       persist_dir: Optional[str],
                       keep_tmp_dir: bool,
                       pdb_dirs: Optional[str],
                       no_esmfold: bool) -> bool:
    existing_pdb = _find_existing_pdb(sample_id, pdb_dirs)
    if existing_pdb:
        protein_path, _ = sample_manager.get_protein_path(sample_id)
        os.makedirs(protein_path.parent, exist_ok=True)
        try:
            if not os.path.exists(protein_path) or not os.path.samefile(existing_pdb, protein_path):
                shutil.copyfile(existing_pdb, protein_path)
        except FileNotFoundError:
            shutil.copyfile(existing_pdb, protein_path)
    else:
        if no_esmfold:
            return False
        protein_path, _ = sample_manager.get_protein_path(sample_id)
        pdb_content = structure_processor.predict_structure_with_sample_id(sample_id, sequence, None)
        if not pdb_content:
            return False
    return run_preprocess(
        uniprot_id=sample_id,
        smiles=smiles,
        prot_pdb_path=str(protein_path),
        output_pocket_path=pocket_pdb,
        index=0,
        gpu_id=gpu_id,
        validate_pose=validate_pose,
        persist_dir=persist_dir,
        keep_tmp_dir=keep_tmp_dir
    )


def dock_with_chai(sample_id: str,
                   smiles: str,
                   sequence: str,
                   docking_dir: str,
                   pocket_pdb: str,
                   pocket_cutoff: float) -> bool:
    available, msg = check_chai1_available()
    if not available:
        raise RuntimeError(msg)
    result = dock_with_chai1(
        protein_sequence=sequence,
        ligand_smiles=smiles,
        output_dir=docking_dir,
        extract_pocket=True,
        pocket_cutoff=pocket_cutoff
    )
    if not result.get("success") or not result.get("pocket_pdb"):
        return False
    if result["pocket_pdb"] != pocket_pdb:
        shutil.copyfile(result["pocket_pdb"], pocket_pdb)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-dock kcat_test_new with DiffDock/Chai-1")
    parser.add_argument("--input-csv", default="data/raw/kcat_test_results.csv")
    parser.add_argument("--output-pt", default="data/processed/kcat_test_new_diffdock.pt")
    parser.add_argument("--output-csv", default="data/processed/kcat_test_new_diffdock.csv")
    parser.add_argument("--method", choices=["diffdock", "chai1"], default="diffdock")
    parser.add_argument("--sample-data-dir", default="sample_data")
    parser.add_argument("--temperature", type=float, default=303.15)
    parser.add_argument("--pocket-cutoff", type=float, default=5.0)
    parser.add_argument("--samples-per-complex", type=int, default=5)
    parser.add_argument("--top-k-poses", type=int, default=1,
                        help="Save top-K poses (pocket_pose*.pdb) in addition to best pose")
    parser.add_argument("--gpu-ids", type=str, default=None, help="Comma-separated GPU ids for DiffDock")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--protein-pdb-dirs", type=str, default=None,
                        help="Comma-separated dirs containing {sample_id}/{sample_id}_protein.pdb")
    parser.add_argument("--no-esmfold", action="store_true",
                        help="Disable ESMFold prediction; skip samples without existing PDB")
    parser.add_argument("--diffdock-python", type=str, default=None,
                        help="Override DiffDock python interpreter (sets DIFFDOCK_PYTHON)")
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-register", action="store_true")
    parser.add_argument("--shard-index", type=int, default=None,
                        help="Shard index for parallel runs (0-based)")
    parser.add_argument("--shard-count", type=int, default=None,
                        help="Total number of shards for parallel runs")
    args = parser.parse_args()

    if args.diffdock_python:
        os.environ["DIFFDOCK_PYTHON"] = args.diffdock_python
    gpu_id_list = None
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        gpu_id_list = [int(x) for x in args.gpu_ids.split(",") if x.strip().isdigit()]

    df = pd.read_csv(args.input_csv)
    for col in ["sample_id", "sequence", "smiles"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    if args.limit:
        df = df.head(args.limit)

    sample_manager = SampleManager(args.sample_data_dir)
    if not args.skip_register:
        sample_manager.register_samples_from_csv(args.input_csv, temperature=args.temperature)

    structure_processor = ProteinStructureProcessor(sample_manager=sample_manager)
    from docking import get_config
    get_config().samples_per_complex = args.samples_per_complex
    get_config().top_k_poses = args.top_k_poses
    get_config().pocket_cutoff = args.pocket_cutoff
    log10_col, linear_col = resolve_kcat_columns(df)

    dataset = []
    successful_rows = []
    total = len(df)
    success_count = 0
    fail_count = 0
    keep_count = 0

    skip_smiles_path = os.path.join("logs", "rdkit_failed_smiles.txt")
    os.makedirs(os.path.dirname(skip_smiles_path), exist_ok=True)
    if args.no_esmfold and args.protein_pdb_dirs:
        df = df[df["sample_id"].apply(lambda sid: _find_existing_pdb(str(sid), args.protein_pdb_dirs) is not None)]
        df = df.reset_index(drop=True)

    if args.shard_index is not None and args.shard_count is not None:
        if args.shard_count <= 0:
            raise ValueError("--shard-count must be > 0")
        if args.shard_index < 0 or args.shard_index >= args.shard_count:
            raise ValueError("--shard-index must be in [0, shard-count)")
        df = df.iloc[[i for i in range(len(df)) if i % args.shard_count == args.shard_index]]
        df = df.reset_index(drop=True)

    for idx, row in enumerate(df.iterrows(), start=1):
        row = row[1]
        sample_id = str(row["sample_id"])
        sequence = str(row["sequence"])
        smiles = str(row["smiles"])
        keep_full = ((idx - 1) % 100) < 5
        gpu_id = args.gpu
        if gpu_id is None and gpu_id_list:
            gpu_id = gpu_id_list[(idx - 1) % len(gpu_id_list)]

        docking_dir = sample_manager.get_docking_dir(sample_id)
        persist_dir = docking_dir / "diffdock_output"
        pocket_name = compute_pocket_name(sample_id, smiles, args.pocket_cutoff)
        pocket_pdb = docking_dir / pocket_name

        if pocket_pdb.exists() and not args.overwrite:
            dock_ok = True
        else:
            if args.method == "diffdock":
                dock_ok = dock_with_diffdock(
                    sample_id=sample_id,
                    smiles=smiles,
                    sequence=sequence,
                    sample_manager=sample_manager,
                    structure_processor=structure_processor,
                    pocket_pdb=str(pocket_pdb),
                    gpu_id=gpu_id,
                    validate_pose=not args.no_validate,
                    persist_dir=str(persist_dir),
                    keep_tmp_dir=keep_full,
                    pdb_dirs=args.protein_pdb_dirs,
                    no_esmfold=args.no_esmfold
                )
            else:
                dock_ok = dock_with_chai(
                    sample_id=sample_id,
                    smiles=smiles,
                    sequence=sequence,
                    docking_dir=str(docking_dir),
                    pocket_pdb=str(pocket_pdb),
                    pocket_cutoff=args.pocket_cutoff
                )

        if not dock_ok or not pocket_pdb.exists():
            sample_manager.log_failure(sample_id, "docking", "docking_failed", len(sequence))
            fail_count += 1
            print(f"[{idx}/{total}] {sample_id} docking_failed | gpu {gpu_id} | success {success_count} fail {fail_count}")
            continue

        try:
            atoms = parse_pocket(str(pocket_pdb))
            if len(atoms) < 3:
                sample_manager.log_failure(sample_id, "graph_build", "too_few_atoms", len(sequence))
                fail_count += 1
                print(f"[{idx}/{total}] {sample_id} too_few_atoms | gpu {gpu_id} | success {success_count} fail {fail_count}")
                continue
            data = enhanced_build_graph(atoms, temperature=args.temperature)
            if log10_col:
                log10_value = float(row[log10_col])
            else:
                log10_value = float(torch.log10(torch.tensor([row[linear_col]], dtype=torch.float)).item())
            data.y = torch.tensor([log10_value], dtype=torch.float)
            data.sample_id = sample_id
            data.pdb_id = pocket_name
            data.ec = int(row["ec"]) if "ec" in row else 1
            meta_path = persist_dir / "docking_meta.json"
            if meta_path.exists():
                try:
                    import json
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    best_conf = meta.get("best_confidence")
                    if best_conf is None:
                        confs = meta.get("confidence_values", [])
                        valid_confs = [c for c in confs if c is not None]
                        best_conf = max(valid_confs) if valid_confs else None
                    if best_conf is not None:
                        data.docking_confidence = torch.tensor([best_conf], dtype=torch.float)
                except Exception:
                    pass
            dataset.append(data)
            successful_rows.append(row.to_dict())
            success_count += 1
            if keep_full:
                keep_count += 1
                print(f"[{idx}/{total}] {sample_id} ok (kept tmp) | gpu {gpu_id} | success {success_count} fail {fail_count}")
            else:
                print(f"[{idx}/{total}] {sample_id} ok | gpu {gpu_id} | success {success_count} fail {fail_count}")
        except Exception as e:
            err_str = str(e)
            if "Invariant Violation" in err_str or "UFFTYPER" in err_str:
                with open(skip_smiles_path, "a") as f:
                    f.write(f"{sample_id}\t{smiles}\n")
            sample_manager.log_failure(sample_id, "graph_build", str(e), len(sequence))
            fail_count += 1
            print(f"[{idx}/{total}] {sample_id} graph_build_failed | gpu {gpu_id} | success {success_count} fail {fail_count}")

        if idx % 10 == 0:
            print(f"Progress: {idx}/{total} | success {success_count} fail {fail_count} kept {keep_count}")

    os.makedirs(os.path.dirname(args.output_pt), exist_ok=True)
    torch.save(dataset, args.output_pt)
    pd.DataFrame(successful_rows).to_csv(args.output_csv, index=False)
    print(f"Saved {len(dataset)} samples -> {args.output_pt}")


if __name__ == "__main__":
    main()
