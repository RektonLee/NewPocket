#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   conda activate env2
#   bash scripts/run_homology_baselines.sh

FASTA="data/processed/kcat_merged_hom40_sequences.fasta"
DATASET="data/processed/kcat_merged_hom40_merged.pt"
THREADS="${THREADS:-8}"
SEED="${SEED:-42}"
RATIOS="${RATIOS:-0.8,0.1,0.1}"

if [ ! -f "$FASTA" ]; then
  echo "Missing fasta: $FASTA" >&2
  exit 1
fi
if [ ! -f "$DATASET" ]; then
  echo "Missing dataset: $DATASET" >&2
  exit 1
fi

for ID in 0.4 0.7 0.9; do
  OUT_DIR="data/processed/mmseqs_${ID}"
  mkdir -p "$OUT_DIR"
  TSV="$OUT_DIR/clusters_cluster.tsv"

  if [ ! -f "$TSV" ]; then
    echo "[MMseqs2] clustering @ identity=${ID}"
    mmseqs easy-cluster "$FASTA" "$OUT_DIR/clusters" "$OUT_DIR/tmp" \
      --min-seq-id "$ID" -c 0.8 --threads "$THREADS"
  else
    echo "[MMseqs2] reuse existing: $TSV"
  fi

  echo "[Train] homology=${ID}"
  python3 src/train.py \
    --dataset "$DATASET" \
    --cluster_tsv "$TSV" \
    --split_ratios "$RATIOS" \
    --split_seed "$SEED" \
    --exp_name "kcat_hom${ID//./}" 
done
