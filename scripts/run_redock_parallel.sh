#!/usr/bin/env bash
set -eo pipefail

# Parallel redock using 2 GPUs. Adjust SHARDS/GPU_IDS as needed.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Avoid OpenMP shared memory errors on some systems
export KMP_SHM_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_INIT_AT_FORK=FALSE
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

INPUT_CSV="${INPUT_CSV:-$ROOT/data/raw/kcat_test_results.csv}"
SAMPLES_PER_COMPLEX="${SAMPLES_PER_COMPLEX:-3}"
GPU_IDS="${GPU_IDS:-0,1}"
PROTEIN_PDB_DIRS="${PROTEIN_PDB_DIRS:-$ROOT/sample_data/samples,/home/lizihao/Work/enzyme_prediction/PGNN/sample_data/samples,/home/lizihao/Work/enzyme_prediction/PGNN/kcat_full_after/samples}"

SHARDS=2

source ~/miniforge3/etc/profile.d/conda.sh
conda activate env2

export PYTHONPATH="$ROOT/src"
export DIFFDOCK_PYTHON="$(which python3)"

for SHARD_IDX in 0 1; do
  LOG_FILE="$LOG_DIR/redock_diffdock_shard${SHARD_IDX}.log"
  nohup python3 "$ROOT/scripts/redock_test_new.py" \
    --method diffdock \
    --samples-per-complex "$SAMPLES_PER_COMPLEX" \
    --gpu-ids "$GPU_IDS" \
    --protein-pdb-dirs "$PROTEIN_PDB_DIRS" \
    --no-esmfold \
    --shard-index "$SHARD_IDX" \
    --shard-count "$SHARDS" \
    > "$LOG_FILE" 2>&1 &
  echo "Started shard $SHARD_IDX -> $LOG_FILE"
done
