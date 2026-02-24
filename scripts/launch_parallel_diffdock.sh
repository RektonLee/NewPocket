#!/bin/bash
# Launch 4 workers for samples that have PDB files (0-4072)

cd /home/lizihao/Work/enzyme_prediction/PGNN_clean

mkdir -p logs

echo "Starting 4 workers for samples with PDB (total 4072)..."

# Each GPU gets ~1018 samples
# GPU 0: 0-1018
CUDA_VISIBLE_DEVICES=0 nohup python scripts/batch_diffdock_clean.py \
  --start_from 0 --max_samples 1018 \
  > logs/diffdock_gpu0.log 2>&1 &
echo "GPU 0: PID $! (samples 0-1018)"

# GPU 1: 1018-2036
CUDA_VISIBLE_DEVICES=1 nohup python scripts/batch_diffdock_clean.py \
  --start_from 1018 --max_samples 1018 \
  > logs/diffdock_gpu1.log 2>&1 &
echo "GPU 1: PID $! (samples 1018-2036)"

# GPU 2: 2036-3054
CUDA_VISIBLE_DEVICES=2 nohup python scripts/batch_diffdock_clean.py \
  --start_from 2036 --max_samples 1018 \
  > logs/diffdock_gpu2.log 2>&1 &
echo "GPU 2: PID $! (samples 2036-3054)"

# GPU 3: 3054-4072
CUDA_VISIBLE_DEVICES=3 nohup python scripts/batch_diffdock_clean.py \
  --start_from 3054 --max_samples 1018 \
  > logs/diffdock_gpu3.log 2>&1 &
echo "GPU 3: PID $! (samples 3054-4072)"

sleep 3

echo ""
echo "✅ All 4 workers launched for 4072 samples with PDB!"
echo ""
echo "Monitor: python scripts/check_diffdock_progress.py"
echo "Stop: bash scripts/stop_diffdock.sh"
