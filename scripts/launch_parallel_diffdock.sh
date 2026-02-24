#!/bin/bash
# Launch 4 parallel DiffDock workers on different GPUs

cd /home/lizihao/Work/enzyme_prediction/PGNN_clean

# Create logs directory
mkdir -p logs

# Kill any existing background processes
pkill -f "batch_diffdock_clean.py" || true

echo "Starting 4 parallel DiffDock workers..."

# GPU 0: samples 0-2265
CUDA_VISIBLE_DEVICES=0 nohup python scripts/batch_diffdock_clean.py \
  --start_from 0 --max_samples 2265 \
  > logs/diffdock_gpu0.log 2>&1 &
echo "GPU 0: PID $! (samples 0-2265)"

# GPU 1: samples 2265-4530
CUDA_VISIBLE_DEVICES=1 nohup python scripts/batch_diffdock_clean.py \
  --start_from 2265 --max_samples 2265 \
  > logs/diffdock_gpu1.log 2>&1 &
echo "GPU 1: PID $! (samples 2265-4530)"

# GPU 2: samples 4530-6795
CUDA_VISIBLE_DEVICES=2 nohup python scripts/batch_diffdock_clean.py \
  --start_from 4530 --max_samples 2265 \
  > logs/diffdock_gpu2.log 2>&1 &
echo "GPU 2: PID $! (samples 4530-6795)"

# GPU 3: samples 6795-9062
CUDA_VISIBLE_DEVICES=3 nohup python scripts/batch_diffdock_clean.py \
  --start_from 6795 --max_samples 2267 \
  > logs/diffdock_gpu3.log 2>&1 &
echo "GPU 3: PID $! (samples 6795-9062)"

sleep 2

echo ""
echo "All workers launched!"
echo "Monitor progress with:"
echo "  tail -f logs/diffdock_gpu*.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep batch_diffdock_clean"
echo ""
echo "Check progress:"
echo "  python scripts/check_diffdock_progress.py"
