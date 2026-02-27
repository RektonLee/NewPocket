#!/bin/bash
# Launch DiffDock batch processing on 8 GPUs (3090 cluster)
# Each GPU processes ~303 samples from the remaining 2421 samples

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(dirname "$SCRIPT_DIR")"
CSV_FILE="$WORK_DIR/data/processed/kcat_full_1213.csv"
TOTAL_SAMPLES=4072
REMAINING_SAMPLES=2421

# Calculate samples per GPU (roughly equal distribution)
SAMPLES_PER_GPU=$((REMAINING_SAMPLES / 8))
echo "📊 Distribution Plan:"
echo "   Total target: $TOTAL_SAMPLES"
echo "   Remaining: $REMAINING_SAMPLES"
echo "   Per GPU: ~$SAMPLES_PER_GPU samples"
echo ""

# Define start indices for each GPU (to cover remaining gaps)
# Based on remaining_samples.txt, we need to process scattered indices
# Strategy: Each GPU scans the entire range but only processes its share
declare -a START_INDICES=(0 509 1018 1527 2036 2545 3054 3563)

echo "🚀 Launching 8 GPU workers..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for gpu in {0..7}; do
    start_idx=${START_INDICES[$gpu]}
    log_file="$WORK_DIR/logs/diffdock_gpu${gpu}_8gpu_run.log"

    echo "GPU $gpu: Starting from index $start_idx → $log_file"

    # Launch background process with specific GPU
    CUDA_VISIBLE_DEVICES=$gpu nohup python "$SCRIPT_DIR/batch_diffdock_clean.py" \
        --start_from $start_idx \
        --max_samples $SAMPLES_PER_GPU \
        > "$log_file" 2>&1 &

    pid=$!
    echo "   PID: $pid"

    # Small delay to avoid race conditions
    sleep 2
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All 8 workers launched!"
echo ""
echo "📋 Monitor progress:"
echo "   tail -f logs/diffdock_gpu0_8gpu_run.log"
echo "   tail -f logs/diffdock_gpu1_8gpu_run.log"
echo "   ... (gpu0-7)"
echo ""
echo "🔍 Check all workers:"
echo "   ps aux | grep batch_diffdock_clean.py"
echo ""
echo "📊 Combined progress:"
echo "   python scripts/check_8gpu_progress.py"
echo ""
echo "⏹️  Stop all workers:"
echo "   pkill -f batch_diffdock_clean.py"
echo ""
