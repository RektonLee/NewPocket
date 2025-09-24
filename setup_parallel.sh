#!/bin/bash

# Setup script for parallel prediction
echo "Setting up parallel prediction environment..."

# Make scripts executable
chmod +x run_parallel_fixed.sh
chmod +x pred_range_fixed.py

echo "✅ Scripts are now executable"
echo ""
echo "Usage:"
echo "  ./run_parallel_fixed.sh [num_chunks]"
echo ""
echo "Examples:"
echo "  ./run_parallel_fixed.sh 8    # 8 chunks (default)"
echo "  ./run_parallel_fixed.sh 16   # 16 chunks"
echo ""
echo "The script will:"
echo "  1. Split your dataset into chunks"
echo "  2. Run all chunks in parallel (no limit on parallel processes)"
echo "  3. Automatically merge all results"
echo "  4. Generate final statistics"
echo ""
echo "Final results will be in: results/parallel_YYYYMMDD_HHMMSS/final_results/"
echo "  - predictions.csv: All results"
echo "  - successful_predictions.csv: Only successful predictions (what you need)"
echo "  - failed_predictions.csv: Failed/skipped samples"
echo "  - final_stats.csv: Overall statistics"
