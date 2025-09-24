#!/bin/bash

# Parallel prediction script for km_test_data_with_values.csv
# Usage: ./run_parallel_km_test.sh [num_chunks]

# Default parameters
NUM_CHUNKS=${1:-8}       # Default 8 chunks
INPUT_FILE="km_test_data_with_values.csv"
MODEL_PATH="/home/lizihao/Work/enzyme_prediction/src/simple2/outputs/nopqr_attention_rbf/best_model.pt"
BASE_OUTPUT_DIR="results/parallel_km_test_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "Parallel prediction for km_test_data_with_values.csv"
echo "Input file: $INPUT_FILE"
echo "Model path: $MODEL_PATH"
echo "Output directory: $BASE_OUTPUT_DIR"
echo "Number of chunks: $NUM_CHUNKS"
echo "=========================================="

# Check input file
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found"
    exit 1
fi

# Check model file
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file $MODEL_PATH not found"
    exit 1
fi

# Create output directory
mkdir -p "$BASE_OUTPUT_DIR"

# Calculate chunk size
TOTAL_LINES=$(wc -l < "$INPUT_FILE")
TOTAL_SAMPLES=$((TOTAL_LINES - 1))  # Subtract header line
CHUNK_SIZE=$((TOTAL_SAMPLES / NUM_CHUNKS))
REMAINDER=$((TOTAL_SAMPLES % NUM_CHUNKS))

echo "Total samples: $TOTAL_SAMPLES"
echo "Chunk size: $CHUNK_SIZE"
echo "Remainder: $REMAINDER"

# Start parallel tasks
PIDS=()
START_IDX=1  # Start from 1, because row 0 is header

for i in $(seq 1 $NUM_CHUNKS); do
    # Calculate current chunk end index
    if [ $i -le $REMAINDER ]; then
        CURRENT_CHUNK_SIZE=$((CHUNK_SIZE + 1))
    else
        CURRENT_CHUNK_SIZE=$CHUNK_SIZE
    fi
    
    END_IDX=$((START_IDX + CURRENT_CHUNK_SIZE))
    OUTPUT_DIR="$BASE_OUTPUT_DIR/chunk_${i}_results"
    
    echo "Starting chunk $i: samples $START_IDX-$((END_IDX-1)) (total $CURRENT_CHUNK_SIZE samples)"
    
    # Start background task
    python3 pred_range_fixed.py \
        --input "$INPUT_FILE" \
        --model "$MODEL_PATH" \
        --output "$OUTPUT_DIR" \
        --start $START_IDX \
        --end $END_IDX \
        --use-sample-manager \
        --sample-data-dir "sample_data" \
        --temperature 303.15 \
        > "$BASE_OUTPUT_DIR/chunk_${i}.log" 2>&1 &
    
    PIDS+=($!)
    START_IDX=$END_IDX
done

# Wait for all tasks to complete
echo "Waiting for all tasks to complete..."
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo "=========================================="
echo "All parallel tasks completed, starting result merging..."

# Merge results
FINAL_OUTPUT_DIR="$BASE_OUTPUT_DIR/final_results"
mkdir -p "$FINAL_OUTPUT_DIR"

# Merge predictions.csv
echo "Merging prediction results..."
FIRST_CHUNK_FILE="$BASE_OUTPUT_DIR/chunk_1_results/predictions.csv"
if [ -f "$FIRST_CHUNK_FILE" ]; then
    head -1 "$FIRST_CHUNK_FILE" > "$FINAL_OUTPUT_DIR/predictions.csv"
    for i in $(seq 1 $NUM_CHUNKS); do
        CHUNK_FILE="$BASE_OUTPUT_DIR/chunk_${i}_results/predictions.csv"
        if [ -f "$CHUNK_FILE" ]; then
            tail -n +2 "$CHUNK_FILE" >> "$FINAL_OUTPUT_DIR/predictions.csv"
        fi
    done
fi

# Merge successful_predictions.csv
echo "Merging successful prediction results..."
FIRST_SUCCESS_FILE="$BASE_OUTPUT_DIR/chunk_1_results/successful_predictions.csv"
if [ -f "$FIRST_SUCCESS_FILE" ]; then
    head -1 "$FIRST_SUCCESS_FILE" > "$FINAL_OUTPUT_DIR/successful_predictions.csv"
    for i in $(seq 1 $NUM_CHUNKS); do
        CHUNK_FILE="$BASE_OUTPUT_DIR/chunk_${i}_results/successful_predictions.csv"
        if [ -f "$CHUNK_FILE" ]; then
            tail -n +2 "$CHUNK_FILE" >> "$FINAL_OUTPUT_DIR/successful_predictions.csv"
        fi
    done
fi

# Merge failed_predictions.csv
echo "Merging failed prediction results..."
FIRST_FAILED_FILE="$BASE_OUTPUT_DIR/chunk_1_results/failed_predictions.csv"
if [ -f "$FIRST_FAILED_FILE" ]; then
    head -1 "$FIRST_FAILED_FILE" > "$FINAL_OUTPUT_DIR/failed_predictions.csv"
    for i in $(seq 1 $NUM_CHUNKS); do
        CHUNK_FILE="$BASE_OUTPUT_DIR/chunk_${i}_results/failed_predictions.csv"
        if [ -f "$CHUNK_FILE" ]; then
            tail -n +2 "$CHUNK_FILE" >> "$FINAL_OUTPUT_DIR/failed_predictions.csv"
        fi
    done
fi

# Generate final statistics
echo "Generating final statistics..."
python3 -c "
import pandas as pd
import os

# Read merged results
predictions_file = '$FINAL_OUTPUT_DIR/predictions.csv'
if os.path.exists(predictions_file):
    df = pd.read_csv(predictions_file)
    valid_predictions = df.dropna(subset=['kcat_pred', 'km_pred'])
    failed_predictions = df[df['kcat_pred'].isna() | df['km_pred'].isna()]
    
    stats = {
        'Total Samples': len(df),
        'Successful Predictions': len(valid_predictions),
        'Failed/Skipped': len(failed_predictions),
        'Success Rate': f'{len(valid_predictions)/len(df)*100:.1f}%',
        'Parallel Chunks': $NUM_CHUNKS
    }
    
    if len(valid_predictions) > 0:
        stats.update({
            'kcat Prediction Range': f'{valid_predictions[\"kcat_pred\"].min():.2f} - {valid_predictions[\"kcat_pred\"].max():.2f}',
            'Km Prediction Range': f'{valid_predictions[\"km_pred\"].min():.2e} - {valid_predictions[\"km_pred\"].max():.2e}'
        })
        
        # Add comparison analysis if experimental values exist
        if 'experimental_km_log10' in valid_predictions.columns:
            comparison_data = valid_predictions.dropna(subset=['experimental_km_log10'])
            if len(comparison_data) > 0:
                stats.update({
                    'Samples with Experimental Values': len(comparison_data),
                    'Mean Absolute Error (log10)': comparison_data['km_error_log10'].mean(),
                    'Median Absolute Error (log10)': comparison_data['km_error_log10'].median(),
                    'Mean Relative Error': comparison_data['km_error_relative'].mean(),
                    'R² Correlation': comparison_data['km_pred_log10'].corr(comparison_data['experimental_km_log10'])**2
                })
    
    pd.DataFrame([stats]).to_csv('$FINAL_OUTPUT_DIR/final_stats.csv', index=False)
    print('Final statistics report generated')
else:
    print('No predictions file found for statistics')
"

echo "=========================================="
echo "Parallel prediction task completed!"
echo "Results saved to: $FINAL_OUTPUT_DIR/"
echo "=========================================="

# Show result files
echo "Generated files:"
ls -la "$FINAL_OUTPUT_DIR/"

# Show brief statistics
if [ -f "$FINAL_OUTPUT_DIR/final_stats.csv" ]; then
    echo ""
    echo "Prediction statistics:"
    cat "$FINAL_OUTPUT_DIR/final_stats.csv"
fi
