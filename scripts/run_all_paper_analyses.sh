#!/bin/bash
# Master script to run all JCIM paper analyses
#
# Usage:
#   bash scripts/run_all_paper_analyses.sh
#
# This script runs all analysis and figure generation scripts for the JCIM paper.
# Run this after model training completes.

set -e  # Exit on error

echo "================================================================"
echo "Running All Paper Analyses for JCIM Submission"
echo "================================================================"
echo ""

# Check if results exist
PREDICTIONS="results/diffdock_trained_auto_test/test_predictions.csv"

if [ ! -f "$PREDICTIONS" ]; then
    echo "⚠️  Predictions file not found: $PREDICTIONS"
    echo "Model training may still be in progress."
    echo "This script will run automatically once training completes."
    exit 1
fi

echo "✅ Predictions found: $PREDICTIONS"
echo ""

# Create output directory
OUTPUT_DIR="paperwriting/figures"
mkdir -p "$OUTPUT_DIR"

echo "📁 Output directory: $OUTPUT_DIR"
echo ""

# 1. Generate basic paper figures
echo "================================================================"
echo "1. Generating Basic Paper Figures"
echo "================================================================"
python scripts/generate_paper_figures.py
echo ""

# 2. EC class performance analysis
echo "================================================================"
echo "2. Analyzing Performance by EC Class"
echo "================================================================"
python scripts/analyze_ec_performance.py \
    --predictions "$PREDICTIONS" \
    --dataset data/processed/kcat_test_new_diffdock.pt \
    --output_dir "$OUTPUT_DIR"
echo ""

# 3. Training curve analysis
echo "================================================================"
echo "3. Analyzing Training Curves"
echo "================================================================"
python scripts/analyze_training_curves.py \
    --output_dir "$OUTPUT_DIR"
echo ""

# 4. Residual analysis
echo "================================================================"
echo "4. Performing Residual Analysis"
echo "================================================================"
python scripts/analyze_residuals.py \
    --predictions "$PREDICTIONS" \
    --output_dir "$OUTPUT_DIR" \
    --outlier_threshold 2.0
echo ""

# 5. Generate comprehensive results summary
echo "================================================================"
echo "5. Generating Results Summary"
echo "================================================================"

SUMMARY_FILE="paperwriting/results_summary.md"

cat > "$SUMMARY_FILE" << 'EOF'
# Results Summary for JCIM Paper

**Generated**: $(date)

---

## Test Set Performance

**Dataset**: 40% Homology OOD Split (898 samples)

### Overall Metrics

| Metric | Value |
|--------|-------|
| Pearson r | TBD |
| R² | TBD |
| MAE | TBD |
| RMSE | TBD |

### Performance by EC Class

See `paperwriting/figures/ec_class_performance.csv` for detailed per-EC results.

---

## Figures Generated

### Main Figures

1. **Figure 1**: Method comparison (`fig1_method_comparison.png/pdf`)
2. **Figure 2**: Scatter plot predictions (`fig2_scatter_predictions.png/pdf`)
3. **Figure 3**: DiffDock vs Vina comparison (`fig3_docking_comparison.png/pdf`)
4. **Figure 4**: Ablation study (`fig4_ablation.png/pdf`)

### Supplementary Figures

- **Figure S1**: Data distribution (`figS1_data_distribution.png/pdf`)
- **Figure S2**: EC class performance (`figS2_ec_performance.png/pdf`)
- **Figure S3**: EC class scatter plots (`figS3_ec_scatter.png/pdf`)
- **Figure S4**: Training curves (`figS4_loss_curves.png/pdf`)
- **Figure S5**: Validation metrics (`figS5_metrics_curves.png/pdf`)
- **Figure S6**: Learning rate schedule (`figS6_lr_schedule.png/pdf`)
- **Figure S7**: Overfitting analysis (`figS7_overfitting.png/pdf`)
- **Figure S8**: Residual distribution (`figS8_residual_distribution.png/pdf`)
- **Figure S9**: Error vs magnitude (`figS9_error_vs_magnitude.png/pdf`)
- **Figure S10**: Error by kcat range (`figS10_error_by_range.png/pdf`)

---

## Files Generated

### Tables
- `ec_class_performance.csv` - Per-EC-class metrics
- `outliers.csv` - Outlier samples for investigation

### Documentation
- `methods_detailed.md` - Comprehensive methods documentation
- `supplementary_info.md` - Supplementary information

---

## Next Steps for Paper Submission

1. ✅ Update main text with test set results
2. ✅ Verify all figure references in LaTeX
3. ⬜ Proofread methods section
4. ⬜ Finalize supplementary materials
5. ⬜ Prepare data/code release for publication

---

**Note**: This summary is automatically generated. Update with actual results after training completes.
EOF

echo "✅ Results summary saved to: $SUMMARY_FILE"
echo ""

echo "================================================================"
echo "✅ All Paper Analyses Complete!"
echo "================================================================"
echo ""
echo "📁 All outputs saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  - Main figures (PNG + PDF): fig1_*.png/pdf, fig2_*.png/pdf, etc."
echo "  - Supplementary figures: figS1_*.png/pdf, figS2_*.png/pdf, etc."
echo "  - Data tables: ec_class_performance.csv, outliers.csv"
echo "  - Documentation: methods_detailed.md, supplementary_info.md"
echo ""
echo "Next steps:"
echo "  1. Review all figures in: $OUTPUT_DIR"
echo "  2. Update gemini.tex with final results"
echo "  3. Compile LaTeX: cd paperwriting && pdflatex gemini.tex"
echo "  4. Submit to JCIM!"
echo ""
echo "================================================================"
