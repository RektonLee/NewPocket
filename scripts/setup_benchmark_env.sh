#!/bin/bash
# ============================================================================
# Setup Benchmark Environments for SOTA Comparison
# ============================================================================

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BENCHMARK_DIR="$PROJECT_ROOT/benchmark_tools"

echo "=============================================="
echo "Setting up benchmark environments"
echo "Project root: $PROJECT_ROOT"
echo "Benchmark dir: $BENCHMARK_DIR"
echo "=============================================="

# ============================================================================
# 1. Setup CatPred Environment
# ============================================================================
echo ""
echo "[1/4] Setting up CatPred environment..."

cd "$BENCHMARK_DIR/CatPred"

# Create conda environment if not exists
if ! conda env list | grep -q "^catpred "; then
    echo "Creating catpred conda environment..."
    conda env create -f environment.yml
else
    echo "catpred environment already exists, skipping creation"
fi

# Install CatPred as package
echo "Installing CatPred package..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate catpred
pip install -e . --quiet
conda deactivate

echo "✅ CatPred environment setup complete"

# ============================================================================
# 2. Setup CataPro Environment
# ============================================================================
echo ""
echo "[2/4] Setting up CataPro environment..."

cd "$BENCHMARK_DIR/CataPro"

# Create conda environment if not exists
if ! conda env list | grep -q "^catapro "; then
    echo "Creating catapro conda environment..."
    conda create -n catapro python=3.9 -y
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate catapro
    pip install torch>=1.13.0 transformers numpy pandas rdkit-pypi sentencepiece --quiet
    conda deactivate
else
    echo "catapro environment already exists, skipping creation"
fi

echo "✅ CataPro environment setup complete"

# ============================================================================
# 3. Download CatPred Data
# ============================================================================
echo ""
echo "[3/4] Downloading CatPred data..."

CATPRED_DATA_DIR="$BENCHMARK_DIR/CatPred/capsule_data"

if [ ! -d "$CATPRED_DATA_DIR" ]; then
    cd "$BENCHMARK_DIR/CatPred"
    echo "Downloading CatPred pre-trained models and data (~2GB)..."
    wget -q --show-progress https://catpred.s3.us-east-1.amazonaws.com/capsule_data_update.tar.gz
    echo "Extracting..."
    tar -xzf capsule_data_update.tar.gz
    rm capsule_data_update.tar.gz
    echo "✅ CatPred data downloaded"
else
    echo "CatPred data already exists, skipping download"
fi

# ============================================================================
# 4. Download CataPro Pre-trained Models (ProtT5 + MolT5)
# ============================================================================
echo ""
echo "[4/4] Setting up CataPro pre-trained models..."

CATAPRO_MODELS_DIR="$BENCHMARK_DIR/CataPro/models"

# Check if ProtT5 model exists
if [ ! -d "$CATAPRO_MODELS_DIR/prot_t5_xl_uniref50" ]; then
    echo "Note: CataPro requires pre-trained models (ProtT5-XL, MolT5-base)."
    echo "These will be downloaded automatically on first run."
    echo "Alternatively, download from HuggingFace:"
    echo "  - https://huggingface.co/Rostlab/prot_t5_xl_uniref50"
    echo "  - https://huggingface.co/laituan245/molt5-base-smiles2caption"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Available environments:"
echo "  - catpred: For running CatPred baseline"
echo "  - catapro: For running CataPro baseline"
echo "  - pocketgnn (your existing env): For running PocketGNN"
echo ""
echo "Data locations:"
echo "  - CatPred data: $BENCHMARK_DIR/CatPred/capsule_data/"
echo "  - CataPro data: $BENCHMARK_DIR/CataPro/datasets/"
echo ""
echo "Next steps:"
echo "  1. Run: python scripts/download_catpred_db.py"
echo "  2. Run: python scripts/convert_catpred_to_pocketgnn.py"
echo ""

