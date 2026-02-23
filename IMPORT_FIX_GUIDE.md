# Import Path Fix & Environment Setup Guide

**Date:** 2026-02-10

## ✅ Import Paths Fixed

All scripts in `scripts/` have been updated to correctly import from `src/` modules.

**Fix applied:**
```python
import sys
import os

# Add src/ to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Now can import from src/
from build_graph_dataset import enhanced_build_graph
from GNN_model import PocketGNNKcatOnly
```

---

## 🔧 Environment Issues

### Current Problem

Your system is encountering a library compatibility issue:

```
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.31' not found
```

**Cause:** RDKit requires a newer version of libstdc++ than your system has.

### Solutions

#### Option 1: Use a Conda Environment with Proper Dependencies (Recommended)

```bash
# Create a new environment with all dependencies
conda create -n pocketgnn python=3.9 pytorch pytorch-geometric rdkit biopython pandas -c pytorch -c conda-forge

# Activate it
conda activate pocketgnn

# Install additional packages
pip install wandb transformers

# Test
python scripts/redock_test_new.py --help
```

#### Option 2: Fix libstdc++ in Your Current Environment

```bash
# Option A: Update libstdc++ via conda
conda install -c conda-forge libstdcxx-ng

# Option B: Use conda's libstdc++
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Test
python scripts/redock_test_new.py --help
```

#### Option 3: Use Docker (Most Reliable)

```bash
# Create a Dockerfile
cat > Dockerfile << 'EOF'
FROM continuumio/miniconda3

WORKDIR /app

# Install dependencies
RUN conda install -y python=3.9 pytorch pytorch-geometric rdkit biopython pandas -c pytorch -c conda-forge
RUN pip install wandb transformers

# Copy code
COPY . /app

CMD ["/bin/bash"]
EOF

# Build and run
docker build -t pocketgnn .
docker run -it -v $(pwd):/app pocketgnn
```

---

## 🐍 Environment Details

### Your Current Environments

```bash
$ conda env list

base                 * /home/lizihao/miniforge3        # Python 3.12 (has libstdc++ issue)
env2                   .../envs/env2                   # Python 2.7 (too old)
diffdock               .../envs/diffdock               # Missing PyTorch
pocket                 .../envs/pocket                 # Unknown
...
```

### Recommended Environment

**Name:** `pocketgnn` (or use existing `pocket` env if properly configured)

**Python Version:** 3.9 or 3.10 (best compatibility)

**Core Dependencies:**
```
python=3.9
pytorch>=1.12
torch-geometric>=2.0
rdkit>=2022.03
biopython>=1.79
pandas>=1.3
numpy>=1.21
scikit-learn>=1.0
```

**Optional Dependencies:**
```
wandb>=0.12          # For experiment tracking
transformers>=4.20   # For ESM-2 embeddings
scipy>=1.7           # For statistical tests
matplotlib>=3.5      # For visualization
seaborn>=0.11        # For enhanced plots
```

---

## ✅ Verification Steps

### Step 1: Check Python Version

```bash
conda activate pocketgnn  # or your environment
python --version
# Should be Python 3.9 or 3.10
```

### Step 2: Test Core Imports

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')

import torch_geometric
print(f'PyG: {torch_geometric.__version__}')

from rdkit import Chem
print('RDKit: OK')

print('✓ All core dependencies working')
"
```

### Step 3: Test Script Imports

```bash
python -c "
import sys
sys.path.insert(0, 'src')

from build_graph_dataset import enhanced_build_graph
from GNN_model import PocketGNNKcatOnly
from data_loader import ProteinStructureProcessor

print('✓ All script imports working')
"
```

### Step 4: Test a Full Script

```bash
python scripts/redock_test_new.py --help
# Should show help message without errors
```

---

## 🔍 Debugging Import Issues

### Issue: "ModuleNotFoundError: No module named 'build_graph_dataset'"

**Diagnosis:**
```bash
python -c "import sys; print('\\n'.join(sys.path[:5]))"
```

**Fix:** Script should have:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
```

### Issue: "ImportError: libstdc++.so.6: version `GLIBCXX_X.X.XX' not found"

**Diagnosis:**
```bash
strings /lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
conda list libstdcxx-ng
```

**Fix:** Install proper libstdc++:
```bash
conda install -c conda-forge libstdcxx-ng
```

### Issue: "SyntaxError" on type annotations

**Diagnosis:**
```bash
python --version
# If < 3.7, this is the problem
```

**Fix:** Use Python 3.9+:
```bash
conda create -n pocketgnn python=3.9
```

---

## 📊 Script Status After Fix

### ✅ Import Paths Fixed (All Scripts)

These scripts now correctly import from `src/`:

**Core Training/Testing:**
- ✅ `scripts/redock_test_new.py`
- ✅ `scripts/build_test_dataset.py`
- ✅ `scripts/quick_benchmark.py`
- ✅ `scripts/predict_and_plot.py`

**Benchmarking:**
- ✅ `scripts/convert_catapro_to_pocketgnn.py`
- ✅ `scripts/eval_pocketgat_on_dlkcat.py`
- ✅ `scripts/eval_on_dlkcat_testset.py`
- ✅ `scripts/process_dlkcat_testset.py`
- ✅ `scripts/convert_testset_for_sota.py`

**Debugging:**
- ✅ `scripts/debug_test.py`
- ✅ `scripts/deep_debug.py`
- ✅ `scripts/analyze_model_output.py`
- ✅ `scripts/fix_model_output.py`

**Standalone Scripts (no src/ imports needed):**
- ✅ `scripts/merge_and_split_by_homology.py`
- ✅ `scripts/check_data_leakage.py`
- ✅ `scripts/run_catapro_baseline.py`
- ✅ `scripts/run_dlkcat.py`
- ✅ `scripts/run_unikp.py`
- ✅ And others...

---

## 🚀 Quick Fix for Your Command

Your original command should now work with a proper environment:

```bash
# Option 1: Create new environment (recommended)
conda create -n pocketgnn python=3.9 pytorch pytorch-geometric rdkit biopython pandas -c pytorch -c conda-forge
conda activate pocketgnn
pip install -r requirements.txt

# Option 2: Fix existing environment
conda activate base  # or your preferred env
conda install -c conda-forge libstdcxx-ng

# Now run your command
python scripts/redock_test_new.py \
    --method diffdock \
    --skip-docking \
    --output-pt data/processed/kcat_test_new_diffdock_full.pt \
    --output-csv data/processed/kcat_test_new_diffdock_full.csv
```

---

## 📝 Summary

| Issue | Status | Fix |
|-------|--------|-----|
| Import paths | ✅ Fixed | Added `sys.path.insert(0, 'src')` |
| libstdc++ version | ⚠️ Environment issue | Install `libstdcxx-ng` or create new env |
| Python 2.7 (env2) | ⚠️ Too old | Use Python 3.9+ |
| Missing torch (diffdock env) | ⚠️ Incomplete env | Install missing packages |

---

## 💡 Recommendations

1. **Create a dedicated `pocketgnn` conda environment** with all dependencies
2. **Use Python 3.9 or 3.10** (best compatibility)
3. **Install from conda-forge** when possible (better compatibility)
4. **Test core imports first** before running full scripts
5. **Keep environments separate** (don't mix Python 2.7 and 3.x)

---

## 📚 Related Documentation

- `requirements.txt` - List of all required packages
- `README.md` - Installation instructions
- `GUIDE.md` - Detailed setup guide
- `PROJECT_STRUCTURE.md` - Project organization

---

**Last Updated:** 2026-02-10
**Status:** Import paths fixed ✅, Environment setup needed ⚠️
