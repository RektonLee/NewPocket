#!/bin/bash
# 迁移后验证脚本
# 在新服务器上运行此脚本，验证所有文件是否完整

set -e

echo "=========================================="
echo "PocketGNN 迁移验证脚本"
echo "=========================================="
echo "运行时间: $(date)"
echo ""

ERRORS=0

# 检查函数
check_file() {
    if [ -f "$1" ]; then
        SIZE=$(du -h "$1" | cut -f1)
        echo "  ✓ $1 (${SIZE})"
    else
        echo "  ✗ 缺失: $1"
        ((ERRORS++))
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        COUNT=$(find "$1" -type f 2>/dev/null | wc -l)
        SIZE=$(du -sh "$1" 2>/dev/null | cut -f1)
        echo "  ✓ $1 (${SIZE}, ${COUNT} 文件)"
    else
        echo "  ✗ 缺失: $1"
        ((ERRORS++))
    fi
}

# 1. 核心代码
echo "1. 核心代码目录"
echo "----------------------------------------"
check_dir "src"
check_dir "scripts"
check_dir "docs"
check_file "requirements.txt"
check_file "CLAUDE.md"
check_file "PROJECT_DOCUMENTATION.md"
echo ""

# 2. 关键数据文件
echo "2. 关键数据文件"
echo "----------------------------------------"
check_file "data/processed/kcat_test_new.csv"
check_file "data/processed/kcat_test_new.pt"
check_file "data/processed/kcat_full_1213.csv"
check_file "data/processed/kcat_full_1213.pt"
check_file "data/processed/kcat_test_new_diffdock.pt"
check_file "data/processed/kcat_test_new_diffdock.csv"
echo ""

# 3. 最佳模型
echo "3. 最佳模型"
echo "----------------------------------------"
check_file "outputs/kcat_esm_norm_fix/best_model.pt"
check_file "outputs/kcat_esm_norm_fix/config.json"
check_file "outputs/kcat_20251213_151558/best_model.pt"
check_file "outputs/kcat_20251213_151558/config.json"
check_file "outputs/kcat_enhanced_run/best_model.pt"
check_file "outputs/kcat_enhanced_run/config.json"
echo ""

# 4. 样本数据（可选）
echo "4. 样本数据（可选）"
echo "----------------------------------------"
if [ -d "sample_data/samples" ]; then
    SAMPLE_COUNT=$(find sample_data/samples -maxdepth 1 -type d -name "kcat_*" 2>/dev/null | wc -l)
    SAMPLE_SIZE=$(du -sh sample_data/samples 2>/dev/null | cut -f1)
    echo "  ✓ sample_data/samples (${SAMPLE_SIZE}, ${SAMPLE_COUNT} 样本目录)"
else
    echo "  ⚠ sample_data/samples 不存在 (可选，如果不需要可以忽略)"
fi
echo ""

# 5. Python 环境测试
echo "5. Python 环境测试"
echo "----------------------------------------"
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo "  ✓ Python: ${PYTHON_VERSION}"

    # 测试关键库
    python -c "import torch; print('  ✓ PyTorch:', torch.__version__)" 2>/dev/null || echo "  ✗ PyTorch 未安装"
    python -c "import torch_geometric; print('  ✓ PyG:', torch_geometric.__version__)" 2>/dev/null || echo "  ✗ PyTorch Geometric 未安装"
    python -c "import rdkit; print('  ✓ RDKit 已安装')" 2>/dev/null || echo "  ✗ RDKit 未安装"
else
    echo "  ✗ Python 未找到"
    ((ERRORS++))
fi
echo ""

# 6. 数据完整性测试
echo "6. 数据加载测试"
echo "----------------------------------------"
if [ -f "data/processed/kcat_test_new.pt" ]; then
    python -c "
import torch
try:
    data = torch.load('data/processed/kcat_test_new.pt')
    print(f'  ✓ kcat_test_new.pt: {len(data)} 样本')
except Exception as e:
    print(f'  ✗ 加载失败: {e}')
    exit(1)
" || ((ERRORS++))
fi

if [ -f "data/processed/kcat_full_1213.pt" ]; then
    python -c "
import torch
try:
    data = torch.load('data/processed/kcat_full_1213.pt')
    print(f'  ✓ kcat_full_1213.pt: {len(data)} 样本')
except Exception as e:
    print(f'  ✗ 加载失败: {e}')
    exit(1)
" || ((ERRORS++))
fi
echo ""

# 7. 模型加载测试
echo "7. 模型加载测试"
echo "----------------------------------------"
if [ -f "outputs/kcat_esm_norm_fix/best_model.pt" ]; then
    python -c "
import torch
try:
    checkpoint = torch.load('outputs/kcat_esm_norm_fix/best_model.pt', map_location='cpu')
    print(f'  ✓ 最佳模型加载成功')
    if 'epoch' in checkpoint:
        print(f'    训练轮数: {checkpoint[\"epoch\"]}')
    if 'best_val_loss' in checkpoint:
        print(f'    验证损失: {checkpoint[\"best_val_loss\"]:.4f}')
except Exception as e:
    print(f'  ✗ 模型加载失败: {e}')
    exit(1)
" || ((ERRORS++))
fi
echo ""

# 总结
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo "✓ 验证通过！所有文件完整。"
    echo "=========================================="
    echo ""
    echo "下一步: 运行测试"
    echo "----------------------------------------"
    echo "python src/test.py \\"
    echo "    --test_dataset data/processed/kcat_test_new.pt \\"
    echo "    --model outputs/kcat_esm_norm_fix/best_model.pt"
    echo ""
    echo "预期结果: R² ≈ 0.44, Pearson ≈ 0.67"
    exit 0
else
    echo "✗ 发现 ${ERRORS} 个错误！"
    echo "请检查上述缺失的文件。"
    exit 1
fi
