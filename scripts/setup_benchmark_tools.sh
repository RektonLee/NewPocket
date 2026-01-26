#!/bin/bash
# 安装 DLKcat 和 UniKP 用于基准测试
# 使用方法: bash scripts/setup_benchmark_tools.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TOOLS_DIR="$PROJECT_ROOT/benchmark_tools"

echo "=================================================="
echo "安装 DLKcat 和 UniKP 基准测试工具"
echo "=================================================="
echo "安装目录: $TOOLS_DIR"
mkdir -p "$TOOLS_DIR"
cd "$TOOLS_DIR"

# ==================== DLKcat ====================
echo ""
echo "==================== DLKcat ===================="

if [ -d "DLKcat" ]; then
    echo "DLKcat 已存在，跳过克隆"
else
    echo "克隆 DLKcat..."
    git clone https://github.com/SysBioChalmers/DLKcat.git
fi

echo "DLKcat 安装完成: $TOOLS_DIR/DLKcat"
echo "注意: DLKcat 使用说明见 DLKcat/DeeplearningApproach/Code/README.md"

# ==================== UniKP ====================
echo ""
echo "==================== UniKP ===================="

if [ -d "UniKP" ]; then
    echo "UniKP 已存在，跳过克隆"
else
    echo "克隆 UniKP..."
    git clone https://github.com/Luo-SynBioLab/UniKP.git
fi

echo "UniKP 安装完成: $TOOLS_DIR/UniKP"
echo "注意: UniKP 需要下载预训练模型，见 UniKP/README.md"

# ==================== 输出使用说明 ====================
echo ""
echo "=================================================="
echo "安装完成！"
echo "=================================================="
echo ""
echo "使用 DLKcat:"
echo "  export DLKCAT_PATH=$TOOLS_DIR/DLKcat"
echo "  cd DLKcat/DeeplearningApproach/Code"
echo "  # 参考 README.md 运行预测"
echo ""
echo "使用 UniKP:"
echo "  export UNIKP_PATH=$TOOLS_DIR/UniKP"
echo "  cd UniKP"
echo "  # 1. 创建环境: conda env create -f environment.yml"
echo "  # 2. 下载预训练模型（见 README.md）"
echo "  # 3. 运行预测"
echo ""
echo "运行基准测试:"
echo "  python scripts/benchmark_dlkcat_unikp.py \\"
echo "    --dataset data/processed/kcat_full_1213.csv \\"
echo "    --dlkcat_path $TOOLS_DIR/DLKcat \\"
echo "    --unikp_path $TOOLS_DIR/UniKP \\"
echo "    --log_transform"
echo ""







