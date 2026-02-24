#!/bin/bash

# PocketGNN 精简迁移脚本（只打包核心文件）
# 用法: bash migrate_to_new_server_minimal.sh

set -e

echo "========================================"
echo "PocketGNN 精简迁移打包脚本"
echo "========================================"
echo ""

# 时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_NAME="pgnn_minimal_${TIMESTAMP}.tar.gz"

# 检查当前目录
if [ ! -f "CLAUDE.md" ]; then
    echo "错误: 请在 PGNN_clean 项目根目录运行此脚本"
    exit 1
fi

echo "📦 开始打包核心文件..."
echo ""

# 创建临时目录
TEMP_DIR=$(mktemp -d)
echo "创建临时目录: $TEMP_DIR"

# 复制文件到临时目录（保持目录结构）
mkdir -p "$TEMP_DIR/PGNN_clean"

echo "✓ 复制源代码..."
cp -r src/ "$TEMP_DIR/PGNN_clean/"
cp -r scripts/ "$TEMP_DIR/PGNN_clean/"

echo "✓ 复制配置文件..."
cp requirements.txt "$TEMP_DIR/PGNN_clean/"
cp CLAUDE.md "$TEMP_DIR/PGNN_clean/"
cp PROJECT_DOCUMENTATION.md "$TEMP_DIR/PGNN_clean/"
cp README.md "$TEMP_DIR/PGNN_clean/" 2>/dev/null || true
cp runs_registry.csv "$TEMP_DIR/PGNN_clean/" 2>/dev/null || true

echo "✓ 复制文档..."
cp -r docs/ "$TEMP_DIR/PGNN_clean/"

echo "✓ 复制数据文件（仅6个核心文件）..."
mkdir -p "$TEMP_DIR/PGNN_clean/data/processed"

# 只复制指定的6个文件
FILES_TO_COPY=(
    "data/processed/kcat_test_new.csv"
    "data/processed/kcat_test_new.pt"
    "data/processed/kcat_full_1213.csv"
    "data/processed/kcat_full_1213.pt"
    "data/processed/kcat_test_new_diffdock.csv"
    "data/processed/kcat_test_new_diffdock.pt"
)

for file in "${FILES_TO_COPY[@]}"; do
    if [ -f "$file" ]; then
        echo "  - $(basename $file) ($(du -h $file | cut -f1))"
        cp "$file" "$TEMP_DIR/PGNN_clean/$file"
    else
        echo "  ⚠️  警告: $file 不存在"
    fi
done

echo "✓ 复制最佳模型..."
mkdir -p "$TEMP_DIR/PGNN_clean/outputs"

# 复制3个最佳模型
MODELS=(
    "outputs/kcat_esm_norm_fix"
    "outputs/kcat_20251213_151558"
    "outputs/kcat_enhanced_run"
)

for model in "${MODELS[@]}"; do
    if [ -d "$model" ]; then
        echo "  - $(basename $model)"
        cp -r "$model" "$TEMP_DIR/PGNN_clean/$model"
    else
        echo "  ⚠️  警告: $model 不存在"
    fi
done

echo ""
echo "📊 打包文件大小统计:"
du -sh "$TEMP_DIR/PGNN_clean"/* | sort -h

echo ""
echo "🗜️  压缩打包中..."
cd "$TEMP_DIR"
tar -czf "$ARCHIVE_NAME" PGNN_clean/

# 移动到原始目录
mv "$ARCHIVE_NAME" "$OLDPWD/"
cd "$OLDPWD"

# 清理临时目录
rm -rf "$TEMP_DIR"

echo ""
echo "✅ 打包完成!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 文件名: $ARCHIVE_NAME"
echo "📊 大小:   $(du -h $ARCHIVE_NAME | cut -f1)"
echo "📁 位置:   $(pwd)/$ARCHIVE_NAME"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📤 接下来:"
echo "  1. 传输到新服务器:"
echo "     scp $ARCHIVE_NAME user@new_server:/path/to/"
echo ""
echo "  2. 在新服务器解压:"
echo "     tar -xzf $ARCHIVE_NAME"
echo "     cd PGNN_clean"
echo ""
echo "  3. 验证迁移:"
echo "     bash verify_migration.sh"
echo ""
