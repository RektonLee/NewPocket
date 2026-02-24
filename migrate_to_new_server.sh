#!/bin/bash
# PocketGNN 项目迁移脚本
# 生成时间: 2026-02-24
# 用途: 打包所有必需文件以迁移到新服务器

set -e  # 遇到错误立即退出

# 配置
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_NAME="pgnn_migration_${TIMESTAMP}.tar.gz"
PROJECT_ROOT="/home/lizihao/Work/enzyme_prediction/PGNN_clean"

echo "=========================================="
echo "PocketGNN 项目迁移打包脚本"
echo "=========================================="
echo "时间: $(date)"
echo "输出: ${ARCHIVE_NAME}"
echo ""

cd "${PROJECT_ROOT}"

# 创建临时目录列表文件
echo "正在收集文件列表..."

# 核心代码和配置
cat > /tmp/pgnn_migration_files.txt <<EOF
src/
scripts/
docs/
configs/
benchmark_tools/
requirements.txt
CLAUDE.md
PROJECT_DOCUMENTATION.md
DOCKING_METHODS_COMPARISON.md
README.md
runs_registry.csv
.gitignore
EOF

# 关键数据文件（精简版）
cat >> /tmp/pgnn_migration_files.txt <<EOF
data/processed/kcat_test_new.csv
data/processed/kcat_test_new.pt
data/processed/kcat_full_1213.csv
data/processed/kcat_full_1213.pt
data/processed/kcat_test_new_diffdock.pt
data/processed/kcat_test_new_diffdock.csv
EOF

# 最佳模型（3个）
cat >> /tmp/pgnn_migration_files.txt <<EOF
outputs/kcat_esm_norm_fix/
outputs/kcat_20251213_151558/
outputs/kcat_enhanced_run/
EOF

echo ""
echo "=========================================="
echo "文件清单汇总"
echo "=========================================="

# 统计文件大小
echo ""
echo "核心代码:"
du -sh src/ scripts/ docs/ 2>/dev/null || echo "  (部分目录不存在)"

echo ""
echo "关键数据文件:"
ls -lh data/processed/kcat_test_new.csv 2>/dev/null || echo "  ✗ kcat_test_new.csv"
ls -lh data/processed/kcat_test_new.pt 2>/dev/null || echo "  ✗ kcat_test_new.pt"
ls -lh data/processed/kcat_full_1213.csv 2>/dev/null || echo "  ✗ kcat_full_1213.csv"
ls -lh data/processed/kcat_full_1213.pt 2>/dev/null || echo "  ✗ kcat_full_1213.pt"
ls -lh data/processed/kcat_test_new_diffdock.pt 2>/dev/null || echo "  ✗ kcat_test_new_diffdock.pt"
ls -lh data/processed/kcat_test_new_diffdock.csv 2>/dev/null || echo "  ✗ kcat_test_new_diffdock.csv"

echo ""
echo "最佳模型:"
du -sh outputs/kcat_esm_norm_fix/ 2>/dev/null || echo "  ✗ kcat_esm_norm_fix/"
du -sh outputs/kcat_20251213_151558/ 2>/dev/null || echo "  ✗ kcat_20251213_151558/"
du -sh outputs/kcat_enhanced_run/ 2>/dev/null || echo "  ✗ kcat_enhanced_run/"

echo ""
echo "=========================================="
read -p "确认开始打包？[y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消打包"
    exit 1
fi

echo ""
echo "开始打包..."

# 打包（排除不需要的文件）
tar -czf "${ARCHIVE_NAME}" \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='wandb' \
    --exclude='temp' \
    --exclude='tmp' \
    --exclude='.vscode' \
    --exclude='.idea' \
    -T /tmp/pgnn_migration_files.txt

# 检查打包结果
if [ -f "${ARCHIVE_NAME}" ]; then
    FILE_SIZE=$(du -h "${ARCHIVE_NAME}" | cut -f1)
    echo ""
    echo "=========================================="
    echo "✓ 打包完成！"
    echo "=========================================="
    echo "文件: ${ARCHIVE_NAME}"
    echo "大小: ${FILE_SIZE}"
    echo "位置: ${PROJECT_ROOT}/${ARCHIVE_NAME}"
    echo ""
    echo "下一步:"
    echo "1. 传输到新服务器:"
    echo "   scp ${ARCHIVE_NAME} user@new_server:/path/to/"
    echo ""
    echo "2. 在新服务器解压:"
    echo "   tar -xzf ${ARCHIVE_NAME} -C /path/to/PGNN_clean/"
    echo ""
    echo "3. 验证文件完整性:"
    echo "   bash verify_migration.sh"
    echo "=========================================="
else
    echo "✗ 打包失败！"
    exit 1
fi

# 清理临时文件
rm -f /tmp/pgnn_migration_files.txt
