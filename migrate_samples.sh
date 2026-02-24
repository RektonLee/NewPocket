#!/bin/bash
# 样本数据迁移脚本（单独处理，因为数据量较大）
# 生成时间: 2026-02-24

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_ROOT="/home/lizihao/Work/enzyme_prediction/PGNN_clean"
PGNN_ROOT="/home/lizihao/Work/enzyme_prediction/PGNN"

echo "=========================================="
echo "样本数据迁移脚本"
echo "=========================================="
echo ""

# 方案1: 打包 PGNN_clean 的样本数据（测试集）
echo "方案1: 打包 PGNN_clean 测试集样本 (516 MB)"
echo "----------------------------------------"
cd "${PROJECT_ROOT}"
ARCHIVE_CLEAN="pgnn_samples_clean_${TIMESTAMP}.tar.gz"

if [ -d "sample_data/samples" ]; then
    echo "正在打包 sample_data/samples/ ..."
    tar -czf "${ARCHIVE_CLEAN}" sample_data/samples/
    echo "✓ 完成: ${ARCHIVE_CLEAN} ($(du -h ${ARCHIVE_CLEAN} | cut -f1))"
else
    echo "✗ 目录不存在: sample_data/samples/"
fi

echo ""

# 方案2: 打包 PGNN 的样本数据（训练集，1.7GB）
echo "方案2: 打包 PGNN 训练集样本 (1.7 GB)"
echo "----------------------------------------"
ARCHIVE_PGNN="pgnn_samples_train_${TIMESTAMP}.tar.gz"

if [ -d "${PGNN_ROOT}/sample_data/samples" ]; then
    echo "警告: 这个包很大 (1.7GB)，打包需要几分钟..."
    read -p "是否继续？[y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "${PGNN_ROOT}"
        echo "正在打包 sample_data/samples/ (kcat_xxxxxx 格式)..."
        tar -czf "${PROJECT_ROOT}/${ARCHIVE_PGNN}" \
            --exclude='*.log' \
            --exclude='*.tmp' \
            sample_data/samples/kcat_[0-9]*
        echo "✓ 完成: ${ARCHIVE_PGNN} ($(du -h ${PROJECT_ROOT}/${ARCHIVE_PGNN} | cut -f1))"
    else
        echo "已跳过 PGNN 训练集样本打包"
    fi
else
    echo "✗ 目录不存在: ${PGNN_ROOT}/sample_data/samples/"
fi

echo ""
echo "=========================================="
echo "样本数据打包完成"
echo "=========================================="
echo ""
echo "生成的文件:"
ls -lh "${PROJECT_ROOT}"/pgnn_samples_*.tar.gz 2>/dev/null || echo "  (没有生成样本包)"
echo ""
echo "传输命令:"
echo "  scp pgnn_samples_*.tar.gz user@new_server:/path/to/"
echo ""
echo "解压命令:"
echo "  tar -xzf pgnn_samples_clean_*.tar.gz -C /path/to/PGNN_clean/"
echo "  tar -xzf pgnn_samples_train_*.tar.gz -C /path/to/backup/"
echo "=========================================="
