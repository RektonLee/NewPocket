#!/bin/bash
# 合并诊断分支到 master 的专业流程
# 使用方法: bash MERGE_COMMANDS.sh

set -e  # 遇到错误立即退出

echo "🚀 开始合并诊断分支到 master..."

# 1. 检查当前分支
CURRENT_BRANCH=$(git branch --show-current)
echo "📍 当前分支: $CURRENT_BRANCH"

if [ "$CURRENT_BRANCH" != "diag/label-permutation" ]; then
    echo "⚠️  警告: 当前不在 diag/label-permutation 分支"
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 2. 检查是否有未提交的更改
if ! git diff-index --quiet HEAD --; then
    echo "📝 检测到未提交的更改，需要先提交..."
    echo "请运行:"
    echo "  git add src/train.py Todiagnose.md DEV_GUIDE.md"
    echo "  git commit -m 'feat: Add diagnostic tests and improve train.py'"
    exit 1
fi

# 3. 切换到 master 分支
echo "🔄 切换到 master 分支..."
git checkout master

# 4. 拉取最新更改（如果有远程仓库）
if git remote | grep -q origin; then
    echo "📥 拉取 master 最新更改..."
    git pull origin master || echo "⚠️  无法拉取，继续本地合并..."
fi

# 5. 合并诊断分支
echo "🔀 合并 diag/label-permutation 分支..."
git merge diag/label-permutation --no-ff -m "Merge diagnostic branch: Add tests and improvements

- Add Label Permutation Test and Frozen Encoder Test
- Fix Pearson correlation NaN handling
- Add fixed random seed for reproducibility
- Add command-line arguments (batch_size, lr, max_epochs)
- Add load_checkpoint functionality
- Improve W&B configuration (group, tags, notes)
- Add exp_name parameter for experiment management
- Add Todiagnose.md diagnostic guide
- Update DEV_GUIDE.md with diagnostic test info"

# 6. 验证合并结果
echo "✅ 合并完成！"
echo ""
echo "📊 验证合并结果:"
git log --oneline -3
echo ""
echo "🧪 建议运行以下命令验证:"
echo "  python src/train.py --help"
echo "  python src/train.py --dataset data/processed/kcat_full.pt --max_epochs 1"

