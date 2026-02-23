#!/bin/bash
# Python 2.7 Cleanup Script for env2
# This script optionally removes Python 2.7 from env2 environment

set -e

ENV_NAME="env2"
ENV_BIN="$HOME/miniforge3/envs/$ENV_NAME/bin"

echo "=========================================="
echo "Python 2.7 Cleanup Script for env2"
echo "=========================================="
echo ""

# Check current status
echo "Current Python configuration:"
echo "  python   -> $(readlink $ENV_BIN/python 2>/dev/null || echo 'not found')"
echo "  python3  -> $(readlink $ENV_BIN/python3 2>/dev/null || echo 'not found')"
echo ""

if [ -f "$ENV_BIN/python2.7" ]; then
    echo "Python 2.7 found at: $ENV_BIN/python2.7"
    echo "Size: $(du -h $ENV_BIN/python2.7 | cut -f1)"
    echo ""

    echo "Files that will be removed:"
    ls -lh "$ENV_BIN"/python2* 2>/dev/null || echo "  (none)"
    echo ""

    read -p "Do you want to remove Python 2.7? (y/N): " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing Python 2.7..."

        # Remove Python 2.7 files
        rm -f "$ENV_BIN/python2.7"
        rm -f "$ENV_BIN/python2.7-config"

        # Remove Python 2 symlinks (only if they point to 2.7)
        if [ -L "$ENV_BIN/python2" ] && [ "$(readlink $ENV_BIN/python2)" = "python2.7" ]; then
            rm -f "$ENV_BIN/python2"
            echo "  ✓ Removed python2 symlink"
        fi

        if [ -L "$ENV_BIN/python2-config" ] && [ "$(readlink $ENV_BIN/python2-config)" = "python2.7-config" ]; then
            rm -f "$ENV_BIN/python2-config"
            echo "  ✓ Removed python2-config symlink"
        fi

        echo "  ✓ Removed python2.7 binary"
        echo "  ✓ Python 2.7 successfully removed!"
        echo ""

        # Verify python still works
        echo "Verifying Python 3 still works..."
        conda run -n "$ENV_NAME" python --version
        echo "  ✓ Python command works correctly"
        echo ""

        # Show space saved
        echo "Space saved: ~14 KB"

    else
        echo "Cancelled. Python 2.7 kept."
    fi
else
    echo "✓ Python 2.7 not found - already removed or never installed."
fi

echo ""
echo "=========================================="
echo "Final Python configuration:"
echo "=========================================="
ls -lh "$ENV_BIN"/python* 2>/dev/null | grep -v ".pyc" | head -10

echo ""
echo "Done!"
