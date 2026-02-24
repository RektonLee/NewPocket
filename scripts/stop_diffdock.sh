#!/bin/bash
# Stop all DiffDock workers

echo "Stopping all DiffDock batch workers..."
pkill -f "batch_diffdock_clean.py"

sleep 2

# Check if any still running
remaining=$(ps aux | grep batch_diffdock_clean | grep -v grep | wc -l)

if [ $remaining -eq 0 ]; then
    echo "✅ All workers stopped"
else
    echo "⚠️  $remaining workers still running, force killing..."
    pkill -9 -f "batch_diffdock_clean.py"
fi

echo ""
echo "Run check_diffdock_progress.py to see final progress"
