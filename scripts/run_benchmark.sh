#!/bin/bash
# Block Benchmark Script
# Chạy benchmark các block types trên HRNet thuần

echo "========================================"
echo "Block Benchmark - Pure HRNet"  
echo "========================================"

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Working dir: $(pwd)"

# Config
EPOCHS=200
BATCH_SIZE=16
LR=0.0001
DATA_DIR="preprocessed_data/ACDC"

# Add src to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Chạy benchmark
python src/training/benchmark_blocks.py \
    --data_dir $DATA_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --blocks basic convnext dcn inverted_residual swin fno wavelet rwkv

echo ""
echo "✓ Benchmark completed!"
