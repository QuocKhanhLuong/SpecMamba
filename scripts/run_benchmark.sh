#!/bin/bash
# Block Benchmark Script
# Chạy benchmark các block types trên HRNet thuần

echo "========================================"
echo "Block Benchmark - Pure HRNet"  
echo "========================================"

cd "$(dirname "$0")/../.."

# Cài đặt
EPOCHS=200
BATCH_SIZE=8
LR=0.0001
DATA_DIR="preprocessed_data/ACDC"

# Chạy benchmark
python -m src.training.benchmark_blocks \
    --data_dir $DATA_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --blocks none convnext dcn inverted_residual swin fno wavelet

echo ""
echo "✓ Benchmark completed!"
