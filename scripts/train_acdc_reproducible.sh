#!/usr/bin/env bash
set -euo pipefail

python src/training/train_acdc.py --config configs/acdc_reproducible.yaml "$@"
