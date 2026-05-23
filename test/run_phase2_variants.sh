#!/usr/bin/env bash
set -euo pipefail

# Run all Phase Test 2 block-architecture ablations from the repository root.
# Override any setting with environment variables, for example:
#   EPOCHS=80 BATCH_SIZE=4 DEVICE=cuda bash test/run_phase2_variants.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-test/configs/ssr_phase2_acdc_224.yaml}"
DEVICE="${DEVICE:-cuda}"

DEFAULT_VARIANTS=(
  baseline_ssr
  ssr_se
  ssr_se_bounded
  ssr_se_lk
  ssr_se_dcn
  ssr_full
)

if [[ "$#" -gt 0 ]]; then
  VARIANTS=("$@")
else
  VARIANTS=("${DEFAULT_VARIANTS[@]}")
fi

for variant in "${VARIANTS[@]}"; do
  run_name="ssr_phase2_acdc_224_${variant}"
  cmd=(
    "${PYTHON_BIN}"
    test/train_ssr_acdc.py
    --config "${CONFIG}"
    --variant "${variant}"
    --run_name "${run_name}"
    --device "${DEVICE}"
  )

  if [[ -n "${EPOCHS:-}" ]]; then
    cmd+=(--epochs "${EPOCHS}")
  fi
  if [[ -n "${BATCH_SIZE:-}" ]]; then
    cmd+=(--batch_size "${BATCH_SIZE}")
  fi
  if [[ -n "${IMAGE_SIZE:-}" ]]; then
    cmd+=(--image_size "${IMAGE_SIZE}")
  fi
  if [[ -n "${DATA_ROOT:-}" ]]; then
    cmd+=(--data_root "${DATA_ROOT}")
  fi
  if [[ -n "${OUTPUT_ROOT:-}" ]]; then
    cmd+=(--output_root "${OUTPUT_ROOT}")
  fi
  if [[ -n "${NUM_WORKERS:-}" ]]; then
    cmd+=(--num_workers "${NUM_WORKERS}")
  fi

  printf '\n==> Running %s\n' "${variant}"
  printf '    %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
done
