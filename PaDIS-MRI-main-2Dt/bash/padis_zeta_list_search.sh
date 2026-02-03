#!/usr/bin/env bash
set -euo pipefail

# --- user settings
NETWORK_DIR=/home/rohan/PaDIS/training-runs/00086-aapm_3-uncond-ddpmpp-pedm-gpus1-batch4-fp32
OUTDIR=../mri/mri_complex
IMAGE_DIR=/data/datasets/fastmri/processed_images/
IMAGE_SIZE=384
VIEWS=20
NAME=ct_parbeam
STEPS=104
SIGMA_MIN=0.003
SIGMA_MAX=10
PAD=64
PSIZE=64
GPUS=0

# single checkpoint to test
CHECKPOINT=900
PKL=$(printf "%s/network-snapshot-%06d.pkl" "$NETWORK_DIR" "$CHECKPOINT")

# list of zeta values to sweep
ZETA_LIST=(
  5.0
  5.5
  6.0
  6.5
)

SUMMARY=zeta_sweep_summary_checkpoint${CHECKPOINT}.csv

# write CSV header
echo "zeta,psnr1,psnr2,psnr3,psnr4,best1,best2,best3,best4,step1,step2,step3,step4" \
  > "$SUMMARY"

mkdir -p logs

for z in "${ZETA_LIST[@]}"; do
  LOG="logs/zeta_${z}.log"
  echo "=== Testing zeta=$z (checkpoint $CHECKPOINT) ==="

  CUDA_VISIBLE_DEVICES="$GPUS" python3 inverse_nodist_mri_DIFF_SEED.py \
    --network="$PKL" \
    --outdir="$OUTDIR" \
    --image_dir="$IMAGE_DIR" \
    --image_size="$IMAGE_SIZE" \
    --views="$VIEWS" \
    --name="$NAME" \
    --steps="$STEPS" \
    --sigma_min="$SIGMA_MIN" \
    --sigma_max="$SIGMA_MAX" \
    --zeta="$z" \
    --pad="$PAD" \
    --psize="$PSIZE" 2>&1 | tee "$LOG"

  # collect exactly four METRICS lines
  mapfile -t METRICS < <(grep '^METRICS' "$LOG")
  if [ "${#METRICS[@]}" -ne 4 ]; then
    echo "ERROR: expected 4 METRICS lines, got ${#METRICS[@]}" >&2
    exit 1
  fi

  PSNRS=(); BESTPS=(); BESTSTEPS=()
  for line in "${METRICS[@]}"; do
    # METRICS,checkpoint,final_psnr,best_psnr,best_step
    IFS=',' read -r _ c fpsnr bpsnr bstp <<< "$line"
    PSNRS+=("$fpsnr")
    BESTPS+=("$bpsnr")
    BESTSTEPS+=("$bstp")
  done

  # append one CSV row
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$z" \
    "${PSNRS[0]}" "${PSNRS[1]}" "${PSNRS[2]}" "${PSNRS[3]}" \
    "${BESTPS[0]}" "${BESTPS[1]}" "${BESTPS[2]}" "${BESTPS[3]}" \
    "${BESTSTEPS[0]}" "${BESTSTEPS[1]}" "${BESTSTEPS[2]}" "${BESTSTEPS[3]}" \
    >> "$SUMMARY"
done

echo "Done. Results in $SUMMARY"
