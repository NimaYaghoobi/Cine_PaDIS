#!/usr/bin/env bash
set -euo pipefail

# --- user settings
NETWORK_DIR=/home/rohan/EDM-FastMRI/models/edm/brain/32dB/00007--uncond-ddpmpp-edm-gpus1-batch4-fp32-container_test
OUTDIR=../mri/mri_complex
IMAGE_DIR=/data/datasets/fastmri/processed_images/
IMAGE_SIZE=384
VIEWS=20
NAME=ct_parbeam
STEPS=1000
SIGMA_MIN=0.003
SIGMA_MAX=10
PAD=64
PSIZE=64
GPUS=2

# single checkpoint to test
CHECKPOINT=650
PKL=$(printf "%s/network-snapshot-%06d.pkl" "$NETWORK_DIR" "$CHECKPOINT")

# list of zeta values to sweep
ZETA_LIST=(
  4.5
  5.0
  5.5
  6.0
  6.5
  7.0
)

SUMMARY=zeta_edmf200_sweep_summary_checkpoint${CHECKPOINT}.csv

# write CSV header
echo "zeta,psnr1,psnr2,psnr3,psnr4,best1,best2,best3,best4,step1,step2,step3,step4" \
  > "$SUMMARY"

mkdir -p logs

for z in "${ZETA_LIST[@]}"; do
  LOG="logs/zeta_${z}.log"
  echo "=== Testing zeta=$z (checkpoint $CHECKPOINT) ==="

  CUDA_VISIBLE_DEVICES="$GPUS" python3 debug_full_image.py \
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
