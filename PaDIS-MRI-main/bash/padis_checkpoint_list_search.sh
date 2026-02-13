#!/usr/bin/env bash
set -euo pipefail

# --- user settings
NETWORK_DIR=/home/rohan/PaDIS/training-runs/00081-aapm_3-uncond-ddpmpp-pedm-gpus1-batch4-fp32
OUTDIR=../mri/mri_complex
IMAGE_DIR=/data/datasets/fastmri/processed_images/
IMAGE_SIZE=384
VIEWS=20
NAME=ct_parbeam
STEPS=110
SIGMA_MIN=0.003
SIGMA_MAX=10
ZETA=3.0
PAD=64
PSIZE=64
GPUS=5

# instead of seq START/INCR/END, list them explicitly here:
CHECKPOINT_LIST=(
  1161
  1281
  1361
  1461
  1581
  1661
  1761
  1841
)

SUMMARY=checkpoint_search_summary_padis200_list.csv

# header: checkpoint, 4 finals, 4 bests, 4 steps
echo "checkpoint,psnr1,psnr2,psnr3,psnr4,best1,best2,best3,best4,step1,step2,step3,step4" \
  > "$SUMMARY"
mkdir -p logs

for ckpt in "${CHECKPOINT_LIST[@]}"; do
  PKL=$(printf "%s/network-snapshot-%06d.pkl" "$NETWORK_DIR" "$ckpt")
  LOG="logs/ckpt_${ckpt}.log"

  echo "=== Testing checkpoint $ckpt ==="
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
    --zeta="$ZETA" \
    --pad="$PAD" \
    --psize="$PSIZE" \
  2>&1 | tee "$LOG"

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
    "$ckpt" \
    "${PSNRS[0]}" "${PSNRS[1]}" "${PSNRS[2]}" "${PSNRS[3]}" \
    "${BESTPS[0]}" "${BESTPS[1]}" "${BESTPS[2]}" "${BESTPS[3]}" \
    "${BESTSTEPS[0]}" "${BESTSTEPS[1]}" "${BESTSTEPS[2]}" "${BESTSTEPS[3]}" \
    >> "$SUMMARY"
done

echo "Done. Results in $SUMMARY"
