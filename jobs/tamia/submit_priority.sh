#!/bin/bash
# Submit the three priorities toward ~0.50 macro (keep ViT-B):
#   [1] low-LR fine-tune of the tamIA f64 e75 checkpoint
#   [2] multi-view re-eval of the current best Nibi FT (and [1] after it finishes)
#   [3] RGB-only VideoMAE dense run (VideoGLUE-style) + multi-view eval
#
#   bash jobs/tamia/submit_priority.sh
#   bash jobs/tamia/submit_priority.sh --skip-rgb
#   bash jobs/tamia/submit_priority.sh --skip-ft
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

SKIP_FT=0
SKIP_RGB=0
SKIP_MV=0
for arg in "$@"; do
  case "$arg" in
    --skip-ft)  SKIP_FT=1 ;;
    --skip-rgb) SKIP_RGB=1 ;;
    --skip-mv)  SKIP_MV=1 ;;
    *) echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

TWOSCRIPT=jobs/tamia/train_videomae_twostream_8gpu_h200.sh
EVALSCRIPT=jobs/tamia/eval_multiview_twostream.sh
RGBSCRIPT=jobs/tamia/train_videomae_rgb_8gpu_h200.sh

# Prefer the new tamIA from-scratch run; fall back to Nibi best FT.
TAMIA_E75=runs/train_test/videomae_twostream_vitb_f64_tamia_h200x8_e75/best.pth
NIBI_FT=runs/train_test/videomae_twostream_vitb_f64_ddp2_ft_lr5e5_e100/best.pth

CPUS="${CPUS_PER_TASK:-64}"

echo "================================================================"
echo " tamIA priority jobs → target ~0.50 macro (ViT-B, no model change)"
echo "================================================================"

FT_JOB=""
if [[ "$SKIP_FT" -eq 0 ]]; then
  if [[ ! -f "$TAMIA_E75" ]]; then
    echo "  [1] SKIP FT — missing $TAMIA_E75"
  else
    FT_JOB=$(sbatch --parsable --cpus-per-task="$CPUS" \
      --export=ALL,NUM_FRAMES=64,BATCH_SIZE=4,GRAD_ACCUM_STEPS=1,GRAD_CHECKPOINT=1,FRAMES_FPS=6,EPOCHS=100,LR=5e-5,RESUME="$TAMIA_E75",RUN_SUFFIX=_tamia_h200x8_ft_lr5e5_e100 \
      "$TWOSCRIPT")
    echo "  [1] f64 low-LR FT of tamIA e75     -> job $FT_JOB"
  fi
else
  echo "  [1] SKIP FT (--skip-ft)"
fi

# Multi-view on the current best Nibi FT (immediate, independent).
if [[ "$SKIP_MV" -eq 0 ]]; then
  if [[ -f "$NIBI_FT" ]]; then
    MV1=$(sbatch --parsable \
      --export=ALL,CKPT="$NIBI_FT",FRAMES_FPS=6,NUM_SEGMENT=5,NUM_CROP=3 \
      "$EVALSCRIPT")
    echo "  [2a] multi-view eval Nibi best FT   -> job $MV1"
  else
    echo "  [2a] SKIP multi-view Nibi — missing $NIBI_FT"
  fi

  # Chain multi-view on the new FT when [1] was submitted.
  if [[ -n "$FT_JOB" ]]; then
    FT_OUT=runs/train_test/videomae_twostream_vitb_f64_tamia_h200x8_ft_lr5e5_e100/best.pth
    MV2=$(sbatch --parsable --dependency=afterok:"$FT_JOB" \
      --export=ALL,CKPT="$FT_OUT",FRAMES_FPS=6,NUM_SEGMENT=5,NUM_CROP=3 \
      "$EVALSCRIPT")
    echo "  [2b] multi-view eval after FT       -> job $MV2 (afterok:$FT_JOB)"
  fi
else
  echo "  [2] SKIP multi-view (--skip-mv)"
fi

if [[ "$SKIP_RGB" -eq 0 ]]; then
  RGB_JOB=$(sbatch --parsable --cpus-per-task="$CPUS" \
    --export=ALL,NUM_FRAMES=16,BATCH_SIZE=8,GRAD_ACCUM_STEPS=4,FRAMES_FPS=6,EPOCHS=75,LR=1e-3,WARMUP_EPOCHS=5,RUN_SUFFIX=_tamia_h200x8_rgb,NUM_SEGMENT=5,NUM_CROP=3 \
    "$RGBSCRIPT")
  echo "  [3] RGB VideoMAE dense + 5x3 eval   -> job $RGB_JOB"
else
  echo "  [3] SKIP RGB (--skip-rgb)"
fi

echo ""
echo "Monitor:  squeue -u \$USER"
echo "Results:  python src/summarize.py --protocol train_test"
echo "================================================================"
