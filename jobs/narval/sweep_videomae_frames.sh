#!/bin/bash
# VideoMAE V2 frame-count sweep: 16 / 32 / 48 / 64 frames (RGB only).
#   FRAMES_FPS=6 sbatch jobs/narval/sweep_videomae_frames.sh
#SBATCH --account=def-hervete
#SBATCH --job-name=charades_vmae_frames
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=runs/%x-%A_%a.out

export TARGET_FPS="${FRAMES_FPS:-6}"

cd "${SLURM_SUBMIT_DIR:?run sbatch from the project root}"
source jobs/narval/_common.sh

export VIDEOMAE_CKPT="${VIDEOMAE_CKPT:-$DATA_DIR/videomae/vit_b_k710_dl_from_giant.pth}"
if [ ! -f "$VIDEOMAE_CKPT" ]; then
  echo "ERROR: checkpoint not found: $VIDEOMAE_CKPT"
  exit 1
fi

FRAMES=(16 32 48 64)
NF="${FRAMES[$SLURM_ARRAY_TASK_ID]}"

case "$NF" in
  16) BATCH=8 ;;
  32) BATCH=4 ;;
  48) BATCH=2 ;;
  64) BATCH=1 ;;
  *) echo "Unknown NF=$NF" >&2; exit 1 ;;
esac

OUT="${RUN_ROOT}/videomae_vitb_f${NF}"
echo "==> VideoMAE frame sweep | num_frames=$NF | batch=$BATCH | ${TARGET_FPS}fps"

python src/train_videomae.py --model vit_base_patch16_224 \
    --finetune "$VIDEOMAE_CKPT" --protocol "$PROTOCOL" --out-dir "$OUT" \
    --num-frames "$NF" --batch-size "$BATCH" \
    --epochs 30 --warmup-epochs 2 --lr 1e-3 --layer-decay 0.75 \
    --weight-decay 0.05 --drop-path 0.1 --workers "$SLURM_CPUS_PER_TASK"

python src/evaluate_videomae.py --ckpt "$OUT/best.pth" --out-dir "$OUT" \
    --batch-size "$BATCH" --workers "$SLURM_CPUS_PER_TASK"
