#!/bin/bash
# RGB-only VideoMAE V2 baseline on Nibi (for comparison with two-stream).
# Larger batch sizes than Narval thanks to H100 VRAM (80 GB).
#
#   sbatch jobs/nibi/train_videomae_rgb.sh
#SBATCH --account=def-hervete
#SBATCH --job-name=nibi_vmae_rgb
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=runs/%x-%j.out

export TARGET_FPS="${FRAMES_FPS:-}"

cd "${SLURM_SUBMIT_DIR:?run sbatch from the project root}"
source jobs/nibi/_common.sh

export VIDEOMAE_CKPT="${VIDEOMAE_CKPT:-$DATA_DIR/videomae/vit_b_k710_dl_from_giant.pth}"
if [ ! -f "$VIDEOMAE_CKPT" ]; then
  echo "ERROR: checkpoint not found: $VIDEOMAE_CKPT"
  exit 1
fi

NF="${NUM_FRAMES:-16}"
BATCH="${BATCH_SIZE:-12}"
OUT="${RUN_ROOT}/videomae_vitb_f${NF}"

python src/train_videomae.py --model vit_base_patch16_224 \
    --finetune "$VIDEOMAE_CKPT" --protocol "$PROTOCOL" --out-dir "$OUT" \
    --epochs 30 --warmup-epochs 2 --batch-size "$BATCH" --num-frames "$NF" \
    --lr 1e-3 --layer-decay 0.75 --weight-decay 0.05 --drop-path 0.1 \
    --workers "$SLURM_CPUS_PER_TASK"

python src/evaluate_videomae.py --ckpt "$OUT/best.pth" --out-dir "$OUT" \
    --batch-size "$BATCH" --workers "$SLURM_CPUS_PER_TASK"
