#!/bin/bash
# VideoMAE RGB sweep: 32 / 64 frames (12 fps subset required).
#   bash jobs/fir/submit_frames_32_64.sh train
#SBATCH --account=def-hervete
#SBATCH --job-name=fir_vmae_f32_64
#SBATCH --array=0-1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=runs/%x-%A_%a.out

export TARGET_FPS=12

cd "${SLURM_SUBMIT_DIR:?run sbatch from the project root}"
source jobs/fir/_common.sh

export VIDEOMAE_CKPT="${VIDEOMAE_CKPT:-$DATA_DIR/videomae/vit_b_k710_dl_from_giant.pth}"
[ -f "$VIDEOMAE_CKPT" ] || { echo "Run: bash data/download_videomae_ckpt.sh"; exit 1; }

FRAMES=(32 64)
NF="${FRAMES[$SLURM_ARRAY_TASK_ID]}"

case "$NF" in
  32) BATCH=8 ;;
  64) BATCH=4 ;;
  *) echo "Unknown NF=$NF" >&2; exit 1 ;;
esac

OUT="${RUN_ROOT}/videomae_vitb_f${NF}"
echo "==> VideoMAE RGB | num_frames=$NF | batch=$BATCH | 12fps subset"

python src/train_videomae.py --model vit_base_patch16_224 \
    --finetune "$VIDEOMAE_CKPT" --protocol "$PROTOCOL" --out-dir "$OUT" \
    --num-frames "$NF" --batch-size "$BATCH" \
    --epochs 30 --warmup-epochs 2 --lr 1e-3 --layer-decay 0.75 \
    --weight-decay 0.05 --drop-path 0.1 --workers "$SLURM_CPUS_PER_TASK"

python src/evaluate_videomae.py --ckpt "$OUT/best.pth" --out-dir "$OUT" \
    --batch-size "$BATCH" --workers "$SLURM_CPUS_PER_TASK"
