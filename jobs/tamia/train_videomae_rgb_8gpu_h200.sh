#!/bin/bash
# RGB-only VideoMAE V2 on 8× H200 — VideoGLUE-style dense sampling + large
# effective batch. Uses the same 6 fps JPEG subset (dense enough for multi-clip).
#
#   sbatch jobs/tamia/train_videomae_rgb_8gpu_h200.sh
#
# Defaults aim at VideoGLUE-ish recipe: eff batch 256, lr 1e-3, 75 epochs,
# then multi-view eval (5×3).
#SBATCH --account=aip-hervete
#SBATCH --job-name=tamia_vmae_rgb_h200x8
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=23:59:00
#SBATCH --output=runs/%x-%j.out

export TARGET_FPS="${FRAMES_FPS:-6}"

cd "${SLURM_SUBMIT_DIR:?run sbatch from the project root}"
source jobs/tamia/_common.sh
source jobs/_helpers.sh

NF="${NUM_FRAMES:-16}"
BATCH="${BATCH_SIZE:-8}"
ACCUM="${GRAD_ACCUM_STEPS:-4}"
EPOCHS="${EPOCHS:-75}"
LR="${LR:-1e-3}"
WARMUP="${WARMUP_EPOCHS:-5}"
SEED="${SEED:-42}"
RUN_SUFFIX="${RUN_SUFFIX:-_tamia_h200x8_rgb}"
NGPU="${NUM_GPUS:-8}"
OUT="${RUN_ROOT}/videomae_vitb_f${NF}${RUN_SUFFIX}"

export VIDEOMAE_CKPT="${VIDEOMAE_CKPT:-$DATA_DIR/videomae/vit_b_k710_dl_from_giant.pth}"
if [ ! -f "$VIDEOMAE_CKPT" ]; then
  echo "ERROR: checkpoint not found: $VIDEOMAE_CKPT"
  exit 1
fi

GRAD_CKPT_FLAG=""
if [[ "${GRAD_CHECKPOINT:-0}" == "1" ]]; then
  GRAD_CKPT_FLAG="--grad-checkpoint"
fi

if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  DETECTED="${SLURM_GPUS_ON_NODE##*:}"
  if [[ "$DETECTED" =~ ^[0-9]+$ ]]; then
    NGPU="$DETECTED"
  fi
fi

TOTAL_CPUS="${SLURM_CPUS_ON_NODE:-$(nproc)}"
WORKERS=$(( TOTAL_CPUS / NGPU ))
EFF_BATCH=$((BATCH * NGPU * ACCUM))

echo "==> tamIA RGB VideoMAE | GPUs=$NGPU | frames=$NF | per_gpu_batch=$BATCH | accum=$ACCUM | effective_batch=$EFF_BATCH | epochs=$EPOCHS | lr=$LR | out=${OUT##*/}"

torchrun --standalone --nproc_per_node="$NGPU" \
  src/train_videomae.py --model vit_base_patch16_224 \
    --finetune "$VIDEOMAE_CKPT" --protocol "$PROTOCOL" --out-dir "$OUT" \
    --epochs "$EPOCHS" --warmup-epochs "$WARMUP" --batch-size "$BATCH" \
    --num-frames "$NF" --seed "$SEED" --lr "$LR" \
    --grad-accum-steps "$ACCUM" $GRAD_CKPT_FLAG \
    --layer-decay 0.75 --weight-decay 0.05 --drop-path 0.1 \
    --workers "$WORKERS"

# Multi-view final eval (paper-style). Always run — this is the headline number.
SEG="${NUM_SEGMENT:-5}"
CROP="${NUM_CROP:-3}"
echo "==> Multi-view eval segments=$SEG crops=$CROP"
python src/evaluate_videomae.py --ckpt "$OUT/best.pth" --out-dir "$OUT" \
  --num-segment "$SEG" --num-crop "$CROP" \
  --batch-size "$BATCH" --workers "$WORKERS"
