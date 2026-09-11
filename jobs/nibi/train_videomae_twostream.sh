#!/bin/bash
# Reproducible two-stream VideoMAE V2 training from EPOCH 1 on Nibi H100.
#
#   sbatch jobs/nibi/train_videomae_twostream.sh
#
# Hyperparameters are pinned in src/train_videomae_twostream.py defaults + below.
# Full recipe: docs/nibi/REPRODUCE.md
#SBATCH --account=def-hervete
#SBATCH --job-name=nibi_vmae_2stream
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=runs/%x-%j.out

export TARGET_FPS="${FRAMES_FPS:-}"

cd "${SLURM_SUBMIT_DIR:?run sbatch from the project root}"
source jobs/nibi/_common.sh
source jobs/_helpers.sh

export VIDEOMAE_CKPT="${VIDEOMAE_CKPT:-$DATA_DIR/videomae/vit_b_k710_dl_from_giant.pth}"
if [ ! -f "$VIDEOMAE_CKPT" ]; then
  echo "ERROR: checkpoint not found: $VIDEOMAE_CKPT"
  exit 1
fi

NF="${NUM_FRAMES:-16}"
BATCH="${BATCH_SIZE:-6}"
SEED="${SEED:-42}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
ACCUM="${GRAD_ACCUM_STEPS:-1}"
OUT="${RUN_ROOT}/videomae_twostream_vitb_f${NF}${RUN_SUFFIX}"

GRAD_CKPT_FLAG=""
if [[ "${GRAD_CHECKPOINT:-0}" == "1" ]]; then
  GRAD_CKPT_FLAG="--grad-checkpoint"
fi

echo "==> Two-stream VideoMAE (epoch 1→30) | frames=$NF | batch=$BATCH | accum=$ACCUM | seed=$SEED | out=${OUT##*/}"

python src/train_videomae_twostream.py --model vit_base_patch16_224 \
    --finetune "$VIDEOMAE_CKPT" --protocol "$PROTOCOL" --out-dir "$OUT" \
    --fusion concat --epochs 30 --warmup-epochs 2 --batch-size "$BATCH" \
    --num-frames "$NF" --seed "$SEED" \
    --freeze-flow-embed-epochs 5 --lr 5e-4 --grad-clip 1.0 \
    --grad-accum-steps "$ACCUM" $GRAD_CKPT_FLAG \
    --layer-decay 0.75 --weight-decay 0.05 --drop-path 0.1 \
    --workers "$SLURM_CPUS_PER_TASK"

maybe_final_eval_twostream "$OUT" "$BATCH" "$SLURM_CPUS_PER_TASK"
