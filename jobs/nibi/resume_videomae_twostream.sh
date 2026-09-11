#!/bin/bash
# OPTIONAL: resume a crashed run mid-training. For a clean reproducible rerun,
# use jobs/nibi/train_videomae_twostream.sh from epoch 1 instead (see REPRODUCE.md).
#SBATCH --account=def-hervete
#SBATCH --job-name=nibi_vmae_2stream_resume
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=runs/%x-%j.out

export TARGET_FPS="${FRAMES_FPS:-}"

cd "${SLURM_SUBMIT_DIR:?run sbatch from the project root}"
source jobs/nibi/_common.sh
source jobs/_helpers.sh

NF="${NUM_FRAMES:-16}"
BATCH="${BATCH_SIZE:-6}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
OUT="${RUN_ROOT}/videomae_twostream_vitb_f${NF}${RUN_SUFFIX}"
RESUME="${RESUME:-$OUT/best.pth}"

if [ ! -f "$RESUME" ]; then
  echo "ERROR: resume checkpoint not found: $RESUME"
  exit 1
fi

LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-30}"
ACCUM="${GRAD_ACCUM_STEPS:-1}"
GRAD_CKPT_FLAG=""
if [[ "${GRAD_CHECKPOINT:-0}" == "1" ]]; then
  GRAD_CKPT_FLAG="--grad-checkpoint"
fi

echo "==> Resume two-stream VideoMAE | from $RESUME | out=${OUT##*/} | lr=$LR | target=$EPOCHS | accum=$ACCUM | grad_ckpt=${GRAD_CHECKPOINT:-0}"

python src/train_videomae_twostream.py --model vit_base_patch16_224 \
    --resume "$RESUME" --protocol "$PROTOCOL" --out-dir "$OUT" \
    --fusion concat --epochs "$EPOCHS" --warmup-epochs 0 --batch-size "$BATCH" \
    --num-frames "$NF" --freeze-flow-embed-epochs 0 \
    --lr "$LR" --grad-clip 1.0 --grad-accum-steps "$ACCUM" $GRAD_CKPT_FLAG \
    --layer-decay 0.75 --weight-decay 0.05 --drop-path 0.1 \
    --workers "$SLURM_CPUS_PER_TASK"

maybe_final_eval_twostream "$OUT" "$BATCH" "$SLURM_CPUS_PER_TASK"
