#!/bin/bash
# Multi-GPU (DDP) two-stream VideoMAE V2 on Fir H100.
#
#   sbatch jobs/fir/train_videomae_twostream_multigpu.sh
#   NUM_GPUS=2 BATCH_SIZE=8 SCALE_LR=1 sbatch jobs/fir/train_videomae_twostream_multigpu.sh
#
#SBATCH --account=def-hervete
#SBATCH --job-name=fir_vmae_2s_ddp
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=runs/%x-%j.out

export TARGET_FPS="${FRAMES_FPS:-}"

cd "${SLURM_SUBMIT_DIR:?run sbatch from the project root}"
source jobs/fir/_common.sh
source jobs/_helpers.sh

export VIDEOMAE_CKPT="${VIDEOMAE_CKPT:-$DATA_DIR/videomae/vit_b_k710_dl_from_giant.pth}"
[ -f "$VIDEOMAE_CKPT" ] || { echo "Run: bash data/download_videomae_ckpt.sh"; exit 1; }

NF="${NUM_FRAMES:-16}"
BATCH="${BATCH_SIZE:-8}"
SEED="${SEED:-42}"
RUN_SUFFIX="${RUN_SUFFIX:-_ddp2}"
ACCUM="${GRAD_ACCUM_STEPS:-1}"
NGPU="${NUM_GPUS:-2}"
OUT="${RUN_ROOT}/videomae_twostream_vitb_f${NF}${RUN_SUFFIX}"

GRAD_CKPT_FLAG=""
if [[ "${GRAD_CHECKPOINT:-0}" == "1" ]]; then
  GRAD_CKPT_FLAG="--grad-checkpoint"
fi

SCALE_LR_FLAG=""
if [[ "${SCALE_LR:-0}" == "1" ]]; then
  SCALE_LR_FLAG="--scale-lr"
fi

if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  DETECTED="${SLURM_GPUS_ON_NODE##*:}"
  if [[ "$DETECTED" =~ ^[0-9]+$ ]]; then
    NGPU="$DETECTED"
  fi
elif command -v nvidia-smi >/dev/null 2>&1; then
  DETECTED=$(nvidia-smi -L 2>/dev/null | wc -l)
  if [[ "$DETECTED" -ge 1 ]]; then
    NGPU="$DETECTED"
  fi
fi

WORKERS="${SLURM_CPUS_PER_TASK:-16}"
EFF_BATCH=$((BATCH * NGPU * ACCUM))

echo "==> Multi-GPU two-stream VideoMAE (Fir) | GPUs=$NGPU | frames=$NF | per_gpu_batch=$BATCH | effective_batch=$EFF_BATCH | out=${OUT##*/}"

torchrun --standalone --nproc_per_node="$NGPU" \
  src/train_videomae_twostream.py --model vit_base_patch16_224 \
    --finetune "$VIDEOMAE_CKPT" --protocol "$PROTOCOL" --out-dir "$OUT" \
    --fusion concat --epochs 30 --warmup-epochs 2 --batch-size "$BATCH" \
    --num-frames "$NF" --seed "$SEED" \
    --freeze-flow-embed-epochs 5 --lr 5e-4 --grad-clip 1.0 \
    --grad-accum-steps "$ACCUM" $GRAD_CKPT_FLAG $SCALE_LR_FLAG \
    --layer-decay 0.75 --weight-decay 0.05 --drop-path 0.1 \
    --workers "$WORKERS"

maybe_final_eval_twostream "$OUT" "$BATCH" "$WORKERS"
