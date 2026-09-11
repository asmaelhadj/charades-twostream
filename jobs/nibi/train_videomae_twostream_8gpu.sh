#!/bin/bash
# 8-GPU (DDP) two-stream VideoMAE V2 on a full Nibi H100 node.
#
# Nibi GPU nodes: 8x H100 (80GB) + 192 CPU cores + 2 TB RAM per node.
# Requesting all 8 GPUs = whole node (highest single-node performance).
#
#   sbatch jobs/nibi/train_videomae_twostream_8gpu.sh
#
# Effective batch = BATCH_SIZE x 8 x GRAD_ACCUM_STEPS.
#   default:            6 x 8 x 1  = 48
#   VideoGLUE-like ~256: BATCH_SIZE=4 GRAD_ACCUM_STEPS=8  -> 4 x 8 x 8 = 256
#
# Large effective batch usually needs a higher LR and/or more epochs:
#   SCALE_LR=1        -> peak LR x world_size (aggressive; watch for NaN)
#   EPOCHS=50         -> match VideoGLUE-style longer schedule
#
#SBATCH --account=def-hervete
#SBATCH --job-name=nibi_vmae_2s_ddp8
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
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
RUN_SUFFIX="${RUN_SUFFIX:-_ddp8}"
ACCUM="${GRAD_ACCUM_STEPS:-1}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-5e-4}"
WARMUP="${WARMUP_EPOCHS:-2}"
NGPU="${NUM_GPUS:-8}"
OUT="${RUN_ROOT}/videomae_twostream_vitb_f${NF}${RUN_SUFFIX}"

GRAD_CKPT_FLAG=""
if [[ "${GRAD_CHECKPOINT:-0}" == "1" ]]; then
  GRAD_CKPT_FLAG="--grad-checkpoint"
fi

SCALE_LR_FLAG=""
if [[ "${SCALE_LR:-0}" == "1" ]]; then
  SCALE_LR_FLAG="--scale-lr"
fi

# Prefer Slurm-reported GPU count; fall back to nvidia-smi.
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

WORKERS="${SLURM_CPUS_PER_TASK:-64}"
EFF_BATCH=$((BATCH * NGPU * ACCUM))

echo "==> 8-GPU two-stream VideoMAE | GPUs=$NGPU | frames=$NF | per_gpu_batch=$BATCH | accum=$ACCUM | effective_batch=$EFF_BATCH | epochs=$EPOCHS | lr=$LR | warmup=$WARMUP | scale_lr=${SCALE_LR:-0} | out=${OUT##*/}"

torchrun --standalone --nproc_per_node="$NGPU" \
  src/train_videomae_twostream.py --model vit_base_patch16_224 \
    --finetune "$VIDEOMAE_CKPT" --protocol "$PROTOCOL" --out-dir "$OUT" \
    --fusion concat --epochs "$EPOCHS" --warmup-epochs "$WARMUP" --batch-size "$BATCH" \
    --num-frames "$NF" --seed "$SEED" \
    --freeze-flow-embed-epochs 5 --lr "$LR" --grad-clip 1.0 \
    --grad-accum-steps "$ACCUM" $GRAD_CKPT_FLAG $SCALE_LR_FLAG \
    --layer-decay 0.75 --weight-decay 0.05 --drop-path 0.1 \
    --workers "$WORKERS"

maybe_final_eval_twostream "$OUT" "$BATCH" "$WORKERS"
