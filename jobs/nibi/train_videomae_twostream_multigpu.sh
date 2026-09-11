#!/bin/bash
# Multi-GPU (DDP) two-stream VideoMAE V2 on Nibi H100.
#
#   sbatch jobs/nibi/train_videomae_twostream_multigpu.sh
#   NUM_GPUS=4 BATCH_SIZE=6 EPOCHS=50 LR=7e-4 sbatch jobs/nibi/train_videomae_twostream_multigpu.sh
#
# Effective batch = BATCH_SIZE × NUM_GPUS × GRAD_ACCUM_STEPS.
# Optional: SCALE_LR=1, GRAD_CHECKPOINT=1, RESUME=path/to/best.pth
#
#SBATCH --account=def-hervete
#SBATCH --job-name=nibi_vmae_2s_ddp
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
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
RUN_SUFFIX="${RUN_SUFFIX:-_ddp2}"
ACCUM="${GRAD_ACCUM_STEPS:-1}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-5e-4}"
WARMUP="${WARMUP_EPOCHS:-2}"
FREEZE="${FREEZE_FLOW_EMBED_EPOCHS:-5}"
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

RESUME_ARGS=(--finetune "$VIDEOMAE_CKPT")
if [[ -n "${RESUME:-}" ]]; then
  if [[ ! -f "$RESUME" ]]; then
    echo "ERROR: RESUME checkpoint not found: $RESUME"
    exit 1
  fi
  RESUME_ARGS=(--resume "$RESUME")
  FREEZE=0
  WARMUP=0
fi

# Detect allocated GPUs (prefer Slurm env, fall back to nvidia-smi / NUM_GPUS)
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

echo "==> Multi-GPU two-stream | GPUs=$NGPU | frames=$NF | batch=$BATCH | accum=$ACCUM | eff=$EFF_BATCH | epochs=$EPOCHS | lr=$LR | warmup=$WARMUP | freeze=$FREEZE | out=${OUT##*/}"

torchrun --standalone --nproc_per_node="$NGPU" \
  src/train_videomae_twostream.py --model vit_base_patch16_224 \
    "${RESUME_ARGS[@]}" \
    --protocol "$PROTOCOL" --out-dir "$OUT" \
    --fusion concat --epochs "$EPOCHS" --warmup-epochs "$WARMUP" --batch-size "$BATCH" \
    --num-frames "$NF" --seed "$SEED" \
    --freeze-flow-embed-epochs "$FREEZE" --lr "$LR" --grad-clip 1.0 \
    --grad-accum-steps "$ACCUM" $GRAD_CKPT_FLAG $SCALE_LR_FLAG \
    --layer-decay 0.75 --weight-decay 0.05 --drop-path 0.1 \
    --workers "$WORKERS"

maybe_final_eval_twostream "$OUT" "$BATCH" "$WORKERS"
