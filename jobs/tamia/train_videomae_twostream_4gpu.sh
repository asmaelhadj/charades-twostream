#!/bin/bash
# 4-GPU (DDP) two-stream VideoMAE V2 on a full tamIA node.
#
# tamIA GPU nodes: 4x H100 SXM (80GB). Nodes are allocated WHOLE and every
# job must use all 4 GPUs of each node it gets -> this script always takes
# one full node. Max walltime on tamIA is 24h.
#
#   sbatch jobs/tamia/train_videomae_twostream_4gpu.sh
#
# Effective batch = BATCH_SIZE x 4 x GRAD_ACCUM_STEPS.
#   default:            6 x 4 x 1  = 24
#   VideoGLUE-like ~256: BATCH_SIZE=4 GRAD_ACCUM_STEPS=16 -> 4 x 4 x 16 = 256
#
#SBATCH --account=aip-hervete
#SBATCH --job-name=tamia_vmae_2s_ddp4
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
# tamIA gives whole nodes but only assigns the CPUs you request; without this
# the job gets 1 CPU -> 0 dataloader workers -> NCCL timeout. H200 nodes have
# 64 CPUs; 48 here is conservative for H100 nodes — raise after checking
# `scontrol show node <h100-node> | grep CPUTot`.
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=23:59:00
#SBATCH --output=runs/%x-%j.out

export TARGET_FPS="${FRAMES_FPS:-}"

cd "${SLURM_SUBMIT_DIR:?run sbatch from the project root}"
source jobs/tamia/_common.sh
source jobs/_helpers.sh

export VIDEOMAE_CKPT="${VIDEOMAE_CKPT:-$DATA_DIR/videomae/vit_b_k710_dl_from_giant.pth}"
if [ ! -f "$VIDEOMAE_CKPT" ]; then
  echo "ERROR: checkpoint not found: $VIDEOMAE_CKPT"
  echo "No internet on tamIA compute nodes; download it on the login node first."
  exit 1
fi

NF="${NUM_FRAMES:-16}"
BATCH="${BATCH_SIZE:-6}"
SEED="${SEED:-42}"
RUN_SUFFIX="${RUN_SUFFIX:-_tamia_ddp4}"
ACCUM="${GRAD_ACCUM_STEPS:-1}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-5e-4}"
WARMUP="${WARMUP_EPOCHS:-2}"
NGPU="${NUM_GPUS:-4}"
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

# Whole node -> all CPUs are ours; split dataloader workers across GPUs.
TOTAL_CPUS="${SLURM_CPUS_ON_NODE:-$(nproc)}"
WORKERS=$(( TOTAL_CPUS / NGPU ))
EFF_BATCH=$((BATCH * NGPU * ACCUM))

echo "==> tamIA 4-GPU two-stream VideoMAE | GPUs=$NGPU | frames=$NF | per_gpu_batch=$BATCH | accum=$ACCUM | effective_batch=$EFF_BATCH | epochs=$EPOCHS | lr=$LR | warmup=$WARMUP | scale_lr=${SCALE_LR:-0} | workers/gpu=$WORKERS | out=${OUT##*/}"

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
