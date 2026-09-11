#!/bin/bash
# 8-GPU (DDP) two-stream VideoMAE V2 on a full tamIA H200 node.
#
# tamIA H200 nodes: 8x H200 (141GB each, ~1.75x the memory of an H100-80G).
# Same GPU count as the cancelled Nibi 8xH100 job, but with room for larger
# per-GPU batches (or 64-frame runs without gradient checkpointing).
# Nodes are allocated WHOLE; max walltime on tamIA is 24h.
#
#   sbatch jobs/tamia/train_videomae_twostream_8gpu_h200.sh
#
# Effective batch = BATCH_SIZE x 8 x GRAD_ACCUM_STEPS.
#   default:            10 x 8 x 1 = 80
#   VideoGLUE-like ~256: BATCH_SIZE=8 GRAD_ACCUM_STEPS=4 -> 8 x 8 x 4 = 256
#
#SBATCH --account=aip-hervete
#SBATCH --job-name=tamia_vmae_2s_h200x8
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=23:59:00
#SBATCH --output=runs/%x-%j.out

export TARGET_FPS="${FRAMES_FPS:-}"

cd "${SLURM_SUBMIT_DIR:?run sbatch from the project root}"
source jobs/tamia/_common.sh
source jobs/_helpers.sh

NF="${NUM_FRAMES:-16}"
# 141GB per GPU: batch 10 at 16 frames is conservative; lower if OOM.
BATCH="${BATCH_SIZE:-10}"
SEED="${SEED:-42}"
RUN_SUFFIX="${RUN_SUFFIX:-_tamia_h200x8}"
ACCUM="${GRAD_ACCUM_STEPS:-1}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-5e-4}"
WARMUP="${WARMUP_EPOCHS:-2}"
FREEZE="${FREEZE_FLOW_EMBED_EPOCHS:-5}"
NGPU="${NUM_GPUS:-8}"
OUT="${RUN_ROOT}/videomae_twostream_vitb_f${NF}${RUN_SUFFIX}"

# RESUME=<path/to/best.pth> switches to low-LR fine-tune of a previous run
# (no warmup, no flow-embed freeze), like jobs/nibi resume/ft runs.
RESUME="${RESUME:-}"
if [[ -n "$RESUME" ]]; then
  if [ ! -f "$RESUME" ]; then
    echo "ERROR: resume checkpoint not found: $RESUME"
    exit 1
  fi
  INIT_ARGS=(--resume "$RESUME")
  WARMUP=0
  FREEZE=0
else
  export VIDEOMAE_CKPT="${VIDEOMAE_CKPT:-$DATA_DIR/videomae/vit_b_k710_dl_from_giant.pth}"
  if [ ! -f "$VIDEOMAE_CKPT" ]; then
    echo "ERROR: checkpoint not found: $VIDEOMAE_CKPT"
    echo "No internet on tamIA compute nodes; download it on the login node first."
    exit 1
  fi
  INIT_ARGS=(--finetune "$VIDEOMAE_CKPT")
fi

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

echo "==> tamIA 8x H200 two-stream VideoMAE | GPUs=$NGPU | frames=$NF | per_gpu_batch=$BATCH | accum=$ACCUM | effective_batch=$EFF_BATCH | epochs=$EPOCHS | lr=$LR | warmup=$WARMUP | resume=${RESUME:-none} | scale_lr=${SCALE_LR:-0} | workers/gpu=$WORKERS | out=${OUT##*/}"

torchrun --standalone --nproc_per_node="$NGPU" \
  src/train_videomae_twostream.py --model vit_base_patch16_224 \
    "${INIT_ARGS[@]}" --protocol "$PROTOCOL" --out-dir "$OUT" \
    --fusion concat --epochs "$EPOCHS" --warmup-epochs "$WARMUP" --batch-size "$BATCH" \
    --num-frames "$NF" --seed "$SEED" \
    --freeze-flow-embed-epochs "$FREEZE" --lr "$LR" --grad-clip 1.0 \
    --grad-accum-steps "$ACCUM" $GRAD_CKPT_FLAG $SCALE_LR_FLAG \
    --layer-decay 0.75 --weight-decay 0.05 --drop-path 0.1 \
    --workers "$WORKERS"

maybe_final_eval_twostream "$OUT" "$BATCH" "$WORKERS"
