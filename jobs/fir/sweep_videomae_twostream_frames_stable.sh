#!/bin/bash
# Stable two-stream f32/f64 on Fir H100 (12 fps dense subset).
#
#   FRAMES_FPS=12 sbatch jobs/fir/sweep_videomae_twostream_frames_stable.sh
#SBATCH --account=def-hervete
#SBATCH --job-name=fir_vmae2s_stable
#SBATCH --array=0-1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=runs/%x-%A_%a.out

export TARGET_FPS="${FRAMES_FPS:-12}"

cd "${SLURM_SUBMIT_DIR:?run sbatch from the project root}"
source jobs/fir/_common.sh
source jobs/_helpers.sh

export VIDEOMAE_CKPT="${VIDEOMAE_CKPT:-$DATA_DIR/videomae/vit_b_k710_dl_from_giant.pth}"
[ -f "$VIDEOMAE_CKPT" ] || { echo "Run: bash data/download_videomae_ckpt.sh"; exit 1; }

FRAMES=(32 64)
NF="${FRAMES[$SLURM_ARRAY_TASK_ID]}"

case "$NF" in
  32) BATCH=2; ACCUM=2 ;;
  64) BATCH=1; ACCUM=4 ;;
  *) echo "Unknown NF=$NF" >&2; exit 1 ;;
esac

OUT="${RUN_ROOT}/videomae_twostream_vitb_f${NF}"
echo "==> Stable two-stream | num_frames=$NF | batch=$BATCH | accum=$ACCUM | ${TARGET_FPS}fps | lr=5e-4 | grad_checkpoint"

python src/train_videomae_twostream.py --model vit_base_patch16_224 \
    --finetune "$VIDEOMAE_CKPT" --protocol "$PROTOCOL" --out-dir "$OUT" \
    --fusion concat --epochs 30 --warmup-epochs 2 --batch-size "$BATCH" \
    --num-frames "$NF" --seed 42 \
    --freeze-flow-embed-epochs 5 --lr 5e-4 --grad-clip 1.0 \
    --grad-accum-steps "$ACCUM" --grad-checkpoint \
    --layer-decay 0.75 --weight-decay 0.05 --drop-path 0.1 \
    --workers "$SLURM_CPUS_PER_TASK"

maybe_final_eval_twostream "$OUT" "$BATCH" "$SLURM_CPUS_PER_TASK"
