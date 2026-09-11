#!/bin/bash
# Stable re-run of f32 and f64 two-stream VideoMAE (after f16 is complete).
#
# Fixes vs the original sweep (sweep_videomae_twostream_frames.sh):
#   - LR 5e-4 (was 1e-3)
#   - freeze flow embed 5 epochs (was 3)
#   - grad clip 1.0
#   - OOM-safe batch sizes: f32=2, f64=1
#   - 48 h wall time (was 36 h)
#
# Requires the dense 6 fps subset:
#   sbatch data/build_subset_fps.sh
#
#   FRAMES_FPS=6 sbatch jobs/nibi/sweep_videomae_twostream_frames_stable.sh
#SBATCH --account=def-hervete
#SBATCH --job-name=nibi_vmae2s_stable
#SBATCH --array=0-1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=runs/%x-%A_%a.out

export TARGET_FPS="${FRAMES_FPS:-6}"

cd "${SLURM_SUBMIT_DIR:?run sbatch from the project root}"
source jobs/nibi/_common.sh
source jobs/_helpers.sh

export VIDEOMAE_CKPT="${VIDEOMAE_CKPT:-$DATA_DIR/videomae/vit_b_k710_dl_from_giant.pth}"
if [ ! -f "$VIDEOMAE_CKPT" ]; then
  echo "ERROR: checkpoint not found: $VIDEOMAE_CKPT"
  exit 1
fi

FRAMES=(32 64)
NF="${FRAMES[$SLURM_ARRAY_TASK_ID]}"

case "$NF" in
  32) BATCH=2; ACCUM=2 ;;   # effective batch 4
  64) BATCH=1; ACCUM=4 ;;   # effective batch 4
  *) echo "Unknown NF=$NF" >&2; exit 1 ;;
esac

OUT="${RUN_ROOT}/videomae_twostream_vitb_f${NF}"
echo "==> Stable two-stream rerun | num_frames=$NF | batch=$BATCH | accum=$ACCUM | ${TARGET_FPS}fps | lr=5e-4 | grad_checkpoint"

python src/train_videomae_twostream.py --model vit_base_patch16_224 \
    --finetune "$VIDEOMAE_CKPT" --protocol "$PROTOCOL" --out-dir "$OUT" \
    --fusion concat --epochs 30 --warmup-epochs 2 --batch-size "$BATCH" \
    --num-frames "$NF" --seed 42 \
    --freeze-flow-embed-epochs 5 --lr 5e-4 --grad-clip 1.0 \
    --grad-accum-steps "$ACCUM" --grad-checkpoint \
    --layer-decay 0.75 --weight-decay 0.05 --drop-path 0.1 \
    --workers "$SLURM_CPUS_PER_TASK"

maybe_final_eval_twostream "$OUT" "$BATCH" "$SLURM_CPUS_PER_TASK"
