#!/bin/bash
# Frame-count sweep for two-stream VideoMAE V2 on Nibi H100.
# Tests num_frames = 16 / 32 / 48 / 64 with batch sizes tuned for dual ViT-B.
#
# Requires the dense 6 fps subset (~180 frames/video):
#   sbatch data/build_subset_fps.sh
#   FRAMES_FPS=6 sbatch jobs/nibi/sweep_videomae_twostream_frames.sh
#SBATCH --account=def-hervete
#SBATCH --job-name=nibi_vmae2s_frames
#SBATCH --array=0-3
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
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

FRAMES=(16 32 48 64)
NF="${FRAMES[$SLURM_ARRAY_TASK_ID]}"

case "$NF" in
  16) BATCH=6 ;;
  32) BATCH=4 ;;
  48) BATCH=2 ;;
  64) BATCH=2 ;;
  *) echo "Unknown NF=$NF" >&2; exit 1 ;;
esac

OUT="${RUN_ROOT}/videomae_twostream_vitb_f${NF}"
echo "==> Two-stream frame sweep | num_frames=$NF | batch=$BATCH | ${TARGET_FPS}fps subset"

python src/train_videomae_twostream.py --model vit_base_patch16_224 \
    --finetune "$VIDEOMAE_CKPT" --protocol "$PROTOCOL" --out-dir "$OUT" \
    --fusion concat --epochs 30 --warmup-epochs 2 --batch-size "$BATCH" \
    --num-frames "$NF" --freeze-flow-embed-epochs 3 \
    --lr 1e-3 --layer-decay 0.75 --weight-decay 0.05 --drop-path 0.1 \
    --workers "$SLURM_CPUS_PER_TASK"

maybe_final_eval_twostream "$OUT" "$BATCH" "$SLURM_CPUS_PER_TASK"
