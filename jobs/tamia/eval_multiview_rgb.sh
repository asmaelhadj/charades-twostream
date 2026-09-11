#!/bin/bash
# Multi-view re-eval of an RGB VideoMAE checkpoint (student or RGB teacher).
#
#   CKPT=runs/train_test/videomae_vits_distill_from_2s/best.pth \
#     sbatch jobs/tamia/eval_multiview_rgb.sh
#
#SBATCH --account=aip-hervete
#SBATCH --job-name=tamia_eval_rgb
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=6:00:00
#SBATCH --output=runs/%x-%j.out

export TARGET_FPS="${FRAMES_FPS:-6}"

cd "${SLURM_SUBMIT_DIR:?run sbatch from the project root}"
source jobs/tamia/_common.sh
source jobs/_helpers.sh

CKPT="${CKPT:?set CKPT=path/to/best.pth}"
if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT"
  exit 1
fi

OUT_DIR="${OUT_DIR:-$(dirname "$CKPT")}"
SEG="${NUM_SEGMENT:-5}"
CROP="${NUM_CROP:-3}"
BATCH="${BATCH_SIZE:-8}"
WORKERS="${SLURM_CPUS_PER_TASK:-16}"

echo "==> RGB multi-view eval | ckpt=$CKPT | segments=$SEG crops=$CROP | out=$OUT_DIR"
python src/evaluate_videomae.py \
  --ckpt "$CKPT" --out-dir "$OUT_DIR" \
  --num-segment "$SEG" --num-crop "$CROP" \
  --batch-size "$BATCH" --workers "$WORKERS"

echo "==> Done. python src/summarize.py --protocol train_test"
