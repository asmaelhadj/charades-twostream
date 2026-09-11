#!/bin/bash
# Dense controlled frame sweep: sample 24 / 48 / 96 frames from the 12 fps subset.
# Only num_rgb changes; epochs, batch, subset tar, and augmentations are fixed.
#
# Prerequisite (~360 frames stored per video on 12 fps tar):
#   bash jobs/narval/submit_frames_dense.sh build
#
# Train (20 epochs, array tasks 0–2 → 24 / 48 / 96 frames):
#   bash jobs/narval/submit_frames_dense.sh train
#
# Summarize:
#   python src/summarize.py --protocol train_test
#SBATCH --account=def-hervete
#SBATCH --job-name=charades_dense_frames
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=runs/%x-%A_%a.out

cd "${SLURM_SUBMIT_DIR:?run sbatch from the project root}"

export PROTOCOL="${PROTOCOL:-train_test}"
export FRAMES_FPS=12
export COUNTS="24 48 96"
export EPOCHS=20
export FIXED_BATCH="${FIXED_BATCH:-8}"
export TEMPORAL_CHUNK="${TEMPORAL_CHUNK:-16}"
export SWEEP="${SWEEP:-rgb}"
export GRAD_CHECKPOINT=1

source jobs/narval/_common.sh

read -r -a COUNTS_ARR <<< "$COUNTS"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
N="${COUNTS_ARR[$TASK]}"

COMMON=(
  --protocol "$PROTOCOL"
  --epochs "$EPOCHS"
  --batch-size "$FIXED_BATCH"
  --temporal-chunk "$TEMPORAL_CHUNK"
  --workers "$SLURM_CPUS_PER_TASK"
  --grad-checkpoint
)

if [ "$SWEEP" = "rgb" ]; then
  OUT="${RUN_ROOT}/rgb_n${N}_dense_e${EPOCHS}"
  echo "==> Dense RGB | sample $N / ~360 stored | 12fps subset | epochs=$EPOCHS | batch=$FIXED_BATCH"
  python src/train.py --stream rgb "${COMMON[@]}" \
      --num-rgb "$N" --num-flow 10 --out-dir "$OUT"
  python src/evaluate.py --ckpt "${OUT}/best.pth" --out-dir "$OUT" \
      --batch-size "$FIXED_BATCH" --workers "$SLURM_CPUS_PER_TASK"

elif [ "$SWEEP" = "flow" ]; then
  OUT="${RUN_ROOT}/flow_n${N}_dense_e${EPOCHS}"
  echo "==> Dense flow | sample $N pairs | 12fps subset | epochs=$EPOCHS | batch=$FIXED_BATCH"
  python src/train.py --stream flow "${COMMON[@]}" \
      --num-rgb 8 --num-flow "$N" --out-dir "$OUT"
  python src/evaluate.py --ckpt "${OUT}/best.pth" --out-dir "$OUT" \
      --batch-size "$FIXED_BATCH" --workers "$SLURM_CPUS_PER_TASK"

else
  echo "SWEEP=$SWEEP not supported here (use rgb or flow)" >&2
  exit 1
fi
