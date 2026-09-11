#!/bin/bash
# Frame-count ablation (Slurm array).
#   sbatch jobs/narval/sweep_frames.sh
#   SWEEP=flow sbatch jobs/narval/sweep_frames.sh
#
# For a fair ablation (only frame count changes, fixed batch/hyperparams), use:
#   sbatch jobs/narval/sweep_frames_controlled.sh
#SBATCH --account=def-hervete
#SBATCH --job-name=charades_frames
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=runs/%x-%A_%a.out

cd "${SLURM_SUBMIT_DIR:?run sbatch from the project root}"
source jobs/narval/_common.sh

SWEEP="${SWEEP:-rgb}"
COUNTS=(2 4 8 12 16)
N="${COUNTS[$SLURM_ARRAY_TASK_ID]}"

if [ "$N" -le 8 ]; then
  BATCH=32
elif [ "$N" -le 12 ]; then
  BATCH=24
else
  BATCH=16
fi

if [ "$SWEEP" = "rgb" ]; then
  OUT="${RUN_ROOT}/rgb_n${N}"
  echo "==> RGB sweep | num_rgb=$N | batch=$BATCH | protocol=$PROTOCOL"
  python src/train.py --stream rgb --protocol "$PROTOCOL" \
      --num-rgb "$N" --num-flow 10 --out-dir "$OUT" \
      --epochs 25 --batch-size "$BATCH" --workers "$SLURM_CPUS_PER_TASK"
  python src/evaluate.py --ckpt "${OUT}/best.pth" --out-dir "$OUT" \
      --batch-size "$BATCH" --workers "$SLURM_CPUS_PER_TASK"
elif [ "$SWEEP" = "flow" ]; then
  OUT="${RUN_ROOT}/flow_n${N}"
  echo "==> Flow sweep | num_flow=$N | batch=$BATCH | protocol=$PROTOCOL"
  python src/train.py --stream flow --protocol "$PROTOCOL" \
      --num-rgb 8 --num-flow "$N" --out-dir "$OUT" \
      --epochs 25 --batch-size "$BATCH" --workers "$SLURM_CPUS_PER_TASK"
  python src/evaluate.py --ckpt "${OUT}/best.pth" --out-dir "$OUT" \
      --batch-size "$BATCH" --workers "$SLURM_CPUS_PER_TASK"
else
  echo "Unknown SWEEP=$SWEEP (use rgb or flow)" >&2
  exit 1
fi
