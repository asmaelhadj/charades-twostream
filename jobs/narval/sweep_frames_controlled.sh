#!/bin/bash
# Controlled frame-count ablation: ONLY num_rgb / num_flow changes.
#
# Everything else is fixed:
#   - same compact subset (default 16 frames/video, or FRAMES_FPS dense tar)
#   - batch size, epochs, lr, temporal_chunk, protocol, augmentations
#   - frames are evenly sampled + random jitter (dataset._pick), never "use all"
#
# Usage:
#   sbatch jobs/narval/sweep_frames_controlled.sh
#   SWEEP=flow sbatch jobs/narval/sweep_frames_controlled.sh
#   COUNTS="2 4 8 12 16" FIXED_BATCH=32 sbatch jobs/narval/sweep_frames_controlled.sh
#
# For more than 16 sampled frames, build a dense subset first:
#   bash jobs/narval/submit_frames_dense.sh build
#   bash jobs/narval/submit_frames_dense.sh train    # 24 / 48 / 96 frames, 20 epochs
#
# Do NOT use sweep_fps.sh for this question — that script changes batch size,
# uses num_rgb=0 (all frames), and reads different subset tars per fps level.
#SBATCH --account=def-hervete
#SBATCH --job-name=charades_frames_ctrl
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=runs/%x-%A_%a.out

cd "${SLURM_SUBMIT_DIR:?run sbatch from the project root}"
source jobs/narval/_common.sh

SWEEP="${SWEEP:-rgb}"
FIXED_BATCH="${FIXED_BATCH:-32}"
EPOCHS="${EPOCHS:-25}"
TEMPORAL_CHUNK="${TEMPORAL_CHUNK:-16}"

# Optional: FRAMES_FPS=12 loads charades_2stream_subset_12fps.tar (~360 frames/video).
export TARGET_FPS="${FRAMES_FPS:-}"

if [ -n "${COUNTS:-}" ]; then
  read -r -a COUNTS_ARR <<< "$COUNTS"
else
  COUNTS_ARR=(2 4 8 12 16)
fi

TASK="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$TASK" -ge "${#COUNTS_ARR[@]}" ]; then
  echo "Array task $TASK out of range for COUNTS (${COUNTS_ARR[*]})" >&2
  exit 1
fi
N="${COUNTS_ARR[$TASK]}"

COMMON=(
  --protocol "$PROTOCOL"
  --epochs "$EPOCHS"
  --batch-size "$FIXED_BATCH"
  --temporal-chunk "$TEMPORAL_CHUNK"
  --workers "$SLURM_CPUS_PER_TASK"
)
[ -n "${GRAD_CHECKPOINT:-}" ] && COMMON+=(--grad-checkpoint)

if [ "$SWEEP" = "rgb" ]; then
  OUT="${RUN_ROOT}/rgb_n${N}_ctrl"
  echo "==> Controlled RGB | num_rgb=$N | batch=$FIXED_BATCH | chunk=$TEMPORAL_CHUNK | subset=${TARGET_FPS:-default16}"
  python src/train.py --stream rgb "${COMMON[@]}" \
      --num-rgb "$N" --num-flow 10 --out-dir "$OUT"
  python src/evaluate.py --ckpt "${OUT}/best.pth" --out-dir "$OUT" \
      --batch-size "$FIXED_BATCH" --workers "$SLURM_CPUS_PER_TASK"

elif [ "$SWEEP" = "flow" ]; then
  OUT="${RUN_ROOT}/flow_n${N}_ctrl"
  echo "==> Controlled flow | num_flow=$N | batch=$FIXED_BATCH | chunk=$TEMPORAL_CHUNK | subset=${TARGET_FPS:-default16}"
  python src/train.py --stream flow "${COMMON[@]}" \
      --num-rgb 8 --num-flow "$N" --out-dir "$OUT"
  python src/evaluate.py --ckpt "${OUT}/best.pth" --out-dir "$OUT" \
      --batch-size "$FIXED_BATCH" --workers "$SLURM_CPUS_PER_TASK"

elif [ "$SWEEP" = "fusion" ]; then
  OUT="${RUN_ROOT}/fusion_average_n${N}_ctrl"
  echo "==> Controlled fusion (average) | num_rgb=$N num_flow=$N | batch=$FIXED_BATCH"
  python src/train.py --stream fusion --fusion average "${COMMON[@]}" \
      --num-rgb "$N" --num-flow "$N" --out-dir "$OUT" \
      --batch-size "${FUSION_BATCH:-24}"
  python src/evaluate.py --ckpt "${OUT}/best.pth" --out-dir "$OUT" \
      --batch-size "${FUSION_BATCH:-24}" --workers "$SLURM_CPUS_PER_TASK"

else
  echo "Unknown SWEEP=$SWEEP (use rgb, flow, or fusion)" >&2
  exit 1
fi
