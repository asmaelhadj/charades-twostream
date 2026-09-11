#!/bin/bash
# Full two-stream fps sweep at 3 / 6 / 9 / 12 fps.
#   PROTOCOL=train_test sbatch jobs/narval/sweep_fps.sh
#SBATCH --account=def-hervete
#SBATCH --job-name=charades_fps
#SBATCH --array=0-19
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=runs/%x-%A_%a.out

FPS_LEVELS=(3 6 9 12)
FUSIONS=(average weighted concat)

TASK=$SLURM_ARRAY_TASK_ID
FPS_IDX=$(( TASK / 5 ))
SUB=$(( TASK % 5 ))
export TARGET_FPS="${FPS_LEVELS[$FPS_IDX]}"

cd "${SLURM_SUBMIT_DIR:?run sbatch from the project root}"
source jobs/narval/_common.sh

case "$TARGET_FPS" in
  3)  CHUNK=16; RGB_BATCH=2; FLOW_BATCH=2; FUSION_BATCH=1; GC="" ;;
  6)  CHUNK=16; RGB_BATCH=1; FLOW_BATCH=1; FUSION_BATCH=1; GC="" ;;
  9)  CHUNK=8;  RGB_BATCH=1; FLOW_BATCH=1; FUSION_BATCH=1; GC="--grad-checkpoint" ;;
  12) CHUNK=8;  RGB_BATCH=1; FLOW_BATCH=1; FUSION_BATCH=1; GC="--grad-checkpoint" ;;
  *) echo "Unknown TARGET_FPS=$TARGET_FPS" >&2; exit 1 ;;
esac

COMMON=(--protocol "$PROTOCOL" --temporal-chunk "$CHUNK" --target-fps "$TARGET_FPS"
        --epochs 25 --workers "$SLURM_CPUS_PER_TASK")
[ -n "$GC" ] && COMMON+=("$GC")

case "$SUB" in
  0)
    STREAM=rgb; OUT="${RUN_ROOT}/rgb_${TARGET_FPS}fps"; BATCH=$RGB_BATCH
    TRAIN=(--stream rgb --num-rgb 0 --num-flow 10 --batch-size "$BATCH" --out-dir "$OUT")
    ;;
  1)
    STREAM=flow; OUT="${RUN_ROOT}/flow_${TARGET_FPS}fps"; BATCH=$FLOW_BATCH
    TRAIN=(--stream flow --num-rgb 8 --num-flow 0 --batch-size "$BATCH" --out-dir "$OUT")
    ;;
  2|3|4)
    FUSION="${FUSIONS[$(( SUB - 2 ))]}"
    STREAM="fusion_${FUSION}"; OUT="${RUN_ROOT}/fusion_${FUSION}_${TARGET_FPS}fps"; BATCH=$FUSION_BATCH
    TRAIN=(--stream fusion --fusion "$FUSION" --num-rgb 0 --num-flow 0
           --batch-size "$BATCH" --out-dir "$OUT")
    ;;
  *) echo "Bad array task $TASK" >&2; exit 1 ;;
esac

echo "==> ${STREAM} @ ${TARGET_FPS} fps | batch=$BATCH | protocol=$PROTOCOL"
python src/train.py "${COMMON[@]}" "${TRAIN[@]}"
python src/evaluate.py --ckpt "${OUT}/best.pth" --out-dir "$OUT" \
    --batch-size "$BATCH" --workers "$SLURM_CPUS_PER_TASK"
