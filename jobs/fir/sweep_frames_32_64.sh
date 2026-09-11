#!/bin/bash
# Controlled CNN sweep: sample 32 / 64 frames from 12 fps subset.
#   SWEEP=rgb    sbatch jobs/fir/sweep_frames_32_64.sh
#   SWEEP=flow   sbatch jobs/fir/sweep_frames_32_64.sh
#   SWEEP=fusion sbatch jobs/fir/sweep_frames_32_64.sh   # uses FUSION=average|weighted|concat
#SBATCH --account=def-hervete
#SBATCH --job-name=fir_frames_32_64
#SBATCH --array=0-1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=runs/%x-%A_%a.out

cd "${SLURM_SUBMIT_DIR:?run sbatch from the project root}"

export PROTOCOL="${PROTOCOL:-train_test}"
export TARGET_FPS=12
export EPOCHS="${EPOCHS:-20}"
SWEEP="${SWEEP:-rgb}"
FUSION="${FUSION:-average}"

source jobs/fir/_common.sh

COUNTS=(32 64)
N="${COUNTS[$SLURM_ARRAY_TASK_ID]}"

case "$SWEEP" in
  rgb)
    case "$N" in 32) BATCH=16 ;; 64) BATCH=8 ;; esac
    OUT="${RUN_ROOT}/rgb_n${N}_dense_e${EPOCHS}"
    TRAIN=(--stream rgb --num-rgb "$N" --num-flow 10)
    ;;
  flow)
    case "$N" in 32) BATCH=16 ;; 64) BATCH=8 ;; esac
    OUT="${RUN_ROOT}/flow_n${N}_dense_e${EPOCHS}"
    TRAIN=(--stream flow --num-rgb 8 --num-flow "$N")
    ;;
  fusion)
    case "$N" in 32) BATCH=6 ;; 64) BATCH=3 ;; esac
    OUT="${RUN_ROOT}/fusion_${FUSION}_n${N}_dense_e${EPOCHS}"
    TRAIN=(--stream fusion --fusion "$FUSION" --num-rgb "$N" --num-flow "$N")
    ;;
  *)
    echo "Unknown SWEEP=$SWEEP (use rgb, flow, or fusion)" >&2
    exit 1
    ;;
esac

COMMON=(--protocol "$PROTOCOL" --epochs "$EPOCHS" --batch-size "$BATCH"
        --temporal-chunk 16 --workers "$SLURM_CPUS_PER_TASK" --grad-checkpoint)

echo "==> $SWEEP | n=$N | 12fps subset | epochs=$EPOCHS | batch=$BATCH | fusion=${FUSION:-n/a}"

python src/train.py "${TRAIN[@]}" "${COMMON[@]}" --out-dir "$OUT"
python src/evaluate.py --ckpt "${OUT}/best.pth" --out-dir "$OUT" \
    --batch-size "$BATCH" --workers "$SLURM_CPUS_PER_TASK"
