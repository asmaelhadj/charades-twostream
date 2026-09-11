#!/bin/bash
# Controlled 24 / 48 / 96 frame sweep, 20 epochs, 12 fps subset.
#SBATCH --account=def-hervete
#SBATCH --job-name=fir_dense_frames
#SBATCH --array=0-2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=runs/%x-%A_%a.out

cd "${SLURM_SUBMIT_DIR:?run sbatch from the project root}"

export PROTOCOL="${PROTOCOL:-train_test}"
export FRAMES_FPS=12
export TARGET_FPS=12
export COUNTS="24 48 96"
export EPOCHS=20
export FIXED_BATCH="${FIXED_BATCH:-16}"
export TEMPORAL_CHUNK=16
export SWEEP="${SWEEP:-rgb}"
export GRAD_CHECKPOINT=1

source jobs/fir/_common.sh

read -r -a COUNTS_ARR <<< "$COUNTS"
N="${COUNTS_ARR[$SLURM_ARRAY_TASK_ID]}"

COMMON=(--protocol "$PROTOCOL" --epochs "$EPOCHS" --batch-size "$FIXED_BATCH"
        --temporal-chunk "$TEMPORAL_CHUNK" --workers "$SLURM_CPUS_PER_TASK" --grad-checkpoint)

OUT="${RUN_ROOT}/rgb_n${N}_dense_e${EPOCHS}"
python src/train.py --stream rgb "${COMMON[@]}" --num-rgb "$N" --num-flow 10 --out-dir "$OUT"
python src/evaluate.py --ckpt "${OUT}/best.pth" --out-dir "$OUT" \
    --batch-size "$FIXED_BATCH" --workers "$SLURM_CPUS_PER_TASK"
