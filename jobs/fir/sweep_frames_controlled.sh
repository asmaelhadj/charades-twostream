#!/bin/bash
# Frame-count ablation: only num_rgb changes (2/4/8/12/16), fixed batch=32.
#SBATCH --account=def-hervete
#SBATCH --job-name=fir_frames_ctrl
#SBATCH --array=0-4
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=runs/%x-%A_%a.out

cd "${SLURM_SUBMIT_DIR:?run sbatch from the project root}"
source jobs/fir/_common.sh

COUNTS=(2 4 8 12 16)
N="${COUNTS[$SLURM_ARRAY_TASK_ID]}"
OUT="${RUN_ROOT}/rgb_n${N}_ctrl"

python src/train.py --stream rgb --protocol "$PROTOCOL" --epochs 25 \
    --batch-size 32 --temporal-chunk 16 --num-rgb "$N" --num-flow 10 \
    --out-dir "$OUT" --workers "$SLURM_CPUS_PER_TASK"
python src/evaluate.py --ckpt "${OUT}/best.pth" --out-dir "$OUT" \
    --batch-size 32 --workers "$SLURM_CPUS_PER_TASK"
