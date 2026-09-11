#!/bin/bash
# Compare fusion strategies (Slurm array).
#   sbatch jobs/narval/compare_fusion.sh
#SBATCH --account=def-hervete
#SBATCH --job-name=charades_cmp
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=runs/%x-%A_%a.out

cd "${SLURM_SUBMIT_DIR:?run sbatch from the project root}"
source jobs/narval/_common.sh

FUSIONS=(average weighted concat)
FUSION="${FUSIONS[$SLURM_ARRAY_TASK_ID]}"
echo "==> Fusion strategy: $FUSION | protocol: $PROTOCOL"

python src/train.py --stream fusion --fusion "$FUSION" --protocol "$PROTOCOL" \
    --out-dir "${RUN_ROOT}/fusion_${FUSION}" \
    --epochs 25 --batch-size 24 --workers "$SLURM_CPUS_PER_TASK"

python src/evaluate.py --ckpt "${RUN_ROOT}/fusion_${FUSION}/best.pth" \
    --out-dir "${RUN_ROOT}/fusion_${FUSION}" --workers "$SLURM_CPUS_PER_TASK"
