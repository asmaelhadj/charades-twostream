#!/bin/bash
# Two-stream late fusion (default: concat).
#   sbatch jobs/narval/train_fusion.sh
#SBATCH --account=def-hervete
#SBATCH --job-name=charades_fusion
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=runs/%x-%j.out

cd "${SLURM_SUBMIT_DIR:?run sbatch from the project root}"
source jobs/narval/_common.sh

FUSION="${1:-concat}"

python src/train.py --stream fusion --fusion "$FUSION" --protocol "$PROTOCOL" \
    --out-dir "${RUN_ROOT}/fusion_${FUSION}" \
    --epochs 25 --batch-size 24 --workers "$SLURM_CPUS_PER_TASK"

python src/evaluate.py --ckpt "${RUN_ROOT}/fusion_${FUSION}/best.pth" \
    --out-dir "${RUN_ROOT}/fusion_${FUSION}" --workers "$SLURM_CPUS_PER_TASK"
