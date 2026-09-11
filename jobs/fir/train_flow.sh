#!/bin/bash
#SBATCH --account=def-hervete
#SBATCH --job-name=fir_flow
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=runs/%x-%j.out

cd "${SLURM_SUBMIT_DIR:?run sbatch from the project root}"
source jobs/fir/_common.sh

python src/train.py --stream flow --protocol "$PROTOCOL" --out-dir "${RUN_ROOT}/flow" \
    --epochs 25 --batch-size 32 --workers "$SLURM_CPUS_PER_TASK"
python src/evaluate.py --ckpt "${RUN_ROOT}/flow/best.pth" --out-dir "${RUN_ROOT}/flow" \
    --workers "$SLURM_CPUS_PER_TASK"
