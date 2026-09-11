#!/bin/bash
#SBATCH --account=def-hervete
#SBATCH --job-name=fir_videomae
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=runs/%x-%j.out

cd "${SLURM_SUBMIT_DIR:?run sbatch from the project root}"
source jobs/fir/_common.sh

export VIDEOMAE_CKPT="${VIDEOMAE_CKPT:-$DATA_DIR/videomae/vit_b_k710_dl_from_giant.pth}"
[ -f "$VIDEOMAE_CKPT" ] || { echo "Run: bash data/download_videomae_ckpt.sh"; exit 1; }

OUT="${RUN_ROOT}/videomae_vitb"
python src/train_videomae.py --model vit_base_patch16_224 \
    --finetune "$VIDEOMAE_CKPT" --protocol "$PROTOCOL" --out-dir "$OUT" \
    --epochs 30 --warmup-epochs 2 --batch-size 16 --num-frames 16 \
    --lr 1e-3 --layer-decay 0.75 --weight-decay 0.05 --drop-path 0.1 \
    --workers "$SLURM_CPUS_PER_TASK"
python src/evaluate_videomae.py --ckpt "$OUT/best.pth" --out-dir "$OUT" \
    --workers "$SLURM_CPUS_PER_TASK"
