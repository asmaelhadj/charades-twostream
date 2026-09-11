#!/bin/bash
#SBATCH --account=def-hervete
#SBATCH --job-name=fir_vmae_2s
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=runs/%x-%j.out

export TARGET_FPS="${FRAMES_FPS:-}"

cd "${SLURM_SUBMIT_DIR:?run sbatch from the project root}"
source jobs/fir/_common.sh
source jobs/_helpers.sh

export VIDEOMAE_CKPT="${VIDEOMAE_CKPT:-$DATA_DIR/videomae/vit_b_k710_dl_from_giant.pth}"
[ -f "$VIDEOMAE_CKPT" ] || { echo "Run: bash data/download_videomae_ckpt.sh"; exit 1; }

NF="${NUM_FRAMES:-16}"
BATCH="${BATCH_SIZE:-8}"
SEED="${SEED:-42}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
OUT="${RUN_ROOT}/videomae_twostream_vitb_f${NF}${RUN_SUFFIX}"

echo "==> Two-stream VideoMAE (epoch 1→30) | frames=$NF | batch=$BATCH | seed=$SEED | out=${OUT##*/}"

python src/train_videomae_twostream.py --model vit_base_patch16_224 \
    --finetune "$VIDEOMAE_CKPT" --protocol "$PROTOCOL" --out-dir "$OUT" \
    --fusion concat --epochs 30 --warmup-epochs 2 --batch-size "$BATCH" \
    --num-frames "$NF" --seed "$SEED" \
    --freeze-flow-embed-epochs 5 --lr 5e-4 --grad-clip 1.0 \
    --layer-decay 0.75 --weight-decay 0.05 --drop-path 0.1 \
    --workers "$SLURM_CPUS_PER_TASK"
maybe_final_eval_twostream "$OUT" "$BATCH" "$SLURM_CPUS_PER_TASK"
