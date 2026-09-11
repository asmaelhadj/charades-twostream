#!/bin/bash
# Targeted hyperparameter sweep to beat f32_ddp4_lr7e4_e50 (~0.417 macro).
#
# Strategy (based on prior results):
#   - Prefer 4x H100 (8x nodes rarely idle)
#   - Keep f32 as main axis (best so far)
#   - Longer schedule + mild LR variants + resume fine-tune
#   - One f64 longer run (best weighted so far)
#
#   bash jobs/nibi/submit_improve_sweep.sh
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

SBATCH_4="--gres=gpu:h100:4 --cpus-per-task=32 --mem=256G --time=2-00:00:00"
SCRIPT=jobs/nibi/train_videomae_twostream_multigpu.sh

echo "================================================================"
echo " Improve sweep — submit targeted 4-GPU jobs"
echo " Baseline to beat: f32_ddp4_lr7e4_e50  macro≈0.417"
echo "================================================================"

# 1) f32: same recipe, longer (75 epochs) — most promising
J1=$(sbatch --parsable $SBATCH_4 \
  --export=ALL,NUM_GPUS=4,NUM_FRAMES=32,BATCH_SIZE=2,GRAD_ACCUM_STEPS=4,GRAD_CHECKPOINT=1,FRAMES_FPS=6,EPOCHS=75,LR=7e-4,WARMUP_EPOCHS=5,RUN_SUFFIX=_ddp4_lr7e4_e75 \
  "$SCRIPT")
echo "  [1] f32 longer e75 lr7e4     → job $J1"

# 2) f32: slightly higher LR (still below old NaN 1e-3 regime)
J2=$(sbatch --parsable $SBATCH_4 \
  --export=ALL,NUM_GPUS=4,NUM_FRAMES=32,BATCH_SIZE=2,GRAD_ACCUM_STEPS=4,GRAD_CHECKPOINT=1,FRAMES_FPS=6,EPOCHS=50,LR=9e-4,WARMUP_EPOCHS=5,RUN_SUFFIX=_ddp4_lr9e4_e50 \
  "$SCRIPT")
echo "  [2] f32 lr9e4 e50            → job $J2"

# 3) f32: lower LR, longer — smoother late training
J3=$(sbatch --parsable $SBATCH_4 \
  --export=ALL,NUM_GPUS=4,NUM_FRAMES=32,BATCH_SIZE=2,GRAD_ACCUM_STEPS=4,GRAD_CHECKPOINT=1,FRAMES_FPS=6,EPOCHS=75,LR=5e-4,WARMUP_EPOCHS=5,RUN_SUFFIX=_ddp4_lr5e4_e75 \
  "$SCRIPT")
echo "  [3] f32 lr5e4 e75            → job $J3"

# 4) Resume best f32 with low LR fine-tune (cheap, often helps)
BEST_F32="runs/${PROTOCOL:-train_test}/videomae_twostream_vitb_f32_ddp4_lr7e4_e50/best.pth"
if [[ -f "$BEST_F32" ]]; then
  J4=$(sbatch --parsable $SBATCH_4 \
    --export=ALL,NUM_GPUS=4,NUM_FRAMES=32,BATCH_SIZE=2,GRAD_ACCUM_STEPS=4,GRAD_CHECKPOINT=1,FRAMES_FPS=6,EPOCHS=70,LR=1e-4,RESUME="$BEST_F32",RUN_SUFFIX=_ddp4_ft_lr1e4_e70 \
    "$SCRIPT")
  echo "  [4] f32 resume FT lr1e4→e70  → job $J4"
else
  echo "  [4] SKIP resume FT (missing $BEST_F32)"
fi

# 5) f64: same winning recipe but longer (best weighted so far)
J5=$(sbatch --parsable $SBATCH_4 \
  --export=ALL,NUM_GPUS=4,NUM_FRAMES=64,BATCH_SIZE=1,GRAD_ACCUM_STEPS=8,GRAD_CHECKPOINT=1,FRAMES_FPS=6,EPOCHS=75,LR=7e-4,WARMUP_EPOCHS=5,RUN_SUFFIX=_ddp4_lr7e4_e75 \
  "$SCRIPT")
echo "  [5] f64 longer e75 lr7e4     → job $J5"

# 6) f32: longer freeze of flow embed (7 ep) — mild ablation
J6=$(sbatch --parsable $SBATCH_4 \
  --export=ALL,NUM_GPUS=4,NUM_FRAMES=32,BATCH_SIZE=2,GRAD_ACCUM_STEPS=4,GRAD_CHECKPOINT=1,FRAMES_FPS=6,EPOCHS=50,LR=7e-4,WARMUP_EPOCHS=5,FREEZE_FLOW_EMBED_EPOCHS=7,RUN_SUFFIX=_ddp4_lr7e4_frz7_e50 \
  "$SCRIPT")
echo "  [6] f32 freeze_flow=7 e50    → job $J6"

echo ""
echo "Monitor:  squeue -u \$USER"
echo "Results:  python src/summarize.py --protocol train_test"
echo "================================================================"
