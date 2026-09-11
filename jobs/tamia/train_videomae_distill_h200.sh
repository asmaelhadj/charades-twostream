#!/bin/bash
# Charades task distillation on 8× H200:
#   Teacher = best two-stream ViT-B (frozen)
#   Student = VideoMAE V2 ViT-S RGB (~22M) — deployable path
#
#   # download student init (login node):
#   MODEL=vit_s bash data/download_videomae_ckpt.sh
#
#   TEACHER_CKPT=runs/train_test/videomae_twostream_vitb_f64_ddp2_ft_lr5e5_e100/best.pth \
#     sbatch jobs/tamia/train_videomae_distill_h200.sh
#
# Ablation (no KD, same student):
#   ALPHA_KD=0 ALPHA_FEAT=0 sbatch jobs/tamia/train_videomae_distill_h200.sh
#
# Relational KD (Park et al. 2019, distance-wise + angle-wise; on top of the
# default logit KD, in place of or alongside the FitNet-style ALPHA_FEAT):
#   ALPHA_FEAT=0 ALPHA_RKD_D=25 ALPHA_RKD_A=50 sbatch jobs/tamia/train_videomae_distill_h200.sh
#
#SBATCH --account=aip-hervete
#SBATCH --job-name=tamia_vmae_distill
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=23:59:00
#SBATCH --output=runs/%x-%j.out

export TARGET_FPS="${FRAMES_FPS:-6}"

cd "${SLURM_SUBMIT_DIR:?run sbatch from the project root}"
source jobs/tamia/_common.sh
source jobs/_helpers.sh

TEACHER_CKPT="${TEACHER_CKPT:-}"
if [ -z "$TEACHER_CKPT" ]; then
  for cand in \
    runs/train_test/videomae_twostream_vitb_f64_tamia_h200x8_ft_lr5e5_e100/best.pth \
    runs/train_test/videomae_twostream_vitb_f64_ddp2_ft_lr5e5_e100/best.pth
  do
    if [ -f "$cand" ]; then TEACHER_CKPT="$cand"; break; fi
  done
fi
if [ -z "${TEACHER_CKPT}" ] || [ ! -f "$TEACHER_CKPT" ]; then
  echo "ERROR: set TEACHER_CKPT=path/to/best.pth (two-stream or RGB Charades ckpt)"
  exit 1
fi

TEACHER_TYPE="${TEACHER_TYPE:-twostream}"
STUDENT_MODEL="${STUDENT_MODEL:-vit_small_patch16_224}"
STUDENT_CKPT="${STUDENT_CKPT:-$DATA_DIR/videomae/vit_s_k710_dl_from_giant.pth}"
if [ ! -f "$STUDENT_CKPT" ]; then
  echo "ERROR: student init not found: $STUDENT_CKPT"
  echo "On a login node:  MODEL=vit_s bash data/download_videomae_ckpt.sh"
  exit 1
fi

BATCH="${BATCH_SIZE:-4}"
ACCUM="${GRAD_ACCUM_STEPS:-2}"
EPOCHS="${EPOCHS:-75}"
LR="${LR:-5e-4}"
WARMUP="${WARMUP_EPOCHS:-5}"
SEED="${SEED:-42}"
ALPHA_KD="${ALPHA_KD:-0.5}"
ALPHA_FEAT="${ALPHA_FEAT:-0.25}"
ALPHA_RKD_D="${ALPHA_RKD_D:-0}"
ALPHA_RKD_A="${ALPHA_RKD_A:-0}"
TEMP="${TEMPERATURE:-3.0}"
RUN_SUFFIX="${RUN_SUFFIX:-_distill_from_2s}"
NGPU="${NUM_GPUS:-8}"

# Inherit frames from teacher unless NUM_FRAMES is set (>0).
NF_FLAG=()
if [[ -n "${NUM_FRAMES:-}" && "${NUM_FRAMES}" -gt 0 ]]; then
  NF_FLAG=(--num-frames "$NUM_FRAMES")
  OUT="${RUN_ROOT}/videomae_vits_f${NUM_FRAMES}${RUN_SUFFIX}"
else
  OUT="${RUN_ROOT}/videomae_vits${RUN_SUFFIX}"
fi

GRAD_CKPT_FLAG=""
if [[ "${GRAD_CHECKPOINT:-1}" == "1" ]]; then
  GRAD_CKPT_FLAG="--grad-checkpoint"
fi
POS_FLAG=""
if [[ "${POS_WEIGHT:-0}" == "1" ]]; then
  POS_FLAG="--pos-weight"
fi

if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  DETECTED="${SLURM_GPUS_ON_NODE##*:}"
  if [[ "$DETECTED" =~ ^[0-9]+$ ]]; then
    NGPU="$DETECTED"
  fi
fi

TOTAL_CPUS="${SLURM_CPUS_ON_NODE:-$(nproc)}"
WORKERS=$(( TOTAL_CPUS / NGPU ))
EFF_BATCH=$((BATCH * NGPU * ACCUM))

echo "==> Distill | teacher=$TEACHER_TYPE:$TEACHER_CKPT"
echo "==> student=$STUDENT_MODEL init=$STUDENT_CKPT | GPUs=$NGPU batch=$BATCH accum=$ACCUM eff=$EFF_BATCH"
echo "==> alpha_kd=$ALPHA_KD alpha_feat=$ALPHA_FEAT alpha_rkd_d=$ALPHA_RKD_D alpha_rkd_a=$ALPHA_RKD_A T=$TEMP | out=${OUT##*/}"

torchrun --standalone --nproc_per_node="$NGPU" \
  src/train_videomae_distill.py \
    --teacher-ckpt "$TEACHER_CKPT" --teacher-type "$TEACHER_TYPE" \
    --student-model "$STUDENT_MODEL" --student-finetune "$STUDENT_CKPT" \
    --protocol "$PROTOCOL" --out-dir "$OUT" \
    --epochs "$EPOCHS" --warmup-epochs "$WARMUP" --batch-size "$BATCH" \
    --lr "$LR" --grad-accum-steps "$ACCUM" --seed "$SEED" \
    --alpha-kd "$ALPHA_KD" --alpha-feat "$ALPHA_FEAT" \
    --alpha-rkd-d "$ALPHA_RKD_D" --alpha-rkd-a "$ALPHA_RKD_A" --temperature "$TEMP" \
    --workers "$WORKERS" $GRAD_CKPT_FLAG $POS_FLAG "${NF_FLAG[@]}"

# Single-view eval (student is RGB-only). Multi-view optional via NUM_SEGMENT/NUM_CROP.
SEG="${NUM_SEGMENT:-1}"
CROP="${NUM_CROP:-1}"
echo "==> Eval student segments=$SEG crops=$CROP"
python src/evaluate_videomae.py --ckpt "$OUT/best.pth" --out-dir "$OUT" \
  --num-segment "$SEG" --num-crop "$CROP" \
  --batch-size "$BATCH" --workers "$WORKERS"

echo "==> Done. Summarize with: python src/summarize.py --protocol $PROTOCOL"
echo "    Cost/latency: $OUT/cost_estimate.json  $OUT/latency.json"
