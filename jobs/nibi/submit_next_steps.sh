#!/bin/bash
# Submit the next-step jobs from the progress report (Nibi login node).
#
# Usage (from project root):
#   bash jobs/nibi/submit_next_steps.sh              # resume + repro f16
#   bash jobs/nibi/submit_next_steps.sh --with-f32f64  # also queue f32/f64 after resume
#   bash jobs/nibi/submit_next_steps.sh --diagnose-only
#
# Options:
#   --diagnose-only   Run failure diagnosis only (no sbatch)
#   --skip-resume     Skip resuming the partial sweep f16 checkpoint
#   --skip-repro      Skip the clean reproducible f16 run from epoch 1
#   --with-f32f64     Submit stable f32/f64 sweep after resume completes
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

DIAGNOSE_ONLY=false
SKIP_RESUME=false
SKIP_REPRO=false
WITH_F32F64=false

for arg in "$@"; do
  case "$arg" in
    --diagnose-only) DIAGNOSE_ONLY=true ;;
    --skip-resume)   SKIP_RESUME=true ;;
    --skip-repro)    SKIP_REPRO=true ;;
    --with-f32f64)   WITH_F32F64=true ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 1
      ;;
  esac
done

echo "================================================================"
echo " Next steps — Nibi two-stream VideoMAE"
echo "================================================================"

echo ""
echo ">>> Step 1/4: Diagnose sweep failures"
bash jobs/nibi/diagnose_sweep_failures.sh

if $DIAGNOSE_ONLY; then
  echo ""
  echo "Diagnose-only mode; no jobs submitted."
  exit 0
fi

RESUME_JOB=""
REPRO_JOB=""

if ! $SKIP_RESUME; then
  echo ""
  echo ">>> Step 2/4: Resume partial sweep f16 (epoch 8 → 30, lr=1e-4)"
  CKPT="runs/${PROTOCOL:-train_test}/videomae_twostream_vitb_f16/best.pth"
  if [[ ! -f "$CKPT" ]]; then
    echo "WARNING: checkpoint not found locally: $CKPT"
    echo "         Submitting anyway — job will fail fast if missing on cluster."
  fi
  RESUME_JOB=$(sbatch --parsable jobs/nibi/resume_videomae_twostream.sh)
  echo "Submitted resume job: $RESUME_JOB"
fi

if ! $SKIP_REPRO; then
  echo ""
  echo ">>> Step 3/4: Reproducible f16 from epoch 1 (lr=5e-4, separate _repro dir)"
  REPRO_JOB=$(sbatch --parsable --export=ALL,RUN_SUFFIX=_repro \
    jobs/nibi/train_videomae_twostream.sh)
  echo "Submitted repro job: $REPRO_JOB  →  videomae_twostream_vitb_f16_repro"
fi

if $WITH_F32F64; then
  echo ""
  echo ">>> Step 4/4: Stable f32/f64 re-run (after resume completes)"
  if [[ -z "$RESUME_JOB" ]]; then
    echo "ERROR: --with-f32f64 requires resume job (do not use --skip-resume)" >&2
    exit 1
  fi
  STABLE_JOB=$(sbatch --parsable --dependency=afterok:"$RESUME_JOB" \
    jobs/nibi/sweep_videomae_twostream_frames_stable.sh)
  echo "Submitted stable f32/f64 sweep: $STABLE_JOB (depends on $RESUME_JOB)"
else
  echo ""
  echo ">>> Step 4/4: Skipped (pass --with-f32f64 to queue after resume finishes)"
  echo "    Or submit manually later:"
  echo "      FRAMES_FPS=6 sbatch jobs/nibi/sweep_videomae_twostream_frames_stable.sh"
fi

echo ""
echo "================================================================"
echo " Submitted jobs:"
[[ -n "$RESUME_JOB" ]] && echo "   resume f16:  $RESUME_JOB"
[[ -n "$REPRO_JOB"  ]] && echo "   repro f16:   $REPRO_JOB"
[[ -n "${STABLE_JOB:-}" ]] && echo "   f32/f64:     $STABLE_JOB"
echo ""
echo " Monitor:  squeue -u \$USER"
echo " Summarize: python src/summarize.py --protocol train_test"
echo "================================================================"
