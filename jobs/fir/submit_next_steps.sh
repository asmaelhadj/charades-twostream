#!/bin/bash
# Optional Fir follow-ups (CNN + RGB VideoMAE are already done per progress report).
#
# Usage (Fir login node):
#   bash jobs/fir/submit_next_steps.sh
#   bash jobs/fir/submit_next_steps.sh --with-f32f64
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

WITH_F32F64=false
for arg in "$@"; do
  case "$arg" in
    --with-f32f64) WITH_F32F64=true ;;
    -h|--help)
      sed -n '2,7p' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

echo "================================================================"
echo " Fir next steps — VideoMAE two-stream (optional)"
echo "================================================================"
echo ""
echo "Already completed on Fir (no action needed):"
echo "  - CNN baselines (rgb, flow, fusion, frame sweeps)"
echo "  - VideoMAE RGB-only: videomae_vitb (0.3259 macro mAP)"
echo ""

F16_JOB=$(sbatch --parsable jobs/fir/train_videomae_twostream.sh)
echo "Submitted reproducible two-stream f16: $F16_JOB"
echo "  → runs/train_test/videomae_twostream_vitb_f16"

if $WITH_F32F64; then
  STABLE_JOB=$(sbatch --parsable --dependency=afterok:"$F16_JOB" \
    jobs/fir/sweep_videomae_twostream_frames_stable.sh)
  echo "Submitted stable f32/f64: $STABLE_JOB (depends on $F16_JOB)"
else
  echo ""
  echo "Skipped f32/f64 (pass --with-f32f64 to queue after f16)"
  echo "  FRAMES_FPS=12 sbatch jobs/fir/sweep_videomae_twostream_frames_stable.sh"
fi

echo ""
echo "Monitor:  squeue -u \$USER"
echo "Summarize: python src/summarize.py --protocol train_test"
echo "================================================================"
