#!/bin/bash
# Build 12 fps subset (if needed) and submit controlled 24 / 48 / 96 frame sweep.
#
#   bash jobs/narval/submit_frames_dense.sh build
#   bash jobs/narval/submit_frames_dense.sh train
#   bash jobs/narval/submit_frames_dense.sh all
#
# Options (train step):
#   EPOCHS=20          fixed at 20 in sweep_frames_dense.sh (override via env before sbatch)
#   FIXED_BATCH=8      same batch for all frame counts (lower if OOM)
#   SWEEP=flow         vary num_flow instead of num_rgb
#   PROTOCOL=train_test
set -euo pipefail

export PROTOCOL="${PROTOCOL:-train_test}"
STEP="${1:-all}"
TAR="${SCRATCH:-/scratch}/charades/charades_2stream_subset_12fps.tar"

case "$STEP" in
  build)
    echo "==> Submitting 12 fps subset build (needed for sampling 24/48/96 frames)"
    sbatch data/build_subset_fps.sh
    echo "When the job finishes, run: bash jobs/narval/submit_frames_dense.sh train"
    ;;
  train)
    if [ ! -f "$TAR" ]; then
      echo "Missing $TAR" >&2
      echo "Run first: bash jobs/narval/submit_frames_dense.sh build" >&2
      exit 1
    fi
    echo "==> Submitting dense frame sweep: 24 / 48 / 96 sampled frames, 20 epochs"
    echo "    subset: 12 fps tar | protocol: $PROTOCOL"
    sbatch jobs/narval/sweep_frames_dense.sh
    echo "Summarize: python src/summarize.py --protocol $PROTOCOL"
    ;;
  all)
    bash "$0" build
    echo ""
    echo "After the build job completes, run: bash jobs/narval/submit_frames_dense.sh train"
    ;;
  *)
    echo "Usage: bash jobs/narval/submit_frames_dense.sh {build|train|all}" >&2
    exit 1
    ;;
esac
