#!/bin/bash
#   bash jobs/fir/submit_frames_dense.sh build
#   bash jobs/fir/submit_frames_dense.sh train
set -euo pipefail
export PROTOCOL="${PROTOCOL:-train_test}"
STEP="${1:-train}"
TAR="${SCRATCH}/charades/charades_2stream_subset_12fps.tar"

case "$STEP" in
  build)
    sbatch data/build_subset_fps.sh
    echo "When done: bash jobs/fir/submit_frames_dense.sh train"
    ;;
  train)
    [ -f "$TAR" ] || { echo "Missing $TAR — run build first"; exit 1; }
    sbatch jobs/fir/sweep_frames_dense.sh
    echo "Summarize: python src/summarize.py --protocol $PROTOCOL"
    ;;
  *)
    echo "Usage: bash jobs/fir/submit_frames_dense.sh {build|train}" >&2
    exit 1
    ;;
esac
