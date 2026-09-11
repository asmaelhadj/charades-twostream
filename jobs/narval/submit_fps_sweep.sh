#!/bin/bash
# Build fps subsets and submit full two-stream fps sweep.
#   bash jobs/narval/submit_fps_sweep.sh build
#   bash jobs/narval/submit_fps_sweep.sh train
set -euo pipefail

export PROTOCOL="${PROTOCOL:-train_test}"
STEP="${1:-all}"

submit_train() {
  for FPS in 3 6 9 12; do
    TAR="${SCRATCH}/charades/charades_2stream_subset_${FPS}fps.tar"
    if [ ! -f "$TAR" ]; then
      echo "Missing $TAR — run 'bash jobs/narval/submit_fps_sweep.sh build' first." >&2
      exit 1
    fi
  done
  sbatch jobs/narval/sweep_fps.sh
}

case "$STEP" in
  build) sbatch data/build_subset_fps.sh
         echo "When done: bash jobs/narval/submit_fps_sweep.sh train" ;;
  train) submit_train ;;
  all)   sbatch data/build_subset_fps.sh
         echo "When done: bash jobs/narval/submit_fps_sweep.sh train" ;;
  *) echo "Usage: bash jobs/narval/submit_fps_sweep.sh {build|train|all}" >&2; exit 1 ;;
esac
