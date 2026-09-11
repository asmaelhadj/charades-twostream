#!/bin/bash
# Build 12 fps subset (if needed) and submit 32 / 64 frame experiments.
#
#   bash jobs/fir/submit_frames_32_64.sh build
#   bash jobs/fir/submit_frames_32_64.sh train-rgb
#   bash jobs/fir/submit_frames_32_64.sh train-flow
#   bash jobs/fir/submit_frames_32_64.sh train-fusion      # average + weighted + concat
#   bash jobs/fir/submit_frames_32_64.sh train-videomae  # VideoMAE RGB
#   bash jobs/fir/submit_frames_32_64.sh train-videomae-twostream
#   bash jobs/fir/submit_frames_32_64.sh train-all       # everything below
set -euo pipefail

export PROTOCOL="${PROTOCOL:-train_test}"
STEP="${1:-train-all}"
TAR="${SCRATCH}/charades/charades_2stream_subset_12fps.tar"

require_tar() {
  [ -f "$TAR" ] || { echo "Missing $TAR — run: bash jobs/fir/submit_frames_32_64.sh build"; exit 1; }
}

submit_cnn() {
  local sweep="$1"
  export SWEEP="$sweep"
  sbatch jobs/fir/sweep_frames_32_64.sh
}

submit_fusion_all() {
  for FUSION in average weighted concat; do
    export SWEEP=fusion FUSION="$FUSION"
    sbatch --job-name="fir_fus_${FUSION}_32_64" jobs/fir/sweep_frames_32_64.sh
  done
}

case "$STEP" in
  build)
    echo "==> Building 12 fps subset (~360 frames/video)"
    sbatch data/build_subset_fps.sh
    echo "When done: bash jobs/fir/submit_frames_32_64.sh train-all"
    ;;
  train-rgb)       require_tar; submit_cnn rgb ;;
  train-flow)      require_tar; submit_cnn flow ;;
  train-fusion)    require_tar; submit_fusion_all ;;
  train-videomae)  require_tar; sbatch jobs/fir/sweep_videomae_frames_32_64.sh ;;
  train-videomae-twostream)
    require_tar; sbatch jobs/fir/sweep_videomae_twostream_32_64.sh ;;
  train-cnn-all)
    require_tar
    submit_cnn rgb
    submit_cnn flow
    submit_fusion_all
    ;;
  train-all)
    require_tar
    submit_cnn rgb
    submit_cnn flow
    submit_fusion_all
    sbatch jobs/fir/sweep_videomae_frames_32_64.sh
    sbatch jobs/fir/sweep_videomae_twostream_32_64.sh
    echo "Summarize when done: python src/summarize.py --protocol $PROTOCOL"
    ;;
  *)
    echo "Usage: bash jobs/fir/submit_frames_32_64.sh {build|train-rgb|train-flow|train-fusion|train-videomae|train-videomae-twostream|train-cnn-all|train-all}" >&2
    exit 1
    ;;
esac
