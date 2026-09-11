#!/bin/bash
# =============================================================================
# build_subset_fps.sh  ---  Submit with: sbatch data/build_subset_fps.sh
#
# ONE-TIME job: extract full Charades tarballs once, then build compact subset
# tars at 3, 6, 9, and 12 fps (~90 / ~180 / ~270 / ~360 frames per ~30 s video).
#
# Outputs on $SCRATCH:
#   charades/charades_2stream_subset_3fps.tar    (~25-30 GB)
#   charades/charades_2stream_subset_6fps.tar    (~50-60 GB)
#   charades/charades_2stream_subset_9fps.tar    (~75-90 GB)
#   charades/charades_2stream_subset_12fps.tar   (~100-120 GB)
#
# After this finishes, train with:  bash jobs/narval/submit_fps_sweep.sh
# =============================================================================
#SBATCH --account=def-hervete
#SBATCH --job-name=charades_fps_sub
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=runs/%x-%j.out

set -euo pipefail

DATA_DIR="${SCRATCH}/charades"
ARCHIVES="$DATA_DIR/archives"

source "${CHARADES_VENV:-${SCRATCH}/charades_venv}/bin/activate"

echo "==> Copying tarballs to node-local SSD ($SLURM_TMPDIR)"
cp "$ARCHIVES/Charades_v1_rgb.tar"  "$SLURM_TMPDIR/"
cp "$ARCHIVES/Charades_v1_flow.tar" "$SLURM_TMPDIR/"

echo "==> Extracting (local SSD)"
mkdir -p "$SLURM_TMPDIR/raw"
tar -xf "$SLURM_TMPDIR/Charades_v1_rgb.tar"  -C "$SLURM_TMPDIR/raw"
tar -xf "$SLURM_TMPDIR/Charades_v1_flow.tar" -C "$SLURM_TMPDIR/raw"

RGB_ROOT="$SLURM_TMPDIR/raw/Charades_v1_rgb"
FLOW_ROOT="$SLURM_TMPDIR/raw/Charades_v1_flow"
ANNOT="$DATA_DIR/annotations"

for FPS in 3 6 9 12; do
  OUT="$SLURM_TMPDIR/subset_${FPS}fps"
  TAR="$DATA_DIR/charades_2stream_subset_${FPS}fps.tar"
  echo "==> Building ${FPS} fps subset -> $TAR"
  python src/preprocess.py \
      --rgb-root  "$RGB_ROOT" \
      --flow-root "$FLOW_ROOT" \
      --annot-dir "$ANNOT" \
      --out-dir   "$OUT" \
      --target-fps "$FPS" \
      --short-side 256 \
      --workers "$SLURM_CPUS_PER_TASK"
  tar -cf "$TAR" -C "$OUT" .
  du -h "$TAR"
  rm -rf "$OUT"   # free node-local SSD before building the next (denser) fps level
done

echo "==> All fps subsets built."
