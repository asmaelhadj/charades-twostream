#!/bin/bash
# =============================================================================
# build_subset.sh  ---  Submit with: sbatch data/build_subset.sh
#
# ONE-TIME preprocessing (CPU job). The full Charades RGB+flow tarballs contain
# MILLIONS of tiny JPEGs. Reading those directly off /scratch (Lustre) every
# training job is slow and against Alliance filesystem guidelines. So here we:
#
#   1. copy the tarballs to the node-local SSD ($SLURM_TMPDIR),
#   2. extract them there,
#   3. keep only a handful of evenly-spaced frames per video, resized small,
#   4. repack the result into ONE compact tar (~5 GB) back on $SCRATCH.
#
# Training jobs then copy that small tar to $SLURM_TMPDIR (seconds) and read
# from local SSD. Run this ONCE.
# =============================================================================
#SBATCH --account=def-hervete
#SBATCH --job-name=charades_subset
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=runs/%x-%j.out

set -euo pipefail

DATA_DIR="${SCRATCH}/charades"
ARCHIVES="$DATA_DIR/archives"
OUT_TAR="$DATA_DIR/charades_2stream_subset.tar"

source "${CHARADES_VENV:-${SCRATCH}/charades_venv}/bin/activate"

echo "==> Copying tarballs to node-local SSD ($SLURM_TMPDIR)"
cp "$ARCHIVES/Charades_v1_rgb.tar"  "$SLURM_TMPDIR/"
cp "$ARCHIVES/Charades_v1_flow.tar" "$SLURM_TMPDIR/"

echo "==> Extracting (local SSD)"
mkdir -p "$SLURM_TMPDIR/raw"
tar -xf "$SLURM_TMPDIR/Charades_v1_rgb.tar"  -C "$SLURM_TMPDIR/raw"
tar -xf "$SLURM_TMPDIR/Charades_v1_flow.tar" -C "$SLURM_TMPDIR/raw"

echo "==> Subsampling frames"
python src/preprocess.py \
    --rgb-root  "$SLURM_TMPDIR/raw/Charades_v1_rgb" \
    --flow-root "$SLURM_TMPDIR/raw/Charades_v1_flow" \
    --annot-dir "$DATA_DIR/annotations" \
    --out-dir   "$SLURM_TMPDIR/subset" \
    --frames-per-video 16 \
    --short-side 256 \
    --workers "$SLURM_CPUS_PER_TASK"

echo "==> Packing compact subset to $OUT_TAR"
tar -cf "$OUT_TAR" -C "$SLURM_TMPDIR/subset" .

echo "==> Done. Subset tar size:"
du -h "$OUT_TAR"
