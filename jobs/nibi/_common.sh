# Sourced by Nibi Slurm jobs. Stages the compact RGB+flow subset onto node SSD.
# Not a standalone script.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

DATA_DIR="${SCRATCH}/charades"
module purge
module load StdEnv/2023 python/3.11
source "$HOME/charades_venv/bin/activate"

export PROTOCOL="${PROTOCOL:-train_test}"
export RUN_ROOT="runs/${PROTOCOL}"
echo "==> Nibi | PROTOCOL=$PROTOCOL  RUN_ROOT=$RUN_ROOT"

export TORCH_HOME="$DATA_DIR/torch_cache"

if [ -n "${TARGET_FPS:-}" ]; then
  SUBSET_TAR="$DATA_DIR/charades_2stream_subset_${TARGET_FPS}fps.tar"
else
  SUBSET_TAR="$DATA_DIR/charades_2stream_subset.tar"
fi

if [ ! -f "$SUBSET_TAR" ]; then
  echo "ERROR: subset tar not found: $SUBSET_TAR"
  echo "Build on a login node:  sbatch data/build_subset.sh"
  echo "For frame sweeps:       sbatch data/build_subset_fps.sh"
  exit 1
fi

echo "==> Staging $SUBSET_TAR -> $SLURM_TMPDIR"
cp "$SUBSET_TAR" "$SLURM_TMPDIR/"
mkdir -p "$SLURM_TMPDIR/data"
tar -xf "$SLURM_TMPDIR/$(basename "$SUBSET_TAR")" -C "$SLURM_TMPDIR/data"

export DATA_ROOT="$SLURM_TMPDIR/data"
export ANNOT_DIR="$DATA_DIR/annotations"
echo "==> DATA_ROOT=$DATA_ROOT (rgb_frames + flow_frames)"
