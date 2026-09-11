# Sourced by tamIA Slurm jobs. Stages the compact RGB+flow subset onto node SSD.
# Not a standalone script.
#
# tamIA specifics vs Nibi:
#   - account is aip-hervete (AIP allocation), NOT def-hervete
#   - nodes are allocated whole: 4x H100 SXM 80GB per node, all 4 must be used
#   - walltime: min 1h, max 24h (test jobs: min 5 min)
#   - compute nodes have NO internet access -> everything (weights, data, pip
#     wheels) must already be on $SCRATCH / $HOME before submitting
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

DATA_DIR="${SCRATCH}/charades"
module purge
module load StdEnv/2023 python/3.11
source "$HOME/charades_venv/bin/activate"

export PROTOCOL="${PROTOCOL:-train_test}"
export RUN_ROOT="runs/${PROTOCOL}"
echo "==> tamIA | PROTOCOL=$PROTOCOL  RUN_ROOT=$RUN_ROOT"

export TORCH_HOME="$DATA_DIR/torch_cache"
# No internet on compute nodes: make sure nothing tries to phone home.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ -n "${TARGET_FPS:-}" ]; then
  SUBSET_TAR="$DATA_DIR/charades_2stream_subset_${TARGET_FPS}fps.tar"
else
  SUBSET_TAR="$DATA_DIR/charades_2stream_subset.tar"
fi

if [ ! -f "$SUBSET_TAR" ]; then
  echo "ERROR: subset tar not found: $SUBSET_TAR"
  echo "Copy it from Nibi/Fir (login node has internet):"
  echo "  rsync -av nibi.alliancecan.ca:\$SCRATCH/charades/charades_2stream_subset*.tar \$SCRATCH/charades/"
  echo "Or rebuild:  sbatch data/build_subset.sh"
  exit 1
fi

echo "==> Staging $SUBSET_TAR -> $SLURM_TMPDIR"
cp "$SUBSET_TAR" "$SLURM_TMPDIR/"
mkdir -p "$SLURM_TMPDIR/data"
tar -xf "$SLURM_TMPDIR/$(basename "$SUBSET_TAR")" -C "$SLURM_TMPDIR/data"

export DATA_ROOT="$SLURM_TMPDIR/data"
export ANNOT_DIR="$DATA_DIR/annotations"
echo "==> DATA_ROOT=$DATA_ROOT (rgb_frames + flow_frames)"
