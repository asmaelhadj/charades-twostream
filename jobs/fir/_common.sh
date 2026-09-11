# Sourced by Fir Slurm jobs (H100 80 GB). Stages compact subset onto node SSD.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

DATA_DIR="${SCRATCH}/charades"

module --force purge
module load StdEnv/2023 python/3.11 cuda/13.2
source "${CHARADES_VENV:-${SCRATCH}/charades_venv}/bin/activate"

echo "==> module list:"; module list 2>&1
python -c "import torch; print('torch', torch.__version__, 'build', torch.version.cuda, 'cuda_available', torch.cuda.is_available())"
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "WARNING: torch.cuda.is_available() is False (nvidia-smi may still show a GPU)" >&2
  nvidia-smi || true
fi

export PROTOCOL="${PROTOCOL:-train_test}"
export RUN_ROOT="runs/${PROTOCOL}"
echo "==> Fir (H100) | PROTOCOL=$PROTOCOL  RUN_ROOT=$RUN_ROOT"

export TORCH_HOME="$DATA_DIR/torch_cache"

if [ -n "${TARGET_FPS:-}" ]; then
  SUBSET_TAR="$DATA_DIR/charades_2stream_subset_${TARGET_FPS}fps.tar"
else
  SUBSET_TAR="$DATA_DIR/charades_2stream_subset.tar"
fi

if [ ! -f "$SUBSET_TAR" ]; then
  echo "ERROR: subset tar not found: $SUBSET_TAR"
  echo "Build: sbatch data/build_subset.sh"
  echo "Dense fps subsets: sbatch data/build_subset_fps.sh"
  exit 1
fi

echo "==> Staging $SUBSET_TAR -> $SLURM_TMPDIR"
cp "$SUBSET_TAR" "$SLURM_TMPDIR/"
mkdir -p "$SLURM_TMPDIR/data"
tar -xf "$SLURM_TMPDIR/$(basename "$SUBSET_TAR")" -C "$SLURM_TMPDIR/data"

export DATA_ROOT="$SLURM_TMPDIR/data"
export ANNOT_DIR="$DATA_DIR/annotations"
echo "==> DATA_ROOT=$DATA_ROOT"
