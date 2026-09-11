#!/bin/bash
# =============================================================================
# setup_env.sh  ---  Run ONCE on a cluster LOGIN node (needs internet).
#
#   cd $SCRATCH/narval_charades
#   bash env/setup_env.sh
#
# Venv defaults to $SCRATCH/charades_venv (not $HOME — Fir /home can be flaky).
# Override:  VENV_DIR=/path/to/venv bash env/setup_env.sh
# =============================================================================
set -euo pipefail

# ----- Where things live ----------------------------------------------------
VENV_DIR="${VENV_DIR:-${SCRATCH:-$HOME/scratch}/charades_venv}"
TORCH_CACHE="${SCRATCH:-$HOME/scratch}/charades/torch_cache"
# Keep virtualenv's wheel cache off $HOME (avoids I/O errors on some clusters).
export VIRTUALENV_OVERRIDE_APP_DATA="${VIRTUALENV_OVERRIDE_APP_DATA:-${SCRATCH:-$HOME/scratch}/.virtualenv}"

echo "==> Loading modules"
module --force purge
module load StdEnv/2023
module load python/3.11
module load cuda/13.2

echo "==> Creating virtual environment at: $VENV_DIR"
mkdir -p "$(dirname "$VENV_DIR")" "$VIRTUALENV_OVERRIDE_APP_DATA"
rm -rf "$VENV_DIR"
virtualenv --no-download "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Installing packages from the Alliance wheelhouse (--no-index)"
pip install --no-index --upgrade pip
pip install --no-index torch torchvision numpy pandas scikit-learn pillow tqdm matplotlib seaborn

echo "==> Pre-caching ImageNet pretrained ResNet-50 weights (login node has internet)"
mkdir -p "$TORCH_CACHE"
export TORCH_HOME="$TORCH_CACHE"
python - <<'PY'
import torchvision.models as models
# Triggers the one-time download into $TORCH_HOME/hub/checkpoints
models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
print("ResNet-50 ImageNet weights cached OK.")
PY

cat <<EOF

============================================================================
 Environment ready.
   venv         : $VENV_DIR
   torch cache  : $TORCH_CACHE   (set TORCH_HOME to this inside jobs)
 Activate later with:
   module load StdEnv/2023 python/3.11
   source $VENV_DIR/bin/activate
============================================================================
EOF
