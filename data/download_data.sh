#!/bin/bash
# =============================================================================
# download_data.sh  ---  Run on a Narval LOGIN node (needs internet).
#
# Downloads the Charades annotations + PRE-EXTRACTED RGB frames + optical flow
# straight into $SCRATCH. We use the pre-extracted tarballs from AllenAI so we
# never have to compute optical flow ourselves.
#
#   Annotations : Charades.zip                ~ 3 MB
#   RGB frames  : Charades_v1_rgb.tar         ~ 76 GB
#   Optical flow: Charades_v1_flow.tar        ~ 45 GB
#
# Total ~120 GB. $SCRATCH on Narval has a 20 TB quota, so this is fine, but the
# transfer takes a while. Run it inside `tmux` or `screen` so it survives a
# dropped SSH connection:   tmux new -s dl   ->   bash data/download_data.sh
# =============================================================================
set -euo pipefail

DATA_DIR="${SCRATCH:-$HOME/scratch}/charades"
ARCHIVES="$DATA_DIR/archives"
mkdir -p "$ARCHIVES"
cd "$ARCHIVES"

BASE="https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades"

echo "==> Target: $ARCHIVES"

# -c = resume a partial download if the script is re-run.
echo "==> Annotations"
wget -c "$BASE/Charades.zip"

echo "==> RGB frames (~76 GB) — this is the long one"
wget -c "$BASE/Charades_v1_rgb.tar"

echo "==> Optical flow (~45 GB)"
wget -c "$BASE/Charades_v1_flow.tar"

echo "==> Unzipping annotations"
mkdir -p "$DATA_DIR/annotations"
unzip -o Charades.zip -d "$DATA_DIR/annotations" >/dev/null
# Charades.zip contains a top-level Charades/ folder; flatten it.
if [ -d "$DATA_DIR/annotations/Charades" ]; then
  mv "$DATA_DIR/annotations/Charades/"* "$DATA_DIR/annotations/" 2>/dev/null || true
  rmdir "$DATA_DIR/annotations/Charades" 2>/dev/null || true
fi

cat <<EOF

============================================================================
 Download complete.
   archives    : $ARCHIVES
                   Charades_v1_rgb.tar
                   Charades_v1_flow.tar
   annotations : $DATA_DIR/annotations
                   Charades_v1_train.csv / Charades_v1_test.csv
                   Charades_v1_classes.txt
 Next: build the compact training subset (Slurm job):
   sbatch data/build_subset.sh
============================================================================
EOF
