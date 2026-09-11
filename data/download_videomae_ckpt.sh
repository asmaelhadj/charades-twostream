#!/bin/bash
# =============================================================================
# download_videomae_ckpt.sh  ---  Run on a Narval LOGIN node (needs internet).
#
# Downloads the VideoMAE V2 pretrained backbone used to fine-tune on Charades.
# COMPUTE NODES HAVE NO INTERNET, so the weights must live on $SCRATCH before
# any GPU job starts. Default = ViT-B distilled from the giant K710 model
# (~174 MB), which the jobs/narval/train_videomae.sh script expects.
#
#   bash data/download_videomae_ckpt.sh           # ViT-B (default)
#   MODEL=vit_s bash data/download_videomae_ckpt.sh   # smaller ViT-S
#
# Source: official mirror https://huggingface.co/OpenGVLab/VideoMAE2
# =============================================================================
set -euo pipefail

DATA_DIR="${SCRATCH:-$HOME/scratch}/charades"
CKPT_DIR="$DATA_DIR/videomae"
mkdir -p "$CKPT_DIR"

MODEL="${MODEL:-vit_b}"
case "$MODEL" in
  vit_b) FILE="vit_b_k710_dl_from_giant.pth" ;;
  vit_s) FILE="vit_s_k710_dl_from_giant.pth" ;;
  *) echo "Unknown MODEL=$MODEL (use vit_b or vit_s)"; exit 1 ;;
esac

URL="https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/distill/${FILE}"

echo "==> Downloading $FILE -> $CKPT_DIR"
wget -c "$URL" -O "$CKPT_DIR/$FILE"

cat <<EOF

============================================================================
 VideoMAE V2 checkpoint ready:
   $CKPT_DIR/$FILE
 Train jobs read it via VIDEOMAE_CKPT (set in jobs/narval/train_videomae.sh).
============================================================================
EOF
