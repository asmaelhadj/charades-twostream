# Shared helpers for Slurm job scripts. Source after cd to project root.
#
#   source jobs/_helpers.sh

maybe_final_eval_twostream() {
  # Usage: maybe_final_eval_twostream <out_dir> <batch_size> <workers>
  # Under train_test, checkpoint selection already uses the official test set,
  # so a second full pass is redundant. Override with FORCE_FINAL_EVAL=1.
  # Set NUM_SEGMENT/NUM_CROP > 1 to force a multi-view pass instead.
  local out="${1:?}"
  local batch="${2:-6}"
  local workers="${3:-8}"
  local protocol="${PROTOCOL:-train_test}"
  local seg="${NUM_SEGMENT:-1}"
  local crop="${NUM_CROP:-1}"

  if [[ "$seg" -gt 1 || "$crop" -gt 1 ]]; then
    echo "==> Multi-view final eval | segments=$seg crops=$crop | out=$out"
    python src/evaluate_videomae_twostream.py --ckpt "$out/best.pth" --out-dir "$out" \
        --batch-size "$batch" --workers "$workers" \
        --num-segment "$seg" --num-crop "$crop"
    return 0
  fi

  if [[ "$protocol" == "train_test" && "${FORCE_FINAL_EVAL:-0}" != "1" ]]; then
    echo "==> Skip final eval (protocol=$protocol; val already = test during training)."
    echo "    Set FORCE_FINAL_EVAL=1 or NUM_SEGMENT/NUM_CROP for multi-view."
    return 0
  fi

  echo "==> Final eval on test | out=$out | protocol=$protocol"
  python src/evaluate_videomae_twostream.py --ckpt "$out/best.pth" --out-dir "$out" \
      --batch-size "$batch" --workers "$workers"
}
