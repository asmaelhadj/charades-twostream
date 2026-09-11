#!/bin/bash
# Diagnose why the Nibi two-stream frame sweep stopped early.
# Run on a Nibi login node (no GPU):
#   bash jobs/nibi/diagnose_sweep_failures.sh
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
PROTOCOL="${PROTOCOL:-train_test}"
RUN_ROOT="runs/${PROTOCOL}"

echo "================================================================"
echo " Nibi two-stream frame sweep — failure diagnosis"
echo " $(date -Iseconds 2>/dev/null || date)"
echo "================================================================"

echo ""
echo "--- Slurm logs: runs/nibi_vmae2s_frames-*.out ---"
shopt -s nullglob
LOGS=(runs/nibi_vmae2s_frames-*.out)
if ((${#LOGS[@]} == 0)); then
  echo "  (no sweep logs found under runs/)"
else
  for log in "${LOGS[@]}"; do
    echo ""
    echo ">> $log"
    tail -n 30 "$log" | sed 's/^/    /'
    echo "    --- patterns ---"
    for pat in "OutOfMemory" "CUDA out of memory" "nan" "NaN" "TIMEOUT" "CANCELLED" "FAILED" "error" "Error" "Killed"; do
      if grep -qi "$pat" "$log" 2>/dev/null; then
        echo "    FOUND: $pat"
        grep -ni "$pat" "$log" | tail -n 3 | sed 's/^/      /'
      fi
    done
  done
fi

echo ""
echo "--- sacct (recent nibi_vmae2s_frames jobs) ---"
if command -v sacct >/dev/null 2>&1; then
  sacct -u "$USER" -n -P \
    --format=JobID,JobName%20,State,ExitCode,Elapsed,Timelimit,MaxRSS,ReqMem \
    -S "$(date -d '30 days ago' +%Y-%m-%d 2>/dev/null || date -v-30d +%Y-%m-%d 2>/dev/null || echo 2026-01-01)" \
    2>/dev/null | grep -i "vmae2s_frames\|vmae_2stream" || echo "  (no matching sacct rows)"
else
  echo "  sacct not available (run on cluster login node)"
fi

echo ""
echo "--- Run directories under $RUN_ROOT ---"
for nf in 16 32 48 64; do
  dir="${RUN_ROOT}/videomae_twostream_vitb_f${nf}"
  echo ""
  echo ">> videomae_twostream_vitb_f${nf}"
  if [[ ! -d "$dir" ]]; then
    echo "    (directory missing)"
    continue
  fi
  if [[ -f "$dir/history.json" ]]; then
    python3 - "$dir/history.json" <<'PY'
import json, sys
h = json.load(open(sys.argv[1]))
vals = h.get("val_map") or []
best = h.get("best_map")
print(f"    epochs completed: {len(vals)}")
if vals:
    best_ep = max(range(1, len(vals)+1), key=lambda i: vals[i-1])
    print(f"    best val mAP: {best:.4f} at epoch {best_ep}")
    print(f"    last epoch val mAP: {vals[-1]:.4f}")
PY
  else
    echo "    (no history.json)"
  fi
  if [[ -f "$dir/best.pth" ]]; then
    python3 - "$dir/best.pth" <<'PY'
import torch, sys
ckpt = torch.load(sys.argv[1], map_location="cpu")
print(f"    best.pth epoch: {ckpt.get('epoch', '?')}  val_map: {ckpt.get('val_map', '?')}")
PY
  else
    echo "    (no best.pth)"
  fi
  if [[ -f "$dir/per_class_ap.json" ]]; then
    echo "    per_class_ap.json: present"
  else
    echo "    per_class_ap.json: MISSING (eval may not have run)"
  fi
  if [[ -f "$dir/train_config.json" ]]; then
    python3 - "$dir/train_config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
keys = ["lr", "batch_size", "num_frames", "freeze_flow_embed_epochs", "grad_clip", "epochs"]
print("    config: " + ", ".join(f"{k}={c.get(k)}" for k in keys if k in c))
PY
  fi
done

echo ""
echo "================================================================"
echo " Done. See docs/nibi/REPRODUCE.md for the stable rerun recipe."
echo "================================================================"
