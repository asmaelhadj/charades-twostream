#!/bin/bash
# Print epoch counts for all VideoMAE two-stream runs.
# Usage: bash jobs/nibi/check_epochs.sh
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
PROTOCOL="${PROTOCOL:-train_test}"
ROOT="runs/${PROTOCOL}"

python3 - "$ROOT" <<'PY'
import glob, json, os, sys

root = sys.argv[1]
dirs = sorted(glob.glob(os.path.join(root, "videomae_twostream_vitb_f*")))
if not dirs:
    print(f"No runs under {root}/videomae_twostream_vitb_f*")
    raise SystemExit(0)

print(f"{'run':40s} {'epochs':>7s} {'target':>7s} {'best@ep':>8s} {'best mAP':>10s} {'last mAP':>10s}")
print("-" * 88)

for d in dirs:
    name = os.path.basename(d)
    target = cfg_epochs = "?"
    hist_path = os.path.join(d, "history.json")
    cfg_path = os.path.join(d, "train_config.json")

    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        target = cfg.get("epochs", "?")
        cfg_epochs = target

    n_epochs = 0
    best_map = last_map = None
    best_ep = "?"

    if os.path.exists(hist_path):
        with open(hist_path) as f:
            h = json.load(f)
        vals = h.get("val_map") or []
        n_epochs = len(vals)
        best_map = h.get("best_map")
        if vals:
            last_map = vals[-1]
            best_ep = max(range(1, len(vals) + 1), key=lambda i: vals[i - 1])

    ckpt = os.path.join(d, "best.pth")
    if os.path.exists(ckpt):
        import torch
        ck = torch.load(ckpt, map_location="cpu")
        best_ep = ck.get("epoch", best_ep)

    def fmt(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) else "   -   "

    print(f"{name:40s} {n_epochs:7d} {str(target):>7s} {str(best_ep):>8s} {fmt(best_map):>10s} {fmt(last_map):>10s}")

    has_test = os.path.exists(os.path.join(d, "per_class_ap.json"))
    if not has_test:
        print(f"  ⚠ no per_class_ap.json (eval may not have run)")

PY
