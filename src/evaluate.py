"""Evaluate a trained checkpoint: overall mAP, per-class AP table, and plots.

    python src/evaluate.py --ckpt runs/fusion_concat/best.pth --out-dir runs/fusion_concat
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import annotations  # noqa: E402
from dataset import CharadesTwoStream  # noqa: E402
from metrics import charades_map  # noqa: E402
from models import build_model  # noqa: E402


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", default=os.environ.get("DATA_ROOT", ""))
    p.add_argument("--annot-dir", default=os.environ.get("ANNOT_DIR", ""))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    targs = ckpt["args"]
    stream, fusion = targs["stream"], targs.get("fusion", "concat")
    num_flow = targs.get("num_flow", 10)
    protocol = targs.get("protocol", "train_test")
    print(f"Loaded {args.ckpt} | stream={stream} fusion={fusion} protocol={protocol} "
          f"(best checkpoint mAP={ckpt.get('val_map'):.4f})", flush=True)
    print(f"  {annotations.describe_protocol(protocol)}", flush=True)
    if protocol == "train_test":
        print("  Note: test set was also used to pick best.pth → expect val mAP ≈ test mAP", flush=True)
    else:
        print("  Eval split: official Charades test (not used during training)", flush=True)

    load_rgb = stream in ("rgb", "fusion")
    load_flow = stream in ("flow", "fusion")
    ds = CharadesTwoStream(args.data_root, annot_dir=args.annot_dir, split="test",
                           img_size=targs.get("img_size", 224),
                           num_rgb=targs.get("num_rgb", 8), num_flow=num_flow,
                           train=False, load_rgb=load_rgb, load_flow=load_flow,
                           protocol=protocol)
    loader = DataLoader(ds, args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    model = build_model(stream, num_classes=annotations.NUM_CLASSES,
                        proj_dim=targs.get("proj_dim", 512),
                        num_flow=num_flow, fusion=fusion,
                        temporal_chunk=targs.get("temporal_chunk", 16)).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()

    scores, targets = [], []
    with torch.no_grad():
        for rgb, flow, y in loader:
            with torch.cuda.amp.autocast():
                if stream == "rgb":
                    out = model(rgb.to(device))
                elif stream == "flow":
                    out = model(flow.to(device))
                else:
                    out = model(rgb.to(device), flow.to(device))
            scores.append(torch.sigmoid(out).float().cpu().numpy())
            targets.append(y.numpy())
    scores = np.concatenate(scores)
    targets = np.concatenate(targets)

    maps = charades_map(targets, scores)
    macro_map, weighted_map, aps = maps["macro"], maps["weighted"], maps["per_class_ap"]
    print(f"Test macro mAP:    {macro_map:.4f}", flush=True)
    print(f"Test weighted mAP: {weighted_map:.4f}", flush=True)

    _, names = annotations.load_classes(args.annot_dir)
    order = np.argsort(-np.nan_to_num(aps))
    rows = [{"class": names[i], "ap": None if np.isnan(aps[i]) else round(float(aps[i]), 4),
             "n_pos": int(targets[:, i].sum())} for i in order]
    with open(os.path.join(args.out_dir, "per_class_ap.json"), "w") as f:
        json.dump({
            "macro_mAP": macro_map,
            "weighted_mAP": weighted_map,
            "mAP": macro_map,  # backward compat (macro = Charades standard)
            "per_class": rows,
        }, f, indent=2)

    print("\nTop-10 classes by AP:")
    for r in rows[:10]:
        print(f"  {r['ap']}  (n={r['n_pos']:4d})  {r['class']}")
    print("\nBottom-10 classes by AP:")
    for r in [r for r in rows if r["ap"] is not None][-10:]:
        print(f"  {r['ap']}  (n={r['n_pos']:4d})  {r['class']}")

    # Plot training curves if history is present next to the checkpoint.
    hist_path = os.path.join(os.path.dirname(args.ckpt), "history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            h = json.load(f)
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        ax[0].plot(h["train_loss"], label="train")
        ax[0].plot(h["val_loss"], label="val")
        ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].plot(h["val_map"], color="green", label="macro")
        if h.get("val_weighted_map"):
            ax[1].plot(h["val_weighted_map"], color="orange", label="weighted")
        ax[1].set_title("Val mAP"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "curves.png"), dpi=150)
        print(f"\nSaved curves.png and per_class_ap.json to {args.out_dir}")


if __name__ == "__main__":
    main()
