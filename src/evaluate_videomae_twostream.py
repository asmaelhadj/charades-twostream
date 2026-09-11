"""Evaluate a fine-tuned two-stream VideoMAE V2 checkpoint on Charades test.

Supports multi-view inference (temporal segments × spatial crops), averaging
sigmoid scores — same idea as VideoMAE ``test_num_segment`` / ``test_num_crop``.

    python src/evaluate_videomae_twostream.py --ckpt .../best.pth --out-dir ...
    python src/evaluate_videomae_twostream.py --ckpt .../best.pth --out-dir ... \
        --num-segment 5 --num-crop 3
"""
import argparse
import json
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import annotations  # noqa: E402
from dataset_videomae_twostream import CharadesVideoMAETwoStream  # noqa: E402
from metrics import charades_map  # noqa: E402
from videomae_model import build_twostream_videomae  # noqa: E402


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", default=os.environ.get("DATA_ROOT", ""))
    p.add_argument("--annot-dir", default=os.environ.get("ANNOT_DIR", ""))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=6,
                   help="views per forward (not videos); keep modest for VRAM")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--num-segment", type=int, default=1,
                   help="temporal clips per video (paper often uses 5)")
    p.add_argument("--num-crop", type=int, default=1,
                   help="spatial crops per clip (1=center, 3=L/C/R)")
    p.add_argument("--out-json", default="",
                   help="override output json name (default: per_class_ap.json "
                        "or per_class_ap_s{N}c{M}.json for multi-view)")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    targs = ckpt["args"]
    protocol = targs.get("protocol", "train_test")
    n_views = args.num_segment * args.num_crop
    print(f"Loaded {args.ckpt} | model={targs.get('model')} fusion={targs.get('fusion')} "
          f"protocol={protocol} (best val mAP={ckpt.get('val_map'):.4f})", flush=True)
    print(f"  {annotations.describe_protocol(protocol)}", flush=True)
    print(f"  multi-view: segments={args.num_segment} crops={args.num_crop} "
          f"({n_views} views/video)", flush=True)

    ds = CharadesVideoMAETwoStream(
        args.data_root, annot_dir=args.annot_dir, split="test",
        num_frames=targs.get("num_frames", 16),
        img_size=targs.get("img_size", 224),
        train=False, protocol=protocol,
        num_segment=args.num_segment, num_crop=args.num_crop,
    )

    model = build_twostream_videomae(
        model=targs.get("model", "vit_base_patch16_224"),
        num_classes=annotations.NUM_CLASSES,
        all_frames=targs.get("num_frames", 16),
        tubelet_size=targs.get("tubelet_size", 2),
        img_size=targs.get("img_size", 224),
        fusion=targs.get("fusion", "concat"),
        proj_dim=targs.get("proj_dim", 512),
        drop_path_rate=targs.get("drop_path", 0.1),
        rgb_ckpt=None, flow_ckpt=None, verbose=False,
    ).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()

    scores, targets = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            views, y = ds.get_views(i)
            view_probs = []
            for start in range(0, len(views), args.batch_size):
                batch = views[start:start + args.batch_size]
                rgb = torch.stack([v[0] for v in batch]).to(device, non_blocking=True)
                flow = torch.stack([v[1] for v in batch]).to(device, non_blocking=True)
                out = model(rgb, flow).float()
                if not torch.isfinite(out).all():
                    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
                view_probs.append(torch.sigmoid(out).cpu())
            probs = torch.cat(view_probs, dim=0).mean(dim=0).numpy()
            scores.append(probs)
            targets.append(y.numpy())
            if (i + 1) % 200 == 0 or i + 1 == len(ds):
                print(f"  evaluated {i + 1}/{len(ds)}", flush=True)

    scores = np.stack(scores)
    targets = np.stack(targets)

    maps = charades_map(targets, scores)
    macro_map, weighted_map, aps = maps["macro"], maps["weighted"], maps["per_class_ap"]
    print(f"Test macro mAP:    {macro_map:.4f}", flush=True)
    print(f"Test weighted mAP: {weighted_map:.4f}", flush=True)

    _, names = annotations.load_classes(args.annot_dir)
    order = np.argsort(-np.nan_to_num(aps))
    rows = [{"class": names[i], "ap": None if np.isnan(aps[i]) else round(float(aps[i]), 4),
             "n_pos": int(targets[:, i].sum())} for i in order]

    if args.out_json:
        json_name = args.out_json
    elif n_views > 1:
        json_name = f"per_class_ap_s{args.num_segment}c{args.num_crop}.json"
    else:
        json_name = "per_class_ap.json"

    payload = {
        "macro_mAP": macro_map,
        "weighted_mAP": weighted_map,
        "mAP": macro_map,
        "num_segment": args.num_segment,
        "num_crop": args.num_crop,
        "num_views": n_views,
        "per_class": rows,
    }
    with open(os.path.join(args.out_dir, json_name), "w") as f:
        json.dump(payload, f, indent=2)

    # Also refresh the default name so summarize.py picks multi-view up when
    # the user explicitly asked for multi-view.
    if n_views > 1 and json_name != "per_class_ap.json":
        with open(os.path.join(args.out_dir, "per_class_ap.json"), "w") as f:
            json.dump(payload, f, indent=2)

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
        print(f"\nSaved curves.png and {json_name} to {args.out_dir}")


if __name__ == "__main__":
    main()
