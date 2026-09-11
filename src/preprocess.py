"""Subsample + resize Charades frames into a compact per-video subset.

Reads the (locally extracted) pre-extracted RGB frames and optical flow, keeps
evenly-spaced frames per video, resizes them so the short side is `short-side`
px, and writes them to --out-dir with the SAME naming convention so the Dataset
can glob them.

Two ways to choose how many frames to keep per video:
  --frames-per-video N   fixed count (evenly spaced)
  --target-fps F         ~F frames/second from source (Charades tarballs are 24 fps)

Naming convention:

    out/rgb_frames/<id>/<id>-NNNNNN.jpg
    out/flow_frames/<id>/<id>-NNNNNNx.jpg   (horizontal component)
    out/flow_frames/<id>/<id>-NNNNNNy.jpg   (vertical component)
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

import annotations

SOURCE_FPS = 24.0  # AllenAI Charades_v1_rgb / flow tarballs


def evenly_spaced(items, k):
    if not items:
        return []
    if len(items) <= k:
        return items
    idx = np.linspace(0, len(items) - 1, k, dtype=int)
    return [items[i] for i in idx]


def resize_short_side(img, short):
    w, h = img.size
    s = short / min(w, h)
    if s < 1.0:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    return img


def list_rgb(rgb_root, vid):
    d = os.path.join(rgb_root, vid)
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.lower().endswith(".jpg"))


def list_flow(flow_root, vid):
    d = os.path.join(flow_root, vid)
    if not os.path.isdir(d):
        return [], []
    files = sorted(os.listdir(d))
    xs = [f for f in files if f.endswith("x.jpg")]
    ys = [f for f in files if f.endswith("y.jpg")]
    return xs, ys


def frames_for_fps(n_source, target_fps, source_fps=SOURCE_FPS):
    """Map target temporal rate to a frame count for one video."""
    if n_source <= 0:
        return 0
    return max(1, min(n_source, round(n_source * target_fps / source_fps)))


def process_video(args):
    vid, rgb_root, flow_root, out_dir, target_fps, fixed_k, source_fps, short = args
    try:
        rgb_all = list_rgb(rgb_root, vid)
        k = (frames_for_fps(len(rgb_all), target_fps, source_fps)
             if target_fps is not None else fixed_k)
        # ---- RGB ----
        rgb_files = evenly_spaced(rgb_all, k)
        if rgb_files:
            od = os.path.join(out_dir, "rgb_frames", vid)
            os.makedirs(od, exist_ok=True)
            for f in rgb_files:
                img = Image.open(os.path.join(rgb_root, vid, f)).convert("RGB")
                resize_short_side(img, short).save(os.path.join(od, f), quality=90)

        # ---- Flow (keep x/y pairs aligned) ----
        xs, ys = list_flow(flow_root, vid)
        n = min(len(xs), len(ys))
        if n:
            pairs = list(zip(xs[:n], ys[:n]))
            pairs = evenly_spaced(pairs, k)
            od = os.path.join(out_dir, "flow_frames", vid)
            os.makedirs(od, exist_ok=True)
            for fx, fy in pairs:
                ix = Image.open(os.path.join(flow_root, vid, fx)).convert("L")
                iy = Image.open(os.path.join(flow_root, vid, fy)).convert("L")
                resize_short_side(ix, short).save(os.path.join(od, fx), quality=90)
                resize_short_side(iy, short).save(os.path.join(od, fy), quality=90)
        return (vid, bool(rgb_files), bool(n))
    except Exception as e:  # noqa: BLE001
        return (vid, False, False, str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-root", required=True)
    ap.add_argument("--flow-root", required=True)
    ap.add_argument("--annot-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frames-per-video", type=int, default=None,
                    help="fixed evenly-spaced frame count per video")
    ap.add_argument("--target-fps", type=float, default=None,
                    help="target sampling rate (source is --source-fps, default 24)")
    ap.add_argument("--source-fps", type=float, default=SOURCE_FPS)
    ap.add_argument("--short-side", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if (args.frames_per_video is None) == (args.target_fps is None):
        ap.error("set exactly one of --frames-per-video or --target-fps")
    if args.frames_per_video is None:
        args.frames_per_video = 16  # unused when target_fps is set

    train_ids, _ = annotations.build_labels(args.annot_dir, "train")
    test_ids, _ = annotations.build_labels(args.annot_dir, "test")
    vids = list(dict.fromkeys(train_ids + test_ids))
    mode = (f"target_fps={args.target_fps} (source {args.source_fps} fps)"
            if args.target_fps is not None else f"frames_per_video={args.frames_per_video}")
    print(f"Videos to process: {len(vids)} | {mode}", flush=True)

    tasks = [
        (v, args.rgb_root, args.flow_root, args.out_dir,
         args.target_fps, args.frames_per_video, args.source_fps, args.short_side)
        for v in vids
    ]

    ok_rgb = ok_flow = done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_video, t) for t in tasks]
        for fut in as_completed(futs):
            res = fut.result()
            done += 1
            ok_rgb += int(res[1])
            ok_flow += int(res[2])
            if done % 500 == 0:
                print(f"  {done}/{len(vids)}  rgb_ok={ok_rgb} flow_ok={ok_flow}", flush=True)

    meta = {
        "source_fps": args.source_fps,
        "short_side": args.short_side,
        "videos": len(vids),
    }
    if args.target_fps is not None:
        meta["target_fps"] = args.target_fps
        meta["approx_frames_per_30s"] = frames_for_fps(
            round(30.1 * args.source_fps), args.target_fps, args.source_fps)
    else:
        meta["frames_per_video"] = args.frames_per_video
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "subset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"DONE: {done} videos | rgb_ok={ok_rgb} | flow_ok={ok_flow}", flush=True)


if __name__ == "__main__":
    main()
