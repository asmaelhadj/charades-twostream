"""Train an RGB baseline, a flow baseline, or a two-stream fusion model.

Examples (run inside a Slurm job, see jobs/*.sh):
    python src/train.py --stream rgb    --out-dir runs/rgb
    python src/train.py --stream flow   --out-dir runs/flow
    python src/train.py --stream fusion --fusion concat --out-dir runs/fusion_concat
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import annotations
from dataset import CharadesTwoStream
from metrics import charades_map
from models import build_model


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stream", choices=["rgb", "flow", "fusion"], required=True)
    p.add_argument("--fusion", choices=["average", "weighted", "concat"], default="concat")
    p.add_argument("--protocol", choices=annotations.PROTOCOLS, default="train_test",
                   help="train_test: test for both (default) | train_val_test: val holdout + test eval")
    p.add_argument("--data-root", default=os.environ.get("DATA_ROOT", ""))
    p.add_argument("--annot-dir", default=os.environ.get("ANNOT_DIR", ""))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-rgb", type=int, default=8,
                   help="RGB frames per video; 0 = use every frame in the subset")
    p.add_argument("--num-flow", type=int, default=10,
                   help="flow pairs per video; 0 = use every stored flow pair")
    p.add_argument("--temporal-chunk", type=int, default=16,
                   help="RGB frames processed per ResNet forward pass (VRAM control)")
    p.add_argument("--target-fps", type=float, default=None,
                   help="recorded in logs/checkpoint only (subset built at this fps)")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--proj-dim", type=int, default=512)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-pos-weight", action="store_true",
                   help="disable BCE class re-weighting")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="checkpoint the backbone to save GPU memory (needed for long/high-fps clips)")
    return p.parse_args()


def make_loaders(args):
    load_rgb = args.stream in ("rgb", "fusion")
    load_flow = args.stream in ("flow", "fusion")
    val_split = annotations.checkpoint_split(args.protocol)
    common = dict(annot_dir=args.annot_dir, img_size=args.img_size,
                  num_rgb=args.num_rgb, num_flow=args.num_flow,
                  load_rgb=load_rgb, load_flow=load_flow,
                  protocol=args.protocol)
    train_ds = CharadesTwoStream(args.data_root, split="train", train=True, **common)
    val_ds = CharadesTwoStream(args.data_root, split=val_split, train=False, **common)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader


def forward(model, stream, rgb, flow, device):
    if stream == "rgb":
        return model(rgb.to(device, non_blocking=True))
    if stream == "flow":
        return model(flow.to(device, non_blocking=True))
    return model(rgb.to(device, non_blocking=True), flow.to(device, non_blocking=True))


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | stream={args.stream} | fusion={args.fusion} | "
          f"protocol={args.protocol}", flush=True)
    print(f"  {annotations.describe_protocol(args.protocol)}", flush=True)
    print(f"  checkpoint split during training: {annotations.checkpoint_split(args.protocol)}", flush=True)

    train_loader, val_loader = make_loaders(args)

    model = build_model(args.stream, num_classes=annotations.NUM_CLASSES,
                        proj_dim=args.proj_dim, num_flow=args.num_flow,
                        fusion=args.fusion, temporal_chunk=args.temporal_chunk,
                        grad_checkpoint=args.grad_checkpoint).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}", flush=True)

    # BCE with optional per-class positive weighting for class imbalance.
    pos_weight = None
    if not args.no_pos_weight:
        pos = annotations.positive_counts(args.annot_dir, protocol=args.protocol)
        n_train = len(train_loader.dataset)
        pw = np.clip((n_train - pos) / np.maximum(pos, 1.0), 1.0, 20.0)
        pos_weight = torch.tensor(pw, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    history = {"protocol": args.protocol, "checkpoint_split": annotations.checkpoint_split(args.protocol),
               "train_loss": [], "val_loss": [],
               "val_map": [], "val_weighted_map": []}
    best_map = 0.0
    ckpt_path = os.path.join(args.out_dir, "best.pth")

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        running = 0.0
        for rgb, flow, y in train_loader:
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                out = forward(model, args.stream, rgb, flow, device)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * y.size(0)
        train_loss = running / len(train_loader.dataset)

        # ---- validate ----
        model.eval()
        vloss, scores, targets = 0.0, [], []
        with torch.no_grad():
            for rgb, flow, y in val_loader:
                y = y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    out = forward(model, args.stream, rgb, flow, device)
                    loss = criterion(out, y)
                vloss += loss.item() * y.size(0)
                scores.append(torch.sigmoid(out).float().cpu().numpy())
                targets.append(y.cpu().numpy())
        val_loss = vloss / len(val_loader.dataset)
        maps = charades_map(np.concatenate(targets), np.concatenate(scores))
        val_map, val_weighted = maps["macro"], maps["weighted"]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_map"].append(val_map)
        history["val_weighted_map"].append(val_weighted)

        marker = ""
        if val_map > best_map:
            best_map = val_map
            torch.save({"epoch": epoch, "state": model.state_dict(),
                        "val_map": val_map, "val_weighted_map": val_weighted,
                        "args": vars(args)}, ckpt_path)
            marker = "  <- best"
        print(f"Ep {epoch:3d}/{args.epochs} | train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_macro_mAP {val_map:.4f} | "
              f"val_weighted_mAP {val_weighted:.4f}{marker}", flush=True)

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump({"best_map": best_map, "best_weighted_map": max(history["val_weighted_map"]),
                   **history}, f, indent=2)
    print(f"Best val macro mAP: {best_map:.4f}  ({ckpt_path})", flush=True)


if __name__ == "__main__":
    main()
