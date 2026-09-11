"""Fine-tune two-stream VideoMAE V2 (RGB + optical flow) on Charades.

Late-fusion dual ViT backbones reading the precomputed rgb_frames/ and
flow_frames/ subset (no RAFT extraction needed).

Single GPU:
    python src/train_videomae_twostream.py --finetune $CKPT \\
        --out-dir runs/train_test/videomae_twostream_vitb

Multi-GPU (DDP via torchrun):
    torchrun --standalone --nproc_per_node=2 src/train_videomae_twostream.py \\
        --finetune $CKPT --out-dir runs/train_test/videomae_twostream_vitb_ddp2

See jobs/nibi/train_videomae_twostream.sh (1 GPU) or
jobs/nibi/train_videomae_twostream_multigpu.sh (2+ GPUs).
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import annotations
from dataset_videomae_twostream import CharadesVideoMAETwoStream
from metrics import charades_map
from reproducibility import save_train_config, seed_worker, set_seed
from videomae_model import build_twostream_videomae


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="vit_base_patch16_224",
                   choices=["vit_small_patch16_224", "vit_base_patch16_224",
                            "vit_large_patch16_224", "vit_huge_patch16_224"])
    p.add_argument("--finetune", default=os.environ.get("VIDEOMAE_CKPT", ""),
                   help="pretrained VideoMAE V2 checkpoint for RGB + flow init")
    p.add_argument("--protocol", choices=annotations.PROTOCOLS, default="train_test")
    p.add_argument("--data-root", default=os.environ.get("DATA_ROOT", ""))
    p.add_argument("--annot-dir", default=os.environ.get("ANNOT_DIR", ""))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fusion", choices=["concat", "average", "weighted"], default="concat")
    p.add_argument("--proj-dim", type=int, default=512)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=6,
                   help="per-GPU micro-batch size")
    p.add_argument("--lr", type=float, default=5e-4,
                   help="peak LR (5e-4 is stable for dual ViT-B; 1e-3 can NaN after flow unfreeze)")
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--layer-decay", type=float, default=0.75)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-accum-steps", type=int, default=1,
                   help="accumulate gradients over N micro-batches "
                        "(effective batch = batch_size * world_size * N)")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="checkpoint ViT blocks to save VRAM (slower, needed for long clips)")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--tubelet-size", type=int, default=2)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42,
                   help="random seed for reproducible training from epoch 1")
    p.add_argument("--freeze-flow-embed-epochs", type=int, default=5,
                   help="freeze flow patch_embed for the first N epochs")
    p.add_argument("--resume", default="",
                   help="optional: path to best.pth to continue (default: train epoch 1)")
    p.add_argument("--pos-weight", action="store_true")
    p.add_argument("--scale-lr", action="store_true",
                   help="multiply peak/min LR by world_size (linear scaling rule)")
    return p.parse_args()


def init_distributed():
    """Detect torchrun / Slurm DDP. Returns (enabled, rank, world_size, local_rank, device)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, 0, device

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if "LOCAL_RANK" not in os.environ and "SLURM_LOCALID" in os.environ:
        local_rank = int(os.environ["SLURM_LOCALID"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    return True, rank, world_size, local_rank, device


def is_main(rank):
    return rank == 0


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def make_loaders(args, distributed, rank, world_size):
    val_split = annotations.checkpoint_split(args.protocol)
    common = dict(annot_dir=args.annot_dir, num_frames=args.num_frames,
                  img_size=args.img_size, protocol=args.protocol)
    train_ds = CharadesVideoMAETwoStream(args.data_root, split="train", train=True, **common)
    val_ds = CharadesVideoMAETwoStream(args.data_root, split=val_split, train=False, **common)

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed,
    ) if distributed else None

    g = torch.Generator()
    g.manual_seed(args.seed)
    workers = args.workers
    if distributed and world_size > 1:
        workers = max(1, args.workers // world_size)

    train_loader = DataLoader(
        train_ds, args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=workers, pin_memory=True, drop_last=True,
        generator=g, worker_init_fn=seed_worker,
    )
    # Validation on rank 0 only (full set) — avoids gather complexity.
    val_loader = None
    if is_main(rank):
        val_loader = DataLoader(
            val_ds, args.batch_size, shuffle=False,
            num_workers=workers, pin_memory=True,
        )
    return train_loader, val_loader, train_sampler


def _strip_prefix(name):
    for prefix in ("rgb_net.", "flow_net."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _layer_id(name, num_layers):
    base = _strip_prefix(name)
    if base in ("pos_embed", "cls_token") or base.startswith("patch_embed"):
        return 0
    if base.startswith("blocks."):
        return int(base.split(".")[1]) + 1
    return num_layers + 1


def build_param_groups(model, base_lr, weight_decay, layer_decay):
    raw = unwrap_model(model)
    depth = raw.get_num_layers()
    num_layers = depth
    scales = [layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)]
    groups = {}
    for name, p in raw.named_parameters():
        if not p.requires_grad:
            continue
        no_decay = p.ndim <= 1 or name.endswith(".bias")
        lid = _layer_id(name, num_layers)
        key = (lid, "no_decay" if no_decay else "decay")
        if key not in groups:
            groups[key] = {
                "params": [],
                "weight_decay": 0.0 if no_decay else weight_decay,
                "lr": base_lr * scales[lid],
                "lr_scale": scales[lid],
            }
        groups[key]["params"].append(p)
    return list(groups.values())


def flow_embed_param_groups(model, base_lr, weight_decay, layer_decay):
    """Param groups for flow_net.patch_embed only (layer id 0), used on unfreeze."""
    raw = unwrap_model(model)
    depth = raw.get_num_layers()
    scale = layer_decay ** (depth + 1)  # lid == 0
    groups = {}
    for name, p in raw.named_parameters():
        if not name.startswith("flow_net.patch_embed"):
            continue
        if not p.requires_grad:
            continue
        no_decay = p.ndim <= 1 or name.endswith(".bias")
        key = "no_decay" if no_decay else "decay"
        if key not in groups:
            groups[key] = {
                "params": [],
                "weight_decay": 0.0 if no_decay else weight_decay,
                "lr": base_lr * scale,
                "lr_scale": scale,
            }
        groups[key]["params"].append(p)
    return list(groups.values())


def set_flow_embed_trainable(model, trainable):
    for p in unwrap_model(model).flow_net.patch_embed.parameters():
        p.requires_grad = trainable


def lr_at(epoch, args):
    if epoch < args.warmup_epochs:
        return args.lr * (epoch + 1) / max(1, args.warmup_epochs)
    progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return args.min_lr + (args.lr - args.min_lr) * cos


def _weights_finite(model):
    return all(torch.isfinite(p).all() for p in unwrap_model(model).parameters())


def _save_history(out_dir, best_map, history):
    path = os.path.join(out_dir, "history.json")
    with open(path, "w") as f:
        json.dump({"best_map": best_map,
                   "best_weighted_map": max(history["val_weighted_map"]) if history["val_weighted_map"] else 0.0,
                   **history}, f, indent=2)


def _load_history(out_dir):
    path = os.path.join(out_dir, "history.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _save_per_class_ap(out_dir, maps, targets, annot_dir):
    """Write per_class_ap.json from the checkpoint-split eval (avoids a second test pass)."""
    _, names = annotations.load_classes(annot_dir)
    aps = maps["per_class_ap"]
    order = np.argsort(-np.nan_to_num(aps))
    rows = [{"class": names[i],
             "ap": None if np.isnan(aps[i]) else round(float(aps[i]), 4),
             "n_pos": int(targets[:, i].sum())} for i in order]
    path = os.path.join(out_dir, "per_class_ap.json")
    with open(path, "w") as f:
        json.dump({"macro_mAP": maps["macro"], "weighted_mAP": maps["weighted"],
                   "mAP": maps["macro"], "per_class": rows,
                   "note": "from checkpoint-split eval during training"}, f, indent=2)
    return path


def _save_curves(out_dir, history):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(history["val_map"], color="green", label="macro")
    if history.get("val_weighted_map"):
        ax[1].plot(history["val_weighted_map"], color="orange", label="weighted")
    ax[1].set_title("Val mAP")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "curves.png"), dpi=150)
    plt.close(fig)


def _log(rank, msg):
    if is_main(rank):
        print(msg, flush=True)


def main():
    args = get_args()
    distributed, rank, world_size, local_rank, device = init_distributed()

    if is_main(rank):
        os.makedirs(args.out_dir, exist_ok=True)
    if distributed:
        dist.barrier()

    if args.resume and not args.finetune:
        args.finetune = os.environ.get("VIDEOMAE_CKPT", "")

    set_seed(args.seed + rank)

    if args.scale_lr and world_size > 1:
        args.lr = args.lr * world_size
        args.min_lr = args.min_lr * world_size

    cfg = dict(vars(args))
    cfg["world_size"] = world_size
    cfg["distributed"] = distributed
    cfg["effective_batch"] = args.batch_size * world_size * max(1, args.grad_accum_steps)
    if is_main(rank):
        cfg_path = save_train_config(args.out_dir, cfg)
    else:
        cfg_path = os.path.join(args.out_dir, "train_config.json")

    _log(rank, f"Device: {device} | DDP={distributed} | world_size={world_size} | "
         f"model={args.model} | fusion={args.fusion} | protocol={args.protocol} | seed={args.seed}")
    _log(rank, f"  config saved to {cfg_path}")
    _log(rank, f"  {annotations.describe_protocol(args.protocol)}")
    _log(rank, f"  per_gpu_batch={args.batch_size} | accum={args.grad_accum_steps} | "
         f"effective_batch={cfg['effective_batch']} | "
         f"grad_checkpoint={args.grad_checkpoint} | scale_lr={args.scale_lr}")
    if args.resume:
        _log(rank, f"  RESUME mode from {args.resume}")
    elif not args.finetune:
        _log(rank, "  WARNING: --finetune empty → training from random init")
    else:
        _log(rank, f"  training from epoch 1 | lr={args.lr} | freeze_flow={args.freeze_flow_embed_epochs} ep")

    train_loader, val_loader, train_sampler = make_loaders(
        args, distributed, rank, world_size,
    )

    start_epoch = 1
    best_map = 0.0
    history = {"protocol": args.protocol,
               "checkpoint_split": annotations.checkpoint_split(args.protocol),
               "world_size": world_size,
               "train_loss": [], "val_loss": [], "val_map": [], "val_weighted_map": []}

    build_kwargs = dict(
        model=args.model, num_classes=annotations.NUM_CLASSES,
        all_frames=args.num_frames, tubelet_size=args.tubelet_size,
        img_size=args.img_size, fusion=args.fusion, proj_dim=args.proj_dim,
        drop_path_rate=args.drop_path, grad_checkpoint=args.grad_checkpoint,
    )

    if args.resume:
        _log(rank, f"  resuming from {args.resume}")
        model = build_twostream_videomae(
            rgb_ckpt=None, flow_ckpt=None, verbose=is_main(rank), **build_kwargs,
        ).to(device)
        resume_ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(resume_ckpt["state"])
        start_epoch = int(resume_ckpt["epoch"]) + 1
        best_map = float(resume_ckpt.get("val_map", 0.0))
        set_flow_embed_trainable(model, True)
        args.freeze_flow_embed_epochs = 0
        if is_main(rank):
            prev = _load_history(args.out_dir)
            if prev:
                history = {k: prev.get(k, history.get(k)) for k in history}
                best_map = max(best_map, float(prev.get("best_map", best_map)))
        _log(rank, f"  resume epoch {start_epoch} | best mAP so far {best_map:.4f}")
    else:
        model = build_twostream_videomae(
            rgb_ckpt=args.finetune or None,
            flow_ckpt=args.finetune or None,
            verbose=is_main(rank),
            **build_kwargs,
        ).to(device)
        if args.freeze_flow_embed_epochs > 0:
            set_flow_embed_trainable(model, False)
            _log(rank, f"  flow patch_embed frozen for first {args.freeze_flow_embed_epochs} epochs")

    if start_epoch > args.epochs:
        _log(rank, f"  already at epoch {start_epoch - 1} >= {args.epochs}, nothing to do")
        if distributed:
            dist.destroy_process_group()
        return

    if distributed:
        # find_unused_parameters: flow embed may be frozen for the first N epochs
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)

    n_params = sum(p.numel() for p in unwrap_model(model).parameters() if p.requires_grad)
    _log(rank, f"Trainable params: {n_params:,}")

    pos_weight = None
    if args.pos_weight:
        pos = annotations.positive_counts(args.annot_dir, protocol=args.protocol)
        n_train = len(train_loader.dataset)
        pw = np.clip((n_train - pos) / np.maximum(pos, 1.0), 1.0, 20.0)
        pos_weight = torch.tensor(pw, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    param_groups = build_param_groups(model, args.lr, args.weight_decay, args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    ckpt_path = os.path.join(args.out_dir, "best.pth")
    flow_embed_in_optim = args.freeze_flow_embed_epochs <= 0

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if (args.freeze_flow_embed_epochs > 0
                and epoch == args.freeze_flow_embed_epochs + 1
                and not flow_embed_in_optim):
            set_flow_embed_trainable(model, True)
            new_groups = flow_embed_param_groups(
                model, args.lr, args.weight_decay, args.layer_decay)
            n_added = sum(p.numel() for g in new_groups for p in g["params"])
            for g in new_groups:
                optimizer.add_param_group(g)
            flow_embed_in_optim = True
            n_params = sum(p.numel() for p in unwrap_model(model).parameters() if p.requires_grad)
            _log(rank, f"  unfroze flow patch_embed at epoch {epoch} "
                 f"(+{n_added:,} params → {n_params:,} trainable)")

        lr = lr_at(epoch - 1, args)
        for g in optimizer.param_groups:
            g["lr"] = lr * g["lr_scale"]

        model.train()
        running = 0.0
        n_seen = 0
        optimizer.zero_grad(set_to_none=True)
        accum = max(1, args.grad_accum_steps)
        for step, (rgb, flow, y) in enumerate(train_loader, start=1):
            rgb = rgb.to(device, non_blocking=True)
            flow = flow.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                out = model(rgb, flow)
                loss = criterion(out, y) / accum
            if not torch.isfinite(loss):
                _log(rank, f"  warning: non-finite train loss at epoch {epoch}, skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            if step % accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running += loss.item() * accum * y.size(0)
            n_seen += y.size(0)

        # Average train loss across ranks
        if distributed:
            stats = torch.tensor([running, float(n_seen)], device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            running, n_seen = stats[0].item(), int(stats[1].item())
        train_loss = running / max(n_seen, 1)

        if not _weights_finite(model):
            _log(rank, f"  ERROR: non-finite weights after epoch {epoch} train — stopping")
            break

        val_loss = float("nan")
        val_map = 0.0
        val_weighted = 0.0
        if is_main(rank) and val_loader is not None:
            raw = unwrap_model(model)
            raw.eval()
            vloss, scores, targets = 0.0, [], []
            with torch.no_grad():
                for rgb, flow, y in val_loader:
                    rgb = rgb.to(device, non_blocking=True)
                    flow = flow.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    out = raw(rgb, flow).float()
                    loss = criterion(out, y)
                    vloss += loss.item() * y.size(0)
                    prob = torch.sigmoid(out)
                    if not torch.isfinite(prob).all():
                        prob = torch.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)
                    scores.append(prob.cpu().numpy())
                    targets.append(y.cpu().numpy())
            val_loss = vloss / len(val_loader.dataset)
            y_true = np.concatenate(targets)
            maps = charades_map(y_true, np.concatenate(scores))
            val_map, val_weighted = maps["macro"], maps["weighted"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_map"].append(val_map)
            history["val_weighted_map"].append(val_weighted)

            marker = ""
            if val_map > best_map:
                best_map = val_map
                torch.save({"epoch": epoch, "state": raw.state_dict(),
                            "val_map": val_map, "val_weighted_map": val_weighted,
                            "args": cfg}, ckpt_path)
                _save_per_class_ap(args.out_dir, maps, y_true, args.annot_dir)
                marker = "  <- best"
            print(f"Ep {epoch:3d}/{args.epochs} | lr {lr:.2e} | train_loss {train_loss:.4f} | "
                  f"val_loss {val_loss:.4f} | val_macro_mAP {val_map:.4f} | "
                  f"val_weighted_mAP {val_weighted:.4f}{marker}", flush=True)
            _save_history(args.out_dir, best_map, history)

        if distributed:
            dist.barrier()

    if is_main(rank):
        _save_curves(args.out_dir, history)
        _log(rank, f"Best val macro mAP: {best_map:.4f}  ({ckpt_path})")
        _log(rank, f"  wrote history.json, per_class_ap.json (on best), curves.png")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
