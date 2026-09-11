"""Fine-tune VideoMAE V2 (ViT) on Charades for multi-label classification.

RGB-only single clip [B, 3, T, H, W]. Supports single-GPU and torchrun DDP.

    python src/train_videomae.py --finetune $CKPT --out-dir runs/<proto>/videomae_vitb
    torchrun --standalone --nproc_per_node=8 src/train_videomae.py ...
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
from dataset_videomae import CharadesVideoClips
from metrics import charades_map
from reproducibility import seed_worker, set_seed
from videomae_model import build_videomae


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="vit_base_patch16_224",
                   choices=["vit_small_patch16_224", "vit_base_patch16_224",
                            "vit_large_patch16_224", "vit_huge_patch16_224"])
    p.add_argument("--finetune", default=os.environ.get("VIDEOMAE_CKPT", ""),
                   help="path to pretrained VideoMAE V2 checkpoint (no head)")
    p.add_argument("--protocol", choices=annotations.PROTOCOLS, default="train_test")
    p.add_argument("--data-root", default=os.environ.get("DATA_ROOT", ""))
    p.add_argument("--annot-dir", default=os.environ.get("ANNOT_DIR", ""))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8, help="per-GPU micro-batch")
    p.add_argument("--lr", type=float, default=1e-3, help="peak LR (top layer)")
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--layer-decay", type=float, default=0.75)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--tubelet-size", type=int, default=2)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pos-weight", action="store_true")
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--grad-checkpoint", action="store_true")
    return p.parse_args()


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, 0, device
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return True, rank, world_size, local_rank, torch.device("cuda", local_rank)


def is_main(rank):
    return rank == 0


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def make_loaders(args, distributed, rank, world_size):
    val_split = annotations.checkpoint_split(args.protocol)
    common = dict(annot_dir=args.annot_dir, num_frames=args.num_frames,
                  img_size=args.img_size, protocol=args.protocol)
    train_ds = CharadesVideoClips(args.data_root, split="train", train=True, **common)
    val_ds = CharadesVideoClips(args.data_root, split=val_split, train=False, **common)

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed,
    ) if distributed else None

    workers = args.workers
    if distributed and world_size > 1:
        workers = max(1, args.workers // world_size)

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=workers, pin_memory=True, drop_last=True,
        generator=g, worker_init_fn=seed_worker,
    )
    val_loader = None
    if is_main(rank):
        val_loader = DataLoader(
            val_ds, args.batch_size, shuffle=False,
            num_workers=workers, pin_memory=True,
        )
    return train_loader, val_loader, train_sampler


def _layer_id(name, num_layers):
    if name in ("pos_embed", "cls_token") or name.startswith("patch_embed"):
        return 0
    if name.startswith("blocks."):
        return int(name.split(".")[1]) + 1
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
        no_decay = p.ndim <= 1 or name.endswith(".bias") or name in ("pos_embed",)
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


def lr_at(epoch, args):
    if epoch < args.warmup_epochs:
        return args.lr * (epoch + 1) / max(1, args.warmup_epochs)
    progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return args.min_lr + (args.lr - args.min_lr) * cos


def main():
    args = get_args()
    distributed, rank, world_size, local_rank, device = init_distributed()
    set_seed(args.seed + rank)
    if is_main(rank):
        os.makedirs(args.out_dir, exist_ok=True)

    eff = args.batch_size * world_size * max(1, args.grad_accum_steps)
    if is_main(rank):
        print(f"Device: {device} | DDP={distributed} | world_size={world_size} | "
              f"model={args.model} | protocol={args.protocol}", flush=True)
        print(f"  {annotations.describe_protocol(args.protocol)}", flush=True)
        print(f"  per_gpu_batch={args.batch_size} | accum={args.grad_accum_steps} | "
              f"effective_batch={eff} | frames={args.num_frames}", flush=True)
        if not args.finetune:
            print("  WARNING: --finetune empty → random init", flush=True)

    train_loader, val_loader, train_sampler = make_loaders(
        args, distributed, rank, world_size,
    )

    model = build_videomae(
        model=args.model, num_classes=annotations.NUM_CLASSES,
        all_frames=args.num_frames, tubelet_size=args.tubelet_size,
        img_size=args.img_size, drop_path_rate=args.drop_path,
        finetune=args.finetune or None, verbose=is_main(rank),
        grad_checkpoint=args.grad_checkpoint,
    ).to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    if is_main(rank):
        n_params = sum(p.numel() for p in unwrap_model(model).parameters() if p.requires_grad)
        print(f"Trainable params: {n_params:,}", flush=True)

    pos_weight = None
    if args.pos_weight:
        pos = annotations.positive_counts(args.annot_dir, protocol=args.protocol)
        n_train = len(train_loader.dataset)
        pw = np.clip((n_train - pos) / np.maximum(pos, 1.0), 1.0, 20.0)
        pos_weight = torch.tensor(pw, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    param_groups = build_param_groups(model, args.lr, args.weight_decay, args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler()

    history = {"protocol": args.protocol,
               "checkpoint_split": annotations.checkpoint_split(args.protocol),
               "world_size": world_size,
               "effective_batch": eff,
               "train_loss": [], "val_loss": [], "val_map": [], "val_weighted_map": []}
    best_map = 0.0
    ckpt_path = os.path.join(args.out_dir, "best.pth")

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        lr = lr_at(epoch - 1, args)
        for g in optimizer.param_groups:
            g["lr"] = lr * g["lr_scale"]

        model.train()
        running = 0.0
        n_seen = 0
        optimizer.zero_grad(set_to_none=True)
        accum = max(1, args.grad_accum_steps)
        for step, (clip, y) in enumerate(train_loader, start=1):
            clip = clip.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                out = model(clip)
                loss = criterion(out, y) / accum
            scaler.scale(loss).backward()
            if step % accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running += loss.item() * accum * y.size(0)
            n_seen += y.size(0)
        train_loss = running / max(1, n_seen)

        # ---- validate on rank 0 ----
        val_loss = val_map = val_weighted = float("nan")
        if is_main(rank) and val_loader is not None:
            unwrap_model(model).eval()
            vloss, scores, targets = 0.0, [], []
            with torch.no_grad():
                for clip, y in val_loader:
                    clip = clip.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    with torch.cuda.amp.autocast():
                        out = unwrap_model(model)(clip)
                        loss = criterion(out, y)
                    vloss += loss.item() * y.size(0)
                    scores.append(torch.sigmoid(out).float().cpu().numpy())
                    targets.append(y.cpu().numpy())
            val_loss = vloss / len(val_loader.dataset)
            maps = charades_map(np.concatenate(targets), np.concatenate(scores))
            val_map, val_weighted = maps["macro"], maps["weighted"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_map"].append(val_map)
            history["val_weighted_map"].append(val_weighted)

            marker = ""
            if val_map > best_map:
                best_map = val_map
                torch.save({
                    "epoch": epoch,
                    "state": unwrap_model(model).state_dict(),
                    "val_map": val_map,
                    "val_weighted_map": val_weighted,
                    "args": vars(args),
                }, ckpt_path)
                marker = "  <- best"
            print(f"Ep {epoch:3d}/{args.epochs} | lr {lr:.2e} | train_loss {train_loss:.4f} | "
                  f"val_loss {val_loss:.4f} | val_macro_mAP {val_map:.4f} | "
                  f"val_weighted_mAP {val_weighted:.4f}{marker}", flush=True)

        if distributed:
            dist.barrier()

    if is_main(rank):
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump({
                "best_map": best_map,
                "best_weighted_map": max(history["val_weighted_map"]) if history["val_weighted_map"] else None,
                **history,
            }, f, indent=2)
        print(f"Best val macro mAP: {best_map:.4f}  ({ckpt_path})", flush=True)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
