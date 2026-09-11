"""Charades task distillation: freeze a strong teacher, train a small RGB student.

Typical MVP (deployability):
  Teacher = best two-stream ViT-B Charades ckpt (frozen)
  Student = VideoMAE V2 ViT-S RGB (~22M), same #frames as teacher

    python src/train_videomae_distill.py \\
      --teacher-ckpt runs/train_test/.../best.pth --teacher-type twostream \\
      --student-finetune $SCRATCH/charades/videomae/vit_s_k710_dl_from_giant.pth \\
      --out-dir runs/train_test/videomae_vits_distill_from_2s

Also supports an RGB teacher (--teacher-type rgb) and a no-KD control
(--alpha-kd 0) for fair ablation.
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
from dataset_videomae_twostream import CharadesVideoMAETwoStream
from distill_loss import MultiLabelDistillLoss
from metrics import charades_map
from model_cost import benchmark_latency, count_params, summarize_cost
from reproducibility import seed_worker, set_seed
from videomae_model import build_twostream_videomae, build_videomae


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-ckpt", required=True)
    p.add_argument("--teacher-type", choices=["twostream", "rgb"], default="twostream")
    p.add_argument("--student-model", default="vit_small_patch16_224",
                   choices=["vit_small_patch16_224", "vit_base_patch16_224"])
    p.add_argument("--student-finetune", default="",
                   help="K710-distilled student init (e.g. vit_s_k710_dl_from_giant.pth)")
    p.add_argument("--protocol", choices=annotations.PROTOCOLS, default="train_test")
    p.add_argument("--data-root", default=os.environ.get("DATA_ROOT", ""))
    p.add_argument("--annot-dir", default=os.environ.get("ANNOT_DIR", ""))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=75)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4, help="per-GPU micro-batch")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--layer-decay", type=float, default=0.75)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--num-frames", type=int, default=0,
                   help="0 = inherit from teacher ckpt args")
    p.add_argument("--tubelet-size", type=int, default=2)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pos-weight", action="store_true")
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--alpha-kd", type=float, default=0.5)
    p.add_argument("--alpha-feat", type=float, default=0.25,
                   help="point-wise (FitNet-style) feature KD weight; 0 to disable")
    p.add_argument("--alpha-rkd-d", type=float, default=0.0,
                   help="relational KD distance-wise weight (Park et al. 2019); 0 to disable")
    p.add_argument("--alpha-rkd-a", type=float, default=0.0,
                   help="relational KD angle-wise weight (Park et al. 2019); 0 to disable")
    p.add_argument("--temperature", type=float, default=3.0)
    p.add_argument("--skip-latency", action="store_true")
    return p.parse_args()


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, 0, device
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, dist.get_rank(), dist.get_world_size(), local_rank, torch.device("cuda", local_rank)


def is_main(rank):
    return rank == 0


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def _layer_id(name, num_layers):
    if name in ("pos_embed", "cls_token") or name.startswith("patch_embed"):
        return 0
    if name.startswith("blocks."):
        return int(name.split(".")[1]) + 1
    return num_layers + 1


def build_param_groups(model, base_lr, weight_decay, layer_decay, extra_params=None):
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
    if extra_params:
        groups[("extra", "decay")] = {
            "params": list(extra_params),
            "weight_decay": weight_decay,
            "lr": base_lr,
            "lr_scale": 1.0,
        }
    return list(groups.values())


def lr_at(epoch, args):
    if epoch < args.warmup_epochs:
        return args.lr * (epoch + 1) / max(1, args.warmup_epochs)
    progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return args.min_lr + (args.lr - args.min_lr) * cos


def load_teacher_meta(teacher_ckpt):
    ckpt = torch.load(teacher_ckpt, map_location="cpu")
    targs = ckpt.get("args", {})
    return ckpt, targs


def load_teacher(args, device, verbose=True):
    ckpt, targs = load_teacher_meta(args.teacher_ckpt)
    nf = args.num_frames or int(targs.get("num_frames", 16))
    teacher_nf = int(targs.get("num_frames", nf))
    if args.num_frames and args.num_frames != teacher_nf:
        raise ValueError(
            f"--num-frames={args.num_frames} does not match teacher "
            f"num_frames={teacher_nf}. Rebuild/choose a matching teacher, "
            "or omit --num-frames to inherit."
        )
    nf = teacher_nf if not args.num_frames else args.num_frames
    if args.teacher_type == "twostream":
        teacher = build_twostream_videomae(
            model=targs.get("model", "vit_base_patch16_224"),
            num_classes=annotations.NUM_CLASSES,
            all_frames=nf,
            tubelet_size=targs.get("tubelet_size", args.tubelet_size),
            img_size=targs.get("img_size", args.img_size),
            fusion=targs.get("fusion", "concat"),
            proj_dim=targs.get("proj_dim", 512),
            drop_path_rate=targs.get("drop_path", 0.1),
            rgb_ckpt=None, flow_ckpt=None, verbose=verbose,
        )
        teacher.load_state_dict(ckpt["state"], strict=True)
        teacher_embed = teacher.rgb_net.embed_dim
    else:
        teacher = build_videomae(
            model=targs.get("model", "vit_base_patch16_224"),
            num_classes=annotations.NUM_CLASSES,
            all_frames=nf,
            tubelet_size=targs.get("tubelet_size", args.tubelet_size),
            img_size=targs.get("img_size", args.img_size),
            drop_path_rate=targs.get("drop_path", 0.1),
            finetune=None, verbose=verbose,
        )
        teacher.load_state_dict(ckpt["state"], strict=True)
        teacher_embed = teacher.embed_dim
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    meta = {
        "num_frames": nf,
        "teacher_args": targs,
        "teacher_val_map": ckpt.get("val_map"),
        "teacher_embed": teacher_embed,
    }
    return teacher, meta


def load_teacher_meta_only(args):
    """For KD-ablation runs that still need teacher frame count / naming."""
    ckpt, targs = load_teacher_meta(args.teacher_ckpt)
    nf = args.num_frames or int(targs.get("num_frames", 16))
    return {
        "num_frames": nf,
        "teacher_args": targs,
        "teacher_val_map": ckpt.get("val_map"),
        "teacher_embed": 768,
    }


@torch.no_grad()
def teacher_forward(teacher, teacher_type, rgb, flow=None, return_features=False):
    if teacher_type == "twostream":
        logits = teacher(rgb, flow)
        if not return_features:
            return logits, None
        feats = teacher.rgb_net.forward_features(rgb)
        return logits, feats
    if return_features:
        return teacher(rgb, return_features=True)
    return teacher(rgb), None


def make_loaders(args, num_frames, distributed, rank, world_size):
    val_split = annotations.checkpoint_split(args.protocol)
    common = dict(annot_dir=args.annot_dir, num_frames=num_frames,
                  img_size=args.img_size, protocol=args.protocol)
    if args.teacher_type == "twostream":
        train_ds = CharadesVideoMAETwoStream(args.data_root, split="train", train=True, **common)
        val_ds = CharadesVideoMAETwoStream(args.data_root, split=val_split, train=False, **common)
    else:
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


def unpack_batch(batch, teacher_type, device):
    if teacher_type == "twostream":
        rgb, flow, y = batch
        return rgb.to(device, non_blocking=True), flow.to(device, non_blocking=True), y.to(device, non_blocking=True)
    rgb, y = batch
    return rgb.to(device, non_blocking=True), None, y.to(device, non_blocking=True)


def main():
    args = get_args()
    distributed, rank, world_size, local_rank, device = init_distributed()
    set_seed(args.seed + rank)
    if is_main(rank):
        os.makedirs(args.out_dir, exist_ok=True)

    need_feats = args.alpha_feat > 0 or args.alpha_rkd_d > 0 or args.alpha_rkd_a > 0
    need_teacher = args.alpha_kd > 0 or need_feats
    if need_teacher:
        teacher, tmeta = load_teacher(args, device, verbose=is_main(rank))
    else:
        teacher = None
        tmeta = load_teacher_meta_only(args)
        if is_main(rank):
            print("  NOTE: alpha_kd=alpha_feat=alpha_rkd_*=0 → student-only fine-tune (KD control)", flush=True)
    num_frames = tmeta["num_frames"]
    use_feat = need_feats

    eff = args.batch_size * world_size * max(1, args.grad_accum_steps)
    if is_main(rank):
        print(f"Device: {device} | DDP={distributed} | world_size={world_size}", flush=True)
        print(f"  Teacher: {args.teacher_type} | {args.teacher_ckpt} | "
              f"val_map={tmeta['teacher_val_map']}", flush=True)
        print(f"  Student: {args.student_model} | frames={num_frames} | "
              f"alpha_kd={args.alpha_kd} alpha_feat={args.alpha_feat} "
              f"alpha_rkd_d={args.alpha_rkd_d} alpha_rkd_a={args.alpha_rkd_a} "
              f"T={args.temperature}", flush=True)
        print(f"  {annotations.describe_protocol(args.protocol)}", flush=True)
        print(f"  per_gpu_batch={args.batch_size} | accum={args.grad_accum_steps} | "
              f"effective_batch={eff}", flush=True)

    train_loader, val_loader, train_sampler = make_loaders(
        args, num_frames, distributed, rank, world_size,
    )

    student = build_videomae(
        model=args.student_model, num_classes=annotations.NUM_CLASSES,
        all_frames=num_frames, tubelet_size=args.tubelet_size,
        img_size=args.img_size, drop_path_rate=args.drop_path,
        finetune=args.student_finetune or None, verbose=is_main(rank),
        grad_checkpoint=args.grad_checkpoint,
    ).to(device)

    # Only the point-wise (FitNet-style) feature loss needs a dimension-
    # bridging projector; the RKD relational losses compare dimensionless
    # relations computed separately in each model's own embedding space.
    projector = None
    if args.alpha_feat > 0:
        projector = nn.Linear(student.embed_dim, tmeta["teacher_embed"]).to(device)

    if distributed:
        student = DDP(student, device_ids=[local_rank], output_device=local_rank,
                      find_unused_parameters=False)
        if projector is not None:
            # Must be DDP-wrapped too, or each rank trains an uncoordinated
            # local copy with no gradient sync across GPUs.
            projector = DDP(projector, device_ids=[local_rank], output_device=local_rank)

    pos_weight = None
    if args.pos_weight:
        pos = annotations.positive_counts(args.annot_dir, protocol=args.protocol)
        n_train = len(train_loader.dataset)
        pw = np.clip((n_train - pos) / np.maximum(pos, 1.0), 1.0, 20.0)
        pos_weight = torch.tensor(pw, dtype=torch.float32, device=device)

    criterion = MultiLabelDistillLoss(
        alpha_kd=args.alpha_kd, alpha_feat=args.alpha_feat,
        alpha_rkd_d=args.alpha_rkd_d, alpha_rkd_a=args.alpha_rkd_a,
        temperature=args.temperature, pos_weight=pos_weight,
    )

    extra = list(projector.parameters()) if projector is not None else None
    param_groups = build_param_groups(
        student, args.lr, args.weight_decay, args.layer_decay, extra_params=extra,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler()

    if is_main(rank):
        n_s = count_params(unwrap_model(student), trainable_only=True)
        cost = summarize_cost(args.student_model, num_frames, streams=1, params=n_s)
        print(f"  Student trainable params: {n_s:,} (~{cost['params_m']}M) | "
              f"est GFLOPs/clip: {cost['gflops_est']}", flush=True)
        cost_t = None
        if teacher is not None:
            n_t = count_params(teacher)
            cost_t = summarize_cost(
                tmeta["teacher_args"].get("model", "vit_base_patch16_224"),
                num_frames,
                streams=2 if args.teacher_type == "twostream" else 1,
                params=n_t,
            )
            print(f"  Teacher params: {n_t:,} (~{cost_t['params_m']}M) | "
                  f"est GFLOPs/clip: {cost_t['gflops_est']}", flush=True)
        with open(os.path.join(args.out_dir, "cost_estimate.json"), "w") as f:
            json.dump({"student": cost, "teacher": cost_t}, f, indent=2)

    history = {
        "protocol": args.protocol,
        "teacher_ckpt": args.teacher_ckpt,
        "teacher_type": args.teacher_type,
        "student_model": args.student_model,
        "num_frames": num_frames,
        "alpha_kd": args.alpha_kd,
        "alpha_feat": args.alpha_feat,
        "alpha_rkd_d": args.alpha_rkd_d,
        "alpha_rkd_a": args.alpha_rkd_a,
        "temperature": args.temperature,
        "world_size": world_size,
        "effective_batch": eff,
        "train_loss": [], "val_loss": [], "val_map": [], "val_weighted_map": [],
    }
    best_map = 0.0
    ckpt_path = os.path.join(args.out_dir, "best.pth")

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        lr = lr_at(epoch - 1, args)
        for g in optimizer.param_groups:
            g["lr"] = lr * g["lr_scale"]

        student.train()
        if teacher is not None:
            teacher.eval()
        running = 0.0
        n_seen = 0
        optimizer.zero_grad(set_to_none=True)
        accum = max(1, args.grad_accum_steps)

        for step, batch in enumerate(train_loader, start=1):
            rgb, flow, y = unpack_batch(batch, args.teacher_type, device)
            with torch.cuda.amp.autocast():
                if use_feat:
                    s_logits, s_feats = student(rgb, return_features=True)
                else:
                    s_logits = student(rgb)
                    s_feats = None
                if need_teacher:
                    t_logits, t_feats = teacher_forward(
                        teacher, args.teacher_type, rgb, flow, return_features=use_feat,
                    )
                else:
                    t_logits, t_feats = None, None
                loss, _parts = criterion(
                    s_logits, t_logits, y,
                    student_feats=s_feats, teacher_feats=t_feats, projector=projector,
                )
                loss = loss / accum
            scaler.scale(loss).backward()
            if step % accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                params = list(student.parameters())
                if projector is not None:
                    params += list(projector.parameters())
                nn.utils.clip_grad_norm_(params, args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running += loss.item() * accum * y.size(0)
            n_seen += y.size(0)
        train_loss = running / max(1, n_seen)

        val_loss = val_map = val_weighted = float("nan")
        if is_main(rank) and val_loader is not None:
            unwrap_model(student).eval()
            vloss, scores, targets = 0.0, [], []
            bce = nn.BCEWithLogitsLoss()
            with torch.no_grad():
                for batch in val_loader:
                    rgb, flow, y = unpack_batch(batch, args.teacher_type, device)
                    with torch.cuda.amp.autocast():
                        out = unwrap_model(student)(rgb)
                        loss = bce(out, y)
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
                payload = {
                    "epoch": epoch,
                    "state": unwrap_model(student).state_dict(),
                    "val_map": val_map,
                    "val_weighted_map": val_weighted,
                    "args": {
                        **vars(args),
                        "model": args.student_model,
                        "num_frames": num_frames,
                        "finetune": args.student_finetune,
                        "drop_path": args.drop_path,
                    },
                }
                if projector is not None:
                    payload["projector"] = unwrap_model(projector).state_dict()
                torch.save(payload, ckpt_path)
                marker = "  <- best"
            print(f"Ep {epoch:3d}/{args.epochs} | lr {lr:.2e} | train_loss {train_loss:.4f} | "
                  f"val_loss {val_loss:.4f} | val_macro_mAP {val_map:.4f} | "
                  f"val_weighted_mAP {val_weighted:.4f}{marker}", flush=True)

        if distributed:
            dist.barrier()

    if is_main(rank):
        # Latency on student only (deployment path = RGB student).
        latency = None
        if not args.skip_latency and device.type == "cuda":
            ex = torch.randn(1, 3, num_frames, args.img_size, args.img_size, device=device)
            latency = benchmark_latency(unwrap_model(student), ex, device)
            print(f"Student latency (batch=1, CUDA): median={latency['median_ms']:.1f} ms "
                  f"p90={latency['p90_ms']:.1f} ms", flush=True)
            with open(os.path.join(args.out_dir, "latency.json"), "w") as f:
                json.dump(latency, f, indent=2)

        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump({
                "best_map": best_map,
                "best_weighted_map": max(history["val_weighted_map"]) if history["val_weighted_map"] else None,
                "latency": latency,
                **history,
            }, f, indent=2)
        print(f"Best val macro mAP: {best_map:.4f}  ({ckpt_path})", flush=True)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
