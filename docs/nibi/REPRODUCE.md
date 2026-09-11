# Reproducible Two-Stream VideoMAE Training

Standard recipe: **train from epoch 1**, fixed hyperparameters, saved config.

---

## One command (Nibi H100)

```bash
cd ~/narval_charades
source ~/charades_venv/bin/activate

# Clean rerun (optional — removes old partial run)
rm -rf runs/train_test/videomae_twostream_vitb_f16

sbatch jobs/nibi/train_videomae_twostream.sh
```

---

## Pinned hyperparameters (epoch 1 → 30)

| Setting | Value | Notes |
|---------|-------|-------|
| Model | VideoMAE ViT-B × 2 (RGB + flow) | `vit_base_patch16_224` |
| Fusion | concat | late fusion MLP |
| Pretrain | `vit_b_k710_dl_from_giant.pth` | K710 backbone |
| Frames | 16 | from `charades_2stream_subset.tar` |
| Batch (Nibi) | 6 | 3 on Narval 40 GB |
| Epochs | 30 | |
| Warmup | 2 epochs | |
| Peak LR | **5e-4** | stable for dual ViT (1e-3 can NaN) |
| Min LR | 1e-6 | cosine decay |
| Layer decay | 0.75 | |
| Weight decay | 0.05 | |
| Drop path | 0.1 | |
| Grad clip | 1.0 | |
| Freeze flow embed | **5 epochs** | then unfreeze |
| Seed | **42** | Python / NumPy / PyTorch |
| Protocol | `train_test` | override: `PROTOCOL=train_val_test` |
| Loss | BCE multi-label | macro mAP metric |

All values are written to `runs/<protocol>/videomae_twostream_vitb_f16/train_config.json` at job start.

---

## Outputs (after successful run)

```
runs/train_test/videomae_twostream_vitb_f16/
├── train_config.json   ← exact args for reproduction
├── history.json        ← updated every epoch
├── best.pth            ← best checkpoint
├── per_class_ap.json   ← test evaluation
└── curves.png
```

Summarize:

```bash
python src/summarize.py --protocol train_test
```

---

## Prerequisites

1. `$SCRATCH/charades/charades_2stream_subset.tar` (RGB + flow)
2. `$SCRATCH/charades/annotations/`
3. `$SCRATCH/charades/videomae/vit_b_k710_dl_from_giant.pth`
4. `~/charades_venv` (`bash env/setup_env.sh`)

---

## Narval (40 GB A100)

Same recipe, smaller batch:

```bash
sbatch jobs/narval/train_videomae_twostream.sh
```

---

## Optional: resume (not the default recipe)

Only if a run crashed mid-training:

```bash
sbatch jobs/nibi/resume_videomae_twostream.sh
```

For a **clean reproducible rerun**, always use `train_videomae_twostream.sh` from epoch 1 with a fresh `--out-dir`.
Use `RUN_SUFFIX=_repro` to avoid overwriting a partial sweep checkpoint:

```bash
sbatch --export=ALL,RUN_SUFFIX=_repro jobs/nibi/train_videomae_twostream.sh
```

---

## Next steps (progress report)

Run all report follow-ups from a Nibi login node:

```bash
cd ~/narval_charades   # or your project path on Nibi
sed -i 's/\r$//' jobs/nibi/*.sh   # if synced from Windows

# Diagnose + submit resume f16 + reproducible f16
bash jobs/nibi/submit_next_steps.sh

# Also queue stable f32/f64 after resume finishes
bash jobs/nibi/submit_next_steps.sh --with-f32f64

# Diagnosis only
bash jobs/nibi/submit_next_steps.sh --diagnose-only
```

| Step | Script | Output dir |
|------|--------|------------|
| Resume sweep f16 | `resume_videomae_twostream.sh` | `videomae_twostream_vitb_f16` |
| Repro f16 epoch 1 | `train_videomae_twostream.sh` + `RUN_SUFFIX=_repro` | `videomae_twostream_vitb_f16_repro` |
| Diagnose failures | `diagnose_sweep_failures.sh` | (stdout only) |
| Stable f32/f64 | `sweep_videomae_twostream_frames_stable.sh` | `videomae_twostream_vitb_f32`, `_f64` |

---

## Multi-GPU (DDP) on Compute Canada

Training now supports PyTorch DistributedDataParallel via `torchrun`. Single-GPU jobs are unchanged (`python src/train_videomae_twostream.py`).

```bash
# 2× H100 on Nibi (default script)
sbatch jobs/nibi/train_videomae_twostream_multigpu.sh

# Same on Fir
sbatch jobs/fir/train_videomae_twostream_multigpu.sh

# 4 GPUs: override Slurm gres + NUM_GPUS
sbatch --gres=gpu:h100:4 --cpus-per-task=32 --mem=256G \
  --export=ALL,NUM_GPUS=4,RUN_SUFFIX=_ddp4 \
  jobs/nibi/train_videomae_twostream_multigpu.sh

# 8 GPUs = full Nibi node (8x H100, 192 cores, 2 TB) — dedicated script
sbatch jobs/nibi/train_videomae_twostream_8gpu.sh

# Optional: linear LR scaling with world size (VideoGLUE-style)
SCALE_LR=1 sbatch jobs/nibi/train_videomae_twostream_multigpu.sh
```

### Full Nibi node (8x H100)

Only Nibi has 8 H100 per node (Fir/Trillium/Rorqual have 4). Requesting all 8
GPUs grabs a whole node (`--mem=0`, 64 CPUs) for the highest single-node batch.

```bash
# Default: effective batch 6 x 8 = 48
sbatch jobs/nibi/train_videomae_twostream_8gpu.sh

# Push toward the VideoGLUE batch (256) via grad accumulation + longer schedule
EPOCHS=50 BATCH_SIZE=4 GRAD_ACCUM_STEPS=8 \
  sbatch jobs/nibi/train_videomae_twostream_8gpu.sh   # effective batch 256

# f64 at 8 GPUs (grad checkpointing for VRAM)
NUM_FRAMES=64 BATCH_SIZE=1 GRAD_ACCUM_STEPS=4 GRAD_CHECKPOINT=1 FRAMES_FPS=6 \
  sbatch jobs/nibi/train_videomae_twostream_8gpu.sh
```

Large effective batch → fewer optimizer steps per epoch. To avoid underfitting,
raise `EPOCHS` (e.g. 50, VideoGLUE-style) and/or enable `SCALE_LR=1` (peak LR ×
world size — aggressive, watch for NaN after flow unfreeze).

| Setting | Meaning |
|---------|---------|
| `--batch-size` / `BATCH_SIZE` | **Per-GPU** micro-batch (same as before) |
| Effective batch | `batch_size × world_size × grad_accum_steps` |
| `--scale-lr` / `SCALE_LR=1` | Multiply peak/min LR by `world_size` |
| Output dir | `videomae_twostream_vitb_f16_ddp2` by default (`RUN_SUFFIX`) |

Checkpoints stay single-process compatible (`state` without `module.` prefix). Eval after training is still single-GPU.

Manual launch (interactive / debug):

```bash
torchrun --standalone --nproc_per_node=2 src/train_videomae_twostream.py \
  --finetune "$VIDEOMAE_CKPT" --out-dir runs/train_test/videomae_twostream_vitb_f16_ddp2 \
  --batch-size 6 --num-frames 16 --epochs 30
```

---

## OOM strategies (two-stream VideoMAE)

Dual ViT-B + long clips is the main VRAM bottleneck. Use these flags:

| Flag / env | Effect | When |
|------------|--------|------|
| `--grad-checkpoint` / `GRAD_CHECKPOINT=1` | Recompute ViT blocks in backward (~30–50% less activation memory, slower) | f32 / f64, or any OOM |
| `--grad-accum-steps N` / `GRAD_ACCUM_STEPS=N` | Effective batch = `batch_size × N` without extra VRAM | Keep training stable at batch=1 |
| `--batch-size` / `BATCH_SIZE` | Lower micro-batch | First lever if OOM |
| `--num-frames` / `NUM_FRAMES` | Shorter clips | Last resort |

Examples:

```bash
# f64-safe: batch 1, accumulate 4, checkpoint blocks
NUM_FRAMES=64 BATCH_SIZE=1 GRAD_ACCUM_STEPS=4 GRAD_CHECKPOINT=1 \
  FRAMES_FPS=6 sbatch jobs/nibi/train_videomae_twostream.sh

# Resume f32 with OOM guards
NUM_FRAMES=32 BATCH_SIZE=2 GRAD_ACCUM_STEPS=2 GRAD_CHECKPOINT=1 LR=1e-4 \
  FRAMES_FPS=6 sbatch jobs/nibi/resume_videomae_twostream.sh
```

The stable f32/f64 sweep already enables checkpointing + accumulation by default.

Already on by default: AMP (`autocast` + `GradScaler`), grad clip 1.0.

---

## What we fixed vs the first Nibi run

| Issue | Fix |
|-------|-----|
| NaN at epoch 7 | LR 5e-4 (was 1e-3), freeze flow 5 ep, grad clip 1.0 |
| Slurm `_common.sh` path | `cd $SLURM_SUBMIT_DIR` |
| Eval crash on NaN | metrics + val without AMP |
| Lost history on crash | `history.json` saved every epoch |
| No reproducibility record | `train_config.json` + seed 42 |
| OOM at 32/64 frames | `--grad-checkpoint` + `--grad-accum-steps` + smaller batch |
| Single-GPU batch too small vs VideoGLUE | DDP multi-GPU (`train_videomae_twostream_multigpu.sh`) |
| Flow embed never trained after “unfreeze” | `optimizer.add_param_group` on unfreeze |
