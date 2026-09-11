# Two-Stream VideoMAE V2 on Nibi (H100)

Cluster-specific guide. Shared code lives in `src/`; jobs in `jobs/nibi/`.
For Narval (CNN baselines, RGB VideoMAE), see [../narval/GUIDE.md](../narval/GUIDE.md).

---

## What was added

| File | Role |
|------|------|
| `src/dataset_videomae_twostream.py` | Loads aligned RGB `[3,T,H,W]` + flow `[2,T,H,W]` from existing subset |
| `src/videomae_model.py` | `TwoStreamVideoMAEv2`, `build_twostream_videomae()` |
| `src/train_videomae_twostream.py` | Fine-tuning (BCE, layer decay, freeze flow embed 3 epochs) |
| `src/evaluate_videomae_twostream.py` | Test mAP + curves |
| `jobs/nibi/_common.sh` | Nibi env + data staging |
| `jobs/nibi/train_videomae_twostream.sh` | Main two-stream job (16 frames) |
| `jobs/nibi/sweep_videomae_twostream_frames.sh` | Sweep 16 / 32 / 48 / 64 frames |
| `jobs/nibi/train_videomae_rgb.sh` | RGB-only baseline (larger batch on H100) |

Nibi requires an explicit GPU type in every job: `#SBATCH --gres=gpu:h100:1`

Flow source: **AllenAI pre-extracted** `flow_frames/*x.jpg, *y.jpg` (same as CNN two-stream). No mp4 download, no RAFT.

---

## Architecture (quick recap)

```
rgb_frames  → rgb_net  (VideoMAE ViT-B, 3 ch)  ──┐
                                                  ├─ concat → MLP → 157 logits
flow_frames → flow_net (VideoMAE ViT-B, 2 ch)  ──┘
```

- Flow backbone initialized from the same K710 RGB checkpoint (first 2 conv channels).
- Flow `patch_embed` frozen for 3 epochs, then unfrozen.

---

## Setup on Nibi (one-time)

### 1. Clone / copy the project

```bash
ssh <user>@nibi.alliancecan.ca
cd ~/projects/def-hervete/$USER
# git clone … or rsync from Narval:
# rsync -av narval:~/projects/def-hervete/$USER/narval_charades/ ./narval_charades/
cd narval_charades
```

Fix Windows line endings if needed:

```bash
sed -i 's/\r$//' jobs/nibi/*.sh data/*.sh env/*.sh
```

### 2. Python environment (login node)

```bash
bash env/setup_env.sh
```

### 3. Data on `$SCRATCH`

You need annotations + the **RGB+flow** subset tar (not RGB-only):

```bash
bash data/download_data.sh          # annotations + full rgb/flow tarballs
sbatch data/build_subset.sh         # 16 frames/video, ~5 GB tar

# For 32+ frame sweeps, also build the dense fps subsets:
sbatch data/build_subset_fps.sh     # 3/6/9/12 fps tars
```

Copy from Narval if already built:

```bash
rsync -av narval:$SCRATCH/charades/charades_2stream_subset*.tar $SCRATCH/charades/
rsync -av narval:$SCRATCH/charades/annotations/ $SCRATCH/charades/annotations/
```

### 4. VideoMAE pretrained weights

```bash
bash data/download_videomae_ckpt.sh
# → $SCRATCH/charades/videomae/vit_b_k710_dl_from_giant.pth
```

---

## Run jobs

```bash
# Two-stream VideoMAE (16 frames, batch 6) — 1× H100
sbatch jobs/nibi/train_videomae_twostream.sh

# Multi-GPU DDP (2× H100) — larger effective batch
sbatch jobs/nibi/train_videomae_twostream_multigpu.sh

# RGB-only baseline for comparison (larger batch on H100)
sbatch jobs/nibi/train_videomae_rgb.sh

# Frame sweep 16/32/48/64 (uses 6 fps subset by default)
sbatch jobs/nibi/sweep_videomae_twostream_frames.sh

# Optional overrides
PROTOCOL=train_test sbatch jobs/nibi/train_videomae_twostream.sh
NUM_FRAMES=32 BATCH_SIZE=4 sbatch jobs/nibi/train_videomae_twostream.sh
FRAMES_FPS=9 sbatch jobs/nibi/sweep_videomae_twostream_frames.sh
SCALE_LR=1 sbatch jobs/nibi/train_videomae_twostream_multigpu.sh
```

Monitor:

```bash
squeue -u $USER
tail -f runs/nibi_vmae_2stream-*.out
python src/summarize.py
```

**Next steps** (resume f16, repro run, diagnose sweep, stable f32/f64):

```bash
bash jobs/nibi/submit_next_steps.sh
bash jobs/nibi/submit_next_steps.sh --with-f32f64   # f32/f64 after resume
```

See [REPRODUCE.md](REPRODUCE.md) for details.

Outputs: `runs/<protocol>/videomae_twostream_vitb_f16/` (or `_f32`, `_f16_repro`, etc.)

---

## Batch sizes (H100 vs Narval A100 40 GB)

| Config | Narval (RGB only) | Nibi two-stream | Nibi RGB only |
|--------|-------------------|-----------------|---------------|
| 16 frames | 8 | **6** | **12** |
| 32 frames | 4 | **4** | **8** |
| 48 frames | 2 | **2** | **4** |
| 64 frames | 1 | **2** | **2** |

If OOM: lower `BATCH_SIZE` env var or `--batch-size`.

---

## Compare runs

`summarize.py` picks up all runs under `runs/<protocol>/<name>/`:

```
videomae_vitb_f16              # RGB-only baseline
videomae_twostream_vitb_f16    # RGB + flow fusion
fusion_concat                  # CNN two-stream (if trained)
```

---

## Meeting talking points (Éric)

1. **No RAFT** — we reuse AllenAI flow already on disk.
2. **Same data** as ResNet two-stream; only the encoders change (ViT vs ResNet).
3. **Nibi (H100)** — 80 GB VRAM lets us run dual ViT-B at 32–64 frames without batch=1.
4. **Open experiment** — does flow help VideoMAE as much as it helps CNNs?
