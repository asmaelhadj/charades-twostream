# VideoMAE V2 on Charades — Narval (Compute Canada)

**Author:** Asma El Hadj · **Supervisor:** Prof. Éric Hervet · **Cluster:** Narval (`def-hervete`)

This adds a **VideoMAE V2** (ViT) fine-tuning pipeline for multi-label Charades,
sitting *alongside* the existing two-stream CNN code. It reuses the same
annotations, mAP metric, split protocols, compact data subset, and Slurm
plumbing — so the only genuinely new pieces are the model, the clip dataset,
and a train/eval pair.

VideoMAE V2 is a self-supervised video pretraining method (dual masking, CVPR
2023). We do **not** pretrain — we take the released **ViT-B distilled from the
giant K710 model** and fine-tune its classification head (+ backbone) on
Charades with sigmoid + BCE, reporting macro **mAP**.

---

## 1. What was added

| File | Role |
|------|------|
| `src/videomae_model.py` | Self-contained VideoMAE V2 fine-tuning ViT (B/L/H). Key names match the official checkpoints so released weights load directly. No `timm`/`einops`/`decord` dependency. |
| `src/dataset_videomae.py` | `CharadesVideoClips` — returns one RGB clip `[3, T, H, W]` + multi-hot label `[157]`, reading the **same** `rgb_frames/` subset the two-stream jobs use. |
| `src/train_videomae.py` | Fine-tuning loop: BCE (sigmoid), AdamW with **layer-wise LR decay**, linear-warmup + cosine schedule, AMP. Saves `best.pth` / `history.json` (same layout as `train.py`). |
| `src/evaluate_videomae.py` | Test mAP + per-class AP + `curves.png` (same layout as `evaluate.py`; `summarize.py` picks it up). |
| `data/download_videomae_ckpt.sh` | Login-node download of the pretrained backbone to `$SCRATCH`. |
| `jobs/narval/train_videomae.sh` | Slurm GPU job: train then evaluate. |

### Design choice: reuse the RGB subset (no 55 GB video download)

The official recipe reads `.mp4` files via `decord`. We instead **reuse the
compact `rgb_frames/` subset** already built by `data/build_subset.sh` (16
evenly-spaced JPEG frames per video). That number matches VideoMAE V2's default
`num_frames=16`, so a video = one clip with no extra download, no `decord`, and
no Lustre small-file problem. If you rebuild the subset with more frames,
`--num-frames` evenly-spaced frames are sampled from whatever is available.

---

## 2. How to run on Narval

Assumes the two-stream setup is already done (venv, dataset downloaded, and
`charades_2stream_subset.tar` built — see `README.md` / `docs/narval/GUIDE.md` steps 0–5).
If so, only **two** new steps are needed.

### Step A — download the pretrained backbone (LOGIN node, has internet)

```bash
cd ~/projects/def-hervete/$USER/narval_charades
sed -i 's/\r$//' data/download_videomae_ckpt.sh jobs/narval/train_videomae.sh   # if copied from Windows
bash data/download_videomae_ckpt.sh        # ~174 MB -> $SCRATCH/charades/videomae/
```

This fetches `vit_b_k710_dl_from_giant.pth` from the official Hugging Face
mirror (`OpenGVLab/VideoMAE2`).

### Step B — fine-tune + evaluate (Slurm GPU job)

```bash
sbatch jobs/narval/train_videomae.sh                       # default protocol: train_val_test
PROTOCOL=train_test sbatch jobs/narval/train_videomae.sh   # simple train/test protocol
```

Monitor and summarize:

```bash
squeue -u $USER
tail -f runs/charades_videomae-*.out
python src/summarize.py        # videomae_vitb appears next to rgb/flow/fusion_*
```

Outputs land in `runs/<protocol>/videomae_vitb/`:
`best.pth`, `history.json`, `per_class_ap.json`, `curves.png`.

---

## 3. Training recipe (defaults in `jobs/narval/train_videomae.sh`)

| Setting | Value | Note |
|---------|-------|------|
| Backbone | ViT-B/16, 16 frames, tubelet 2 | `vit_base_patch16_224` |
| Loss | `BCEWithLogitsLoss` (sigmoid) | multi-label; Mixup/CutMix **off** |
| Optimizer | AdamW, wd 0.05 | no weight decay on 1-D params |
| LR | peak `1e-3`, layer-decay `0.75` | lower layers get smaller LR |
| Schedule | 2-epoch warmup → cosine to `1e-6` | |
| Drop path | 0.1 | stochastic depth |
| Epochs / batch | 30 / 8 | lower batch if OOM |
| Metric | macro mAP (+ weighted) | only classes with ≥1 positive |

`pos_weight` BCE re-weighting is **off** by default here (the standard VideoMAE
recipe uses plain BCE, comparable to the published 44.0 mAP). Enable it with
`--pos-weight` if rare classes are being ignored.

### Expected ballpark (full Charades, official recipe)

| Backbone | Pretrain | mAP |
|----------|----------|-----|
| VideoMAE V2 ViT-B | UnlabeledHybrid + K710 | ~44 |
| VideoMAE V2 ViT-L | UnlabeledHybrid + K710 | ~51 |

Our numbers will be a bit lower than the paper because we fine-tune on a
**16-frame subset** (single clip), not full multi-clip / multi-crop testing.
Still, ViT-B here should clearly beat the ResNet-50 two-stream baselines.

---

## 3b. Frame-count sweep (16 / 32 / 48 / 64)

`jobs/narval/sweep_videomae_frames.sh` fine-tunes ViT-B at four temporal lengths and
compares mAP. Unlike the two-stream **fps** sweep (which changes how many frames
the CNN averages), this changes the **model's own clip length**.

Why it needs no checkpoint surgery: VideoMAE V2's positional embedding is
**sinusoidal** (computed analytically, so it regenerates for any token count),
and the patch-embed `Conv3d` kernel is frame-count independent. More frames just
means more tokens (16 frames → 1568 tokens, 64 frames → 6272). Every pretrained
weight still loads; only the classification head is (re)initialised.

All levels read from one **dense subset** (default the 6 fps one, ~180 frames/
video — enough for 64). Build it first if you have not already:

```bash
sbatch data/build_subset_fps.sh        # builds 3/6/9/12 fps subsets
```

Then submit the sweep (on the GPU queue):

```bash
sbatch jobs/narval/sweep_videomae_frames.sh                 # reads 6 fps subset
FRAMES_FPS=9 sbatch jobs/narval/sweep_videomae_frames.sh    # read a denser subset instead
python src/summarize.py                              # videomae_vitb_f16 / _f32 / _f48 / _f64
```

Notes:
- Self-attention cost grows ~quadratically with frames, so the job lowers the
  batch size automatically (16→8, 32→4, 48→2, 64→1). Lower further on OOM.
- 48/64-frame runs are slow (more tokens, tiny batch); reduce `--epochs` if a
  run risks exceeding the 24 h walltime.

## 3c. Two-stream VideoMAE V2 (RGB + optical flow)

Dual VideoMAE ViT backbones with late fusion, reading the **same precomputed**
`rgb_frames/` + `flow_frames/` as the CNN two-stream jobs (AllenAI flow, not RAFT).

| File | Role |
|------|------|
| `src/dataset_videomae_twostream.py` | RGB clip `[3,T,H,W]` + flow clip `[2,T,H,W]` |
| `src/videomae_model.py` | `TwoStreamVideoMAEv2`, `build_twostream_videomae()` |
| `src/train_videomae_twostream.py` | Training loop |
| `src/evaluate_videomae_twostream.py` | Test evaluation |

**Nibi (H100):** see [docs/nibi/GUIDE.md](../nibi/GUIDE.md) — larger batches and frame sweeps with two ViT-B
backbones. Narval jobs can use the same Python entry points with lower `--batch-size`.

## 4. Common pitfalls (already handled)

1. **No internet on compute nodes** → backbone is pre-downloaded to `$SCRATCH`;
   the model file vendors its few helpers (no `timm`).
2. **Wrong loss** → sigmoid + BCE (never softmax/CE); labels co-occur.
3. **Head mismatch** → the pretrained 710-class head is dropped automatically;
   the loader prints missing/unexpected keys (only `head.*` should be missing).
4. **OOM** → lower `--batch-size` (e.g. 4) in `jobs/narval/train_videomae.sh`.
5. **Switch backbone size** → `--model vit_large_patch16_224` and download the
   matching checkpoint (`MODEL=vit_s bash data/download_videomae_ckpt.sh` etc.).
