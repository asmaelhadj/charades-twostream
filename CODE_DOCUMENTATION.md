# Code Documentation — Two-Stream Charades on Narval

**Project:** Multi-label action recognition on Charades with RGB + optical-flow two-stream networks  
**Author:** Asma El Hadj  
**Supervisor:** Prof. Éric Hervet  

This document explains **every part of the codebase**: what each file does, how data flows through the pipeline, and how the neural networks are built.

---

## Table of contents

1. [High-level overview](#1-high-level-overview)
2. [Project structure](#2-project-structure)
3. [End-to-end pipeline](#3-end-to-end-pipeline)
4. [Data layer](#4-data-layer)
5. [Model layer](#5-model-layer)
6. [Training and evaluation](#6-training-and-evaluation)
7. [Shell scripts and Slurm jobs](#7-shell-scripts-and-slurm-jobs)
8. [Outputs and artifacts](#8-outputs-and-artifacts)
9. [Design decisions](#9-design-decisions)
10. [Extension points](#10-extension-points)

---

## 1. High-level overview

### Problem

Given a **30-second domestic activity video**, predict which of **157 action classes** are present. This is **multi-label classification**: a video can have several actions at once (e.g. *Sit*, *Hold book*, *Read book*).

### Approach: Two-stream network

Inspired by Simonyan & Zisserman (2014), we use two parallel CNN streams:

```
                    ┌─────────────────────────────────────┐
                    │           Input video               │
                    └──────────────┬──────────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │                                         │
              ▼                                         ▼
     ┌─────────────────┐                    ┌─────────────────┐
     │  Spatial stream │                    │ Temporal stream │
     │  (RGB frames)   │                    │ (optical flow)  │
     │  ResNet-50      │                    │ ResNet-50       │
     └────────┬────────┘                    └────────┬────────┘
              │                                      │
              │         Late fusion                  │
              └──────────────┬───────────────────────┘
                             ▼
                    157 logits (multi-label)
                             │
                      sigmoid → probabilities
                             │
                         mAP metric
```

### Three experiments

| Experiment | `--stream` | Description |
|------------|------------|-------------|
| RGB baseline | `rgb` | Appearance only |
| Flow baseline | `flow` | Motion only |
| Two-stream fusion | `fusion` | RGB + flow combined |

For fusion, three **late fusion** strategies are compared: `average`, `weighted`, `concat`.

---

## 2. Project structure

```
narval_charades/
│
├── env/
│   └── setup_env.sh              # One-time: venv + cache ResNet weights
│
├── data/
│   ├── download_data.sh          # Download annotations + RGB + flow tarballs
│   └── build_subset.sh           # Slurm job: compact subset tar (~1.7 GB)
│
├── src/                          # Python source code (core logic)
│   ├── annotations.py            # Parse CSV labels, train/val/test splits
│   ├── preprocess.py             # Subsample + resize frames (build_subset)
│   ├── dataset.py                # PyTorch Dataset (load RGB + flow)
│   ├── models.py                 # SpatialStream, TemporalStream, TwoStreamFusion
│   ├── metrics.py                # mAP computation
│   ├── train.py                  # Training loop
│   ├── evaluate.py               # Test evaluation + plots
│   ├── summarize.py              # Compare all runs in a table
│   └── make_split.py             # Create train_val_split.json
│
├── jobs/                         # Slurm batch scripts (by cluster)
│   ├── README.md
│   ├── narval/                   # 40 GB A100 — CNN + VideoMAE RGB
│   │   ├── _common.sh
│   │   ├── train_rgb.sh / train_flow.sh / train_fusion.sh
│   │   ├── compare_fusion.sh
│   │   ├── train_videomae.sh / train_videomae_twostream.sh
│   │   └── submit_*.sh
│   └── nibi/                     # H100 — VideoMAE two-stream
│       ├── _common.sh
│       └── train_videomae_twostream.sh / sweep_*.sh
│
├── docs/
│   ├── narval/GUIDE.md           # Narval walkthrough
│   ├── narval/VIDEOMAE.md        # VideoMAE on Narval
│   └── nibi/GUIDE.md             # Nibi + two-stream VideoMAE
```

---

## 3. End-to-end pipeline

```
[Laptop]  scp project → Narval
              │
              ▼
[Login node]  setup_env.sh
              │  → $HOME/charades_venv
              │  → $SCRATCH/charades/torch_cache (ResNet-50 weights)
              ▼
[Login node]  download_data.sh
              │  → Charades.zip, Charades_v1_rgb.tar, Charades_v1_flow.tar
              ▼
[Slurm CPU]   build_subset.sh → preprocess.py
              │  → charades_2stream_subset.tar (~1.7 GB)
              ▼
[Login node]  make_split.py (train_val_test protocol only)
              │  → train_val_split.json
              ▼
[Slurm GPU]   train.py → evaluate.py
              │  → runs/<protocol>/<model>/best.pth, history.json, ...
              ▼
[Login node]  summarize.py → comparison table
```

---

## 4. Data layer

### 4.1 `annotations.py` — Labels and splits

**Role:** Convert Charades CSV files into multi-hot label vectors and manage train/val/test splits.

#### Charades annotation format

Each row in `Charades_v1_train.csv` / `Charades_v1_test.csv`:

| Column | Example | Meaning |
|--------|---------|---------|
| `id` | `AO8RW` | Video identifier |
| `actions` | `c092 11.90 21.20;c147 0.00 12.60` | Actions with time intervals |

Each action entry: `class_id start_seconds end_seconds`, separated by `;`.

We ignore timestamps and build a **video-level multi-hot vector** of length 157:

```
"c092 11.9 21.2;c147 0 12.6"  →  [0,0,...,1,...,1,...,0]  (157 dims)
                                    ↑ class c092    ↑ class c147
```

#### Key functions

| Function | Purpose |
|----------|---------|
| `load_classes(annot_dir)` | Read `Charades_v1_classes.txt` → dict `{c000: 0, c001: 1, ...}` + class names |
| `parse_actions(cell)` | Parse `"c092 11.9 21.2;c147 ..."` → `['c092', 'c147']` |
| `build_labels(annot_dir, split, protocol)` | Return `(video_ids, label_matrix)` for a split |
| `positive_counts(annot_dir, protocol)` | Per-class positive count on train fold (for loss weighting) |
| `checkpoint_split(protocol)` | Which split name is used to pick `best.pth` during training |

#### Two split protocols

| Protocol | Train | Validation (checkpoint) | Test (final eval) |
|----------|-------|-------------------------|-------------------|
| `train_val_test` | 80% of official train | 20% holdout from train | Official test |
| `train_test` | Full official train | Official test | Official test |

For `train_val_test`, the 80/20 split is saved in `train_val_split.json` (seed=42, reproducible).

```python
# Example usage inside dataset.py
ids, labels = build_labels(annot_dir, split="train", protocol="train_val_test")
# ids:   ['AO8RW', 'Y6R7T', ...]
# labels: np.array shape [N, 157], dtype float32, values 0.0 or 1.0
```

---

### 4.2 `preprocess.py` — Build compact subset

**Role:** Run once by `build_subset.sh`. Reads full extracted RGB + flow folders, subsamples frames, resizes, writes a compact copy.

**Why:** Full Charades tarballs contain millions of small JPEGs. Reading them from Lustre (`/scratch`) every training job is slow. We preprocess once into one ~1.7 GB tar.

#### Algorithm (per video)

1. List all RGB `.jpg` files in `Charades_v1_rgb/<video_id>/`
2. Keep **16 evenly spaced** frames via `np.linspace`
3. Resize so **short side = 256 px** (aspect ratio preserved)
4. Same for flow: keep x/y pairs aligned, 16 pairs
5. Write to output with **same naming convention** as original Charades

#### Output layout

```
subset/
├── rgb_frames/
│   └── AO8RW/
│       ├── AO8RW-000000.jpg
│       ├── AO8RW-000012.jpg
│       └── ...  (16 frames)
└── flow_frames/
    └── AO8RW/
        ├── AO8RW-000000x.jpg   (horizontal flow)
        ├── AO8RW-000000y.jpg   (vertical flow)
        └── ...
```

Uses `ProcessPoolExecutor` for parallel processing across videos (16 workers on Slurm CPU job).

---

### 4.3 `dataset.py` — PyTorch Dataset

**Role:** Load RGB frames + optical flow for one video, apply transforms, return tensors.

#### Class: `CharadesTwoStream(Dataset)`

**Constructor parameters:**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `root` | — | Path to extracted subset (`$SLURM_TMPDIR/data`) |
| `annot_dir` | — | Path to annotations CSVs |
| `split` | — | `'train'`, `'val'`, or `'test'` |
| `protocol` | `train_val_test` | Split protocol (see annotations.py) |
| `num_rgb` | 8 | RGB frames sampled per video per batch item |
| `num_flow` | 10 | Flow frame pairs sampled (→ 20 channels) |
| `img_size` | 224 | Final crop size |
| `train` | True | Enable augmentation if True |
| `load_rgb` / `load_flow` | True | Which modalities to load (baselines disable one) |

**Returns per `__getitem__`:**

| Tensor | Shape | Description |
|--------|-------|-------------|
| `rgb` | `[8, 3, 224, 224]` | 8 RGB frames, ImageNet-normalized |
| `flow` | `[20, 224, 224]` | 10 flow pairs (x+y), normalized to [-1, 1] |
| `label` | `[157]` | Multi-hot float tensor |

#### Frame sampling — `_pick(files, n)`

1. If video has fewer than `n` frames → repeat to fill
2. Select `n` indices evenly spaced: `np.linspace(0, len-1, n)`
3. **Training only:** add random jitter ±2 frames for augmentation

#### Augmentation — `_geom()` (training only)

Shared geometry for RGB and flow so they stay aligned:

1. Resize to `(img_size + 32) × (img_size + 32)` → random crop to `224 × 224`
2. Random horizontal flip (50%)

Validation/test: center crop only, no flip.

#### Normalization

- **RGB:** ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
- **Flow:** `(pixel - 0.5) / 0.5` → range `[-1, 1]`

---

## 5. Model layer

### 5.1 `models.py` — Neural network architectures

All models output **157 raw logits** (no softmax). Probabilities = `sigmoid(logits)`.

#### Shared backbone: ResNet-50 (ImageNet pretrained)

```python
def _resnet50_backbone():
    bb = models.resnet50(weights=IMAGENET1K_V1)
    backbone = Sequential(*list(bb.children())[:-1])  # remove fc layer
    # Output: [batch, 2048, 1, 1]
```

Pretrained weights are cached in `$SCRATCH/charades/torch_cache` because compute nodes have no internet.

---

#### `SpatialStream` — RGB stream

**Input:** `[B, T, 3, H, W]` — batch of T RGB frames per video

**Architecture:**

```
Input [B, 8, 3, 224, 224]
        │
        ▼ reshape to [B*8, 3, 224, 224]
ResNet-50 backbone (no fc)
        │
        ▼ [B*8, 2048]
reshape + temporal average pool
        │
        ▼ [B, 2048]
Dropout(0.5) → Linear(2048→512) → ReLU
        │
        ├─ feats_only=True  → [B, 512] features
        └─ feats_only=False → Linear(512→157) → [B, 157] logits
```

**Temporal pooling:** Average features across the 8 frames → one vector per video.

---

#### `TemporalStream` — Optical flow stream

**Input:** `[B, 20, H, W]` — 10 flow frame pairs stacked as 20 grayscale channels

**Key modification:** Replace ResNet's first conv layer:

```python
# Original: Conv2d(3, 64, kernel_size=7, ...)
# New:      Conv2d(20, 64, kernel_size=7, ...)

# Weight initialization ("inflation"):
avg_w = original_conv.weight.mean(dim=1)   # average RGB filters
new_conv.weight = avg_w.repeat(1, 20, 1, 1) / 20
```

This transfers ImageNet spatial filters to the 20-channel flow input.

**Architecture:**

```
Input [B, 20, 224, 224]
        │
ResNet-50 backbone (modified conv1)
        │
        ▼ [B, 2048]
Dropout(0.5) → Linear(2048→512) → ReLU
        │
        └─ Linear(512→157) → [B, 157] logits
```

No temporal pooling needed — all 20 flow channels are stacked into one image.

---

#### `TwoStreamFusion` — Late fusion

Combines spatial + temporal streams **after** feature extraction (late fusion).

**Three strategies:**

##### 1. `average` fusion

```python
logits = (spatial_logits + temporal_logits) / 2
```

Simple, no extra parameters.

##### 2. `weighted` fusion

```python
wr = sigmoid(learnable_w_rgb)
wf = sigmoid(learnable_w_flow)
logits = (wr * spatial_logits + wf * temporal_logits) / (wr + wf)
```

Two learnable scalar weights — model learns how much to trust each stream.

##### 3. `concat` fusion

```python
f_rgb  = spatial_stream(rgb,  feats_only=True)   # [B, 512]
f_flow = temporal_stream(flow, feats_only=True)  # [B, 512]
f = concat([f_rgb, f_flow])                      # [B, 1024]
logits = MLP(f)                                   # Linear→ReLU→Dropout→Linear → [B, 157]
```

Concatenate features, then a small MLP head learns the combination.

---

#### Factory function

```python
build_model(stream='rgb'|'flow'|'fusion', fusion='average'|'weighted'|'concat', ...)
```

Used by both `train.py` and `evaluate.py` to construct the correct model.

---

### 5.2 `metrics.py` — Evaluation metric

#### Charades mAP (mean Average Precision)

Standard metric for multi-label Charades classification.

```python
def charades_map(y_true, y_score):
    # y_true:  [N, 157] binary ground truth
    # y_score: [N, 157] predicted probabilities (after sigmoid)
    
    for each class c:
        if class c has at least one positive in y_true:
            AP[c] = average_precision_score(y_true[:, c], y_score[:, c])
    
    mAP = mean(AP[c] for all classes with positives)
```

**Average Precision (AP)** for one class: area under the precision-recall curve when thresholding predicted scores for that class.

**mAP:** Mean of AP over all 157 classes (only classes with ≥1 positive in the eval set are included).

---

## 6. Training and evaluation

### 6.1 `train.py` — Training loop

#### Command-line arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--stream` | required | `rgb`, `flow`, or `fusion` |
| `--fusion` | `concat` | Fusion strategy (fusion stream only) |
| `--protocol` | `train_val_test` | Split protocol |
| `--epochs` | 25 | Training epochs |
| `--batch-size` | 32 | Batch size (24 for fusion) |
| `--lr` | 1e-4 | AdamW learning rate |
| `--num-rgb` | 8 | RGB frames per video |
| `--num-flow` | 10 | Flow pairs per video |
| `--img-size` | 224 | Crop size |
| `--no-pos-weight` | False | Disable class imbalance weighting |

#### Training loop (per epoch)

```
for each batch (rgb, flow, labels):
    1. Forward pass (FP16 mixed precision)
    2. BCEWithLogitsLoss(logits, labels)
    3. Backward + gradient clip (max norm 5.0)
    4. AdamW step

for each val batch:
    1. Forward (no grad)
    2. sigmoid(logits) → probabilities
    3. Compute val mAP

if val_mAP > best_map:
    save best.pth (model weights + epoch + args)

cosine LR schedule step
```

#### Loss: `BCEWithLogitsLoss` with `pos_weight`

Multi-label → Binary Cross-Entropy per class (not softmax).

Class imbalance handled via per-class positive weights:

```python
pos_weight[c] = clip((n_train - pos[c]) / pos[c], 1.0, 20.0)
```

Rare classes get higher weight so the model doesn't ignore them.

#### Checkpoint saved in `best.pth`

```python
{
    "epoch": int,
    "state": model.state_dict(),
    "val_map": float,
    "args": {all training arguments}
}
```

#### History saved in `history.json`

```python
{
    "protocol": "train_val_test",
    "checkpoint_split": "val",
    "best_map": 0.1629,
    "train_loss": [0.05, 0.04, ...],   # per epoch
    "val_loss":   [0.06, 0.05, ...],
    "val_map":    [0.12, 0.14, ...]
}
```

---

### 6.2 `evaluate.py` — Final test evaluation

Runs **after training** inside the same Slurm job.

1. Load `best.pth`
2. Reconstruct model from saved `args`
3. Evaluate on **official test split only** (`split="test"`)
4. Compute test mAP + per-class AP
5. Save `per_class_ap.json` and `curves.png`

**Output files:**

| File | Content |
|------|---------|
| `per_class_ap.json` | Overall test mAP + AP for each of 157 classes |
| `curves.png` | Train/val loss and val mAP vs epoch |

---

### 6.3 `summarize.py` — Compare all experiments

Scans `runs/<protocol>/<model>/history.json` and prints a table:

```
Protocol: train_val_test
run                  best val mAP   test mAP
----------------------------------------------
rgb                        0.1580     0.1540
flow                       0.0950     0.0910
fusion_average             0.1610     0.1590
fusion_weighted            0.1605     0.1585
fusion_concat              0.1420     0.1400
```

---

### 6.4 `make_split.py` — Create validation holdout

Run once on login node before training with `train_val_test` protocol:

```bash
python src/make_split.py --annot-dir $SCRATCH/charades/annotations
```

Creates `$SCRATCH/charades/annotations/train_val_split.json`:

```json
{
  "seed": 42,
  "val_frac": 0.2,
  "n_train": 6388,
  "n_val": 1597,
  "train_ids": ["AO8RW", ...],
  "val_ids": ["Y6R7T", ...]
}
```

Not needed for `train_test` protocol.

---

## 7. Shell scripts and Slurm jobs

### 7.1 `env/setup_env.sh`

Run **once** on login node. Creates:

- `$HOME/charades_venv` — Python virtual environment
- `$SCRATCH/charades/torch_cache` — cached ResNet-50 ImageNet weights

Uses Alliance wheelhouse (`pip install --no-index`) because compute nodes have no internet.

---

### 7.2 `data/download_data.sh`

Downloads from AllenAI S3 to `$SCRATCH/charades/archives/`:

| File | Size |
|------|------|
| `Charades.zip` | ~3 MB |
| `Charades_v1_rgb.tar` | ~75 GB |
| `Charades_v1_flow.tar` | ~45 GB |

Also extracts annotations to `$SCRATCH/charades/annotations/`.

---

### 7.3 `data/build_subset.sh`

Slurm **CPU job** (16 cores, 64 GB, 4 h). Pipeline:

1. Copy tarballs to `$SLURM_TMPDIR` (node-local SSD)
2. Extract RGB + flow
3. Run `preprocess.py` (16 frames/video, 256 px short side)
4. Repack to `$SCRATCH/charades/charades_2stream_subset.tar`

---

### 7.4 `jobs/narval/_common.sh`

Sourced by every GPU job. Does:

```bash
module load StdEnv/2023 python/3.11
source ~/charades_venv/bin/activate
export PROTOCOL="${PROTOCOL:-train_val_test}"
export RUN_ROOT="runs/${PROTOCOL}"
export TORCH_HOME="$SCRATCH/charades/torch_cache"
cp charades_2stream_subset.tar → $SLURM_TMPDIR
tar -xf ... → $SLURM_TMPDIR/data
export DATA_ROOT=$SLURM_TMPDIR/data
export ANNOT_DIR=$SCRATCH/charades/annotations
```

---

### 7.5 Training job scripts

| Script | GPU | Time | What it runs |
|--------|-----|------|--------------|
| `train_rgb.sh` | 1 | 10 h | RGB baseline |
| `train_flow.sh` | 1 | 10 h | Flow baseline |
| `train_fusion.sh` | 1 | 16 h | One fusion strategy |
| `compare_fusion.sh` | 3 (array) | 16 h | All 3 fusion strategies in parallel |

Each job runs `train.py` then `evaluate.py` sequentially.

**Submit helpers:**

```bash
bash jobs/narval/submit_all.sh train_val_test    # all models, one protocol
bash jobs/narval/submit_all.sh train_test        # all models, other protocol
bash jobs/narval/submit_both_protocols.sh        # both protocols
```

---

## 8. Outputs and artifacts

After a full experiment batch, directory layout:

```
runs/train_val_test/
├── rgb/
│   ├── best.pth              # Best checkpoint (by val mAP)
│   ├── history.json          # Loss + mAP per epoch
│   ├── per_class_ap.json     # Test mAP + AP per class
│   └── curves.png            # Training curves plot
├── flow/
│   └── ...
├── fusion_average/
├── fusion_weighted/
└── fusion_concat/

runs/charades_rgb-<jobid>.out   # Slurm stdout log
```

---

## 9. Design decisions

### Why pre-extracted flow (not compute on-the-fly)?

AllenAI provides pre-computed optical flow tarballs. Computing Farneback flow from video during training would be extremely slow on CPU and unnecessary.

### Why compact subset tar?

Lustre (`/scratch`) performs poorly with millions of small files. One tar file (~1.7 GB) copies to node-local SSD in seconds; extraction is fast on local disk.

### Why BCE and not CrossEntropy?

Charades is **multi-label**: a video can have multiple actions simultaneously. CrossEntropy assumes exactly one class per sample. BCE treats each of the 157 classes as an independent binary problem.

### Why late fusion (not early fusion)?

Late fusion trains each stream almost independently and combines at the logit/feature level. It's simpler, allows single-stream baselines with the same code, and matches the original two-stream paper architecture.

### Why two protocols?

- `train_test`: simple, matches many Charades papers; test used for model selection
- `train_val_test`: methodologically cleaner; test never seen during training decisions

Having both lets you compare and explain results to your supervisor.

### Why venv in `$HOME` not `$PROJECT`?

Narval project filesystem has a **500K file limit** shared across the lab. A Python venv alone is ~5,000 files. `$HOME` has plenty of room.

---

## 10. Extension points

Places to modify for future experiments:

| Goal | Where to change |
|------|-----------------|
| More frames per video | `--num-rgb`, `--num-flow`; also `--frames-per-video` in `build_subset.sh` |
| Longer training | `--epochs` in job scripts |
| Different backbone | `_resnet50_backbone()` in `models.py` |
| Different fusion | Add strategy in `TwoStreamFusion.forward()` |
| Stronger augmentation | `_geom()` in `dataset.py` |
| Learning rate schedule | `scheduler` in `train.py` |
| Multi-GPU | Wrap model in `DistributedDataParallel` in `train.py` |
| Top-K classes only | Filter in `build_labels()` or `annotations.py` |
| Temporal modeling (LSTM/Transformer) | Replace temporal avg pool in `SpatialStream` |

---

## Quick reference: tensor shapes through the pipeline

```
Disk:  rgb_frames/<id>/*.jpg          flow_frames/<id>/*x.jpg, *y.jpg
         │                                    │
Dataset: [8, 3, 224, 224]              [20, 224, 224]        [157]
         │                                    │
Spatial: [B, 8, 3, 224, 224] → [B, 157] logits
Temporal: [B, 20, 224, 224] → [B, 157] logits
Fusion:  rgb + flow → [B, 157] logits
         │
Loss:    BCEWithLogitsLoss(logits, labels)
         │
Metric:  sigmoid(logits) → mAP
```

---

## 11. VideoMAE V2 pipeline (alternative model)

A transformer alternative to the two-stream CNN lives alongside this code and
**reuses** the same `annotations.py` (labels/splits/protocols), `metrics.py`
(mAP), the compact `rgb_frames/` subset, and the Slurm plumbing in
`jobs/narval/_common.sh`. Full run instructions are in **`docs/narval/VIDEOMAE.md`**.

| File | Role |
|------|------|
| `src/videomae_model.py` | Self-contained VideoMAE V2 fine-tuning ViT (B/L/H). Module/parameter names match the official `OpenGVLab/VideoMAEv2` checkpoints, so released weights load with `strict=False` after the pretrained head is dropped. No `timm`/`einops`/`decord` dependency — the two helpers it needs (`drop_path`, `trunc_normal_`) are vendored, because compute nodes have no internet. |
| `src/dataset_videomae.py` | `CharadesVideoClips`: returns one RGB clip `[3, T, H, W]` + multi-hot `[157]`. Reads the same 16-frame subset; a video becomes one clip (matches VideoMAE's `num_frames=16`), so **no extra video download** is needed. |
| `src/train_videomae.py` | Fine-tuning: sigmoid + `BCEWithLogitsLoss` (Mixup/CutMix off), AdamW with **layer-wise LR decay**, linear-warmup + cosine schedule, AMP. Writes `best.pth` / `history.json` in the same layout as `train.py`. |
| `src/evaluate_videomae.py` | Test mAP + per-class AP + `curves.png`, same layout as `evaluate.py` (so `summarize.py` lists `videomae_vitb` automatically). |
| `data/download_videomae_ckpt.sh` | Login-node download of the pretrained backbone (`vit_b_k710_dl_from_giant.pth`, ~174 MB) to `$SCRATCH`. |
| `jobs/narval/train_videomae.sh` | Slurm GPU job: fine-tune then evaluate. |

### Input / output shapes

```
Subset:  rgb_frames/<id>/*.jpg   (16 frames)
Dataset: [3, 16, 224, 224]                       label [157]
Model:   PatchEmbed3D (tubelet 2) -> 1568 tokens -> 12 ViT blocks -> mean-pool
         -> fc_norm -> head -> logits [B, 157]
Loss:    BCEWithLogitsLoss(logits, labels)
Metric:  sigmoid(logits) -> macro mAP
```

### Why a separate model file instead of cloning the official repo?

Compute nodes have no internet and the Alliance wheelhouse may not carry the
repo's pinned `timm==0.4.12`. Vendoring a faithful, key-compatible ViT keeps the
project install-free beyond what the two-stream code already needs, and the
released weights still load unchanged.

---

*Documentation for the Mitacs project — Two-Stream Networks + VideoMAE V2 on Charades, Narval cluster.*
