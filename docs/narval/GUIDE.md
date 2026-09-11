# Two-Stream Charades on Narval — Complete Guide

**Author:** Asma El Hadj  
**Supervisor:** Prof. Éric Hervet  
**Cluster:** Narval (Alliance Canada / Compute Canada)  
**Account:** `def-hervete`  
**Task:** Multi-label action recognition with RGB + optical-flow two-stream networks

This document explains **what was built**, **why**, and **how to run every step** on Narval from start to finish.

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [What was implemented](#2-what-was-implemented)
3. [Directory layout on Narval](#3-directory-layout-on-narval)
4. [Storage strategy (important)](#4-storage-strategy-important)
5. [Step-by-step workflow](#5-step-by-step-workflow)
6. [What each script does](#6-what-each-script-does)
7. [Training pipeline (technical)](#7-training-pipeline-technical)
8. [Expected outputs](#8-expected-outputs)
9. [Troubleshooting](#9-troubleshooting)
10. [Comparison with the Colab notebook](#10-comparison-with-the-colab-notebook)
11. [Quick command cheat sheet](#11-quick-command-cheat-sheet)

---

## 1. Project overview

### Goal

Implement **two-stream networks** for the **Charades** dataset:

| Component | Role |
|-----------|------|
| **RGB stream (spatial)** | ResNet-50 on appearance / object context |
| **Flow stream (temporal)** | ResNet-50 on motion (pre-computed optical flow) |
| **Late fusion** | Combine both streams after feature extraction |
| **Baselines** | Train RGB-only and flow-only models for comparison |

### Dataset: Charades

- **9,848 videos** (~30 s each), domestic activities filmed at home
- **157 action classes**, multi-label: each video can have several actions
- Official metric: **mAP** (mean Average Precision over classes)
- Source: [AllenAI Charades](https://prior.allenai.org/projects/charades)

### Why Narval instead of Colab?

- Charades is **~120 GB** (RGB + flow) — too large for local storage
- Compute Canada provides **$SCRATCH` (20 TB)** and GPU nodes via **Slurm**
- Same HPC concepts apply even when using a department server later

---

## 2. What was implemented

### Models (`src/models.py`)

```
Video
 ├── RGB frames  ──► SpatialStream (ResNet-50)  ──► logits [157]
 └── Flow frames ──► TemporalStream (ResNet-50) ──► logits [157]
                              │
                    Late fusion (3 strategies)
                              │
                         final logits [157]
                              │
                         sigmoid → probabilities
```

| Model | Description |
|-------|-------------|
| `SpatialStream` | ResNet-50 on 8 RGB frames; temporal average pooling |
| `TemporalStream` | ResNet-50 with inflated `conv1` (20 input channels = 10 flow frames × x/y) |
| `TwoStreamFusion` | Late fusion: `average`, `weighted` (learnable), or `concat` (MLP head) |

### Loss and metrics

- **Loss:** `BCEWithLogitsLoss` with optional per-class `pos_weight` (handles class imbalance)
- **Metric:** Charades **mAP** — mean of per-class Average Precision

### Experiments to run

| Job script | What it trains |
|------------|----------------|
| `jobs/narval/train_rgb.sh` | RGB baseline only |
| `jobs/narval/train_flow.sh` | Flow baseline only |
| `jobs/narval/train_fusion.sh` | Two-stream fusion (default: concat) |
| `jobs/narval/compare_fusion.sh` | Slurm array: all 3 fusion strategies in parallel |

Success criterion: **fusion model should beat both single-stream baselines** on mAP.

---

## 3. Directory layout on Narval

After organizing the project under your lab space:

```
/home/$USER/
├── charades_venv/                          ← Python virtual environment (HOME, not project)
└── projects/def-hervete/$USER/
    └── narval_charades/                    ← CODE — always run commands from here
        ├── env/setup_env.sh
        ├── data/
        │   ├── download_data.sh
        │   └── build_subset.sh
        ├── src/                            ← Python source
        ├── jobs/
        │   ├── narval/                     ← Narval Slurm scripts (40 GB)
        │   └── nibi/                       ← Nibi Slurm scripts (H100)
        └── docs/
            ├── narval/GUIDE.md
            └── nibi/GUIDE.md
        ├── runs/                           ← training outputs
        └── README.md / docs/narval/GUIDE.md

/scratch/$USER/charades/                 ← DATA (large files)
├── archives/
│   ├── Charades.zip                        (~3 MB)
│   ├── Charades_v1_rgb.tar                 (~75 GB)
│   └── Charades_v1_flow.tar                (~45 GB)
├── annotations/
│   ├── Charades_v1_train.csv
│   ├── Charades_v1_test.csv
│   └── Charades_v1_classes.txt
├── torch_cache/                            ← cached ResNet-50 weights
└── charades_2stream_subset.tar             (~5 GB, created by build_subset.sh)
```

---

## 4. Storage strategy (important)

Narval has **three filesystems**. Using the wrong one causes failures.

| Filesystem | Path | Quota (your case) | Use for |
|------------|------|-------------------|---------|
| **Home** | `/home/$USER` | 50 GB, 500K files | venv, small configs |
| **Project** | `/project/def-hervete` | 1 TB shared, **500K files total** | code, checkpoints, logs |
| **Scratch** | `/scratch/$USER` | 20 TB, 1M files | dataset tarballs, caches |

### Rules we follow

1. **Never put the venv in `$PROJECT`** — a venv is ~5,000+ small files; your project was at **499K/500K files** and hit `Disk quota exceeded`.
2. **Keep raw dataset as tarballs on `$SCRATCH`** — do not unpack 120 GB into millions of files on Lustre.
3. **Preprocess once** → one compact ~5 GB tar (`charades_2stream_subset.tar`).
4. **Every GPU job** copies that small tar to `$SLURM_TMPDIR` (node-local SSD) and reads from there.

---

## 5. Step-by-step workflow

### Step 0 — Copy code to Narval (once, from your laptop)

```powershell
scp -r narval_charades <user>@narval.alliancecan.ca:~/
```

Then on Narval, move it into project space:

```bash
rm -rf ~/projects/def-hervete/$USER/two_stream_charades   # old folder, if empty/unused
mv ~/narval_charades ~/projects/def-hervete/$USER/
cd ~/projects/def-hervete/$USER/narval_charades
```

### Step 1 — Fix Windows line endings (once)

Files copied from Windows may have CRLF endings. Linux bash will fail with:

```
set: pipefail: invalid option name
```

Fix all shell scripts:

```bash
sed -i 's/\r$//' env/*.sh data/*.sh jobs/narval/*.sh jobs/nibi/*.sh
# or: dos2unix env/*.sh data/*.sh jobs/*.sh
```

### Step 2 — Verify Slurm account (once)

```bash
sacctmgr show user $USER withassoc format=account
```

Expected: `def-hervete` (already set in all `#SBATCH --account=` lines).

### Step 3 — Create Python environment (once, login node)

```bash
cd ~/projects/def-hervete/$USER/narval_charades
bash env/setup_env.sh
```

**What it does:**

- Loads `StdEnv/2023` + `python/3.11`
- Creates venv at `$HOME/charades_venv`
- Installs packages from Alliance wheelhouse (`pip install --no-index`)
- Downloads and caches ResNet-50 ImageNet weights to `$SCRATCH/charades/torch_cache`

**Activate later:**

```bash
module load StdEnv/2023 python/3.11
source ~/charades_venv/bin/activate
```

### Step 4 — Download dataset (once, login node, use tmux)

~120 GB total. Run inside `tmux` so it survives SSH disconnect:

```bash
tmux new -s dl
cd ~/projects/def-hervete/$USER/narval_charades
bash data/download_data.sh
```

- Detach: **Ctrl-b** then **d**
- Reattach: `tmux attach -t dl`
- Check progress: `ls -lh $SCRATCH/charades/archives/`

**Downloads:**

| File | Size | Content |
|------|------|---------|
| `Charades.zip` | ~3 MB | train/test CSVs + class names |
| `Charades_v1_rgb.tar` | ~75 GB | pre-extracted RGB frames |
| `Charades_v1_flow.tar` | ~45 GB | pre-extracted optical flow (x/y) |

### Step 5 — Build compact subset (once, Slurm CPU job)

After download completes:

```bash
cd ~/projects/def-hervete/$USER/narval_charades
sbatch data/build_subset.sh
squeue -u $USER
```

Monitor output:

```bash
tail -f runs/charades_subset-*.out
```

Verify result:

```bash
ls -lh $SCRATCH/charades/charades_2stream_subset.tar
```

**Why this step exists:** see [Section 6 — build_subset.sh](#build_subsetsh).

### Step 6 — Train models (Slurm GPU jobs)

Submit from the project directory:

```bash
cd ~/projects/def-hervete/$USER/narval_charades

sbatch jobs/narval/train_rgb.sh          # RGB baseline
sbatch jobs/narval/train_flow.sh         # flow baseline
sbatch jobs/narval/compare_fusion.sh     # average + weighted + concat (3 parallel jobs)
```

Monitor:

```bash
squeue -u $USER
tail -f runs/charades_rgb-*.out
tail -f runs/charades_flow-*.out
tail -f runs/charades_cmp-*.out
```

Each job:

1. Stages subset tar to `$SLURM_TMPDIR`
2. Trains for 25 epochs (mixed precision on GPU)
3. Saves best checkpoint by validation mAP
4. Runs evaluation on test split

### Step 7 — Compare results

```bash
cd ~/projects/def-hervete/$USER/narval_charades
module load StdEnv/2023 python/3.11
source ~/charades_venv/bin/activate
python src/summarize.py
```

Example output:

```
run                       best val mAP   test mAP
--------------------------------------------------
rgb                            0.xxxx     0.xxxx
flow                           0.xxxx     0.xxxx
fusion_average                 0.xxxx     0.xxxx
fusion_weighted                0.xxxx     0.xxxx
fusion_concat                  0.xxxx     0.xxxx
```

---

## 6. What each script does

### `env/setup_env.sh`

| | |
|---|---|
| **Where** | Login node |
| **Runs** | Once |
| **Input** | Internet (wheelhouse + PyTorch model weights) |
| **Output** | `$HOME/charades_venv`, `$SCRATCH/charades/torch_cache` |

### `data/download_data.sh`

| | |
|---|---|
| **Where** | Login node |
| **Runs** | Once |
| **Input** | AllenAI S3 URLs |
| **Output** | `$SCRATCH/charades/archives/*.tar`, `$SCRATCH/charades/annotations/` |

### `data/build_subset.sh`

| | |
|---|---|
| **Where** | Slurm compute node (CPU) |
| **Runs** | Once |
| **Resources** | 16 CPUs, 64 GB RAM, 4 h max |

**Pipeline:**

```
Charades_v1_rgb.tar  (75 GB)  ──┐
                                 ├── copy to $SLURM_TMPDIR (local SSD)
Charades_v1_flow.tar (45 GB)  ──┘
                                 │
                                 ▼
                          extract both tarballs
                                 │
                                 ▼
                    preprocess.py (16 frames/video, resize 256px)
                                 │
                                 ▼
              charades_2stream_subset.tar (~5 GB) → $SCRATCH
```

For each annotated video:

- Keep **16 evenly spaced** RGB frames
- Keep matching **flow x/y** pairs
- Resize so short side = **256 px**
- Preserve Charades naming convention for the DataLoader

### `jobs/narval/_common.sh`

Shared setup for all GPU jobs:

- Load modules + activate venv
- Set `TORCH_HOME` (pretrained weights, no internet on compute nodes)
- Copy + extract subset tar to `$SLURM_TMPDIR`
- Set `DATA_ROOT` and `ANNOT_DIR` environment variables

### `jobs/narval/train_rgb.sh` / `train_flow.sh` / `compare_fusion.sh`

| Script | Stream | Fusion |
|--------|--------|--------|
| `train_rgb.sh` | RGB only | — |
| `train_flow.sh` | Flow only | — |
| `train_fusion.sh` | Both | concat (default) |
| `compare_fusion.sh` | Both | average, weighted, concat (array) |

**Resources per GPU job:** 1 GPU, 8–10 CPUs, 48–64 GB RAM, 10–16 h.

### Python modules

| File | Role |
|------|------|
| `src/annotations.py` | Parse CSV → multi-hot labels [157] |
| `src/preprocess.py` | Subsample + resize frames (used by build_subset) |
| `src/dataset.py` | `CharadesTwoStream` PyTorch Dataset |
| `src/models.py` | SpatialStream, TemporalStream, TwoStreamFusion |
| `src/metrics.py` | Charades mAP |
| `src/train.py` | Training loop (rgb / flow / fusion) |
| `src/evaluate.py` | Test mAP, per-class AP, training curves |
| `src/summarize.py` | Table comparing all runs |

---

## 7. Training pipeline (technical)

### Data flow during training

```
$SCRATCH/charades/charades_2stream_subset.tar
        │
        │  (copy, ~seconds)
        ▼
$SLURM_TMPDIR/data/
├── rgb_frames/<video_id>/*.jpg
└── flow_frames/<video_id>/*x.jpg, *y.jpg
        │
        ▼
CharadesTwoStream Dataset
  • sample 8 RGB frames + 10 flow pairs
  • resize/crop to 224×224
  • augment (train only)
        │
        ▼
Model forward → logits [batch, 157]
        │
        ▼
BCEWithLogitsLoss  +  AdamW  +  cosine LR schedule
        │
        ▼
Validation mAP → save best.pth
```

### Train / val / test — two protocols (`--protocol`)

Choose with `--protocol` on `train.py` or `PROTOCOL=...` in Slurm jobs.
Outputs go to `runs/<protocol>/<model>/`.

| Protocol | Train on | Checkpoint selection | Final eval | val mAP vs test mAP |
|----------|----------|----------------------|------------|---------------------|
| **`train_val_test`** (default, recommended) | 80% official train | 20% holdout val | Official test | **Different** |
| **`train_test`** (simple baseline) | Full official train | Official test | Official test | **Same** (expected) |

```bash
# Recommended protocol
bash jobs/narval/submit_all.sh train_val_test

# Simple train/test protocol (your first runs)
bash jobs/narval/submit_all.sh train_test

# Both at once
bash jobs/narval/submit_both_protocols.sh

# Summarize
python src/summarize.py
python src/summarize.py --protocol train_val_test
```

For `train_val_test`, create the split file once:
```bash
python src/make_split.py --annot-dir $SCRATCH/charades/annotations
```

### Hyperparameters (defaults)

| Parameter | Value |
|-----------|-------|
| Epochs | 25 |
| Learning rate | 1e-4 |
| Batch size | 32 (rgb/flow), 24 (fusion) |
| RGB frames sampled | 8 |
| Flow frames sampled | 10 (20 channels) |
| Image size | 224×224 |
| Optimizer | AdamW (weight decay 1e-4) |
| Precision | FP16 mixed precision (AMP) |

### Fusion strategies

| Strategy | How logits are combined |
|----------|-------------------------|
| `average` | `(logits_rgb + logits_flow) / 2` |
| `weighted` | Learnable weights via sigmoid, normalized |
| `concat` | Concatenate 512-d features → MLP → 157 logits |

---

## 8. Expected outputs

After training, each run folder under `runs/` contains:

```
runs/rgb/
├── best.pth              ← best checkpoint (by val mAP)
├── history.json          ← train/val loss and mAP per epoch
├── per_class_ap.json     ← test mAP + AP per class
└── curves.png            ← loss and mAP plots

runs/flow/
runs/fusion_average/
runs/fusion_weighted/
runs/fusion_concat/
```

Slurm stdout logs:

```
runs/charades_rgb-<jobid>.out
runs/charades_flow-<jobid>.out
runs/charades_cmp-<array_jobid>_<taskid>.out
runs/charades_subset-<jobid>.out
```

---

## 9. Troubleshooting

### `set: pipefail: invalid option name`

**Cause:** Windows CRLF line endings in `.sh` files.  
**Fix:** `sed -i 's/\r$//' env/*.sh data/*.sh jobs/narval/*.sh jobs/nibi/*.sh`

### `Disk quota exceeded` during venv install

**Cause:** Project filesystem file-count limit (500K files shared across lab).  
**Fix:** Put venv in `$HOME`:

```bash
rm -rf ~/projects/def-hervete/$USER/charades_venv
sed -i 's|VENV_DIR="$PROJECT_DIR/charades_venv"|VENV_DIR="$HOME/charades_venv"|' env/setup_env.sh
sed -i 's|source "$PROJECT_DIR/charades_venv/bin/activate"|source "$HOME/charades_venv/bin/activate"|' jobs/narval/_common.sh data/build_subset.sh
bash env/setup_env.sh
```

### Job fails: `charades_2stream_subset.tar` not found

**Cause:** `build_subset.sh` not finished or failed.  
**Fix:** Check `runs/charades_subset-*.out`, re-run `sbatch data/build_subset.sh`.

### Job fails: cannot download ResNet weights

**Cause:** `TORCH_HOME` not set or cache missing.  
**Fix:** Re-run `bash env/setup_env.sh` on login node; verify:

```bash
ls $SCRATCH/charades/torch_cache/hub/checkpoints/
```

### Out of GPU memory (OOM)

Lower batch size in the job script, e.g. `--batch-size 16` or `--batch-size 12`.

### Check disk usage anytime

```bash
diskusage_report
```

---

## 10. Comparison with the Colab notebook

| Aspect | Colab (`two_stream_charades.ipynb`) | Narval (`narval_charades`) |
|--------|-------------------------------------|----------------------------|
| Storage | Google Drive | `$SCRATCH` + `$SLURM_TMPDIR` |
| Downloads | `RemoteZip` streaming | `wget` on login node |
| Optical flow | Computed with OpenCV Farneback | Pre-extracted AllenAI flow tar |
| Task | Top-20 **single-label**, accuracy | Full **157-class multi-label**, mAP |
| Loss | CrossEntropy + class weights | BCEWithLogits + pos_weight |
| Execution | Run all cells interactively | Slurm batch jobs |
| Internet on compute | Yes | **No** — weights must be pre-cached |
| Filesystem | Drive (OK for many small files) | Lustre — avoid millions of small files |

---

## 11. Quick command cheat sheet

```bash
# --- Always start here ---
cd ~/projects/def-hervete/$USER/narval_charades

# --- Environment ---
module load StdEnv/2023 python/3.11
source ~/charades_venv/bin/activate

# --- One-time setup ---
sed -i 's/\r$//' env/*.sh data/*.sh jobs/narval/*.sh jobs/nibi/*.sh
bash env/setup_env.sh

# --- Download (login node, tmux) ---
tmux new -s dl
bash data/download_data.sh          # Ctrl-b d to detach

# --- Preprocess (Slurm) ---
sbatch data/build_subset.sh

# --- Train (Slurm) ---
sbatch jobs/narval/train_rgb.sh
sbatch jobs/narval/train_flow.sh
sbatch jobs/narval/compare_fusion.sh

# --- Monitor ---
squeue -u $USER
tail -f runs/charades_rgb-*.out
ls -lh $SCRATCH/charades/archives/
ls -lh $SCRATCH/charades/charades_2stream_subset.tar

# --- Results ---
python src/summarize.py
```

---

## Workflow diagram (full pipeline)

```
[Laptop]  scp narval_charades → Narval
                │
                ▼
[Login node]  setup_env.sh          → venv + torch cache
                │
                ▼
[Login node]  download_data.sh      → $SCRATCH/archives (120 GB)
                │
                ▼
[Slurm CPU]   build_subset.sh       → charades_2stream_subset.tar (~5 GB)
                │
                ▼
[Slurm GPU]   train_rgb.sh          → runs/rgb/
[Slurm GPU]   train_flow.sh         → runs/flow/
[Slurm GPU]   compare_fusion.sh     → runs/fusion_*/
                │
                ▼
[Login node]  summarize.py          → mAP comparison table
```

---

*Document generated for the Mitacs project — Two-Stream Networks on Charades, Narval cluster.*
