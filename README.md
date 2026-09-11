# Charades — Two-Stream Action Recognition (HPC)

Multi-label action recognition on **Charades** (157 classes): CNN two-stream
baselines, VideoMAE V2 fine-tuning, and two-stream VideoMAE (RGB + optical flow).

Runs on **Alliance Canada** clusters. Shared code in `src/` and `data/`;
cluster-specific Slurm jobs in `jobs/<cluster>/`.

Set `#SBATCH --account=` in job scripts to **your** Alliance allocation
(`def-hervete` / `aip-hervete` are this lab’s accounts).

Python packages in `requirements.txt` are unpinned so they match the Alliance
wheelhouse (`pip install --no-index`). For a laptop install, pin versions after
a successful run.

---

## Project layout

```
narval_charades/
├── src/                  # Python — shared by all clusters
├── data/                 # Download + subset build scripts
├── env/setup_env.sh      # One-time venv (any cluster login node)
├── jobs/
│   ├── README.md         # Job index
│   ├── fir/              # H100 80 GB
│   ├── narval/           # A100 40 GB
│   ├── nibi/             # H100
│   └── tamia/            # full-node H100 / H200 (AIP)
├── docs/
│   ├── narval/           # Narval setup + VideoMAE guide
│   ├── nibi/             # Nibi setup (two-stream VideoMAE)
│   └── tamia/            # tamIA setup
├── reports/              # LaTeX notes (source only)
├── runs/                 # Training outputs (gitignored)
├── LICENSE
└── CODE_DOCUMENTATION.md # Full code reference
```

---

## Which cluster?

| | **Fir** | **Narval** | **Nibi** | **tamIA** |
|---|---------|------------|----------|-----------|
| GPU | H100 80 GB | A100 40 GB | H100 | 4× H100 or 8× H200 |
| Best for | CNN + VideoMAE | CNN baselines, RGB VideoMAE | two-stream VideoMAE, 32–64 frames | multi-GPU / distill |
| Jobs | `jobs/fir/` | `jobs/narval/` | `jobs/nibi/` | `jobs/tamia/` |
| Guide | — | [docs/narval/GUIDE.md](docs/narval/GUIDE.md) | [docs/nibi/GUIDE.md](docs/nibi/GUIDE.md) | [docs/tamia/GUIDE.md](docs/tamia/GUIDE.md) |

---

## Quick start

### 1. Setup (once per cluster)

```bash
bash env/setup_env.sh
bash data/download_data.sh
sbatch data/build_subset.sh
bash data/download_videomae_ckpt.sh
```

### 2. Fir — recommended H100

```bash
bash jobs/fir/submit_all.sh train_test
sbatch jobs/fir/train_videomae.sh
```

### 3. Narval — CNN + RGB VideoMAE

```bash
sbatch jobs/narval/train_rgb.sh
sbatch jobs/narval/train_videomae.sh
bash jobs/narval/submit_all.sh train_test
```

### 4. Nibi — two-stream VideoMAE

```bash
sbatch jobs/nibi/train_videomae_twostream.sh
sbatch jobs/nibi/sweep_videomae_twostream_frames.sh
```

### 5. tamIA — multi-GPU

```bash
bash jobs/tamia/submit_priority.sh
```

### 6. Results

```bash
python src/summarize.py
```

---

## Models

| Model | Input | Scripts |
|-------|-------|---------|
| ResNet two-stream | RGB + flow JPEGs | `train.py` · `jobs/narval/train_*.sh` |
| VideoMAE RGB | RGB clip | `train_videomae.py` · `jobs/*/train_videomae*.sh` |
| VideoMAE two-stream | RGB + flow clips | `train_videomae_twostream.py` · `jobs/*/train_videomae_twostream.sh` |

Flow is **precomputed** (AllenAI Charades tarballs) — no RAFT extraction.

---

## Documentation

- [jobs/README.md](jobs/README.md) — Slurm job index
- [docs/narval/GUIDE.md](docs/narval/GUIDE.md) — full Narval walkthrough
- [docs/narval/VIDEOMAE.md](docs/narval/VIDEOMAE.md) — VideoMAE on Narval
- [docs/nibi/GUIDE.md](docs/nibi/GUIDE.md) — Nibi + two-stream VideoMAE
- [docs/tamia/GUIDE.md](docs/tamia/GUIDE.md) — tamIA (AIP allocation)
- [CODE_DOCUMENTATION.md](CODE_DOCUMENTATION.md) — code deep-dive

---

*Mitacs project — Prof. Éric Hervet · Asma El Hadj*
