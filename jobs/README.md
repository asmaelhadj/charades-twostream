# Slurm jobs by cluster

Shared Python code lives in `src/`. Cluster-specific batch scripts live here.

Replace `#SBATCH --account=` with your Alliance allocation before submitting
(`def-hervete` on most clusters, `aip-hervete` on tamIA).

```
jobs/
├── fir/             # H100 80 GB — recommended cluster
│   ├── _common.sh
│   ├── train_rgb.sh / train_flow.sh / compare_fusion.sh
│   ├── sweep_frames_controlled.sh / sweep_frames_dense.sh
│   ├── train_videomae.sh / train_videomae_twostream.sh
│   └── submit_*.sh
├── narval/          # 40 GB A100 — legacy
│   └── ...
├── nibi/            # H100 — VideoMAE two-stream sweeps
│   └── ...
└── tamia/           # nœuds entiers 4x H100 ou 8x H200 — compte aip-hervete
    ├── _common.sh
    ├── train_videomae_twostream_4gpu.sh
    ├── train_videomae_twostream_8gpu_h200.sh
    ├── train_videomae_rgb_8gpu_h200.sh
    ├── train_videomae_distill_h200.sh   # ViT-B teacher → ViT-S student
    ├── eval_multiview_twostream.sh
    └── submit_priority.sh
```

## Quick start

**Fir** (H100 80 GB — use this cluster):

```bash
bash jobs/fir/submit_all.sh train_test
bash jobs/fir/submit_frames_dense.sh build && bash jobs/fir/submit_frames_dense.sh train
sbatch jobs/fir/train_videomae.sh
```

**Narval** (CNN baselines + RGB VideoMAE):

```bash
sbatch jobs/narval/train_rgb.sh
sbatch jobs/narval/train_videomae.sh
bash jobs/narval/submit_all.sh train_test
```

**Nibi** (two-stream VideoMAE, 32–64 frames):

```bash
sbatch jobs/nibi/train_videomae_twostream.sh
sbatch jobs/nibi/sweep_videomae_twostream_frames.sh
```

**tamIA** (nouveau — allocation AIP `aip-hervete`, nœuds entiers, max 24 h):

```bash
bash jobs/tamia/submit_priority.sh                        # FT + multi-view + RGB
bash jobs/tamia/submit_useful.sh                          # only jobs that add new info
sbatch jobs/tamia/train_videomae_twostream_8gpu_h200.sh   # 8x H200 141 GB
sbatch jobs/tamia/train_videomae_twostream_4gpu.sh        # 4x H100 80 GB
sbatch jobs/tamia/train_videomae_rgb_8gpu_h200.sh         # RGB VideoGLUE-style
MODEL=vit_s bash data/download_videomae_ckpt.sh           # login: student init
sbatch jobs/tamia/train_videomae_distill_h200.sh          # distill → ViT-S RGB
```

Guides: [`docs/narval/GUIDE.md`](../docs/narval/GUIDE.md) · [`docs/nibi/GUIDE.md`](../docs/nibi/GUIDE.md) · [`docs/nibi/REPRODUCE.md`](../docs/nibi/REPRODUCE.md) (epoch-1 recipe) · [`docs/tamia/GUIDE.md`](../docs/tamia/GUIDE.md)
