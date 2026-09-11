#!/bin/bash
# Submit ONLY the tamIA jobs that add new information over the Nibi results
# (best so far: f64_ddp2_ft_lr5e5_e100, test wtd 0.5240).
#
# Skipped on purpose: f16/f32 LR & epoch sweeps (saturated ~0.50-0.517 on
# Nibi; results transfer across clusters).
#
#   bash jobs/tamia/submit_useful.sh          # job 1 only
#   bash jobs/tamia/submit_useful.sh --f96    # also submit the f96 probe
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

SCRIPT=jobs/tamia/train_videomae_twostream_8gpu_h200.sh

echo "================================================================"
echo " tamIA useful jobs — baseline to beat: f64 ft, test wtd 0.5240"
echo "================================================================"

# 1) f64 from-scratch, winning recipe (eff batch 32, lr7e4, e75) but on
#    8x H200: batch 4/GPU without gradient checkpointing. On Nibi this
#    needed 4x H100 + batch 1 + grad ckpt; here each epoch is ~8x faster,
#    so the 75-epoch schedule fits inside tamIA's 24h wall.
J1=$(sbatch --parsable \
  --export=ALL,NUM_FRAMES=64,BATCH_SIZE=4,GRAD_ACCUM_STEPS=1,GRAD_CHECKPOINT=1,FRAMES_FPS=6,EPOCHS=75,LR=7e-4,WARMUP_EPOCHS=5,RUN_SUFFIX=_tamia_h200x8_e75 \
  "$SCRIPT")
echo "  [1] f64 e75 lr7e4 8xH200      -> job $J1"

# 2) OPTIONAL: f96 probe — new frame-count axis that 141GB enables.
#    Needs the 12 fps subset tar (f96 @ 12fps = 8s clips):
#    $SCRATCH/charades/charades_2stream_subset_12fps.tar
if [[ "${1:-}" == "--f96" ]]; then
  if [[ -f "$SCRATCH/charades/charades_2stream_subset_12fps.tar" ]]; then
    J2=$(sbatch --parsable \
      --export=ALL,NUM_FRAMES=96,BATCH_SIZE=2,GRAD_ACCUM_STEPS=2,GRAD_CHECKPOINT=1,FRAMES_FPS=12,EPOCHS=50,LR=7e-4,WARMUP_EPOCHS=5,RUN_SUFFIX=_tamia_h200x8_e50 \
      "$SCRIPT")
    echo "  [2] f96 probe e50 8xH200      -> job $J2"
  else
    echo "  [2] SKIP f96 (missing 12fps subset tar; rsync it or run data/build_subset_fps.sh)"
  fi
fi

echo ""
echo "After [1] finishes, the historically best follow-up is a low-LR"
echo "fine-tune of its best.pth (RESUME=... LR=5e-5 EPOCHS=100),"
echo "same as the run that produced 0.5240."
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Results:  python src/summarize.py --protocol train_test"
echo "================================================================"
