#!/bin/bash
# Submit CNN baselines on Fir (H100).
#   bash jobs/fir/submit_all.sh train_test
set -euo pipefail
export PROTOCOL="${1:-train_test}"
echo "==> Fir | PROTOCOL=$PROTOCOL"
sbatch jobs/fir/train_rgb.sh
sbatch jobs/fir/train_flow.sh
sbatch jobs/fir/compare_fusion.sh
echo "Monitor: squeue -u \$USER"
