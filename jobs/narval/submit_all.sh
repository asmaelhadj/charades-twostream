#!/bin/bash
# Submit all CNN training jobs for one protocol.
#   bash jobs/narval/submit_all.sh train_test
set -euo pipefail

PROTOCOL="${1:-train_test}"
export PROTOCOL

echo "Submitting Narval jobs with PROTOCOL=$PROTOCOL"
sbatch jobs/narval/train_rgb.sh
sbatch jobs/narval/train_flow.sh
sbatch jobs/narval/compare_fusion.sh
echo "Done. Monitor with: squeue -u \$USER"
