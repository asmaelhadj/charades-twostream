#!/bin/bash
# Submit train_test protocol: rgb + flow + fusion comparison.
#   bash jobs/narval/submit_train_test.sh
#   bash jobs/narval/submit_train_test.sh --sweep
set -euo pipefail

export PROTOCOL=train_test

echo "=== train_test: rgb + flow + fusion ==="
sbatch jobs/narval/train_rgb.sh
sbatch jobs/narval/train_flow.sh
sbatch jobs/narval/compare_fusion.sh

if [[ "${1:-}" == "--sweep" ]]; then
  sbatch jobs/narval/sweep_frames_controlled.sh
  echo "Optional: SWEEP=flow sbatch jobs/narval/sweep_frames_controlled.sh"
fi

echo "Summarize: python src/summarize.py --protocol train_test"
