#!/bin/bash
# Submit both split protocols.
#   bash jobs/narval/submit_both_protocols.sh
set -euo pipefail

bash jobs/narval/submit_all.sh train_val_test
bash jobs/narval/submit_all.sh train_test
echo "Both protocol batches submitted."
