#!/usr/bin/env bash
set -euo pipefail

# Bash wrapper for the cross-platform Python implementation.
# Usage examples:
#   bash scripts/train_prelabeler.sh --mode train
#   bash scripts/train_prelabeler.sh --mode predict --source data/frames_selected
#   bash scripts/train_prelabeler.sh --mode export

python scripts/train_prelabeler.py "$@"
