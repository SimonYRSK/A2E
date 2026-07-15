#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible wrapper.
# Historically this script trained and evaluated one experiment immediately.
# The pipeline is now split for better GPU utilization:
#   - train only: bash A2E/scripts/train_interface.sh <phase>
#   - eval only:  bash A2E/scripts/eval_interface.sh <phase>
#
# This wrapper keeps old commands working by dispatching to train_interface.sh.

echo "[DEPRECATED] run_interface.sh is now train-only for compatibility."
echo "[DEPRECATED] Use train_interface.sh for training and eval_interface.sh for evaluation."

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "${SCRIPT_DIR}/train_interface.sh" "$@"
