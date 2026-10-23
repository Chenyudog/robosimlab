#!/usr/bin/env bash
set -euo pipefail
cd /home/cyd/pi0_representation
export PYTHONPATH=/home/cyd/lerobot/src
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
PY=/home/cyd/miniforge3/envs/lerobot/bin/python
"$PY" extract_pretrained.py "$@"
"$PY" plot_features.py outputs/pretrained_goal
