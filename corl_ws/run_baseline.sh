#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="$ROOT/runtime/libero_config"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/cyd/lerobot/src
exec "$ROOT/runtime/.venv/bin/python" "$ROOT/scripts/run_baseline.py" "$@"
