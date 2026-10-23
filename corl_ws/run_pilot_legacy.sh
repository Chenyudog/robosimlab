#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VLA_LEGACY_PI0=1
export VLA_LEROBOT_SRC="$ROOT/third_party/lerobot_v044/src"
export VLA_RUN_PREFIX="${VLA_RUN_PREFIX:-legacy_aligned}"
export PYTHONPATH="$VLA_LEROBOT_SRC"
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="$ROOT/runtime/libero_config"
export HF_HUB_OFFLINE=1
exec "$ROOT/runtime/legacy_pi0/bin/python" -u "$ROOT/scripts/run_finetuned_pilot.py"
