#!/usr/bin/env bash
# Source this file: source /home/cyd/ca_repro/activate_libero.sh
CA_REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CA_REPRO_DIR/.venv/bin/activate"
export LIBERO_CONFIG_PATH="$CA_REPRO_DIR/libero_config"
export MUJOCO_GL=egl
unset CA_REPRO_DIR
