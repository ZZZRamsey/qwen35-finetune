#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
# Defaults are the verified 2-node x 4-GPU layout (world size 8).
export QWEN_NNODES="${QWEN_NNODES:-2}"
export QWEN_NPROC="${QWEN_NPROC:-4}"
export QWEN_CONFIG="${QWEN_CONFIG:-$ROOT/configs/qwen35_4b_full.yaml}"
export QWEN_RUNS_ROOT="${QWEN_RUNS_ROOT:-$ROOT/logs}"
exec "${QWEN_PYTHON:-python}" -B "$ROOT/scripts/launch.py" train 4b "$@"
