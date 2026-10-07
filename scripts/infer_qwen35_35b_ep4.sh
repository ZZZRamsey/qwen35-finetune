#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
exec "${QWEN_PYTHON:-python}" -B "$ROOT/scripts/launch.py" infer "35b" "$@"
